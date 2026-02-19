from __future__ import annotations

import difflib
import math
import re
import threading
import time
import wave
from pathlib import Path

import numpy as np
from PySide6.QtCore import QObject, QPointF, QRectF, Qt, QThread, QTimer, Signal
from PySide6.QtGui import QCloseEvent, QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from pyxis_assistant.assistant import finalize_event_from_json, process_user_text
from pyxis_assistant.calendar import authenticate_google_calendar
from pyxis_assistant.logging_utils import get_logger
from pyxis_assistant.nlp import transcribe_audio
from pyxis_assistant.profiles import (
    default_profile_name,
    get_profile_api_key,
    list_profile_names,
    set_profile_api_key,
)
from pyxis_assistant.voice import LocalWakeVADListener, VoiceRecorder, VoiceSpeaker


class AssistantWorker(QObject):
    status = Signal(str)
    transcript = Signal(str)
    log = Signal(str)
    finished = Signal(bool, str)

    def __init__(self, listen_seconds: float = 3.0) -> None:
        super().__init__()
        self.listen_seconds = listen_seconds
        self.recorder = VoiceRecorder()
        self.local_listener = LocalWakeVADListener()
        self.speaker: VoiceSpeaker | None = None
        self.running = True
        self.conversation_active = False
        self.pending_event_json: str | None = None
        self.logger = get_logger("pyxis.worker")
        self._last_wake_log_time = 0.0
        self._last_wake_score = -1.0
        self._suppress_transcribe_until = 0.0
        self._last_spoken_text = ""
        self._tts_thread: threading.Thread | None = None
        self._wake_variants = {
            "pyxis",
            "pixis",
            "piksis",
            "pike sis",
            "pikesis",
            "pike-sis",
            "pick sis",
            "pyxus",
        }

    def stop(self) -> None:
        self.running = False
        if self.speaker is not None:
            self.speaker.stop()
        self.local_listener.stop()

    def spectrum(self) -> np.ndarray:
        if self.local_listener.available:
            return self.local_listener.spectrum()
        return self.recorder.spectrum()

    def _audio_has_speech(self, path: Path, threshold: float = 500.0) -> bool:
        try:
            with wave.open(str(path), "rb") as wav:
                frames = wav.readframes(wav.getnframes())
            if not frames:
                return False
            data = np.frombuffer(frames, dtype=np.int16).astype(np.float32)
            if data.size == 0:
                return False
            rms = float(np.sqrt(np.mean(np.square(data))))
            return rms >= threshold
        except Exception:
            return True

    def _wake_word_detected(self, transcript_text: str) -> bool:
        normalized = "".join(
            ch.lower() if ch.isalnum() or ch.isspace() else " " for ch in transcript_text
        )
        normalized = " ".join(normalized.split())
        if not normalized:
            return False

        for variant in self._wake_variants:
            if variant in normalized:
                return True

        for token in normalized.split():
            ratio = difflib.SequenceMatcher(a=token, b="pyxis").ratio()
            if ratio >= 0.76:
                return True
        return False

    def _is_confirmation_yes(self, text: str) -> bool:
        normalized = re.sub(r"[^a-z0-9\s]", " ", text.lower())
        normalized = " ".join(normalized.split())
        words = normalized.split()
        if words and words[0] in {"yes", "yep", "yeah"} and len(words) <= 4:
            return True
        yes_phrases = {
            "yes",
            "yep",
            "yeah",
            "yes please",
            "yep please",
            "yeah please",
            "confirm",
            "confirmed",
            "please confirm",
            "do it",
            "go ahead",
            "please do",
        }
        return normalized in yes_phrases

    def _is_confirmation_no(self, text: str) -> bool:
        normalized = re.sub(r"[^a-z0-9\s]", " ", text.lower())
        normalized = " ".join(normalized.split())
        words = normalized.split()
        if words and words[0] == "no" and len(words) <= 4:
            return True
        no_phrases = {
            "no",
            "nope",
            "no thanks",
            "cancel",
            "stop",
            "dont do it",
            "do not do it",
            "not now",
        }
        return normalized in no_phrases

    def _emit_log(self, message: str) -> None:
        self.log.emit(message)
        self.logger.info(message)

    def _ensure_speaker(self) -> VoiceSpeaker:
        if self.speaker is None:
            # Initialize TTS engine in the worker thread to avoid COM/thread affinity issues.
            self.speaker = VoiceSpeaker()
        return self.speaker

    def _on_wake_score(self, score: float, source: str, threshold: float) -> None:
        now = time.monotonic()
        should_log = False
        if score >= max(0.2, threshold * 0.6):
            should_log = True
        elif now - self._last_wake_log_time >= 5.0:
            should_log = True
        elif abs(score - self._last_wake_score) >= 0.12:
            should_log = True

        if not should_log:
            return

        if score >= threshold:
            message = (
                f"Wake candidate: source={source}, score={score:.2f}, "
                f"threshold={threshold:.2f} (awaiting stability checks)"
            )
        else:
            message = f"Wake rejected: source={source}, score={score:.2f}, threshold={threshold:.2f}"
        self._last_wake_log_time = now
        self._last_wake_score = score
        self._emit_log(message)

    def _speak_and_suppress(self, text: str) -> None:
        self._last_spoken_text = text.strip().lower()
        # Keep suppression short and allow barge-in while speaking.
        estimated = self._estimate_tts_seconds(text)
        self._suppress_transcribe_until = time.monotonic() + min(1.1, 0.22 + estimated * 0.25)
        if self.local_listener.available:
            self.local_listener.clear_queue()
        speaker = self._ensure_speaker()
        if self._tts_thread is not None and self._tts_thread.is_alive():
            speaker.stop()
            self._tts_thread.join(timeout=0.3)
        self._tts_thread = threading.Thread(target=speaker.speak, args=(text,), daemon=True)
        self._tts_thread.start()

    def _is_speaking(self) -> bool:
        speaker = self.speaker
        return bool(speaker is not None and speaker.is_speaking())

    def _estimate_tts_seconds(self, text: str) -> float:
        words = max(1, len(text.split()))
        # ~170 wpm plus buffer for synthesis/device latency.
        return min(8.0, 0.35 + (words / 2.85))

    def _looks_like_self_echo(self, transcript_text: str) -> bool:
        cleaned = transcript_text.strip().lower()
        if not cleaned or not self._last_spoken_text:
            return False
        ratio = difflib.SequenceMatcher(a=cleaned, b=self._last_spoken_text).ratio()
        return ratio >= 0.72

    def _handle_pending_confirmation(self, transcript_text: str) -> bool:
        if not self.pending_event_json:
            return False

        if self._is_confirmation_yes(transcript_text):
            ok, detail = finalize_event_from_json(self.pending_event_json)
            self.pending_event_json = None
            if ok:
                message = f"Event created: {detail or '(no link returned)'}"
                self.status.emit(message)
                self._emit_log(message)
                self._speak_and_suppress("Event confirmed and added.")
            else:
                message = f"I couldn't add that event: {detail}"
                self.status.emit(message)
                self._emit_log(message)
                self._speak_and_suppress(message)
            return True

        if self._is_confirmation_no(transcript_text):
            self.pending_event_json = None
            message = "Event cancelled. We can keep chatting."
            self.status.emit(message)
            self._emit_log(message)
            self._speak_and_suppress(message)
            return True

        prompt = "Please say yes to confirm the event, or no to cancel."
        self.status.emit(prompt)
        self._emit_log(prompt)
        self._speak_and_suppress(prompt)
        return True

    def run(self) -> None:
        if self.local_listener.available:
            self.status.emit("Local wake-word + VAD enabled. Say 'pyxis' to wake me.")
            self._emit_log("Wake mode: local openWakeWord + webrtcvad")
            try:
                self.local_listener.start()
            except Exception as exc:
                self.status.emit(
                    f"Local wake listener failed. Falling back to cloud wake. Error: {exc}"
                )
                self._emit_log(f"Local wake startup failed: {exc}")
                self.local_listener.available = False

        if self.local_listener.available:
            self._run_local_wake_loop()
        else:
            self.status.emit(
                "Cloud transcript wake enabled. Say 'pyxis' to wake me. "
                "Install openwakeword for lower latency local wake."
            )
            self._emit_log("Wake mode: cloud transcript fallback")
            self._run_fallback_loop()

        self.finished.emit(True, "Background listening stopped.")

    def _run_local_wake_loop(self) -> None:
        while self.running:
            audio_path: Path | None = None
            try:
                if not self.conversation_active:
                    woke = self.local_listener.wait_for_wake_word_with_feedback(
                        lambda: self.running,
                        on_score=self._on_wake_score,
                    )
                    if not woke:
                        continue
                    greeting = "Hey, I'm Pyxis. What do you need?"
                    self.status.emit(greeting)
                    self._emit_log("Wake word detected (local).")
                    self._speak_and_suppress(greeting)
                    self.conversation_active = True
                    continue

                if time.monotonic() < self._suppress_transcribe_until:
                    if self._is_speaking() and self.local_listener.detect_barge_in(
                        lambda: self.running
                    ):
                        self._emit_log("Barge-in detected: interrupting TTS.")
                        if self.speaker is not None:
                            self.speaker.stop()
                        self._suppress_transcribe_until = 0.0
                        self.local_listener.clear_queue()
                        continue
                    # During active suppression, flush playback bleed.
                    if not self._is_speaking():
                        self.local_listener.clear_queue()
                    time.sleep(0.05)
                    continue

                audio_path = self.local_listener.capture_utterance(
                    lambda: self.running,
                    max_seconds=12.0,
                    min_seconds=0.7,
                    silence_seconds=1.35,
                    pre_roll_seconds=0.35,
                )
                if audio_path is None:
                    continue

                try:
                    transcript_text = transcribe_audio(audio_path)
                except ValueError as exc:
                    if "empty text" in str(exc).lower():
                        continue
                    raise
                if not transcript_text.strip():
                    continue
                if self._looks_like_self_echo(transcript_text):
                    self._emit_log(f"Ignored self-echo transcript: {transcript_text}")
                    continue

                self.transcript.emit(transcript_text)
                self._emit_log(f"Heard: {transcript_text}")

                if self._handle_pending_confirmation(transcript_text):
                    continue

                self.status.emit("Thinking...")
                result = process_user_text(transcript_text.strip())
                self.status.emit(result.text)
                self._emit_log(f"Assistant: {result.text}")
                self._speak_and_suppress(result.text)

                if result.requires_event_confirmation and result.pending_event_json:
                    self.pending_event_json = result.pending_event_json

                if result.end_conversation:
                    self.conversation_active = False
                    self.status.emit("Conversation ended. Say 'pyxis' to wake me again.")
            except Exception as exc:
                self.status.emit(f"Listening error: {exc}")
                self._emit_log(f"ERROR listening: {exc}")
                self.logger.exception("Local wake loop error")
            finally:
                if audio_path and audio_path.exists():
                    audio_path.unlink(missing_ok=True)

        self.local_listener.stop()

    def _run_fallback_loop(self) -> None:
        while self.running:
            audio_path: Path | None = None
            try:
                idle_window = 1.2
                active_window = 4.2
                window = active_window if self.conversation_active else idle_window
                if time.monotonic() < self._suppress_transcribe_until:
                    if self._is_speaking() and self.local_listener.available and self.local_listener.detect_barge_in(
                        lambda: self.running
                    ):
                        self._emit_log("Barge-in detected: interrupting TTS.")
                        if self.speaker is not None:
                            self.speaker.stop()
                        self._suppress_transcribe_until = 0.0
                        continue
                    time.sleep(0.05)
                    continue
                audio_path = self.recorder.record_for(window)
                if not self._audio_has_speech(audio_path):
                    continue

                try:
                    transcript_text = transcribe_audio(audio_path)
                except ValueError as exc:
                    if "empty text" in str(exc).lower():
                        continue
                    raise
                if not transcript_text.strip():
                    continue
                if self._looks_like_self_echo(transcript_text):
                    self._emit_log(f"Ignored self-echo transcript: {transcript_text}")
                    continue

                self.transcript.emit(transcript_text)
                self._emit_log(f"Heard: {transcript_text}")

                if not self.conversation_active:
                    if not self._wake_word_detected(transcript_text):
                        continue
                    greeting = "Hey, I'm Pyxis. What do you need?"
                    self.status.emit(greeting)
                    self._emit_log("Wake word detected (fallback).")
                    self._speak_and_suppress(greeting)
                    self.conversation_active = True
                    continue

                if self._handle_pending_confirmation(transcript_text):
                    continue

                self.status.emit("Thinking...")
                result = process_user_text(transcript_text.strip())
                self.status.emit(result.text)
                self._emit_log(f"Assistant: {result.text}")
                self._speak_and_suppress(result.text)

                if result.requires_event_confirmation and result.pending_event_json:
                    self.pending_event_json = result.pending_event_json

                if result.end_conversation:
                    self.conversation_active = False
                    self.status.emit("Conversation ended. Say 'pyxis' to wake me again.")
            except Exception as exc:
                self.status.emit(f"Listening error: {exc}")
                self._emit_log(f"ERROR listening: {exc}")
                self.logger.exception("Fallback wake loop error")
            finally:
                if audio_path and audio_path.exists():
                    audio_path.unlink(missing_ok=True)


class SpectrumWidget(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._values = np.zeros(48, dtype=np.float32)
        self._phase = 0.0
        self.listening = False

        self.setMinimumSize(320, 320)
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(33)

    def _tick(self) -> None:
        self._phase += 0.05
        self.update()

    def set_values(self, values: np.ndarray) -> None:
        if values.size == 0:
            return
        self._values = values.astype(np.float32, copy=True)
        self.update()

    def set_listening(self, listening: bool) -> None:
        self.listening = listening
        if not listening:
            self._values = np.zeros_like(self._values)
        self.update()

    def paintEvent(self, event: object) -> None:
        del event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        rect = self.rect()
        center = QPointF(float(rect.center().x()), float(rect.center().y()))
        radius = min(rect.width(), rect.height()) * 0.24
        bars = int(self._values.size)

        painter.fillRect(rect, QColor(13, 18, 34))
        painter.setPen(QPen(QColor(30, 40, 70), 2))
        painter.setBrush(QColor(18, 26, 48))
        painter.drawEllipse(center, radius + 64, radius + 64)

        for i in range(bars):
            angle = (2.0 * math.pi * i / bars) + self._phase
            value = float(self._values[i])
            idle = 0.18 + (0.08 * math.sin(self._phase + (i * 0.35)))
            magnitude = max(value, idle if not self.listening else 0.05)

            inner = radius + 24
            outer = inner + (magnitude * 72.0)
            start = QPointF(
                center.x() + (inner * math.cos(angle)),
                center.y() + (inner * math.sin(angle)),
            )
            end = QPointF(
                center.x() + (outer * math.cos(angle)),
                center.y() + (outer * math.sin(angle)),
            )

            color = QColor(78, 191, 255) if self.listening else QColor(89, 145, 200)
            painter.setPen(QPen(color, 3, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
            painter.drawLine(start, end)

        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(21, 33, 58))
        painter.drawEllipse(center, radius, radius)

        painter.setPen(QColor(210, 230, 255))
        painter.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
        label = "LISTENING" if self.listening else "READY"
        painter.drawText(
            QRectF(center.x() - 70, center.y() - 14, 140, 28), Qt.AlignmentFlag.AlignCenter, label
        )


class PyxisWindow(QWidget):
    def __init__(
        self, force_redefine_api_key: bool = False, preferred_profile: str | None = None
    ) -> None:
        super().__init__()
        self.force_redefine_api_key = force_redefine_api_key
        self.preferred_profile = preferred_profile
        self.setWindowTitle("Pyxis Assistant")
        self.setFixedSize(420, 700)
        self.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint, True)

        self.spectrum = SpectrumWidget(self)
        self.status_label = QLabel("Idle")
        self.status_label.setWordWrap(True)
        self.transcript_label = QLabel("Transcript will appear here.")
        self.transcript_label.setWordWrap(True)
        self.transcript_label.setStyleSheet("color: #A9BEDC;")
        self.log_view = QTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMinimumHeight(180)
        self.log_view.setStyleSheet("color: #D3E3FF; background-color: #101b31;")

        self.listen_button = QPushButton("Start Background Listening")
        self.listen_button.clicked.connect(self.toggle_listening)

        self.auth_button = QPushButton("Authenticate Google")
        self.auth_button.clicked.connect(self.run_google_auth)

        layout = QVBoxLayout()
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(12)
        layout.addWidget(self.spectrum)
        layout.addWidget(self.status_label)
        layout.addWidget(self.transcript_label)
        layout.addWidget(self.log_view)

        controls = QHBoxLayout()
        controls.addWidget(self.listen_button)
        controls.addWidget(self.auth_button)
        layout.addLayout(controls)
        self.setLayout(layout)

        self._thread: QThread | None = None
        self._worker: AssistantWorker | None = None
        self._visual_timer = QTimer(self)
        self._visual_timer.timeout.connect(self.refresh_spectrum)
        self._visual_timer.start(33)

    def ensure_api_key(self) -> bool:
        if not self.force_redefine_api_key:
            existing = get_profile_api_key(self.preferred_profile)
            if existing:
                self.status_label.setText(f"Using profile '{existing.name}'.")
                return True

        profiles = list_profile_names()
        default_profile = self.preferred_profile or default_profile_name()
        try:
            default_index = profiles.index(default_profile)
        except ValueError:
            default_index = 0

        selected_profile, ok_profile = QInputDialog.getItem(
            self,
            "Select API Profile",
            "Choose or type a profile name:",
            profiles,
            default_index,
            True,
        )
        if not ok_profile:
            return False

        profile_name = selected_profile.strip() or "default"
        key, ok_key = QInputDialog.getText(
            self,
            "OpenAI API Key",
            f"Enter API key for profile '{profile_name}':",
            QLineEdit.EchoMode.Password,
        )
        if not ok_key:
            return False
        if not key.strip():
            self.status_label.setText("No API key entered.")
            return False

        try:
            set_profile_api_key(profile_name=profile_name, api_key=key, make_default=True)
            return True
        except Exception as exc:
            self.status_label.setText(str(exc))
            return False

    def run_google_auth(self) -> None:
        try:
            authenticate_google_calendar()
            self.status_label.setText("Google authentication successful.")
        except Exception as exc:
            self.status_label.setText(str(exc))

    def toggle_listening(self) -> None:
        if self._thread is not None:
            self._stop_background_worker()
            self.listen_button.setText("Start Background Listening")
            self.status_label.setText("Background listening stopped.")
            self.spectrum.set_listening(False)
            return

        if not self.ensure_api_key():
            self.status_label.setText("OpenAI API key is required.")
            return

        self.spectrum.set_listening(True)
        self.listen_button.setText("Stop Background Listening")
        self.status_label.setText("Starting background listener...")

        self._thread = QThread(self)
        self._worker = AssistantWorker()
        self._worker.moveToThread(self._thread)

        self._thread.started.connect(self._worker.run)
        self._worker.status.connect(self.status_label.setText)
        self._worker.transcript.connect(self._on_transcript)
        self._worker.log.connect(self._on_log)
        self._worker.finished.connect(self._on_finished)
        self._worker.finished.connect(self._thread.quit)
        self._thread.finished.connect(self._cleanup_worker)
        self._thread.start()

    def _on_transcript(self, text: str) -> None:
        self.transcript_label.setText(text)

    def _on_log(self, text: str) -> None:
        self.log_view.append(text)

    def _on_finished(self, success: bool, message: str) -> None:
        del success
        self.status_label.setText(message)

    def _cleanup_worker(self) -> None:
        self.spectrum.set_listening(False)
        self.listen_button.setText("Start Background Listening")
        if self._worker is not None:
            self._worker.deleteLater()
        if self._thread is not None:
            self._thread.deleteLater()
        self._worker = None
        self._thread = None

    def refresh_spectrum(self) -> None:
        if self._worker is None:
            return
        self.spectrum.set_values(self._worker.spectrum())

    def _stop_background_worker(self) -> None:
        worker = self._worker
        thread = self._thread
        if thread is None:
            return

        if worker is not None:
            worker.stop()
        thread.quit()
        if not thread.wait(1200):
            thread.terminate()
            thread.wait(800)

    def closeEvent(self, event: QCloseEvent) -> None:
        self._visual_timer.stop()
        self._stop_background_worker()
        QApplication.quit()
        event.accept()


def launch_gui(
    force_redefine_api_key: bool = False,
    preferred_profile: str | None = None,
) -> int:
    app = QApplication.instance()
    owns_app = app is None
    if app is None:
        app = QApplication([])
    QApplication.setQuitOnLastWindowClosed(True)

    window = PyxisWindow(
        force_redefine_api_key=force_redefine_api_key,
        preferred_profile=preferred_profile,
    )
    if not window.ensure_api_key():
        return 1
    window.show()
    window.toggle_listening()

    if owns_app:
        return app.exec()
    return 0
