from __future__ import annotations

import io
import queue
import re
import subprocess
import sys
import tempfile
import threading
import time
import wave
from collections import deque
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pyttsx3
import sounddevice as sd
from openai import OpenAI

from pyxis_assistant.config import (
    PROJECT_ROOT,
    get_custom_wake_threshold,
    get_openai_api_key,
    get_openwake_threshold,
    get_tts_backend,
    get_tts_model,
    get_tts_speed,
    get_tts_voice,
)
from pyxis_assistant.custom_wake import CustomWakeModel
from pyxis_assistant.logging_utils import get_logger

try:
    from openwakeword.model import Model as OpenWakeWordModel
except Exception:
    OpenWakeWordModel = None  # type: ignore[assignment]

try:
    import webrtcvad
except Exception:
    webrtcvad = None  # type: ignore[assignment]


class VoiceRecorder:
    def __init__(self, sample_rate: int = 16000, block_size: int = 1024, bins: int = 48) -> None:
        self.sample_rate = sample_rate
        self.block_size = block_size
        self._bins = bins
        self._frames: list[np.ndarray] = []
        self._stream: sd.InputStream | None = None
        self._lock = threading.Lock()
        self._spectrum = np.zeros(self._bins, dtype=np.float32)

    def _audio_callback(
        self,
        indata: np.ndarray,
        frames: int,
        time_info: object,
        status: sd.CallbackFlags,
    ) -> None:
        del frames, time_info
        if status:
            return

        mono = indata[:, 0].astype(np.float32).copy()
        with self._lock:
            self._frames.append(mono)
            self._spectrum = self._calculate_spectrum(mono)

    def _calculate_spectrum(self, chunk: np.ndarray) -> np.ndarray:
        if chunk.size == 0:
            return np.zeros(self._bins, dtype=np.float32)

        windowed = chunk * np.hanning(chunk.size)
        fft = np.abs(np.fft.rfft(windowed))
        if fft.size == 0:
            return np.zeros(self._bins, dtype=np.float32)

        grouped = np.array_split(fft, self._bins)
        values = np.array([float(group.mean()) if group.size else 0.0 for group in grouped])
        maximum = float(values.max())
        if maximum > 0:
            values = values / maximum
        return values.astype(np.float32)

    def start(self) -> None:
        with self._lock:
            self._frames = []
            self._spectrum = np.zeros(self._bins, dtype=np.float32)

        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            blocksize=self.block_size,
            callback=self._audio_callback,
        )
        self._stream.start()

    def stop(self) -> np.ndarray:
        stream = self._stream
        self._stream = None
        if stream is not None:
            stream.stop()
            stream.close()

        with self._lock:
            if not self._frames:
                return np.zeros(0, dtype=np.float32)
            captured = np.concatenate(self._frames)
            self._frames = []
            return captured

    def record_for(self, seconds: float) -> Path:
        self.start()
        time.sleep(seconds)
        captured = self.stop()
        if captured.size == 0:
            msg = "No microphone audio captured."
            raise RuntimeError(msg)

        clipped = np.clip(captured, -1.0, 1.0)
        pcm = (clipped * 32767).astype(np.int16)

        tmp_file = tempfile.NamedTemporaryFile(prefix="pyxis_", suffix=".wav", delete=False)
        path = Path(tmp_file.name)
        tmp_file.close()

        with wave.open(str(path), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(self.sample_rate)
            wav.writeframes(pcm.tobytes())

        return path

    def spectrum(self) -> np.ndarray:
        with self._lock:
            return self._spectrum.copy()


class VoiceSpeaker:
    def __init__(self) -> None:
        self._logger = get_logger("pyxis.voice")
        self._engine = pyttsx3.init()
        self._lock = threading.Lock()
        self._backend = get_tts_backend()
        self._openai_model = get_tts_model()
        self._openai_voice = get_tts_voice()
        self._openai_speed = get_tts_speed()
        self._openai_client: OpenAI | None = None
        self._use_windows_sapi = sys.platform.startswith("win")
        self._stop_requested = threading.Event()
        self._is_speaking = False
        self._configure_engine()
        self._ensure_openai_client()
        self._logger.info(
            "TTS configured: backend=%s, model=%s, voice=%s, speed=%.2f",
            self._backend,
            self._openai_model,
            self._openai_voice,
            self._openai_speed,
        )

    def _configure_engine(self) -> None:
        try:
            self._engine.setProperty("volume", 1.0)
            # Slightly slower pace to improve intelligibility.
            self._engine.setProperty("rate", 178)
        except Exception as exc:
            self._logger.warning("Failed to configure TTS engine: %s", exc)

    def speak(self, text: str) -> None:
        with self._lock:
            self._stop_requested.clear()
            self._is_speaking = True
            spoken_text = self._normalize_pyxis_pronunciation(text)
            backend_order = self._backend_order()
            try:
                for backend in backend_order:
                    if self._stop_requested.is_set():
                        return
                    if backend == "openai":
                        if self._speak_openai(spoken_text):
                            return
                    elif backend == "windows":
                        if self._use_windows_sapi and self._speak_windows_sapi(spoken_text):
                            return
                    elif backend == "pyttsx3":
                        if self._speak_pyttsx3(spoken_text):
                            return

                # Last resort: force pyttsx3 once.
                self._logger.warning("All configured TTS backends failed; forcing pyttsx3 fallback.")
                self._speak_pyttsx3(spoken_text)
            finally:
                self._is_speaking = False

    def stop(self) -> None:
        self._stop_requested.set()
        try:
            sd.stop()
        except Exception:
            pass
        try:
            self._engine.stop()
        except Exception:
            pass

    def is_speaking(self) -> bool:
        return self._is_speaking

    def _normalize_pyxis_pronunciation(self, text: str) -> str:
        # Helps local engines pronounce the brand name as "pike-sis".
        return re.sub(r"\bpyxis\b", "pike-sis", text, flags=re.IGNORECASE)

    def _backend_order(self) -> list[str]:
        if self._backend == "auto":
            return ["openai", "windows", "pyttsx3"]
        if self._backend == "openai":
            return ["openai", "windows", "pyttsx3"]
        if self._backend == "windows":
            return ["windows", "pyttsx3"]
        if self._backend == "pyttsx3":
            return ["pyttsx3"]
        return ["openai", "windows", "pyttsx3"]

    def _ensure_openai_client(self) -> None:
        if self._openai_client is not None:
            return
        try:
            self._openai_client = OpenAI(api_key=get_openai_api_key())
        except Exception as exc:
            self._openai_client = None
            self._logger.info("OpenAI TTS unavailable; will fall back. %s", exc)

    def _speak_openai(self, text: str) -> bool:
        self._ensure_openai_client()
        if self._openai_client is None:
            return False
        try:
            if self._openai_model.startswith("gpt-4o-mini-tts"):
                response = self._openai_client.audio.speech.create(
                    model=self._openai_model,
                    voice=self._openai_voice,
                    input=text,
                    response_format="wav",
                    speed=self._openai_speed,
                    instructions="Pronounce 'Pyxis' as 'pike-sis' whenever it appears.",
                )
            else:
                response = self._openai_client.audio.speech.create(
                    model=self._openai_model,
                    voice=self._openai_voice,
                    input=text,
                    response_format="wav",
                    speed=self._openai_speed,
                )
            wav_bytes = response.read()
            if not wav_bytes:
                return False
            self._play_wav_bytes(wav_bytes)
            return True
        except Exception as exc:
            self._logger.warning("OpenAI TTS failed; falling back: %s", exc)
            return False

    def _play_wav_bytes(self, wav_bytes: bytes) -> None:
        with wave.open(io.BytesIO(wav_bytes), "rb") as wav_file:
            channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            sample_rate = wav_file.getframerate()
            frames = wav_file.readframes(wav_file.getnframes())

        if sample_width == 1:
            samples = np.frombuffer(frames, dtype=np.uint8)
            if channels > 1:
                samples = samples.reshape(-1, channels)
            audio = (samples.astype(np.float32) - 128.0) / 128.0
        elif sample_width == 2:
            samples = np.frombuffer(frames, dtype=np.int16)
            if channels > 1:
                samples = samples.reshape(-1, channels)
            audio = samples.astype(np.float32) / 32768.0
        elif sample_width == 4:
            samples = np.frombuffer(frames, dtype=np.int32)
            if channels > 1:
                samples = samples.reshape(-1, channels)
            audio = samples.astype(np.float32) / 2147483648.0
        else:
            msg = f"Unsupported WAV sample width: {sample_width}"
            raise RuntimeError(msg)

        sd.stop()
        sd.play(audio, samplerate=sample_rate, blocking=False)
        frame_count = audio.shape[0] if audio.ndim > 0 else 0
        duration_ms = int((frame_count / max(1, sample_rate)) * 1000) + 120
        elapsed_ms = 0
        while elapsed_ms < duration_ms:
            if self._stop_requested.is_set():
                sd.stop()
                return
            sd.sleep(40)
            elapsed_ms += 40
        sd.stop()

    def _speak_pyttsx3(self, text: str) -> bool:
        try:
            if self._stop_requested.is_set():
                return False
            self._engine.say(text)
            self._engine.runAndWait()
            return True
        except Exception as exc:
            self._logger.warning("TTS failed, reinitializing engine: %s", exc)
            self._engine = pyttsx3.init()
            self._configure_engine()
            try:
                if self._stop_requested.is_set():
                    return False
                self._engine.say(text)
                self._engine.runAndWait()
                return True
            except Exception as second_exc:
                self._logger.warning("pyttsx3 fallback failed: %s", second_exc)
                return False

    def _speak_windows_sapi(self, text: str) -> bool:
        safe = text.replace("'", "''")
        command = (
            "Add-Type -AssemblyName System.Speech; "
            "$speak = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
            "$speak.Volume = 100; "
            "$speak.Rate = 0; "
            f"$speak.Speak('{safe}')"
        )
        try:
            result = subprocess.run(
                ["powershell", "-NoProfile", "-Command", command],
                check=False,
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode == 0:
                return True
            self._logger.warning(
                "Windows SAPI returned non-zero (%s): %s",
                result.returncode,
                result.stderr.strip(),
            )
            return False
        except Exception as exc:
            self._logger.warning("Windows SAPI invocation error: %s", exc)
            return False


class LocalWakeVADListener:
    def __init__(
        self,
        sample_rate: int = 16000,
        frame_ms: int = 20,
        wake_threshold: float | None = None,
        vad_aggressiveness: int = 2,
        bins: int = 48,
    ) -> None:
        self.sample_rate = sample_rate
        self.frame_ms = frame_ms
        self.frame_samples = int(sample_rate * frame_ms / 1000)
        self.wake_threshold = (
            get_openwake_threshold() if wake_threshold is None else wake_threshold
        )
        self._bins = bins
        self._queue: queue.Queue[bytes] = queue.Queue(maxsize=256)
        self._stream: sd.RawInputStream | None = None
        self._lock = threading.Lock()
        self._spectrum = np.zeros(self._bins, dtype=np.float32)
        self._logger = get_logger("pyxis.wake")
        self._custom_model_path = PROJECT_ROOT / ".pyxis" / "wake_model.json"
        self._custom_model: CustomWakeModel | None = None
        self._custom_threshold = get_custom_wake_threshold()
        self._custom_window_frames = int(1.2 / (frame_ms / 1000.0))
        self._custom_short_window_frames = int(0.7 / (frame_ms / 1000.0))
        self._custom_consecutive_hits_required = 2
        self._wake_min_voiced_ratio = 0.20
        self._custom_short_margin = 0.08
        self._custom_score_ema_alpha = 0.35
        self._custom_refractory_seconds = 0.7

        self._vad = webrtcvad.Vad(vad_aggressiveness) if webrtcvad is not None else None
        self._load_custom_model()
        self._wake_model = None
        if self._custom_model is None and OpenWakeWordModel is not None:
            self._wake_model = OpenWakeWordModel()
        elif self._custom_model is not None:
            self._logger.info("Custom wake model loaded; openWakeWord disabled.")
        has_wake = self._custom_model is not None or self._wake_model is not None
        self.available = has_wake and self._vad is not None
        self._logger.info(
            "Wake thresholds: custom=%.2f, openwake=%.2f",
            self._custom_threshold,
            self.wake_threshold,
        )

    def _load_custom_model(self) -> None:
        if not self._custom_model_path.exists():
            return
        try:
            self._custom_model = CustomWakeModel.load(self._custom_model_path)
            self._logger.info("Loaded custom wake model: %s", self._custom_model_path)
        except Exception as exc:
            self._custom_model = None
            self._logger.warning("Failed to load custom wake model: %s", exc)

    def _audio_callback(
        self,
        indata: bytes,
        frames: int,
        time_info: object,
        status: sd.CallbackFlags,
    ) -> None:
        del frames, time_info
        if status:
            return

        chunk = bytes(indata)
        try:
            self._queue.put_nowait(chunk)
        except queue.Full:
            return

        data = np.frombuffer(chunk, dtype=np.int16).astype(np.float32)
        if data.size == 0:
            return
        with self._lock:
            self._spectrum = self._calculate_spectrum(data)

    def _calculate_spectrum(self, chunk: np.ndarray) -> np.ndarray:
        if chunk.size == 0:
            return np.zeros(self._bins, dtype=np.float32)
        windowed = chunk * np.hanning(chunk.size)
        fft = np.abs(np.fft.rfft(windowed))
        if fft.size == 0:
            return np.zeros(self._bins, dtype=np.float32)
        grouped = np.array_split(fft, self._bins)
        values = np.array([float(group.mean()) if group.size else 0.0 for group in grouped])
        maximum = float(values.max())
        if maximum > 0:
            values = values / maximum
        return values.astype(np.float32)

    def start(self) -> None:
        self._stream = sd.RawInputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="int16",
            blocksize=self.frame_samples,
            callback=self._audio_callback,
        )
        self._stream.start()

    def stop(self) -> None:
        stream = self._stream
        self._stream = None
        if stream is not None:
            stream.stop()
            stream.close()

    def spectrum(self) -> np.ndarray:
        with self._lock:
            return self._spectrum.copy()

    def _next_frame(self, timeout: float = 0.2) -> bytes | None:
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def clear_queue(self) -> None:
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break

    def detect_barge_in(
        self,
        running: Callable[[], bool],
        listen_seconds: float = 0.35,
        min_consecutive_speech_frames: int = 5,
    ) -> bool:
        if self._vad is None:
            return False
        end_time = time.monotonic() + max(0.1, listen_seconds)
        consecutive = 0
        while running() and time.monotonic() < end_time:
            frame = self._next_frame(timeout=0.05)
            if frame is None:
                continue
            is_speech = bool(self._vad.is_speech(frame, self.sample_rate))
            if is_speech:
                consecutive += 1
                if consecutive >= min_consecutive_speech_frames:
                    return True
            else:
                consecutive = 0
        return False

    def wait_for_wake_word(self, running: Callable[[], bool]) -> bool:
        return self.wait_for_wake_word_with_feedback(running)

    def wait_for_wake_word_with_feedback(
        self,
        running: Callable[[], bool],
        on_score: Callable[[float, str, float], None] | None = None,
    ) -> bool:
        if self._custom_model is None and self._wake_model is None:
            return False

        # openWakeWord works best with ~80ms chunks (1280 @ 16kHz).
        wake_buffer: deque[bytes] = deque(maxlen=4)
        custom_buffer: deque[bytes] = deque(maxlen=self._custom_window_frames)
        custom_short_buffer: deque[bytes] = deque(maxlen=self._custom_short_window_frames)
        speech_buffer: deque[bool] = deque(maxlen=self._custom_window_frames)
        custom_hit_streak = 0
        custom_smoothed = 0.0
        custom_cooldown_until = 0.0
        last_feedback = 0.0
        while running():
            frame = self._next_frame(timeout=0.2)
            if frame is None:
                continue

            wake_buffer.append(frame)
            custom_buffer.append(frame)
            custom_short_buffer.append(frame)
            frame_is_speech = (
                bool(self._vad.is_speech(frame, self.sample_rate)) if self._vad else True
            )
            speech_buffer.append(frame_is_speech)
            score = 0.0
            if (
                self._custom_model is not None
                and len(custom_buffer) >= self._custom_window_frames
                and len(custom_short_buffer) >= self._custom_short_window_frames
            ):
                merged_custom = b"".join(custom_buffer)
                audio_custom = np.frombuffer(merged_custom, dtype=np.int16)
                merged_short = b"".join(custom_short_buffer)
                audio_short = np.frombuffer(merged_short, dtype=np.int16)
                raw_score_long = float(
                    self._custom_model.score(audio_custom, sample_rate=self.sample_rate)
                )
                raw_score_short = float(
                    self._custom_model.score(audio_short, sample_rate=self.sample_rate)
                )
                custom_smoothed = (
                    self._custom_score_ema_alpha * raw_score_long
                    + (1.0 - self._custom_score_ema_alpha) * custom_smoothed
                )
                score = min(custom_smoothed, raw_score_short)
                voiced_ratio = (
                    float(sum(1 for item in speech_buffer if item)) / max(1, len(speech_buffer))
                )
                recent_voiced = (
                    float(
                        sum(
                            1
                            for item in list(speech_buffer)[-self._custom_short_window_frames :]
                            if item
                        )
                    )
                    / max(1, self._custom_short_window_frames)
                )
                now = time.monotonic()
                can_evaluate = now >= custom_cooldown_until
                voiced_ok = (
                    voiced_ratio >= self._wake_min_voiced_ratio
                    and recent_voiced >= max(0.30, self._wake_min_voiced_ratio)
                )
                dual_window_ok = (
                    raw_score_long >= self._custom_threshold
                    and raw_score_short >= (self._custom_threshold - self._custom_short_margin)
                )
                if can_evaluate and voiced_ok and dual_window_ok:
                    custom_hit_streak += 1
                    if custom_hit_streak >= self._custom_consecutive_hits_required:
                        if on_score is not None:
                            on_score(score, "custom", self._custom_threshold)
                        custom_cooldown_until = now + self._custom_refractory_seconds
                        return True
                else:
                    custom_hit_streak = 0

            if self._wake_model is not None and len(wake_buffer) >= 4:
                # Ignore openWakeWord scoring during non-speech ambient windows.
                recent = (
                    list(speech_buffer)[-4:]
                    if len(speech_buffer) >= 4
                    else list(speech_buffer)
                )
                voiced_recent = float(sum(1 for item in recent if item)) / max(1, len(recent))
                if voiced_recent >= self._wake_min_voiced_ratio:
                    merged = b"".join(wake_buffer)
                    audio = np.frombuffer(merged, dtype=np.int16)
                    prediction = self._wake_model.predict(audio)
                    if isinstance(prediction, dict) and prediction:
                        openwake_score = max(float(value) for value in prediction.values())
                        score = max(score, openwake_score)
                        if openwake_score >= self.wake_threshold:
                            if on_score is not None:
                                on_score(openwake_score, "openwakeword", self.wake_threshold)
                            return True

            if on_score is not None:
                now = time.monotonic()
                if now - last_feedback >= 1.0:
                    threshold = (
                        self._custom_threshold
                        if self._custom_model is not None
                        else self.wake_threshold
                    )
                    source = "custom" if self._custom_model is not None else "openwakeword"
                    on_score(score, source, threshold)
                    last_feedback = now

        return False

    def capture_utterance(
        self,
        running: Callable[[], bool],
        max_seconds: float = 8.0,
        min_seconds: float = 0.4,
        silence_seconds: float = 0.9,
        pre_roll_seconds: float = 0.25,
    ) -> Path | None:
        if self._vad is None:
            return None

        frame_duration = self.frame_ms / 1000.0
        max_frames = int(max_seconds / frame_duration)
        min_frames = int(min_seconds / frame_duration)
        silence_frames_required = int(silence_seconds / frame_duration)
        pre_roll_frames = int(pre_roll_seconds / frame_duration)

        pre_roll: deque[bytes] = deque(maxlen=pre_roll_frames)
        captured: list[bytes] = []
        speech_started = False
        silence_count = 0
        frame_count = 0

        while running() and frame_count < max_frames:
            frame = self._next_frame(timeout=0.25)
            if frame is None:
                continue

            frame_count += 1
            pre_roll.append(frame)
            is_speech = bool(self._vad.is_speech(frame, self.sample_rate))

            if not speech_started:
                if is_speech:
                    speech_started = True
                    captured.extend(pre_roll)
                continue

            captured.append(frame)
            if is_speech:
                silence_count = 0
            else:
                silence_count += 1

            if len(captured) >= min_frames and silence_count >= silence_frames_required:
                break

        if not captured:
            return None
        return self._frames_to_wav(captured)

    def _frames_to_wav(self, frames: list[bytes]) -> Path:
        tmp_file = tempfile.NamedTemporaryFile(prefix="pyxis_vad_", suffix=".wav", delete=False)
        path = Path(tmp_file.name)
        tmp_file.close()

        with wave.open(str(path), "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(self.sample_rate)
            wav_file.writeframes(b"".join(frames))
        return path
