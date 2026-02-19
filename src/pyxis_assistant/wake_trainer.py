from __future__ import annotations

import argparse
import random
import re
import time
import wave
from collections.abc import Callable
from pathlib import Path

import numpy as np
import sounddevice as sd

from pyxis_assistant.config import PROJECT_ROOT
from pyxis_assistant.custom_wake import train_custom_wake_model

TRAIN_DIR = PROJECT_ROOT / ".pyxis" / "wake_training"
POS_DIR = TRAIN_DIR / "positive"
NEG_DIR = TRAIN_DIR / "negative"
NOISE_DIR = TRAIN_DIR / "noise"
MODEL_PATH = PROJECT_ROOT / ".pyxis" / "wake_model.json"

STYLES = {
    "fast": "Say it quickly and naturally.",
    "slow": "Say it slowly and clearly.",
    "relaxed": "Say it in your normal relaxed voice.",
    "whisper": "Whisper it softly but audibly.",
}

NOISE_PLAN = [
    ("silent", 40, "Stay as quiet as possible."),
    ("keyboard", 20, "Type on your keyboard during capture."),
    ("mouse", 20, "Use mouse clicks/movement during capture."),
    ("mixed", 20, "Use mixed desk noise (keyboard + mouse + room noise)."),
]

CONFUSER_WORDS = [
    "s",
    "sis",
    "sister",
    "snake",
    "salivation",
    "salad",
    "sizzle",
    "serious",
    "series",
    "science",
    "system",
    "signal",
    "silence",
    "silly",
    "simple",
    "city",
    "season",
    "sunday",
    "secret",
    "sit",
    "six",
    "sick",
    "sink",
    "singer",
    "singing",
    "siren",
    "sugar",
    "super",
    "pizza",
    "piece",
    "peace",
    "peak",
    "peek",
    "pike",
    "pikes",
    "pixie",
    "pixel",
    "pixels",
    "picker",
    "picked",
    "pickle",
    "piston",
    "pisa",
    "pyrex",
    "pike sis",
    "pick sis",
    "pie sis",
    "pie six",
    "pike six",
    "pix sis",
    "please sis",
    "say sis",
    "my sis",
    "his sis",
    "this is",
    "is this",
    "assist",
    "assistance",
    "analysis",
    "physics",
    "phony",
    "phone",
    "focus",
    "foxes",
    "fixes",
    "fix this",
    "picnic",
    "picasso",
    "pickaxe",
    "picky",
    "psyche",
    "psychic",
    "psalm",
    "phoenix",
    "fizz",
    "fizzy",
    "visits",
    "visit",
    "vicious",
    "suspense",
    "special",
    "spacious",
    "space",
    "speech",
    "spectrum",
    "sprinter",
    "split",
    "spike",
    "spikes",
    "spicy",
    "spice",
    "spider",
    "spy",
    "spyglass",
    "sphinx",
    "sprint",
    "spritely",
    "sip",
    "sips",
    "sip tea",
    "sip water",
    "safety",
    "safari",
    "service",
    "server",
    "session",
    "seamless",
    "seaside",
    "salt",
    "salty",
    "solution",
    "salute",
]

FALLBACK_COMMON_WORDS = [
    "the",
    "be",
    "to",
    "of",
    "and",
    "a",
    "in",
    "that",
    "have",
    "i",
    "it",
    "for",
    "not",
    "on",
    "with",
    "he",
    "as",
    "you",
    "do",
    "at",
    "this",
    "but",
    "his",
    "by",
    "from",
    "they",
    "we",
    "say",
    "her",
    "she",
    "or",
    "an",
    "will",
    "my",
    "one",
    "all",
    "would",
    "there",
    "their",
    "what",
    "so",
    "up",
    "out",
    "if",
    "about",
    "who",
    "get",
    "which",
    "go",
    "me",
    "when",
    "make",
    "can",
    "like",
    "time",
    "no",
    "just",
    "him",
    "know",
    "take",
    "people",
    "into",
    "year",
    "your",
    "good",
    "some",
    "could",
    "them",
    "see",
    "other",
    "than",
    "then",
    "now",
    "look",
    "only",
    "come",
    "its",
    "over",
    "think",
    "also",
    "back",
    "after",
    "use",
    "two",
    "how",
    "our",
    "work",
    "first",
    "well",
    "way",
    "even",
    "new",
    "want",
    "because",
    "any",
    "these",
    "give",
    "day",
    "most",
    "us",
    "is",
    "are",
    "was",
    "were",
    "had",
    "did",
    "been",
    "more",
    "many",
    "very",
    "still",
    "such",
    "through",
    "much",
    "before",
    "where",
    "why",
    "while",
    "few",
    "might",
    "should",
    "must",
    "every",
    "each",
    "again",
    "never",
    "always",
    "today",
    "tomorrow",
    "yesterday",
    "morning",
    "night",
    "week",
    "month",
    "home",
    "house",
    "room",
    "door",
    "window",
    "table",
    "chair",
    "kitchen",
    "water",
    "coffee",
    "tea",
    "food",
    "lunch",
    "dinner",
    "breakfast",
    "store",
    "market",
    "street",
    "road",
    "car",
    "train",
    "plane",
    "travel",
    "city",
    "country",
    "world",
    "earth",
    "sun",
    "moon",
    "school",
    "class",
    "study",
    "learn",
    "teach",
    "book",
    "paper",
    "note",
    "list",
    "task",
    "event",
    "calendar",
    "meeting",
    "project",
    "team",
    "office",
    "email",
    "message",
    "phone",
    "call",
    "video",
    "music",
    "movie",
    "game",
    "story",
    "news",
    "question",
    "answer",
    "idea",
    "plan",
    "goal",
    "problem",
    "solution",
    "system",
    "model",
    "voice",
    "audio",
    "microphone",
    "computer",
    "screen",
    "keyboard",
    "mouse",
    "internet",
    "network",
    "server",
    "code",
    "python",
    "number",
    "word",
    "sentence",
    "language",
    "english",
    "family",
    "friend",
    "child",
    "children",
    "person",
    "man",
    "woman",
    "life",
    "health",
    "money",
    "price",
    "cost",
    "minute",
    "hour",
    "second",
    "early",
    "late",
    "soon",
    "ready",
    "start",
    "stop",
    "open",
    "close",
    "add",
    "remove",
    "update",
    "create",
    "delete",
    "check",
    "test",
    "run",
    "build",
    "clean",
    "small",
    "big",
    "large",
    "high",
    "low",
    "long",
    "short",
    "easy",
    "hard",
    "quick",
    "slow",
    "clear",
    "strong",
    "soft",
    "quiet",
    "loud",
    "light",
    "dark",
    "blue",
    "green",
    "red",
    "black",
    "white",
    "brown",
    "cold",
    "hot",
    "warm",
    "rain",
    "snow",
    "wind",
    "cloud",
    "storm",
    "happy",
    "sad",
    "angry",
    "calm",
    "tired",
    "busy",
    "free",
    "help",
    "please",
    "thanks",
    "sorry",
    "yes",
    "okay",
    "right",
    "left",
    "next",
    "last",
    "best",
    "better",
    "same",
    "different",
    "important",
    "simple",
    "basic",
    "final",
    "main",
    "local",
    "global",
    "public",
    "private",
    "safe",
    "ready",
    "done",
    "fine",
    "great",
    "awesome",
    "normal",
    "special",
    "common",
    "example",
    "result",
    "summary",
    "detail",
    "history",
    "memory",
    "future",
    "past",
    "present",
]


def _record_clip(duration: float, sample_rate: int = 16000) -> np.ndarray:
    frames = int(duration * sample_rate)
    recording = sd.rec(frames, samplerate=sample_rate, channels=1, dtype="float32")
    sd.wait()
    mono = recording[:, 0]
    mono = np.clip(mono, -1.0, 1.0)
    return (mono * 32767).astype(np.int16)


def _trim_silence(
    audio: np.ndarray,
    threshold: int = 550,
    min_samples: int = 4000,
) -> np.ndarray:
    if audio.size == 0:
        return audio

    active = np.flatnonzero(np.abs(audio) > threshold)
    if active.size == 0:
        return audio

    start = int(active[0])
    end = int(active[-1]) + 1
    trimmed = audio[start:end]
    if trimmed.size < min_samples:
        return audio
    return trimmed


def _save_wav(path: Path, audio: np.ndarray, sample_rate: int = 16000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio.astype(np.int16).tobytes())


def _load_wavs(path: Path) -> list[np.ndarray]:
    if not path.exists():
        return []
    audios: list[np.ndarray] = []
    for wav_file in sorted(path.rglob("*.wav")):
        with wave.open(str(wav_file), "rb") as wav:
            data = wav.readframes(wav.getnframes())
        audios.append(np.frombuffer(data, dtype=np.int16).copy())
    return audios


def _existing_count(path: Path) -> int:
    if not path.exists():
        return 0
    return len(list(path.rglob("*.wav")))


def _next_index(path: Path, prefix: str) -> int:
    if not path.exists():
        return 1
    pattern = re.compile(rf"^{re.escape(prefix)}_(\d+)\.wav$")
    best = 0
    for file in path.glob("*.wav"):
        match = pattern.match(file.name)
        if match:
            best = max(best, int(match.group(1)))
    return best + 1


def _augment_with_noise(
    positive: list[np.ndarray],
    noises: list[np.ndarray],
    copies_per_sample: int = 2,
) -> list[np.ndarray]:
    if not noises:
        return positive
    output = list(positive)
    rng = random.Random(42)
    for clip in positive:
        for _ in range(copies_per_sample):
            noise = rng.choice(noises)
            if noise.size < clip.size:
                reps = int(np.ceil(clip.size / max(1, noise.size)))
                noise = np.tile(noise, reps)
            start = rng.randint(0, max(0, noise.size - clip.size))
            segment = noise[start : start + clip.size].astype(np.float32)
            clean = clip.astype(np.float32)
            snr_db = rng.uniform(6.0, 18.0)
            clean_rms = float(np.sqrt(np.mean(np.square(clean))) + 1e-6)
            noise_rms = float(np.sqrt(np.mean(np.square(segment))) + 1e-6)
            desired_noise_rms = clean_rms / (10 ** (snr_db / 20.0))
            scaled_noise = segment * (desired_noise_rms / noise_rms)
            mixed = np.clip(clean + scaled_noise, -32768, 32767).astype(np.int16)
            output.append(mixed)
    return output


def _countdown() -> None:
    for value in [3, 2, 1]:
        print(f"{value}...")
        time.sleep(0.35)


def _style_sequence(total: int, rng: random.Random) -> list[str]:
    styles = list(STYLES.keys())
    values: list[str] = []
    for i in range(total):
        values.append(styles[i % len(styles)])
    rng.shuffle(values)
    return values


def _normalize_words(words: list[str]) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()
    for word in words:
        cleaned = " ".join(str(word).strip().lower().split())
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        normalized.append(cleaned)
    return normalized


def _build_confuser_words(limit: int, rng: random.Random) -> list[str]:
    base = _normalize_words(CONFUSER_WORDS)
    onsets = ["pi", "pie", "pike", "pix", "pick", "sis", "six", "s", "psy", "phi", "spi"]
    codas = ["sis", "six", "s", "ser", "sel", "xel", "xis", "sive", "tion", "ster", "zing"]
    generated: list[str] = []
    for onset in onsets:
        for coda in codas:
            word = f"{onset}{coda}"
            if 3 <= len(word) <= 10 and word.isalpha():
                generated.append(word)
    all_words = _normalize_words(base + generated)
    if len(all_words) < limit:
        repeated: list[str] = []
        while len(repeated) < limit:
            repeated.extend(all_words)
        all_words = repeated[:limit]
    rng.shuffle(all_words)
    return all_words[:limit]


def _load_zipf_words(limit: int, rng: random.Random) -> list[str]:
    try:
        from wordfreq import top_n_list  # type: ignore[import-not-found]

        words = [str(item).lower() for item in top_n_list("en", limit * 2)]
        words = [word for word in words if word.isalpha()]
        normalized = _normalize_words(words)
        if len(normalized) >= limit:
            return normalized[:limit]
    except Exception:
        pass

    fallback = _normalize_words(FALLBACK_COMMON_WORDS)
    if len(fallback) < limit:
        repeated: list[str] = []
        while len(repeated) < limit:
            repeated.extend(fallback)
        fallback = repeated[:limit]
    rng.shuffle(fallback)
    return fallback[:limit]


def _record_category(
    path: Path,
    prefix: str,
    target: int,
    sample_duration: float,
    prompt_builder: Callable[[int, int], str],
    auto_trim: bool,
    trim_audio: bool,
) -> None:
    existing = _existing_count(path)
    if existing >= target:
        return
    index = _next_index(path, prefix)

    while existing < target:
        prompt = prompt_builder(existing, target)
        print(prompt)
        _countdown()
        clip = _record_clip(sample_duration)
        if auto_trim and trim_audio:
            clip = _trim_silence(clip)
        _save_wav(path / f"{prefix}_{index:04d}.wav", clip)
        print("Saved sample.\n")
        existing += 1
        index += 1


def _noise_prompt_builder(noise_key: str, instruction: str) -> Callable[[int, int], str]:
    def builder(done: int, total: int) -> str:
        return f"[Noise:{noise_key} {done + 1}/{total}] {instruction}"

    return builder


def _positive_prompt_builder(phrase: str, style_pos: list[str]) -> Callable[[int, int], str]:
    def builder(done: int, total: int) -> str:
        style = style_pos[done]
        return (
            f"[Positive {done + 1}/{total}] Style={style} | {STYLES[style]} "
            f'Say: "{phrase}"'
        )

    return builder


def _negative_prompt_builder(
    label: str,
    words: list[str],
    styles: list[str],
) -> Callable[[int, int], str]:
    def builder(done: int, total: int) -> str:
        style = styles[done]
        word = words[done]
        return (
            f"[Negative:{label} {done + 1}/{total}] Style={style} | "
            f'{STYLES[style]} Say: "{word}"'
        )

    return builder


def run_trainer(
    positives: int,
    zipf_negatives: int,
    confuser_negatives: int,
    phrase: str,
    sample_duration: float,
    auto_trim: bool,
    seed: int,
    train_only: bool,
    epochs: int,
    learning_rate: float,
    hidden_dim: int,
) -> None:
    rng = random.Random(seed)

    print("Pyxis wake-word trainer")
    print(f"Wake phrase: {phrase}")
    print("Styles: fast, slow, relaxed, whisper")
    print("Noise plan: 40 silent, 20 keyboard, 20 mouse, 20 mixed")
    print("Negative plan: 300 common words + 300 confuser words")
    print("")

    zipf_words = _load_zipf_words(zipf_negatives, rng=rng)
    confuser_words = _build_confuser_words(confuser_negatives, rng=rng)
    style_pos = _style_sequence(positives, rng=rng)
    style_zipf = _style_sequence(zipf_negatives, rng=rng)
    style_conf = _style_sequence(confuser_negatives, rng=rng)

    print(
        "Existing samples: "
        f"positive={_existing_count(POS_DIR)}/{positives}, "
        f"zipf_neg={_existing_count(NEG_DIR / 'zipf')}/{zipf_negatives}, "
        f"confuser_neg={_existing_count(NEG_DIR / 'confuser')}/{confuser_negatives}, "
        f"noise_total={_existing_count(NOISE_DIR)}/100"
    )
    for noise_key, noise_target, _ in NOISE_PLAN:
        print(f"- noise_{noise_key}: {_existing_count(NOISE_DIR / noise_key)}/{noise_target}")
    if _existing_count(TRAIN_DIR) > 0:
        print("Resume mode active: existing recordings are kept and new ones appended.\n")

    if not train_only:
        for noise_key, noise_target, instruction in NOISE_PLAN:
            noise_path = NOISE_DIR / noise_key
            _record_category(
                path=noise_path,
                prefix=noise_key,
                target=noise_target,
                sample_duration=sample_duration,
                auto_trim=auto_trim,
                trim_audio=False,
                prompt_builder=_noise_prompt_builder(noise_key, instruction),
            )

        _record_category(
            path=POS_DIR,
            prefix="pos",
            target=positives,
            sample_duration=sample_duration,
            auto_trim=auto_trim,
            trim_audio=True,
            prompt_builder=_positive_prompt_builder(phrase, style_pos),
        )

        _record_category(
            path=NEG_DIR / "zipf",
            prefix="zipf",
            target=zipf_negatives,
            sample_duration=sample_duration,
            auto_trim=auto_trim,
            trim_audio=True,
            prompt_builder=_negative_prompt_builder("zipf", zipf_words, style_zipf),
        )

        _record_category(
            path=NEG_DIR / "confuser",
            prefix="conf",
            target=confuser_negatives,
            sample_duration=sample_duration,
            auto_trim=auto_trim,
            trim_audio=True,
            prompt_builder=_negative_prompt_builder("confuser", confuser_words, style_conf),
        )

    positive_audio = _load_wavs(POS_DIR)
    negative_audio = _load_wavs(NEG_DIR)
    noise_audio = _load_wavs(NOISE_DIR)
    augmented_positive = _augment_with_noise(positive_audio, noise_audio, copies_per_sample=2)

    if not positive_audio or not negative_audio:
        msg = "Not enough existing samples to train. Record data first or disable --train-only."
        raise ValueError(msg)

    model = train_custom_wake_model(
        positive_audios=augmented_positive,
        negative_audios=negative_audio,
        sample_rate=16000,
        epochs=epochs,
        learning_rate=learning_rate,
        hidden_dim=hidden_dim,
        seed=seed,
    )
    model.save(MODEL_PATH)

    print("\nTraining complete.")
    print(f"- Positive raw: {len(positive_audio)}")
    print(f"- Positive augmented: {len(augmented_positive)}")
    print(f"- Negative total: {len(negative_audio)}")
    print(f"- Noise total: {len(noise_audio)}")
    print(f"- Model type: {model.model_type}")
    print(f"- Suggested internal threshold: {model.threshold:.3f}")
    print(f"- Model saved to: {MODEL_PATH}")
    print("Restart `uv run pyxis` to use the custom wake model.")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Collect structured samples and train a custom Pyxis wake model."
    )
    parser.add_argument(
        "--positives",
        type=int,
        default=160,
        help="Number of wake-phrase samples (style-mixed).",
    )
    parser.add_argument(
        "--zipf-negatives",
        type=int,
        default=300,
        help="Number of common-word negative samples.",
    )
    parser.add_argument(
        "--confuser-negatives",
        type=int,
        default=300,
        help="Number of sound-similar negative samples.",
    )
    parser.add_argument("--phrase", default="pyxis", help="Wake phrase prompt.")
    parser.add_argument(
        "--duration",
        type=float,
        default=1.4,
        help="Seconds per recorded sample.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for style/word order.",
    )
    parser.add_argument(
        "--no-auto-trim",
        action="store_true",
        help="Disable trimming leading/trailing silence from spoken clips.",
    )
    parser.add_argument(
        "--train-only",
        action="store_true",
        help="Skip recording and retrain using existing clips in .pyxis/wake_training.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=260,
        help="Training epochs for the custom neural wake model.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.02,
        help="Learning rate for the custom neural wake model.",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=40,
        help="Hidden layer width for the custom neural wake model.",
    )
    args = parser.parse_args()

    run_trainer(
        positives=max(1, args.positives),
        zipf_negatives=max(1, args.zipf_negatives),
        confuser_negatives=max(1, args.confuser_negatives),
        phrase=args.phrase,
        sample_duration=max(0.6, args.duration),
        auto_trim=not args.no_auto_trim,
        seed=args.seed,
        train_only=args.train_only,
        epochs=max(20, args.epochs),
        learning_rate=max(1e-4, args.learning_rate),
        hidden_dim=max(8, args.hidden_dim),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
