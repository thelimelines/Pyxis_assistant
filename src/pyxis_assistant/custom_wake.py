from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def _normalize_audio(audio: np.ndarray) -> np.ndarray:
    if audio.dtype == np.int16:
        normalized = audio.astype(np.float32) / 32768.0
    else:
        normalized = audio.astype(np.float32)
    if normalized.size == 0:
        return normalized
    peak = float(np.max(np.abs(normalized)))
    if peak > 0:
        normalized = normalized / peak
    return np.clip(normalized, -1.0, 1.0)


def extract_wake_features(audio: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
    normalized = _normalize_audio(audio)
    if normalized.size == 0:
        return np.zeros(66, dtype=np.float32)
    if sample_rate != 16000:
        msg = "Custom wake model expects 16kHz audio."
        raise ValueError(msg)

    frame_size = 400  # 25ms
    hop = 160  # 10ms
    if normalized.size < frame_size:
        normalized = np.pad(normalized, (0, frame_size - normalized.size))

    frame_count = 1 + (normalized.size - frame_size) // hop
    frames = np.stack(
        [normalized[i * hop : i * hop + frame_size] for i in range(frame_count)],
        axis=0,
    )

    window = np.hanning(frame_size).astype(np.float32)
    windowed = frames * window
    fft = np.abs(np.fft.rfft(windowed, n=512))
    bands = np.array_split(fft, 32, axis=1)
    band_energy = np.stack([band.mean(axis=1) for band in bands], axis=1)
    band_energy = np.log1p(band_energy)

    mean_vec = band_energy.mean(axis=0)
    std_vec = band_energy.std(axis=0)
    rms = float(np.sqrt(np.mean(np.square(normalized))))
    zcr = float(np.mean(np.abs(np.diff(np.signbit(normalized).astype(np.float32)))))
    features = np.concatenate([mean_vec, std_vec, np.array([rms, zcr], dtype=np.float32)])
    return features.astype(np.float32)


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    an = float(np.linalg.norm(a))
    bn = float(np.linalg.norm(b))
    if an == 0.0 or bn == 0.0:
        return 0.0
    return float(np.dot(a, b) / (an * bn))


def _sigmoid(x: np.ndarray) -> np.ndarray:
    clipped = np.clip(x, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-clipped))


@dataclass
class CustomWakeModel:
    threshold: float
    model_type: str = "mlp"
    feature_mean: np.ndarray | None = None
    feature_std: np.ndarray | None = None
    w1: np.ndarray | None = None
    b1: np.ndarray | None = None
    w2: np.ndarray | None = None
    b2: np.ndarray | None = None
    positive_mean: np.ndarray | None = None
    negative_mean: np.ndarray | None = None

    def _score_legacy(self, features: np.ndarray) -> float:
        if self.positive_mean is None or self.negative_mean is None:
            return 0.0
        raw = _cosine_similarity(features, self.positive_mean) - _cosine_similarity(
            features, self.negative_mean
        )
        scaled = (raw - self.threshold) * 8.0
        return float(1.0 / (1.0 + math.exp(-scaled)))

    def _score_mlp(self, features: np.ndarray) -> float:
        if (
            self.feature_mean is None
            or self.feature_std is None
            or self.w1 is None
            or self.b1 is None
            or self.w2 is None
            or self.b2 is None
        ):
            return 0.0
        x = (features - self.feature_mean) / np.maximum(self.feature_std, 1e-6)
        hidden = np.maximum(0.0, x @ self.w1 + self.b1)
        bias = float(self.b2[0]) if self.b2.size else 0.0
        logit = float(hidden @ self.w2 + bias)
        prob = float(_sigmoid(np.array([logit], dtype=np.float32))[0])
        return prob

    def score(self, audio: np.ndarray, sample_rate: int = 16000) -> float:
        features = extract_wake_features(audio, sample_rate=sample_rate)
        if self.model_type == "legacy":
            return self._score_legacy(features)
        return self._score_mlp(features)

    def to_json(self) -> dict[str, object]:
        if self.model_type == "legacy":
            return {
                "model_type": "legacy",
                "positive_mean": (
                    [] if self.positive_mean is None else self.positive_mean.tolist()
                ),
                "negative_mean": (
                    [] if self.negative_mean is None else self.negative_mean.tolist()
                ),
                "threshold": self.threshold,
            }
        return {
            "model_type": "mlp",
            "threshold": self.threshold,
            "feature_mean": [] if self.feature_mean is None else self.feature_mean.tolist(),
            "feature_std": [] if self.feature_std is None else self.feature_std.tolist(),
            "w1": [] if self.w1 is None else self.w1.tolist(),
            "b1": [] if self.b1 is None else self.b1.tolist(),
            "w2": [] if self.w2 is None else self.w2.tolist(),
            "b2": [] if self.b2 is None else self.b2.tolist(),
        }

    @classmethod
    def from_json(cls, payload: dict[str, object]) -> CustomWakeModel:
        model_type_raw = payload.get("model_type")
        model_type = str(model_type_raw).strip().lower() if model_type_raw is not None else ""

        threshold_raw = payload.get("threshold")
        threshold = float(threshold_raw) if isinstance(threshold_raw, (int, float, str)) else 0.5

        if model_type == "mlp":
            def _arr(name: str) -> np.ndarray:
                raw = payload.get(name)
                if not isinstance(raw, list):
                    msg = f"Invalid wake model field: {name}."
                    raise ValueError(msg)
                return np.array(raw, dtype=np.float32)

            feature_mean = _arr("feature_mean")
            feature_std = _arr("feature_std")
            w1 = _arr("w1")
            b1 = _arr("b1")
            w2 = _arr("w2")
            b2 = _arr("b2")
            if feature_mean.size == 0 or w1.size == 0:
                msg = "Invalid MLP wake model payload."
                raise ValueError(msg)
            hidden_dim = int(b1.size)
            input_dim = feature_mean.size
            if w1.size != input_dim * hidden_dim:
                msg = "Invalid MLP wake model w1 shape."
                raise ValueError(msg)
            if w2.size != hidden_dim:
                msg = "Invalid MLP wake model w2 shape."
                raise ValueError(msg)
            return cls(
                threshold=threshold,
                model_type="mlp",
                feature_mean=feature_mean,
                feature_std=feature_std,
                w1=w1.reshape(input_dim, hidden_dim),
                b1=b1,
                w2=w2.reshape(hidden_dim),
                b2=b2.reshape(1),
            )

        positive_raw = payload.get("positive_mean")
        negative_raw = payload.get("negative_mean")
        if not isinstance(positive_raw, list) or not isinstance(negative_raw, list):
            msg = "Invalid wake model payload."
            raise ValueError(msg)
        positive = np.array([float(value) for value in positive_raw], dtype=np.float32)
        negative = np.array([float(value) for value in negative_raw], dtype=np.float32)
        return cls(
            threshold=threshold,
            model_type="legacy",
            positive_mean=positive,
            negative_mean=negative,
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_json(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> CustomWakeModel:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            msg = "Invalid wake model payload."
            raise ValueError(msg)
        return cls.from_json(payload)


def _balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    pos = y_true == 1
    neg = y_true == 0
    pos_acc = float(np.mean(y_pred[pos] == 1)) if np.any(pos) else 0.0
    neg_acc = float(np.mean(y_pred[neg] == 0)) if np.any(neg) else 0.0
    return 0.5 * (pos_acc + neg_acc)


def _choose_threshold(probs: np.ndarray, y_true: np.ndarray) -> float:
    candidates = np.unique(probs)
    if candidates.size == 0:
        return 0.5
    best = 0.5
    best_score = -1.0
    for threshold in candidates:
        pred = (probs >= threshold).astype(np.int8)
        score = _balanced_accuracy(y_true, pred)
        if score > best_score:
            best_score = score
            best = float(threshold)
    return best


def train_custom_wake_model(
    positive_audios: list[np.ndarray],
    negative_audios: list[np.ndarray],
    sample_rate: int = 16000,
    epochs: int = 260,
    learning_rate: float = 0.02,
    hidden_dim: int = 40,
    seed: int = 42,
) -> CustomWakeModel:
    if not positive_audios or not negative_audios:
        msg = "Need both positive and negative samples to train."
        raise ValueError(msg)

    positive_features = np.stack(
        [extract_wake_features(audio, sample_rate=sample_rate) for audio in positive_audios]
    ).astype(np.float32)
    negative_features = np.stack(
        [extract_wake_features(audio, sample_rate=sample_rate) for audio in negative_audios]
    ).astype(np.float32)

    x = np.vstack([positive_features, negative_features]).astype(np.float32)
    y = np.concatenate(
        [
            np.ones(len(positive_features), dtype=np.float32),
            np.zeros(len(negative_features), dtype=np.float32),
        ]
    )

    rng = np.random.default_rng(seed)
    indices = np.arange(x.shape[0], dtype=np.int32)
    rng.shuffle(indices)
    x = x[indices]
    y = y[indices]

    feature_mean = x.mean(axis=0)
    feature_std = x.std(axis=0) + 1e-6
    x_norm = (x - feature_mean) / feature_std

    input_dim = x_norm.shape[1]
    hidden_dim = max(8, int(hidden_dim))
    w1 = (rng.normal(0.0, 0.08, size=(input_dim, hidden_dim))).astype(np.float32)
    b1 = np.zeros(hidden_dim, dtype=np.float32)
    w2 = (rng.normal(0.0, 0.08, size=(hidden_dim,))).astype(np.float32)
    b2 = np.zeros(1, dtype=np.float32)

    pos_count = float(np.sum(y == 1.0))
    neg_count = float(np.sum(y == 0.0))
    pos_weight = neg_count / max(1.0, pos_count)

    batch_size = min(64, x_norm.shape[0])
    for _ in range(max(20, int(epochs))):
        batch_indices = rng.permutation(x_norm.shape[0])
        for start in range(0, x_norm.shape[0], batch_size):
            idx = batch_indices[start : start + batch_size]
            xb = x_norm[idx]
            yb = y[idx]

            z1 = xb @ w1 + b1
            h = np.maximum(0.0, z1)
            logits = h @ w2 + b2
            probs = _sigmoid(logits).reshape(-1)

            sample_weight = np.where(yb > 0.5, pos_weight, 1.0).astype(np.float32)
            grad_logits = (probs - yb) * sample_weight / max(1, yb.size)

            grad_w2 = h.T @ grad_logits
            grad_b2 = np.array([np.sum(grad_logits)], dtype=np.float32)
            grad_h = grad_logits[:, None] * w2[None, :]
            grad_z1 = grad_h * (z1 > 0).astype(np.float32)
            grad_w1 = xb.T @ grad_z1
            grad_b1 = grad_z1.sum(axis=0)

            w2 -= learning_rate * grad_w2.astype(np.float32)
            b2 -= learning_rate * grad_b2
            w1 -= learning_rate * grad_w1.astype(np.float32)
            b1 -= learning_rate * grad_b1.astype(np.float32)

    final_h = np.maximum(0.0, x_norm @ w1 + b1)
    final_probs = _sigmoid(final_h @ w2 + b2).reshape(-1)
    threshold = _choose_threshold(final_probs, y.astype(np.int8))

    return CustomWakeModel(
        threshold=threshold,
        model_type="mlp",
        feature_mean=feature_mean.astype(np.float32),
        feature_std=feature_std.astype(np.float32),
        w1=w1.astype(np.float32),
        b1=b1.astype(np.float32),
        w2=w2.astype(np.float32),
        b2=b2.astype(np.float32),
    )
