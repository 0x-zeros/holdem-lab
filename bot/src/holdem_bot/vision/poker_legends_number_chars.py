"""Character-level number recognizer prototypes for Poker Legends crops."""

from __future__ import annotations

import argparse
import html
import importlib
import json
import math
import shutil
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, cast

import cv2
import numpy as np
from numpy.typing import NDArray

from holdem_bot.vision import poker_legends_numbers
from holdem_bot.vision.poker_legends_ctc import (
    CtcAlphabet,
    CtcTimeStepBudget,
    ctc_required_timesteps,
    ctc_time_step_budget,
    validate_ctc_log_probs_shape,
)

RgbImage = NDArray[np.uint8]
GrayImage = NDArray[np.uint8]
FloatImage = NDArray[np.float32]
FloatVector = NDArray[np.float32]

DEFAULT_GLYPH_WIDTH = 24
DEFAULT_GLYPH_HEIGHT = 32
DEFAULT_SEQUENCE_WIDTH = 160
DEFAULT_SEQUENCE_HEIGHT = 32
DEFAULT_TEMPLATE_MAX_DISTANCE = 0.060
DEFAULT_TEMPLATE_VOTE_DISTANCE_WINDOW = 0.006
DEFAULT_TEMPLATE_VOTE_NEIGHBORS = 9
DEFAULT_TEMPLATE_CNN_AGREEMENT_MAX_TEMPLATE_DISTANCE = 0.070
DEFAULT_TEMPLATE_CNN_AGREEMENT_MIN_CNN_CONFIDENCE = 0.70
DEFAULT_MLP_MIN_CONFIDENCE = 0.60
DEFAULT_CNN_MIN_CONFIDENCE = 0.80
DEFAULT_CNN_EPOCHS = 60
DEFAULT_CNN_BATCH_SIZE = 64
DEFAULT_CNN_SEED = 1729
DEFAULT_CTC_MIN_CONFIDENCE = 0.80
DEFAULT_CTC_MIN_TIMESTEP_RATIO = 2.0
DEFAULT_CTC_MIN_EFFECTIVE_TIMESTEPS = 24
DEFAULT_CRNN_EPOCHS = 12
DEFAULT_TRANSFORMER_EPOCHS = 12
DEFAULT_CTC_BATCH_SIZE = 32
DEFAULT_CTC_SEED = 2027
DEFAULT_CTC_LEARNING_RATE = 0.001
DEFAULT_CTC_WEIGHT_DECAY = 0.0001
DEFAULT_TEST_FRAME_MODULO = 5
NUMBER_CHAR_RECOGNIZER_SUMMARY = "number_char_recognizer_summary.json"
NUMBER_CHAR_RECOGNIZER_REPORT = "number_char_recognizer_report.md"
NUMBER_CHAR_RECOGNIZER_REVIEW_HTML = "number_char_recognizer_review.html"
NUMBER_CHAR_RECOGNIZER_REVIEW_MD = "number_char_recognizer_review.md"
ALLOWED_CHARS = frozenset("$0123456789+,.KM")
RecognitionMethod = Literal[
    "template",
    "opencv_mlp",
    "cnn",
    "crnn_ctc",
    "transformer_ctc",
    "template_cnn",
    "tesseract",
]
NumberTextTarget = Literal["base", "overlay", "display"]
NUMBER_TEXT_TARGETS: tuple[NumberTextTarget, ...] = ("base", "overlay", "display")
SequenceArchitecture = Literal["crnn_ctc", "transformer_ctc"]


@dataclass(frozen=True, slots=True)
class NumberCharBox:
    x: int
    y: int
    width: int
    height: int
    area: int

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class NumberGlyphSample:
    label: str
    target: NumberTextTarget
    frame_id: str
    row_id: str
    crop_path: str
    glyph_path: str
    box: NumberCharBox

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["box"] = self.box.to_dict()
        return data


@dataclass(frozen=True, slots=True)
class NumberSequenceSample:
    target: NumberTextTarget
    frame_id: str
    row_id: str
    crop_path: str
    text: str
    image: FloatImage


@dataclass(frozen=True, slots=True)
class NumberCharTargets:
    base: str | None
    overlay: str | None
    display: str | None

    def for_target(self, target: NumberTextTarget) -> str | None:
        if target == "base":
            return self.base
        if target == "overlay":
            return self.overlay
        return self.display

    def to_dict(self) -> dict[str, str | None]:
        return {"base": self.base, "overlay": self.overlay, "display": self.display}


@dataclass(frozen=True, slots=True)
class NumberCharRow:
    row_id: str
    frame_id: str
    group: str
    name: str
    crop_variant: str
    crop_path: Path
    expected: NumberCharTargets
    split: str
    clean_status: str = "labeled_visible"


@dataclass(frozen=True, slots=True)
class StringPrediction:
    method: RecognitionMethod
    text: str | None
    confidence: float
    accepted: bool
    reason: str
    char_distances: tuple[float, ...] = ()
    char_confidences: tuple[float, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class _TemplateGlyph:
    label: str
    frame_id: str
    row_id: str
    image: FloatImage


@dataclass(frozen=True, slots=True)
class _TargetEvaluation:
    target: NumberTextTarget
    expected_text: str
    is_positive: bool
    segmentation_boxes: tuple[NumberCharBox, ...]
    segmentation_status: str
    template: StringPrediction
    opencv_mlp: StringPrediction
    cnn: StringPrediction
    crnn_ctc: StringPrediction
    transformer_ctc: StringPrediction
    template_cnn: StringPrediction
    tesseract: StringPrediction


@dataclass(frozen=True, slots=True)
class _EvaluationRow:
    source: NumberCharRow
    targets: tuple[_TargetEvaluation, ...]


class NumberTemplateRecognizer:
    def __init__(
        self,
        samples: Sequence[NumberGlyphSample],
        *,
        glyph_root: str | Path,
        max_distance: float = DEFAULT_TEMPLATE_MAX_DISTANCE,
        vote_distance_window: float = DEFAULT_TEMPLATE_VOTE_DISTANCE_WINDOW,
        vote_neighbors: int = DEFAULT_TEMPLATE_VOTE_NEIGHBORS,
    ) -> None:
        self.max_distance = max_distance
        self.vote_distance_window = vote_distance_window
        self.vote_neighbors = max(1, vote_neighbors)
        root = Path(glyph_root)
        self._glyphs = tuple(
            _TemplateGlyph(
                label=sample.label,
                frame_id=sample.frame_id,
                row_id=sample.row_id,
                image=_load_glyph(root / sample.glyph_path),
            )
            for sample in samples
        )

    def recognize(
        self,
        glyphs: Sequence[FloatImage],
        *,
        exclude_frame_id: str | None = None,
    ) -> StringPrediction:
        if not glyphs:
            return StringPrediction(
                method="template",
                text=None,
                confidence=0.0,
                accepted=False,
                reason="no_glyphs",
            )
        predicted: list[str] = []
        distances: list[float] = []
        for glyph in glyphs:
            match = self._best_match(glyph, exclude_frame_id=exclude_frame_id)
            if match is None:
                return StringPrediction(
                    method="template",
                    text=None,
                    confidence=0.0,
                    accepted=False,
                    reason="no_template",
                    char_distances=tuple(distances),
                )
            label, distance = match
            predicted.append(label)
            distances.append(distance)
        worst_distance = max(distances) if distances else math.inf
        confidence = max(0.0, 1.0 - worst_distance / max(self.max_distance, 1e-9))
        return StringPrediction(
            method="template",
            text="".join(predicted),
            confidence=confidence,
            accepted=worst_distance <= self.max_distance,
            reason="accepted" if worst_distance <= self.max_distance else "distance",
            char_distances=tuple(distances),
        )

    def _best_match(
        self,
        glyph: FloatImage,
        *,
        exclude_frame_id: str | None,
    ) -> tuple[str, float] | None:
        matches: list[tuple[str, float]] = []
        for template in self._glyphs:
            if exclude_frame_id is not None and template.frame_id == exclude_frame_id:
                continue
            distance = float(np.mean((glyph - template.image) ** 2))
            matches.append((template.label, distance))
        if not matches:
            return None
        matches.sort(key=lambda item: item[1])
        best_distance = matches[0][1]
        close_matches = [
            match
            for match in matches[: self.vote_neighbors]
            if match[1] <= best_distance + self.vote_distance_window
        ]
        label_counts = Counter(label for label, _distance in close_matches)
        label_scores: dict[str, float] = {}
        label_distances: dict[str, float] = {}
        for label, distance in close_matches:
            label_scores[label] = label_scores.get(label, 0.0) + 1.0 / (distance + 1e-9)
            label_distances[label] = min(label_distances.get(label, math.inf), distance)
        label = min(
            label_counts,
            key=lambda item: (
                -label_counts[item],
                -label_scores[item],
                label_distances[item],
                item,
            ),
        )
        return label, label_distances[label]


class NumberOpenCvMlpRecognizer:
    def __init__(
        self,
        samples: Sequence[NumberGlyphSample],
        *,
        glyph_root: str | Path,
        min_confidence: float = DEFAULT_MLP_MIN_CONFIDENCE,
    ) -> None:
        self.min_confidence = min_confidence
        self._labels = tuple(sorted({sample.label for sample in samples}))
        self._label_to_index = {label: index for index, label in enumerate(self._labels)}
        self._model: Any | None = self._train(samples, glyph_root=Path(glyph_root))

    @property
    def available(self) -> bool:
        return self._model is not None and bool(self._labels)

    def recognize(self, glyphs: Sequence[FloatImage]) -> StringPrediction:
        if not glyphs:
            return StringPrediction(
                method="opencv_mlp",
                text=None,
                confidence=0.0,
                accepted=False,
                reason="no_glyphs",
            )
        if self._model is None:
            return StringPrediction(
                method="opencv_mlp",
                text=None,
                confidence=0.0,
                accepted=False,
                reason="unavailable",
            )
        predicted: list[str] = []
        confidences: list[float] = []
        for glyph in glyphs:
            sample = _glyph_vector(glyph).reshape(1, -1)
            _ret, output = self._model.predict(sample)
            scores = cast(NDArray[np.float32], output).reshape(-1)
            probabilities = _softmax(scores)
            index = int(np.argmax(probabilities))
            predicted.append(self._labels[index])
            confidences.append(float(probabilities[index]))
        confidence = min(confidences) if confidences else 0.0
        return StringPrediction(
            method="opencv_mlp",
            text="".join(predicted),
            confidence=confidence,
            accepted=confidence >= self.min_confidence,
            reason="accepted" if confidence >= self.min_confidence else "confidence",
            char_confidences=tuple(confidences),
        )

    def _train(
        self,
        samples: Sequence[NumberGlyphSample],
        *,
        glyph_root: Path,
    ) -> Any | None:
        if len(samples) < 2 or len(self._labels) < 2:
            return None
        rows: list[FloatVector] = []
        labels: list[int] = []
        for sample in samples:
            rows.append(_glyph_vector(_load_glyph(glyph_root / sample.glyph_path)))
            labels.append(self._label_to_index[sample.label])
        train_data = np.vstack(rows).astype(np.float32)
        responses = np.zeros((len(labels), len(self._labels)), dtype=np.float32)
        for row_index, label_index in enumerate(labels):
            responses[row_index, label_index] = 1.0
        ml: Any = cv2.ml
        model = ml.ANN_MLP_create()
        model.setLayerSizes(
            np.array([train_data.shape[1], 64, len(self._labels)], dtype=np.int32)
        )
        model.setActivationFunction(ml.ANN_MLP_SIGMOID_SYM)
        model.setTrainMethod(ml.ANN_MLP_BACKPROP, 0.01, 0.1)
        model.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 300, 1e-4))
        ok = model.train(train_data, cv2.ml.ROW_SAMPLE, responses)
        return model if ok else None


class NumberTorchCnnRecognizer:
    def __init__(
        self,
        samples: Sequence[NumberGlyphSample],
        *,
        glyph_root: str | Path,
        min_confidence: float = DEFAULT_CNN_MIN_CONFIDENCE,
        epochs: int = DEFAULT_CNN_EPOCHS,
        batch_size: int = DEFAULT_CNN_BATCH_SIZE,
        seed: int = DEFAULT_CNN_SEED,
    ) -> None:
        self.min_confidence = min_confidence
        self.epochs = max(1, epochs)
        self.batch_size = max(1, batch_size)
        self.seed = seed
        self._labels = tuple(sorted({sample.label for sample in samples}))
        self._label_to_index = {label: index for index, label in enumerate(self._labels)}
        self._torch: Any | None = None
        self._model: Any | None = None
        self._reason = "untrained"
        self._train(samples, glyph_root=Path(glyph_root))

    @property
    def available(self) -> bool:
        return self._model is not None and self._torch is not None and bool(self._labels)

    @property
    def reason(self) -> str:
        return self._reason

    def recognize(self, glyphs: Sequence[FloatImage]) -> StringPrediction:
        if not glyphs:
            return StringPrediction(
                method="cnn",
                text=None,
                confidence=0.0,
                accepted=False,
                reason="no_glyphs",
            )
        if not self.available:
            return StringPrediction(
                method="cnn",
                text=None,
                confidence=0.0,
                accepted=False,
                reason=self._reason,
            )
        torch = self._torch
        model = self._model
        assert torch is not None
        assert model is not None
        batch = np.stack(glyphs).astype(np.float32)[:, None, :, :]
        with torch.inference_mode():
            logits = model(torch.from_numpy(batch))
            probabilities = torch.softmax(logits, dim=1).detach().cpu().numpy()
        predicted: list[str] = []
        confidences: list[float] = []
        for row in cast(NDArray[np.float32], probabilities):
            index = int(np.argmax(row))
            predicted.append(self._labels[index])
            confidences.append(float(row[index]))
        confidence = min(confidences) if confidences else 0.0
        return StringPrediction(
            method="cnn",
            text="".join(predicted),
            confidence=confidence,
            accepted=confidence >= self.min_confidence,
            reason="accepted" if confidence >= self.min_confidence else "confidence",
            char_confidences=tuple(confidences),
        )

    def _train(
        self,
        samples: Sequence[NumberGlyphSample],
        *,
        glyph_root: Path,
    ) -> None:
        if len(samples) < 2 or len(self._labels) < 2:
            self._reason = "insufficient_training_data"
            return
        try:
            torch = cast(Any, importlib.import_module("torch"))
            nn = cast(Any, importlib.import_module("torch.nn"))
            optim = cast(Any, importlib.import_module("torch.optim"))
        except ImportError:
            self._reason = "torch_unavailable"
            return
        rows: list[FloatImage] = []
        labels: list[int] = []
        for sample in samples:
            rows.append(_load_glyph(glyph_root / sample.glyph_path))
            labels.append(self._label_to_index[sample.label])
        train_data = np.stack(rows).astype(np.float32)[:, None, :, :]
        train_labels = np.array(labels, dtype=np.int64)
        torch.manual_seed(self.seed)
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:  # pragma: no cover - older or platform-specific torch builds.
            pass
        model = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32 * (DEFAULT_GLYPH_HEIGHT // 4) * (DEFAULT_GLYPH_WIDTH // 4), 64),
            nn.ReLU(),
            nn.Linear(64, len(self._labels)),
        )
        optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
        criterion = nn.CrossEntropyLoss()
        x = torch.from_numpy(train_data)
        y = torch.from_numpy(train_labels)
        model.train()
        for _epoch in range(self.epochs):
            order = torch.randperm(x.shape[0])
            for start in range(0, x.shape[0], self.batch_size):
                batch_index = order[start : start + self.batch_size]
                logits = model(x[batch_index])
                loss = criterion(logits, y[batch_index])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        model.eval()
        self._torch = torch
        self._model = model
        self._reason = "trained"


class NumberTorchCtcRecognizer:
    def __init__(
        self,
        samples: Sequence[NumberSequenceSample],
        *,
        architecture: SequenceArchitecture,
        min_confidence: float = DEFAULT_CTC_MIN_CONFIDENCE,
        epochs: int = DEFAULT_CRNN_EPOCHS,
        batch_size: int = DEFAULT_CTC_BATCH_SIZE,
        seed: int = DEFAULT_CTC_SEED,
        learning_rate: float = DEFAULT_CTC_LEARNING_RATE,
        weight_decay: float = DEFAULT_CTC_WEIGHT_DECAY,
        progress_callback: Callable[[int, int, float], None] | None = None,
    ) -> None:
        self.architecture = architecture
        self.min_confidence = min_confidence
        self.epochs = max(1, epochs)
        self.batch_size = max(1, batch_size)
        self.seed = seed
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self._progress_callback = progress_callback
        self._alphabet = CtcAlphabet.from_texts(sample.text for sample in samples)
        self._labels = self._alphabet.labels
        self._torch: Any | None = None
        self._model: Any | None = None
        self._reason = "untrained"
        self._input_time_steps: int | None = None
        self._time_step_budgets: tuple[CtcTimeStepBudget, ...] = ()
        self._train(samples)

    @property
    def available(self) -> bool:
        return self._model is not None and self._torch is not None and bool(self._labels)

    @property
    def reason(self) -> str:
        return self._reason

    @property
    def time_step_budget_summary(self) -> dict[str, object]:
        if not self._time_step_budgets:
            return {
                "input_timesteps": self._input_time_steps,
                "checked": 0,
                "failed": 0,
            }
        ratios = [budget.ratio for budget in self._time_step_budgets if math.isfinite(budget.ratio)]
        failed = [budget for budget in self._time_step_budgets if not budget.valid]
        max_required = max(budget.required_timesteps for budget in self._time_step_budgets)
        max_target_length = max(budget.target_length for budget in self._time_step_budgets)
        input_timesteps = [budget.input_timesteps for budget in self._time_step_budgets]
        return {
            "input_timesteps": self._input_time_steps,
            "min_effective_input_timesteps": min(input_timesteps),
            "max_effective_input_timesteps": max(input_timesteps),
            "min_ratio": DEFAULT_CTC_MIN_TIMESTEP_RATIO,
            "checked": len(self._time_step_budgets),
            "failed": len(failed),
            "max_required_timesteps": max_required,
            "max_target_length": max_target_length,
            "min_observed_ratio": min(ratios) if ratios else None,
            "failed_examples": [
                budget.to_dict() for budget in sorted(failed, key=lambda item: item.ratio)[:5]
            ],
        }

    def recognize(self, image: FloatImage) -> StringPrediction:
        if image.size == 0:
            return StringPrediction(
                method=self.architecture,
                text=None,
                confidence=0.0,
                accepted=False,
                reason="empty_image",
            )
        if not self.available:
            return StringPrediction(
                method=self.architecture,
                text=None,
                confidence=0.0,
                accepted=False,
                reason=self._reason,
            )
        torch = self._torch
        model = self._model
        assert torch is not None
        assert model is not None
        batch = image.astype(np.float32)[None, None, :, :]
        with torch.inference_mode():
            logits = model(torch.from_numpy(batch))
            probabilities = torch.softmax(logits, dim=2).detach().cpu().numpy()
        effective_timesteps = _effective_sequence_timesteps(
            image,
            full_timesteps=int(probabilities.shape[0]),
        )
        text, confidences = self._decode_probabilities(
            cast(NDArray[np.float32], probabilities[:effective_timesteps])
        )
        if not text:
            return StringPrediction(
                method=self.architecture,
                text=None,
                confidence=0.0,
                accepted=False,
                reason="blank",
            )
        confidence = min(confidences) if confidences else 0.0
        return StringPrediction(
            method=self.architecture,
            text=text,
            confidence=confidence,
            accepted=confidence >= self.min_confidence,
            reason="accepted" if confidence >= self.min_confidence else "confidence",
            char_confidences=tuple(confidences),
        )

    def _train(self, samples: Sequence[NumberSequenceSample]) -> None:
        if len(samples) < 2 or len(self._labels) < 2:
            self._reason = "insufficient_training_data"
            return
        try:
            torch = cast(Any, importlib.import_module("torch"))
            nn = cast(Any, importlib.import_module("torch.nn"))
            optim = cast(Any, importlib.import_module("torch.optim"))
        except ImportError:
            self._reason = "torch_unavailable"
            return
        encoded_samples: list[tuple[FloatImage, list[int], str]] = []
        for sample in samples:
            encoded = list(self._alphabet.encode(sample.text))
            if encoded:
                encoded_samples.append((sample.image, encoded, sample.text))
        if len(encoded_samples) < 2:
            self._reason = "insufficient_training_data"
            return
        torch.manual_seed(self.seed)
        nn_module = nn
        model = (
            self._build_crnn_model(nn_module, len(self._labels) + 1)
            if self.architecture == "crnn_ctc"
            else self._build_transformer_model(nn_module, len(self._labels) + 1)
        )
        optimizer = optim.AdamW(
            model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        images = np.stack([sample[0] for sample in encoded_samples]).astype(np.float32)
        targets = [sample[1] for sample in encoded_samples]
        texts = [sample[2] for sample in encoded_samples]
        x = torch.from_numpy(images[:, None, :, :])
        with torch.inference_mode():
            sample_logits = model(x[:1])
        shape = validate_ctc_log_probs_shape(
            tuple(int(value) for value in sample_logits.shape),
            batch_size=1,
            class_count=self._alphabet.class_count,
        )
        self._input_time_steps = shape.timesteps
        effective_input_lengths = [
            _effective_sequence_timesteps_for_text(
                image,
                texts[index],
                full_timesteps=shape.timesteps,
            )
            for index, image in enumerate(images)
        ]
        self._time_step_budgets = tuple(
            ctc_time_step_budget(
                text,
                input_timesteps=effective_input_lengths[index],
                min_ratio=DEFAULT_CTC_MIN_TIMESTEP_RATIO,
            )
            for index, text in enumerate(texts)
        )
        if any(not budget.valid for budget in self._time_step_budgets):
            self._reason = "ctc_time_budget_failed"
            return
        criterion = nn.CTCLoss(blank=self._alphabet.blank_index, zero_infinity=False)
        model.train()
        for epoch in range(self.epochs):
            order = torch.randperm(x.shape[0])
            last_loss = math.nan
            for start in range(0, x.shape[0], self.batch_size):
                batch_index = order[start : start + self.batch_size]
                batch_targets = [targets[int(index)] for index in batch_index]
                target_lengths = torch.tensor(
                    [len(target) for target in batch_targets],
                    dtype=torch.long,
                )
                flat_targets = torch.tensor(
                    [item for target in batch_targets for item in target],
                    dtype=torch.long,
                )
                logits = model(x[batch_index])
                validate_ctc_log_probs_shape(
                    tuple(int(value) for value in logits.shape),
                    batch_size=len(batch_targets),
                    class_count=self._alphabet.class_count,
                )
                log_probs = torch.log_softmax(logits, dim=2)
                input_lengths = torch.tensor(
                    [effective_input_lengths[int(index)] for index in batch_index],
                    dtype=torch.long,
                )
                loss = criterion(log_probs, flat_targets, input_lengths, target_lengths)
                last_loss = float(loss.detach().cpu().item())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            if self._progress_callback is not None:
                self._progress_callback(epoch + 1, self.epochs, last_loss)
        model.eval()
        self._torch = torch
        self._model = model
        self._reason = "trained"

    def _decode_probabilities(
        self,
        probabilities: NDArray[np.float32],
    ) -> tuple[str, list[float]]:
        steps = probabilities[:, 0, :]
        indices = np.argmax(steps, axis=1)
        top_probabilities = np.max(steps, axis=1)
        text, confidences = self._alphabet.greedy_decode(
            [int(index) for index in indices],
            [float(confidence) for confidence in top_probabilities],
        )
        return text, list(confidences)

    @staticmethod
    def _build_crnn_model(nn: Any, output_classes: int) -> Any:
        class _CrnnCtcModel(nn.Module):  # type: ignore[misc]
            def __init__(self) -> None:
                super().__init__()
                self.conv = nn.Sequential(
                    nn.Conv2d(1, 16, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                    nn.Conv2d(16, 32, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                    nn.Conv2d(32, 64, kernel_size=3, padding=1),
                    nn.ReLU(),
                )
                self.rnn = nn.LSTM(
                    input_size=64,
                    hidden_size=64,
                    num_layers=1,
                    bidirectional=True,
                )
                self.out = nn.Linear(128, output_classes)

            def forward(self, x: Any) -> Any:
                features = self.conv(x).mean(dim=2)
                sequence = features.permute(2, 0, 1)
                encoded, _state = self.rnn(sequence)
                return self.out(encoded)

        return _CrnnCtcModel()

    @staticmethod
    def _build_transformer_model(nn: Any, output_classes: int) -> Any:
        class _TransformerCtcModel(nn.Module):  # type: ignore[misc]
            def __init__(self) -> None:
                super().__init__()
                self.conv = nn.Sequential(
                    nn.Conv2d(1, 32, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                    nn.Conv2d(32, 64, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                )
                self.positional = nn.Parameter(
                    # Conv/pool stack maps width 160 -> 40.
                    importlib.import_module("torch").zeros(1, 40, 64)
                )
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=64,
                    nhead=4,
                    dim_feedforward=128,
                    dropout=0.0,
                    batch_first=True,
                )
                self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
                self.out = nn.Linear(64, output_classes)

            def forward(self, x: Any) -> Any:
                features = self.conv(x).mean(dim=2)
                sequence = features.permute(0, 2, 1)
                sequence = sequence + self.positional[:, : sequence.shape[1], :]
                encoded = self.encoder(sequence)
                return self.out(encoded).permute(1, 0, 2)

        return _TransformerCtcModel()


def apply_number_text_target_contract(
    prediction: StringPrediction,
    *,
    target: NumberTextTarget,
    is_stack: bool,
) -> StringPrediction:
    if prediction.text is None or not prediction.accepted:
        return prediction
    if _target_contract_accepts_text(prediction.text, target=target, is_stack=is_stack):
        return prediction
    return StringPrediction(
        method=prediction.method,
        text=prediction.text,
        confidence=prediction.confidence,
        accepted=False,
        reason="format",
        char_distances=prediction.char_distances,
        char_confidences=prediction.char_confidences,
    )


def _ctc_training_text(
    text: str,
    *,
    target: NumberTextTarget,
    strip_fixed_prefix: bool,
) -> str:
    if not strip_fixed_prefix:
        return text
    if target == "base" and text.startswith("$"):
        return text[1:]
    if target == "overlay" and text.startswith("+"):
        return text[1:]
    return text


def _restore_ctc_fixed_prefix(
    prediction: StringPrediction,
    *,
    target: NumberTextTarget,
    strip_fixed_prefix: bool,
) -> StringPrediction:
    if not strip_fixed_prefix or prediction.text is None:
        return prediction
    prefix = "$" if target == "base" else "+" if target == "overlay" else ""
    if not prefix or prediction.text.startswith(prefix):
        return prediction
    return StringPrediction(
        method=prediction.method,
        text=f"{prefix}{prediction.text}",
        confidence=prediction.confidence,
        accepted=prediction.accepted,
        reason=prediction.reason,
        char_distances=prediction.char_distances,
        char_confidences=prediction.char_confidences,
    )


def build_and_evaluate_poker_legends_number_char_recognizers(
    manifest_path: str | Path,
    *,
    output_dir: str | Path,
    field_name: str = "hero_stack",
    targets: Sequence[NumberTextTarget] = NUMBER_TEXT_TARGETS,
    max_crops: int | None = None,
    test_frame_modulo: int = DEFAULT_TEST_FRAME_MODULO,
    template_max_distance: float = DEFAULT_TEMPLATE_MAX_DISTANCE,
    mlp_min_confidence: float = DEFAULT_MLP_MIN_CONFIDENCE,
    cnn_min_confidence: float = DEFAULT_CNN_MIN_CONFIDENCE,
    cnn_epochs: int = DEFAULT_CNN_EPOCHS,
    ctc_min_confidence: float = DEFAULT_CTC_MIN_CONFIDENCE,
    crnn_epochs: int = DEFAULT_CRNN_EPOCHS,
    transformer_epochs: int = DEFAULT_TRANSFORMER_EPOCHS,
    ctc_batch_size: int = DEFAULT_CTC_BATCH_SIZE,
    ctc_learning_rate: float = DEFAULT_CTC_LEARNING_RATE,
    ctc_weight_decay: float = DEFAULT_CTC_WEIGHT_DECAY,
    ctc_seed: int = DEFAULT_CTC_SEED,
    ctc_strip_fixed_prefix: bool = False,
    enable_cnn: bool = True,
    enable_ctc: bool = False,
    enable_transformer_ctc: bool = True,
    enable_tesseract: bool = True,
) -> dict[str, object]:
    manifest_file = Path(manifest_path)
    active_targets = tuple(targets)
    if not active_targets:
        raise ValueError("at least one number text target is required")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    glyph_dir = output / "number_char_glyphs"
    glyph_dir.mkdir(parents=True, exist_ok=True)
    preview_crop_dir = output / "review_crops"
    preview_crop_dir.mkdir(parents=True, exist_ok=True)

    source_rows = _load_rows(
        manifest_file,
        field_name=field_name,
        max_crops=max_crops,
        test_frame_modulo=test_frame_modulo,
    )
    glyph_samples: list[NumberGlyphSample] = []
    sequence_samples: list[NumberSequenceSample] = []
    row_glyphs: dict[tuple[str, NumberTextTarget], tuple[FloatImage, ...]] = {}
    row_sequence_images: dict[tuple[str, NumberTextTarget], FloatImage] = {}
    segmentation_by_row: dict[
        tuple[str, NumberTextTarget], tuple[str, tuple[NumberCharBox, ...]]
    ] = {}
    for row in source_rows:
        crop = _load_rgb_image(row.crop_path)
        for target in active_targets:
            expected = row.expected.for_target(target)
            if expected is None:
                if row.clean_status == "no_visible_number":
                    mask = _text_mask(crop, target=target)
                    sequence_image = _normalize_sequence_image(mask)
                    row_sequence_images[(row.row_id, target)] = sequence_image
                    boxes = _segment_number_characters_from_mask(mask, target=target)
                    segmentation_status = (
                        "negative_glyphs" if boxes else "negative_no_glyphs"
                    )
                    segmentation_by_row[(row.row_id, target)] = (
                        segmentation_status,
                        tuple(boxes),
                    )
                    row_glyphs[(row.row_id, target)] = tuple(
                        _normalize_glyph(mask, box) for box in boxes
                    )
                continue
            mask = _text_mask(crop, target=target)
            sequence_image = _normalize_sequence_image(mask)
            row_sequence_images[(row.row_id, target)] = sequence_image
            sequence_samples.append(
                NumberSequenceSample(
                    target=target,
                    frame_id=row.frame_id,
                    row_id=row.row_id,
                    crop_path=str(row.crop_path),
                    text=_ctc_training_text(
                        expected,
                        target=target,
                        strip_fixed_prefix=ctc_strip_fixed_prefix,
                    ),
                    image=sequence_image,
                )
            )
            boxes = _segment_number_characters_from_mask(mask, target=target)
            segmentation_status = "match" if len(boxes) == len(expected) else "mismatch"
            segmentation_by_row[(row.row_id, target)] = (segmentation_status, tuple(boxes))
            if segmentation_status != "match":
                continue
            glyphs = tuple(_normalize_glyph(mask, box) for box in boxes)
            row_glyphs[(row.row_id, target)] = glyphs
            for index, (label, glyph) in enumerate(zip(expected, glyphs, strict=True)):
                glyph_path = (
                    Path("number_char_glyphs")
                    / target
                    / label_to_path_component(label)
                    / f"{row.row_id}__{index:02d}.png"
                )
                glyph_target = output / glyph_path
                glyph_target.parent.mkdir(parents=True, exist_ok=True)
                _write_glyph(glyph_target, glyph)
                glyph_samples.append(
                    NumberGlyphSample(
                        label=label,
                        target=target,
                        frame_id=row.frame_id,
                        row_id=row.row_id,
                        crop_path=str(row.crop_path),
                        glyph_path=str(glyph_path),
                        box=boxes[index],
                    )
                )

    train_frame_ids = {row.frame_id for row in source_rows if row.split == "train"}
    train_samples = [sample for sample in glyph_samples if sample.frame_id in train_frame_ids]
    train_sequence_samples = [
        sample for sample in sequence_samples if sample.frame_id in train_frame_ids
    ]
    train_samples_by_target = {
        target: [sample for sample in train_samples if sample.target == target]
        for target in active_targets
    }
    template_by_target = {
        target: NumberTemplateRecognizer(
            target_train_samples,
            glyph_root=output,
            max_distance=template_max_distance,
        )
        for target, target_train_samples in train_samples_by_target.items()
    }
    mlp_by_target = {
        target: NumberOpenCvMlpRecognizer(
            target_train_samples,
            glyph_root=output,
            min_confidence=mlp_min_confidence,
        )
        for target, target_train_samples in train_samples_by_target.items()
    }
    cnn_by_target = {
        target: (
            NumberTorchCnnRecognizer(
                target_train_samples,
                glyph_root=output,
                min_confidence=cnn_min_confidence,
                epochs=cnn_epochs,
            )
            if enable_cnn
            else None
        )
        for target, target_train_samples in train_samples_by_target.items()
    }
    crnn = (
        NumberTorchCtcRecognizer(
            train_sequence_samples,
            architecture="crnn_ctc",
            min_confidence=ctc_min_confidence,
            epochs=crnn_epochs,
            batch_size=ctc_batch_size,
            seed=ctc_seed,
            learning_rate=ctc_learning_rate,
            weight_decay=ctc_weight_decay,
        )
        if enable_ctc
        else None
    )
    transformer = (
        NumberTorchCtcRecognizer(
            train_sequence_samples,
            architecture="transformer_ctc",
            min_confidence=ctc_min_confidence,
            epochs=transformer_epochs,
            batch_size=ctc_batch_size,
            seed=ctc_seed + 1,
            learning_rate=ctc_learning_rate,
            weight_decay=ctc_weight_decay,
        )
        if enable_ctc and enable_transformer_ctc
        else None
    )

    def evaluate_row(row: NumberCharRow) -> _EvaluationRow:
        target_evaluations: list[_TargetEvaluation] = []
        tesseract_text = _tesseract_text(row.crop_path) if enable_tesseract else None
        for target in active_targets:
            expected = row.expected.for_target(target)
            if expected is None:
                if row.clean_status != "no_visible_number":
                    continue
                segmentation_status, boxes = segmentation_by_row[(row.row_id, target)]
                glyphs = row_glyphs.get((row.row_id, target), ())
                template_prediction = template_by_target[target].recognize(glyphs)
                mlp_prediction = mlp_by_target[target].recognize(glyphs)
                cnn = cnn_by_target[target]
                if cnn is None:
                    cnn_prediction = _unavailable_prediction("cnn", "disabled")
                else:
                    cnn_prediction = cnn.recognize(glyphs)
                sequence_image = row_sequence_images[(row.row_id, target)]
                crnn_prediction = (
                    apply_number_text_target_contract(
                        _restore_ctc_fixed_prefix(
                            crnn.recognize(sequence_image),
                            target=target,
                            strip_fixed_prefix=ctc_strip_fixed_prefix,
                        ),
                        target=target,
                        is_stack=_is_stack_row(row),
                    )
                    if crnn is not None
                    else _unavailable_prediction("crnn_ctc", "disabled")
                )
                transformer_prediction = (
                    apply_number_text_target_contract(
                        _restore_ctc_fixed_prefix(
                            transformer.recognize(sequence_image),
                            target=target,
                            strip_fixed_prefix=ctc_strip_fixed_prefix,
                        ),
                        target=target,
                        is_stack=_is_stack_row(row),
                    )
                    if transformer is not None
                    else _unavailable_prediction("transformer_ctc", "disabled")
                )
                template_cnn_prediction = _template_cnn_consensus_prediction(
                    template_prediction,
                    cnn_prediction,
                    target=target,
                    is_stack=_is_stack_row(row),
                )
                tesseract_prediction = (
                    _tesseract_prediction_from_text(
                        tesseract_text,
                        target=target,
                        is_stack=_is_stack_row(row),
                    )
                    if tesseract_text is not None
                    else _unavailable_prediction("tesseract", "disabled")
                )
                target_evaluations.append(
                    _TargetEvaluation(
                        target=target,
                        expected_text="",
                        is_positive=False,
                        segmentation_boxes=boxes,
                        segmentation_status=segmentation_status,
                        template=template_prediction,
                        opencv_mlp=mlp_prediction,
                        cnn=cnn_prediction,
                        crnn_ctc=crnn_prediction,
                        transformer_ctc=transformer_prediction,
                        template_cnn=template_cnn_prediction,
                        tesseract=tesseract_prediction,
                    )
                )
                continue
            segmentation_status, boxes = segmentation_by_row[(row.row_id, target)]
            glyphs = row_glyphs.get((row.row_id, target), ())
            template_prediction = (
                template_by_target[target].recognize(glyphs)
                if segmentation_status == "match"
                else _segmentation_failed_prediction("template", segmentation_status)
            )
            mlp_prediction = (
                mlp_by_target[target].recognize(glyphs)
                if segmentation_status == "match"
                else _segmentation_failed_prediction("opencv_mlp", segmentation_status)
            )
            if segmentation_status != "match":
                cnn_prediction = _segmentation_failed_prediction("cnn", segmentation_status)
            else:
                cnn = cnn_by_target[target]
                if cnn is None:
                    cnn_prediction = _unavailable_prediction("cnn", "disabled")
                else:
                    cnn_prediction = cnn.recognize(glyphs)
            sequence_image = row_sequence_images[(row.row_id, target)]
            crnn_prediction = (
                apply_number_text_target_contract(
                    _restore_ctc_fixed_prefix(
                        crnn.recognize(sequence_image),
                        target=target,
                        strip_fixed_prefix=ctc_strip_fixed_prefix,
                    ),
                    target=target,
                    is_stack=_is_stack_row(row),
                )
                if crnn is not None
                else _unavailable_prediction("crnn_ctc", "disabled")
            )
            transformer_prediction = (
                apply_number_text_target_contract(
                    _restore_ctc_fixed_prefix(
                        transformer.recognize(sequence_image),
                        target=target,
                        strip_fixed_prefix=ctc_strip_fixed_prefix,
                    ),
                    target=target,
                    is_stack=_is_stack_row(row),
                )
                if transformer is not None
                else _unavailable_prediction("transformer_ctc", "disabled")
            )
            template_cnn_prediction = _template_cnn_consensus_prediction(
                template_prediction,
                cnn_prediction,
                target=target,
                is_stack=_is_stack_row(row),
            )
            tesseract_prediction = (
                _tesseract_prediction_from_text(
                    tesseract_text,
                    target=target,
                    is_stack=_is_stack_row(row),
                )
                if tesseract_text is not None
                else _unavailable_prediction("tesseract", "disabled")
            )
            target_evaluations.append(
                _TargetEvaluation(
                    target=target,
                    expected_text=expected,
                    is_positive=True,
                    segmentation_boxes=boxes,
                    segmentation_status=segmentation_status,
                    template=template_prediction,
                    opencv_mlp=mlp_prediction,
                    cnn=cnn_prediction,
                    crnn_ctc=crnn_prediction,
                    transformer_ctc=transformer_prediction,
                    template_cnn=template_cnn_prediction,
                    tesseract=tesseract_prediction,
                )
            )
        return _EvaluationRow(
            source=row,
            targets=tuple(target_evaluations),
        )

    shadow_rows = [evaluate_row(row) for row in source_rows]
    evaluation_rows = [row for row in shadow_rows if row.source.split == "test"]
    primary_target = active_targets[0]
    target_summaries = _target_summaries(
        source_rows,
        evaluation_rows,
        segmentation_by_row,
        glyph_samples,
        targets=active_targets,
    )
    cnn_summary = _cnn_summary_for_target(
        target_summaries[primary_target],
        cnn=cnn_by_target[primary_target],
        enable_cnn=enable_cnn,
        cnn_epochs=cnn_epochs,
        cnn_min_confidence=cnn_min_confidence,
    )
    for target in active_targets:
        target_summary = target_summaries[target]
        target_summary["cnn"] = _cnn_summary_for_target(
            target_summary,
            cnn=cnn_by_target[target],
            enable_cnn=enable_cnn,
            cnn_epochs=cnn_epochs,
            cnn_min_confidence=cnn_min_confidence,
        )
        target_summary["crnn_ctc"] = _ctc_summary_for_target(
            target_summary,
            key="crnn_ctc",
            recognizer=crnn,
            enable_ctc=enable_ctc,
            epochs=crnn_epochs,
            min_confidence=ctc_min_confidence,
        )
        target_summary["transformer_ctc"] = _ctc_summary_for_target(
            target_summary,
            key="transformer_ctc",
            recognizer=transformer,
            enable_ctc=enable_ctc,
            epochs=transformer_epochs,
            min_confidence=ctc_min_confidence,
        )

    summary: dict[str, object] = {
        "schema_version": 1,
        "manifest": str(manifest_file),
        "field_name": field_name,
        "primary_target": primary_target,
        "active_targets": list(active_targets),
        "max_crops": max_crops,
        "test_frame_modulo": test_frame_modulo,
        "template_max_distance": template_max_distance,
        "mlp_min_confidence": mlp_min_confidence,
        "cnn_min_confidence": cnn_min_confidence,
        "cnn_epochs": cnn_epochs,
        "ctc_min_confidence": ctc_min_confidence,
        "crnn_epochs": crnn_epochs,
        "transformer_epochs": transformer_epochs,
        "ctc_batch_size": ctc_batch_size,
        "ctc_learning_rate": ctc_learning_rate,
        "ctc_weight_decay": ctc_weight_decay,
        "ctc_seed": ctc_seed,
        "ctc_strip_fixed_prefix": ctc_strip_fixed_prefix,
        "enable_cnn": enable_cnn,
        "enable_ctc": enable_ctc,
        "enable_transformer_ctc": enable_transformer_ctc,
        "enable_tesseract": enable_tesseract,
        "rows": len(source_rows),
        "positive_rows": len(
            [
                row
                for row in source_rows
                if any(row.expected.for_target(target) is not None for target in active_targets)
            ]
        ),
        "hard_negative_rows": len(
            [row for row in source_rows if row.clean_status == "no_visible_number"]
        ),
        "train_rows": len([row for row in source_rows if row.split == "train"]),
        "test_rows": len(evaluation_rows),
        "shadow_rows": len(shadow_rows),
        "hard_negative_test_rows": len(
            [
                row
                for row in evaluation_rows
                if row.source.clean_status == "no_visible_number"
            ]
        ),
        "glyph_samples": len(glyph_samples),
        "train_glyph_samples": len(train_samples),
        "sequence_samples": len(sequence_samples),
        "train_sequence_samples": len(train_sequence_samples),
        "target_glyph_samples": dict(
            sorted(Counter(sample.target for sample in glyph_samples).items())
        ),
        "glyph_label_counts": dict(
            sorted(Counter(sample.label for sample in glyph_samples).items())
        ),
        "targets": target_summaries,
        "segmentation": target_summaries[primary_target]["segmentation"],
        "template": target_summaries[primary_target]["template"],
        "opencv_mlp": target_summaries[primary_target]["opencv_mlp"],
        "template_cnn": target_summaries[primary_target]["template_cnn"],
        "tesseract": target_summaries[primary_target]["tesseract"],
        "cnn": cnn_summary,
        "crnn_ctc": target_summaries[primary_target]["crnn_ctc"],
        "transformer_ctc": target_summaries[primary_target]["transformer_ctc"],
        "evaluation_rows": [
            _evaluation_row_to_dict(row, output, preview_crop_dir) for row in evaluation_rows
        ],
        "shadow_evaluation_rows": [
            _evaluation_row_to_dict(row, output, preview_crop_dir) for row in shadow_rows
        ],
        "artifacts": {
            "glyph_dir": str(glyph_dir.relative_to(output)),
            "review_html": NUMBER_CHAR_RECOGNIZER_REVIEW_HTML,
            "review_md": NUMBER_CHAR_RECOGNIZER_REVIEW_MD,
        },
    }
    (output / NUMBER_CHAR_RECOGNIZER_SUMMARY).write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_report(output / NUMBER_CHAR_RECOGNIZER_REPORT, summary)
    _write_review_files(
        output,
        cast(Sequence[Mapping[str, object]], summary["evaluation_rows"]),
        cast(Mapping[str, object], summary),
    )
    return summary


def segment_number_characters(image: RgbImage) -> tuple[NumberCharBox, ...]:
    mask = _text_mask(image)
    return _segment_number_characters_from_mask(mask)


def _segment_number_characters_from_mask(
    mask: GrayImage,
    *,
    target: NumberTextTarget = "base",
) -> tuple[NumberCharBox, ...]:
    columns = (mask > 0).any(axis=0)
    runs = _runs_from_projection(columns, max_gap=1)
    boxes: list[NumberCharBox] = []
    for x1, x2 in runs:
        sub = mask[:, x1:x2]
        ys = np.where(sub > 0)[0]
        if len(ys) == 0:
            continue
        y1 = int(ys.min())
        y2 = int(ys.max() + 1)
        area = int(np.count_nonzero(sub))
        width = int(x2 - x1)
        height = int(y2 - y1)
        if area < 8 or width < 2 or height < 5:
            continue
        boxes.append(NumberCharBox(x=int(x1), y=y1, width=width, height=height, area=area))
    if target == "overlay":
        boxes = [box for box in boxes if box.width <= 32]
    return tuple(boxes)


def label_to_path_component(label: str) -> str:
    if label == "$":
        return "dollar"
    if label == "+":
        return "plus"
    if label == ",":
        return "comma"
    if label == ".":
        return "dot"
    return label


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate Poker Legends character-level numeric recognizer prototypes."
    )
    parser.add_argument("manifest", help="number_crop_dataset_manifest.json")
    parser.add_argument("--out", required=True)
    parser.add_argument("--field-name", default="hero_stack")
    parser.add_argument(
        "--target",
        action="append",
        choices=NUMBER_TEXT_TARGETS,
        dest="targets",
        help="Text target to evaluate; repeatable. Defaults to base, overlay, and display.",
    )
    parser.add_argument("--max-crops", type=int)
    parser.add_argument("--test-frame-modulo", type=int, default=DEFAULT_TEST_FRAME_MODULO)
    parser.add_argument(
        "--template-max-distance",
        type=float,
        default=DEFAULT_TEMPLATE_MAX_DISTANCE,
    )
    parser.add_argument("--mlp-min-confidence", type=float, default=DEFAULT_MLP_MIN_CONFIDENCE)
    parser.add_argument("--cnn-min-confidence", type=float, default=DEFAULT_CNN_MIN_CONFIDENCE)
    parser.add_argument("--cnn-epochs", type=int, default=DEFAULT_CNN_EPOCHS)
    parser.add_argument("--ctc-min-confidence", type=float, default=DEFAULT_CTC_MIN_CONFIDENCE)
    parser.add_argument("--crnn-epochs", type=int, default=DEFAULT_CRNN_EPOCHS)
    parser.add_argument("--transformer-epochs", type=int, default=DEFAULT_TRANSFORMER_EPOCHS)
    parser.add_argument("--ctc-batch-size", type=int, default=DEFAULT_CTC_BATCH_SIZE)
    parser.add_argument("--ctc-learning-rate", type=float, default=DEFAULT_CTC_LEARNING_RATE)
    parser.add_argument("--ctc-weight-decay", type=float, default=DEFAULT_CTC_WEIGHT_DECAY)
    parser.add_argument("--ctc-seed", type=int, default=DEFAULT_CTC_SEED)
    parser.add_argument("--ctc-strip-fixed-prefix", action="store_true")
    parser.add_argument("--disable-cnn", action="store_true")
    parser.add_argument("--enable-ctc", action="store_true")
    parser.add_argument("--disable-transformer-ctc", action="store_true")
    parser.add_argument("--disable-tesseract", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    summary = build_and_evaluate_poker_legends_number_char_recognizers(
        args.manifest,
        output_dir=args.out,
        field_name=args.field_name,
        targets=tuple(args.targets) if args.targets else NUMBER_TEXT_TARGETS,
        max_crops=args.max_crops,
        test_frame_modulo=args.test_frame_modulo,
        template_max_distance=args.template_max_distance,
        mlp_min_confidence=args.mlp_min_confidence,
        cnn_min_confidence=args.cnn_min_confidence,
        cnn_epochs=args.cnn_epochs,
        ctc_min_confidence=args.ctc_min_confidence,
        crnn_epochs=args.crnn_epochs,
        transformer_epochs=args.transformer_epochs,
        ctc_batch_size=args.ctc_batch_size,
        ctc_learning_rate=args.ctc_learning_rate,
        ctc_weight_decay=args.ctc_weight_decay,
        ctc_seed=args.ctc_seed,
        ctc_strip_fixed_prefix=args.ctc_strip_fixed_prefix,
        enable_cnn=not args.disable_cnn,
        enable_ctc=args.enable_ctc,
        enable_transformer_ctc=not args.disable_transformer_ctc,
        enable_tesseract=not args.disable_tesseract,
    )
    print(json.dumps(_stdout_summary(summary), indent=2, sort_keys=True))


def _load_rows(
    manifest_path: Path,
    *,
    field_name: str,
    max_crops: int | None,
    test_frame_modulo: int,
) -> tuple[NumberCharRow, ...]:
    manifest = _read_json_object(manifest_path)
    rows: list[NumberCharRow] = []
    frame_ids: list[str] = []
    selected = [
        row
        for row in _mapping_sequence(manifest.get("rows"))
        if str(row.get("group") or "") == "texts"
        and str(row.get("name") or "") == field_name
        and (
            (
                isinstance(row.get("truth_canonical_text"), str)
                and str(row.get("truth_canonical_text") or "").strip()
            )
            or _row_clean_status(row) == "no_visible_number"
        )
    ]
    if max_crops is not None:
        selected = selected[:max_crops]
    for row in selected:
        frame_id = str(row.get("frame_id") or "")
        if frame_id and frame_id not in frame_ids:
            frame_ids.append(frame_id)
    frame_index = {frame_id: index for index, frame_id in enumerate(frame_ids)}
    modulo = max(2, test_frame_modulo)
    for index, row in enumerate(selected):
        clean_status = _row_clean_status(row)
        expected = (
            NumberCharTargets(base=None, overlay=None, display=None)
            if clean_status == "no_visible_number"
            else _expected_targets(row)
        )
        if expected.display is None and expected.base is None and expected.overlay is None:
            if clean_status != "no_visible_number":
                continue
        frame_id = str(row.get("frame_id") or "")
        manifest_split = str(row.get("split") or "")
        split = (
            manifest_split
            if manifest_split in {"train", "validation", "test"}
            else "test"
            if frame_index.get(frame_id, 0) % modulo == 0
            else "train"
        )
        crop_path = _resolve_crop_path(manifest_path.parent, row.get("crop_path"))
        rows.append(
            NumberCharRow(
                row_id=f"{index:04d}",
                frame_id=frame_id,
                group=str(row.get("group") or ""),
                name=str(row.get("name") or ""),
                crop_variant=str(row.get("crop_variant") or "default"),
                crop_path=crop_path,
                expected=expected,
                split=split,
                clean_status=clean_status,
            )
        )
    return tuple(rows)


def _row_clean_status(row: Mapping[str, object]) -> str:
    status = str(row.get("clean_status") or "")
    if status:
        return status
    if row.get("truth_visible") is False:
        return "no_visible_number"
    return "labeled_visible"


def _expected_targets(row: Mapping[str, object]) -> NumberCharTargets:
    role = str(row.get("role") or "")
    raw = row.get("truth_canonical_text")
    raw_text = _normalize_compare_text(raw) if isinstance(raw, str) else ""
    if role in {"hero_stack", "seat_stack"}:
        normalized_number = _optional_int(row.get("truth_normalized_number"))
        display = _stack_display_text(raw_text)
        overlay = _stack_overlay_text(display)
        base = _stack_base_text(display)
        if normalized_number is not None:
            base = f"${normalized_number:,}"
            display = f"{base}{overlay}" if overlay is not None else base
        return NumberCharTargets(base=base, overlay=overlay, display=display)
    text = raw_text or None
    return NumberCharTargets(base=None, overlay=None, display=text)


def _stack_display_text(text: str) -> str | None:
    if not text:
        return None
    if not text.startswith("$"):
        text = f"${text}"
    if any(char not in ALLOWED_CHARS for char in text):
        return None
    return text


def _stack_base_text(text: str | None) -> str | None:
    if text is None:
        return None
    base = text.split("+", 1)[0]
    if not base:
        return None
    if not base.startswith("$"):
        base = f"${base}"
    return base


def _stack_overlay_text(text: str | None) -> str | None:
    if text is None or "+" not in text:
        return None
    overlay = text.split("+", 1)[1]
    if not overlay:
        return None
    return f"+{overlay}"


def _text_mask(image: RgbImage, *, target: NumberTextTarget = "base") -> GrayImage:
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    white = (image[:, :, 0] > 165) & (image[:, :, 1] > 165) & (image[:, :, 2] > 165)
    cyan = (
        (image[:, :, 0] < 140)
        & (image[:, :, 1] > 130)
        & (image[:, :, 2] > 110)
        & (hsv[:, :, 1] > 45)
    )
    if target == "base":
        selected = white
    elif target == "overlay":
        selected = cyan
    else:
        selected = white | cyan
    mask = selected.astype(np.uint8) * 255
    mask = cast(GrayImage, cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8)))
    if target == "overlay":
        mask = _remove_overlay_rule_lines(mask)
    return mask


def _remove_overlay_rule_lines(mask: GrayImage) -> GrayImage:
    cleaned = mask.copy()
    row_counts = (cleaned > 0).sum(axis=1)
    cleaned[row_counts > 40, :] = 0
    return cleaned


def _normalize_glyph(mask: GrayImage, box: NumberCharBox) -> FloatImage:
    crop = mask[box.y : box.y + box.height, box.x : box.x + box.width]
    canvas = np.zeros((DEFAULT_GLYPH_HEIGHT, DEFAULT_GLYPH_WIDTH), dtype=np.uint8)
    if crop.size == 0:
        return canvas.astype(np.float32) / 255.0
    scale = min(
        (DEFAULT_GLYPH_WIDTH - 4) / max(1, crop.shape[1]),
        (DEFAULT_GLYPH_HEIGHT - 4) / max(1, crop.shape[0]),
    )
    width = max(1, int(round(crop.shape[1] * scale)))
    height = max(1, int(round(crop.shape[0] * scale)))
    resized = cv2.resize(crop, (width, height), interpolation=cv2.INTER_AREA)
    x = (DEFAULT_GLYPH_WIDTH - width) // 2
    y = (DEFAULT_GLYPH_HEIGHT - height) // 2
    canvas[y : y + height, x : x + width] = resized
    return cast(FloatImage, canvas.astype(np.float32) / 255.0)


def _normalize_sequence_image(mask: GrayImage) -> FloatImage:
    canvas = np.zeros((DEFAULT_SEQUENCE_HEIGHT, DEFAULT_SEQUENCE_WIDTH), dtype=np.uint8)
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return cast(FloatImage, canvas.astype(np.float32) / 255.0)
    x1 = max(0, int(xs.min()) - 2)
    x2 = min(mask.shape[1], int(xs.max()) + 3)
    y1 = max(0, int(ys.min()) - 2)
    y2 = min(mask.shape[0], int(ys.max()) + 3)
    crop = mask[y1:y2, x1:x2]
    scale = min(
        (DEFAULT_SEQUENCE_WIDTH - 4) / max(1, crop.shape[1]),
        (DEFAULT_SEQUENCE_HEIGHT - 4) / max(1, crop.shape[0]),
    )
    width = max(1, int(round(crop.shape[1] * scale)))
    height = max(1, int(round(crop.shape[0] * scale)))
    resized = cv2.resize(crop, (width, height), interpolation=cv2.INTER_AREA)
    x = 2
    y = (DEFAULT_SEQUENCE_HEIGHT - height) // 2
    canvas[y : y + height, x : x + width] = resized
    return cast(FloatImage, canvas.astype(np.float32) / 255.0)


def _effective_sequence_timesteps(
    image: FloatImage,
    *,
    full_timesteps: int,
    min_timesteps: int = DEFAULT_CTC_MIN_EFFECTIVE_TIMESTEPS,
) -> int:
    columns = (image > 0.01).any(axis=0)
    nonzero = np.where(columns)[0]
    if len(nonzero) == 0:
        return full_timesteps
    # Include a small right-side blank margin so CTC can terminate the last label.
    image_width = int(image.shape[1])
    effective_width = min(image_width, int(nonzero.max()) + 3)
    scaled = math.ceil(effective_width * full_timesteps / max(1, image_width))
    return max(1, min(full_timesteps, max(scaled, min_timesteps)))


def _effective_sequence_timesteps_for_text(
    image: FloatImage,
    text: str,
    *,
    full_timesteps: int,
) -> int:
    required = math.ceil(ctc_required_timesteps(text) * DEFAULT_CTC_MIN_TIMESTEP_RATIO)
    return _effective_sequence_timesteps(
        image,
        full_timesteps=full_timesteps,
        min_timesteps=max(DEFAULT_CTC_MIN_EFFECTIVE_TIMESTEPS, required),
    )


def _runs_from_projection(values: NDArray[np.bool_], *, max_gap: int) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start: int | None = None
    gap = 0
    for index, value in enumerate(values):
        if bool(value):
            if start is None:
                start = index
            gap = 0
            continue
        if start is None:
            continue
        gap += 1
        if gap > max_gap:
            end = index - gap + 1
            if end > start:
                runs.append((start, end))
            start = None
            gap = 0
    if start is not None:
        runs.append((start, len(values)))
    return runs


def _tesseract_text(crop_path: Path) -> str:
    image = _load_rgb_image(crop_path)
    raw = poker_legends_numbers._best_numeric_text(  # pyright: ignore[reportPrivateUsage]
        image,
        whitelist=poker_legends_numbers._NUMERIC_WHITELIST,  # pyright: ignore[reportPrivateUsage]
    )
    return _normalize_compare_text(raw)


def _tesseract_prediction_from_text(
    raw_text: str,
    *,
    target: NumberTextTarget,
    is_stack: bool,
) -> StringPrediction:
    text = _target_text_from_display_text(
        raw_text,
        target=target,
        is_stack=is_stack,
    )
    return StringPrediction(
        method="tesseract",
        text=text or None,
        confidence=1.0 if text else 0.0,
        accepted=bool(text),
        reason="raw_text" if text else "missing",
    )


def _target_text_from_display_text(
    text: str,
    *,
    target: NumberTextTarget,
    is_stack: bool,
) -> str | None:
    if not text:
        return None
    if not is_stack:
        return text if target == "display" else None
    display = _stack_display_text(text)
    if target == "base":
        return _stack_base_text(display)
    if target == "overlay":
        return _stack_overlay_text(display)
    return display


def _target_contract_accepts_text(
    text: str,
    *,
    target: NumberTextTarget,
    is_stack: bool,
) -> bool:
    if not is_stack:
        return bool(text) and all(char in ALLOWED_CHARS for char in text)
    if target == "base":
        return text.startswith("$") and _valid_numeric_body(text[1:])
    if target == "overlay":
        return text.startswith("+") and _valid_numeric_body(text[1:])
    if not text.startswith("$"):
        return False
    base, separator, overlay = text.partition("+")
    if not _valid_numeric_body(base[1:]):
        return False
    return not separator or _valid_numeric_body(overlay)


def _valid_numeric_body(text: str) -> bool:
    if not text:
        return False
    return any(char.isdigit() for char in text) and all(char in "0123456789,.KM" for char in text)


def _is_stack_row(row: NumberCharRow) -> bool:
    return row.name.endswith("stack")


def _segmentation_failed_prediction(
    method: RecognitionMethod,
    status: str,
) -> StringPrediction:
    return StringPrediction(
        method=method,
        text=None,
        confidence=0.0,
        accepted=False,
        reason=f"segmentation_{status}",
    )


def _unavailable_prediction(method: RecognitionMethod, reason: str) -> StringPrediction:
    return StringPrediction(
        method=method,
        text=None,
        confidence=0.0,
        accepted=False,
        reason=reason,
    )


def _template_cnn_consensus_prediction(
    template: StringPrediction,
    cnn: StringPrediction,
    *,
    target: NumberTextTarget,
    is_stack: bool,
) -> StringPrediction:
    confidence = min(template.confidence, cnn.confidence)
    if template.text is None:
        return StringPrediction(
            method="template_cnn",
            text=None,
            confidence=confidence,
            accepted=False,
            reason=f"template_{template.reason}",
        )
    if cnn.text is None:
        return StringPrediction(
            method="template_cnn",
            text=None,
            confidence=confidence,
            accepted=False,
            reason=f"cnn_{cnn.reason}",
        )
    if template.text != cnn.text:
        return StringPrediction(
            method="template_cnn",
            text=None,
            confidence=confidence,
            accepted=False,
            reason="disagreement",
        )
    if not _target_contract_accepts_text(template.text, target=target, is_stack=is_stack):
        return StringPrediction(
            method="template_cnn",
            text=template.text,
            confidence=confidence,
            accepted=False,
            reason="target_contract",
        )
    if not template.accepted or not cnn.accepted:
        if _template_cnn_relaxed_agreement_accepts(template, cnn):
            return StringPrediction(
                method="template_cnn",
                text=template.text,
                confidence=max(confidence, min(template.confidence, cnn.confidence)),
                accepted=True,
                reason="accepted_relaxed_agreement",
            )
        blocking = "template" if not template.accepted else "cnn"
        prediction = template if blocking == "template" else cnn
        return StringPrediction(
            method="template_cnn",
            text=None,
            confidence=confidence,
            accepted=False,
            reason=f"{blocking}_{prediction.reason}",
        )
    return StringPrediction(
        method="template_cnn",
        text=template.text,
        confidence=confidence,
        accepted=True,
        reason="accepted",
    )


def _template_cnn_relaxed_agreement_accepts(
    template: StringPrediction,
    cnn: StringPrediction,
) -> bool:
    if template.text is None or cnn.text is None or template.text != cnn.text:
        return False
    if not template.char_distances:
        return False
    return (
        max(template.char_distances) <= DEFAULT_TEMPLATE_CNN_AGREEMENT_MAX_TEMPLATE_DISTANCE
        and cnn.confidence >= DEFAULT_TEMPLATE_CNN_AGREEMENT_MIN_CNN_CONFIDENCE
    )


def _target_summaries(
    source_rows: Sequence[NumberCharRow],
    evaluation_rows: Sequence[_EvaluationRow],
    segmentation_by_row: Mapping[
        tuple[str, NumberTextTarget], tuple[str, tuple[NumberCharBox, ...]]
    ],
    glyph_samples: Sequence[NumberGlyphSample],
    *,
    targets: Sequence[NumberTextTarget],
) -> dict[str, dict[str, object]]:
    summaries: dict[str, dict[str, object]] = {}
    for target in targets:
        evaluations = [
            target_evaluation
            for row in evaluation_rows
            for target_evaluation in row.targets
            if target_evaluation.target == target
        ]
        positive_evaluations = [
            target_evaluation for target_evaluation in evaluations if target_evaluation.is_positive
        ]
        negative_evaluations = [
            target_evaluation
            for target_evaluation in evaluations
            if not target_evaluation.is_positive
        ]
        target_rows = [row for row in source_rows if row.expected.for_target(target) is not None]
        negative_rows = [
            row
            for row in source_rows
            if row.clean_status == "no_visible_number"
            and row.expected.for_target(target) is None
        ]
        summaries[target] = {
            "rows": len(target_rows),
            "test_rows": len(positive_evaluations),
            "hard_negative_rows": len(negative_rows),
            "hard_negative_test_rows": len(negative_evaluations),
            "glyph_samples": len([sample for sample in glyph_samples if sample.target == target]),
            "segmentation": _segmentation_summary(source_rows, target, segmentation_by_row),
            "hard_negatives": _hard_negative_summaries(negative_evaluations),
            "template": _method_summary(positive_evaluations, "template"),
            "opencv_mlp": _method_summary(positive_evaluations, "opencv_mlp"),
            "cnn": _method_summary(positive_evaluations, "cnn"),
            "crnn_ctc": _method_summary(positive_evaluations, "crnn_ctc"),
            "transformer_ctc": _method_summary(positive_evaluations, "transformer_ctc"),
            "template_cnn": _method_summary(positive_evaluations, "template_cnn"),
            "tesseract": _method_summary(positive_evaluations, "tesseract"),
        }
    return summaries


def _cnn_summary_for_target(
    target_summary: Mapping[str, object],
    *,
    cnn: NumberTorchCnnRecognizer | None,
    enable_cnn: bool,
    cnn_epochs: int,
    cnn_min_confidence: float,
) -> dict[str, object]:
    summary = dict(cast(Mapping[str, object], target_summary["cnn"]))
    if not enable_cnn:
        summary["status"] = "not_run"
        summary["reason"] = "disabled"
    elif cnn is None or not cnn.available:
        summary["status"] = "not_run"
        summary["reason"] = cnn.reason if cnn is not None else "disabled"
    else:
        summary["status"] = "trained"
        summary["epochs"] = cnn_epochs
        summary["min_confidence"] = cnn_min_confidence
    return summary


def _ctc_summary_for_target(
    target_summary: Mapping[str, object],
    *,
    key: SequenceArchitecture,
    recognizer: NumberTorchCtcRecognizer | None,
    enable_ctc: bool,
    epochs: int,
    min_confidence: float,
) -> dict[str, object]:
    summary = dict(cast(Mapping[str, object], target_summary[key]))
    if not enable_ctc:
        summary["status"] = "not_run"
        summary["reason"] = "disabled"
    elif recognizer is None:
        summary["status"] = "not_run"
        summary["reason"] = "disabled"
    elif not recognizer.available:
        summary["status"] = "not_run"
        summary["reason"] = recognizer.reason
        summary["time_step_budget"] = recognizer.time_step_budget_summary
    else:
        summary["status"] = "trained"
        summary["epochs"] = epochs
        summary["min_confidence"] = min_confidence
        summary["time_step_budget"] = recognizer.time_step_budget_summary
    return summary


def _method_summary(
    rows: Sequence[_TargetEvaluation],
    method: RecognitionMethod,
) -> dict[str, object]:
    evaluated = len(rows)
    exact = 0
    accepted = 0
    accepted_exact = 0
    accepted_wrong = 0
    missing = 0
    for row in rows:
        prediction = _prediction_for_target_method(row, method)
        if prediction.text is None:
            missing += 1
        is_exact = prediction.text == row.expected_text
        if is_exact:
            exact += 1
        if prediction.accepted:
            accepted += 1
            if is_exact:
                accepted_exact += 1
            else:
                accepted_wrong += 1
    return {
        "evaluated": evaluated,
        "exact": exact,
        "missing": missing,
        "accuracy": exact / evaluated if evaluated else None,
        "accepted": accepted,
        "accepted_exact": accepted_exact,
        "accepted_wrong": accepted_wrong,
        "accepted_precision": accepted_exact / accepted if accepted else None,
        "accepted_coverage": accepted / evaluated if evaluated else None,
    }


def _hard_negative_summaries(
    rows: Sequence[_TargetEvaluation],
) -> dict[str, dict[str, object]]:
    methods: tuple[RecognitionMethod, ...] = (
        "template",
        "opencv_mlp",
        "cnn",
        "crnn_ctc",
        "transformer_ctc",
        "template_cnn",
        "tesseract",
    )
    return {
        method: _hard_negative_method_summary(rows, method)
        for method in methods
    }


def _hard_negative_method_summary(
    rows: Sequence[_TargetEvaluation],
    method: RecognitionMethod,
) -> dict[str, object]:
    evaluated = len(rows)
    false_accepts = 0
    accepted_texts: Counter[str] = Counter()
    for row in rows:
        prediction = _prediction_for_target_method(row, method)
        if not prediction.accepted:
            continue
        false_accepts += 1
        accepted_texts[str(prediction.text or "")] += 1
    return {
        "evaluated": evaluated,
        "false_accepts": false_accepts,
        "false_accept_rate": false_accepts / evaluated if evaluated else None,
        "accepted_texts": dict(sorted(accepted_texts.items())),
    }


def _prediction_for_target_method(
    row: _TargetEvaluation,
    method: RecognitionMethod,
) -> StringPrediction:
    if method == "template":
        return row.template
    if method == "opencv_mlp":
        return row.opencv_mlp
    if method == "cnn":
        return row.cnn
    if method == "crnn_ctc":
        return row.crnn_ctc
    if method == "transformer_ctc":
        return row.transformer_ctc
    if method == "template_cnn":
        return row.template_cnn
    return row.tesseract


def _segmentation_summary(
    rows: Sequence[NumberCharRow],
    target: NumberTextTarget,
    segmentation_by_row: Mapping[
        tuple[str, NumberTextTarget], tuple[str, tuple[NumberCharBox, ...]]
    ],
) -> dict[str, object]:
    counts = Counter(
        status
        for row in rows
        if row.expected.for_target(target) is not None
        for status, _boxes in [segmentation_by_row[(row.row_id, target)]]
    )
    test_counts = Counter(
        segmentation_by_row[(row.row_id, target)][0]
        for row in rows
        if row.split == "test"
        and row.expected.for_target(target) is not None
        and (row.row_id, target) in segmentation_by_row
    )
    target_rows = len([row for row in rows if row.expected.for_target(target) is not None])
    return {
        "counts": dict(sorted(counts.items())),
        "test_counts": dict(sorted(test_counts.items())),
        "success_rate": counts["match"] / target_rows if target_rows else None,
    }


def _evaluation_row_to_dict(
    row: _EvaluationRow,
    output_root: Path,
    preview_crop_dir: Path,
) -> dict[str, object]:
    preview_path = preview_crop_dir / f"{row.source.row_id}__{row.source.crop_path.name}"
    shutil.copy2(row.source.crop_path, preview_path)
    targets = {
        target_evaluation.target: _target_evaluation_to_dict(target_evaluation)
        for target_evaluation in row.targets
    }
    return {
        "row_id": row.source.row_id,
        "frame_id": row.source.frame_id,
        "field": f"{row.source.group}.{row.source.name}",
        "crop_variant": row.source.crop_variant,
        "crop_path": str(row.source.crop_path),
        "preview_crop": str(preview_path.relative_to(output_root)),
        "expected": row.source.expected.to_dict(),
        "split": row.source.split,
        "clean_status": row.source.clean_status,
        "targets": targets,
    }


def _target_evaluation_to_dict(row: _TargetEvaluation) -> dict[str, object]:
    return {
        "target": row.target,
        "expected": row.expected_text,
        "is_positive": row.is_positive,
        "segmentation_status": row.segmentation_status,
        "segmentation_boxes": [box.to_dict() for box in row.segmentation_boxes],
        "template": row.template.to_dict(),
        "opencv_mlp": row.opencv_mlp.to_dict(),
        "cnn": row.cnn.to_dict(),
        "crnn_ctc": row.crnn_ctc.to_dict(),
        "transformer_ctc": row.transformer_ctc.to_dict(),
        "template_cnn": row.template_cnn.to_dict(),
        "tesseract": row.tesseract.to_dict(),
    }


def _write_report(path: Path, summary: Mapping[str, object]) -> None:
    target_summaries = cast(Mapping[str, object], summary["targets"])
    active_targets = tuple(
        cast(Sequence[str], summary.get("active_targets") or NUMBER_TEXT_TARGETS)
    )
    lines = [
        "# Poker Legends Number Character Recognizers",
        "",
        "## Summary",
        f"- Manifest: `{summary['manifest']}`",
        f"- Field: `{summary['field_name']}`",
        f"- Targets: `{summary.get('active_targets')}`",
        f"- CTC strip fixed prefix: `{summary.get('ctc_strip_fixed_prefix')}`",
        f"- Rows: {summary['rows']}",
        f"- Positive rows: {summary['positive_rows']}",
        f"- Hard-negative rows: {summary['hard_negative_rows']}",
        f"- Train rows: {summary['train_rows']}",
        f"- Test rows: {summary['test_rows']}",
        f"- Shadow rows: {summary.get('shadow_rows', summary['test_rows'])}",
        f"- Hard-negative test rows: {summary['hard_negative_test_rows']}",
        f"- Glyph samples: {summary['glyph_samples']}",
        f"- Train glyph samples: {summary['train_glyph_samples']}",
        "",
        "## Targets",
    ]
    for target in active_targets:
        target_summary = cast(Mapping[str, object], target_summaries[target])
        lines.extend(
            [
                f"### {target}",
                f"- Rows: {target_summary['rows']}",
                f"- Test rows: {target_summary['test_rows']}",
                f"- Hard-negative rows: {target_summary['hard_negative_rows']}",
                f"- Hard-negative test rows: {target_summary['hard_negative_test_rows']}",
                f"- Glyph samples: {target_summary['glyph_samples']}",
                f"- Segmentation: {target_summary['segmentation']}",
                f"- Template: {_compact_method_summary(target_summary['template'])}",
                f"- CNN: {_compact_method_summary(target_summary['cnn'])}",
                f"- CRNN+CTC: {_compact_method_summary(target_summary['crnn_ctc'])}",
                f"- Transformer+CTC: "
                f"{_compact_method_summary(target_summary['transformer_ctc'])}",
                f"- Template+CNN consensus: "
                f"{_compact_method_summary(target_summary['template_cnn'])}",
                f"- OpenCV MLP: {_compact_method_summary(target_summary['opencv_mlp'])}",
                f"- Tesseract: {_compact_method_summary(target_summary['tesseract'])}",
                "- Hard negatives: "
                f"{_compact_hard_negative_summary(target_summary['hard_negatives'])}",
                "",
            ]
        )
    lines.extend(
        [
            "## Notes",
            "- `base` is the white available-stack text.",
            "- `overlay` is the cyan plus/current-bet text.",
            "- `display` is the combined visible stack string.",
            "- Template, CNN, CRNN+CTC, Transformer+CTC, and OpenCV MLP are offline "
            "prototypes and are not connected to runtime.",
            "- CNN uses PyTorch and trains on the generated train split only.",
            "- Accepted predictions are thresholded; accepted_wrong must stay at zero before "
            "any runtime use.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_review_files(
    output: Path,
    rows: Sequence[Mapping[str, object]],
    summary: Mapping[str, object],
) -> None:
    md_lines = [
        "# Poker Legends Number Character Recognizer Review",
        "",
        "## Rows",
        "",
        "| # | Crop | Expected | Base | Overlay | Display |",
        "|---:|---|---|---|---|---|",
    ]
    html_rows: list[str] = []
    for index, row in enumerate(rows, start=1):
        crop = str(row["preview_crop"])
        expected = cast(Mapping[str, object], row["expected"])
        targets = cast(Mapping[str, object], row["targets"])
        md_lines.append(
            "| "
            + " | ".join(
                [
                    str(index),
                    f'<img src="{crop}" width="180">',
                    _expected_targets_md(expected),
                    _target_review_md(targets.get("base")),
                    _target_review_md(targets.get("overlay")),
                    _target_review_md(targets.get("display")),
                ]
            )
            + " |"
        )
        html_rows.append(
            "<tr>"
            f"<td>{index}</td>"
            f'<td><img src="{html.escape(crop)}" alt="crop {index}"></td>'
            f"<td>{_expected_targets_html(expected)}</td>"
            f"<td>{_target_review_html(targets.get('base'))}</td>"
            f"<td>{_target_review_html(targets.get('overlay'))}</td>"
            f"<td>{_target_review_html(targets.get('display'))}</td>"
            "</tr>"
        )
    (output / NUMBER_CHAR_RECOGNIZER_REVIEW_MD).write_text(
        "\n".join(md_lines) + "\n",
        encoding="utf-8",
    )
    html_doc = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>Poker Legends Number Character Recognizer Review</title>
<style>
body {{ font-family: system-ui, -apple-system, Segoe UI, sans-serif; margin: 24px; color: #111; }}
table {{ border-collapse: collapse; width: 100%; }}
th, td {{ border: 1px solid #ddd; padding: 6px 8px; vertical-align: top; }}
th {{ position: sticky; top: 0; background: #f6f6f6; text-align: left; }}
img {{ min-width: 120px; max-width: 260px; background: #222; }}
code {{ white-space: pre-wrap; }}
.ok {{ color: #116329; }}
.bad {{ color: #b00020; }}
</style></head><body>
<h1>Poker Legends Number Character Recognizer Review</h1>
<p>Offline comparison for <code>{html.escape(str(summary["field_name"]))}</code>.</p>
{_target_summary_html(summary)}
<table><thead><tr><th>#</th><th>Crop</th><th>Expected</th><th>Base</th><th>Overlay</th><th>Display</th></tr></thead>
<tbody>{''.join(html_rows)}</tbody></table>
</body></html>
"""
    (output / NUMBER_CHAR_RECOGNIZER_REVIEW_HTML).write_text(html_doc, encoding="utf-8")


def _target_summary_html(summary: Mapping[str, object]) -> str:
    target_summaries = cast(Mapping[str, object], summary["targets"])
    active_targets = tuple(
        cast(Sequence[str], summary.get("active_targets") or NUMBER_TEXT_TARGETS)
    )
    items: list[str] = []
    for target in active_targets:
        target_summary = cast(Mapping[str, object], target_summaries[target])
        items.append(
            "<li>"
            f"<strong>{html.escape(target)}</strong>: "
            f"seg={html.escape(str(target_summary['segmentation']))}; "
            "hard_neg="
            f"{html.escape(_compact_hard_negative_summary(target_summary['hard_negatives']))}; "
            f"CRNN={html.escape(_compact_method_summary(target_summary['crnn_ctc']))}; "
            f"TxCTC={html.escape(_compact_method_summary(target_summary['transformer_ctc']))}; "
            f"CNN={html.escape(_compact_method_summary(target_summary['cnn']))}; "
            f"T+CNN={html.escape(_compact_method_summary(target_summary['template_cnn']))}; "
            f"Template={html.escape(_compact_method_summary(target_summary['template']))}; "
            f"Tesseract={html.escape(_compact_method_summary(target_summary['tesseract']))}"
            "</li>"
        )
    return "<ul>" + "".join(items) + "</ul>"


def _expected_targets_md(expected: Mapping[str, object]) -> str:
    return "<br>".join(
        f"{target}: `{expected.get(target)}`" for target in NUMBER_TEXT_TARGETS
    )


def _expected_targets_html(expected: Mapping[str, object]) -> str:
    return "".join(
        f"<div><strong>{html.escape(target)}</strong>: "
        f"<code>{html.escape(str(expected.get(target)))}</code></div>"
        for target in NUMBER_TEXT_TARGETS
    )


def _target_review_md(value: object) -> str:
    if not isinstance(value, Mapping):
        return "`n/a`"
    return "<br>".join(
        [
            f"kind: `{'positive' if value.get('is_positive') else 'negative'}`",
            f"seg: `{value.get('segmentation_status')}`",
            f"crnn: {_review_prediction_md(cast(Mapping[str, object], value['crnn_ctc']))}",
            "txctc: "
            f"{_review_prediction_md(cast(Mapping[str, object], value['transformer_ctc']))}",
            f"cnn: {_review_prediction_md(cast(Mapping[str, object], value['cnn']))}",
            "t+cnn: "
            f"{_review_prediction_md(cast(Mapping[str, object], value['template_cnn']))}",
            f"tpl: {_review_prediction_md(cast(Mapping[str, object], value['template']))}",
            f"ocr: {_review_prediction_md(cast(Mapping[str, object], value['tesseract']))}",
        ]
    )


def _target_review_html(value: object) -> str:
    if not isinstance(value, Mapping):
        return "<span class=\"muted\">n/a</span>"
    return (
        "<div><strong>kind</strong>: "
        f"<code>{'positive' if value.get('is_positive') else 'negative'}</code></div>"
        "<div><strong>seg</strong>: "
        f"<code>{html.escape(str(value.get('segmentation_status')))}</code></div>"
        "<div><strong>crnn</strong>: "
        f"{_review_prediction_html(cast(Mapping[str, object], value['crnn_ctc']))}</div>"
        "<div><strong>txctc</strong>: "
        f"{_review_prediction_html(cast(Mapping[str, object], value['transformer_ctc']))}</div>"
        "<div><strong>cnn</strong>: "
        f"{_review_prediction_html(cast(Mapping[str, object], value['cnn']))}</div>"
        "<div><strong>t+cnn</strong>: "
        f"{_review_prediction_html(cast(Mapping[str, object], value['template_cnn']))}</div>"
        "<div><strong>tpl</strong>: "
        f"{_review_prediction_html(cast(Mapping[str, object], value['template']))}</div>"
        "<div><strong>ocr</strong>: "
        f"{_review_prediction_html(cast(Mapping[str, object], value['tesseract']))}</div>"
    )


def _review_prediction_md(prediction: Mapping[str, object]) -> str:
    text = prediction.get("text")
    confidence = _to_float(prediction.get("confidence"))
    accepted = bool(prediction.get("accepted"))
    reason = str(prediction.get("reason") or "")
    return f"`{text}` ({confidence:.2f}, {'accepted' if accepted else reason})"


def _review_prediction_html(prediction: Mapping[str, object]) -> str:
    text = prediction.get("text")
    confidence = _to_float(prediction.get("confidence"))
    accepted = bool(prediction.get("accepted"))
    reason = str(prediction.get("reason") or "")
    klass = "ok" if accepted else "bad"
    suffix = "accepted" if accepted else reason
    return (
        f'<code>{html.escape(str(text))}</code> '
        f'<span class="{klass}">({confidence:.2f}, {html.escape(suffix)})</span>'
    )


def _compact_method_summary(value: object) -> str:
    summary = cast(Mapping[str, object], value)
    return (
        f"accuracy={_optional_ratio(summary.get('accuracy'))}, "
        f"accepted_precision={_optional_ratio(summary.get('accepted_precision'))}, "
        f"accepted_wrong={summary.get('accepted_wrong')}, "
        f"accepted_coverage={_optional_ratio(summary.get('accepted_coverage'))}"
    )


def _compact_hard_negative_summary(value: object) -> str:
    summary = cast(Mapping[str, object], value)
    parts: list[str] = []
    for method in ("cnn", "template_cnn", "template", "tesseract"):
        method_summary = cast(Mapping[str, object], summary.get(method) or {})
        parts.append(
            f"{method}={method_summary.get('false_accepts', 0)}/"
            f"{method_summary.get('evaluated', 0)}"
        )
    return ", ".join(parts)


def _stdout_summary(summary: Mapping[str, object]) -> dict[str, object]:
    return {
        "rows": summary["rows"],
        "positive_rows": summary["positive_rows"],
        "hard_negative_rows": summary["hard_negative_rows"],
        "train_rows": summary["train_rows"],
        "test_rows": summary["test_rows"],
        "shadow_rows": summary.get("shadow_rows", summary["test_rows"]),
        "hard_negative_test_rows": summary["hard_negative_test_rows"],
        "glyph_samples": summary["glyph_samples"],
        "segmentation": summary["segmentation"],
        "template": summary["template"],
        "cnn": summary["cnn"],
        "crnn_ctc": summary["crnn_ctc"],
        "transformer_ctc": summary["transformer_ctc"],
        "template_cnn": summary["template_cnn"],
        "opencv_mlp": summary["opencv_mlp"],
        "tesseract": summary["tesseract"],
        "report": NUMBER_CHAR_RECOGNIZER_REPORT,
        "review_html": NUMBER_CHAR_RECOGNIZER_REVIEW_HTML,
    }


def _normalize_compare_text(text: str) -> str:
    normalized = text.upper().replace(" ", "").replace("\n", "")
    normalized = normalized.translate(str.maketrans("OoI|", "0011"))
    return "".join(char for char in normalized if char in ALLOWED_CHARS)


def _softmax(scores: NDArray[np.float32]) -> NDArray[np.float32]:
    shifted = scores.astype(np.float64) - float(np.max(scores))
    exp = np.exp(shifted)
    total = float(np.sum(exp))
    if total <= 0.0:
        return np.zeros(scores.shape, dtype=np.float32)
    return cast(NDArray[np.float32], (exp / total).astype(np.float32))


def _glyph_vector(glyph: FloatImage) -> FloatVector:
    return cast(FloatVector, glyph.reshape(-1).astype(np.float32))


def _load_glyph(path: Path) -> FloatImage:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"could not read glyph: {path}")
    return cast(FloatImage, image.astype(np.float32) / 255.0)


def _write_glyph(path: Path, glyph: FloatImage) -> None:
    image = np.clip(glyph * 255.0, 0, 255).astype(np.uint8)
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"could not write glyph: {path}")


def _load_rgb_image(path: str | Path) -> RgbImage:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"could not read image: {path}")
    return cast(RgbImage, cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


def _resolve_crop_path(root: Path, value: object) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else root / path


def _read_json_object(path: str | Path) -> dict[str, object]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object: {path}")
    return cast(dict[str, object], data)


def _mapping_sequence(value: object) -> list[Mapping[str, object]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    return [cast(Mapping[str, object], item) for item in value if isinstance(item, Mapping)]


def _to_float(value: object) -> float:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        return float(value)
    return 0.0


def _optional_int(value: object) -> int | None:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str) and value.strip():
        try:
            return int(value)
        except ValueError:
            return None
    return None


def _optional_ratio(value: object) -> str:
    if value is None:
        return "n/a"
    return f"{_to_float(value):.3f}"


if __name__ == "__main__":
    main()
