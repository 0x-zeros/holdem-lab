"""CTC invariants for Poker Legends OCR experiments.

This module is intentionally model-agnostic.  It keeps the CTC bookkeeping that
must be true before any CRNN/Transformer OCR result is interpretable.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass

DEFAULT_CTC_MIN_TIMESTEP_RATIO = 2.0


class CtcAlignmentError(ValueError):
    """Raised when a CTC target/logit layout cannot be interpreted safely."""


@dataclass(frozen=True, slots=True)
class CtcEncodedTarget:
    text: str
    indices: tuple[int, ...]
    target_length: int
    required_timesteps: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class CtcLogProbShape:
    timesteps: int
    batch_size: int
    class_count: int

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class CtcTimeStepBudget:
    text: str
    target_length: int
    required_timesteps: int
    input_timesteps: int
    min_ratio: float
    ratio: float
    valid: bool
    reason: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class CtcAlphabet:
    labels: tuple[str, ...]
    blank_index: int = 0

    def __post_init__(self) -> None:
        if self.blank_index != 0:
            raise CtcAlignmentError("Poker Legends OCR expects CTC blank index 0")
        if len(set(self.labels)) != len(self.labels):
            raise CtcAlignmentError("CTC alphabet labels must be unique")
        for label in self.labels:
            if not label:
                raise CtcAlignmentError("CTC alphabet labels must be non-empty")

    @classmethod
    def from_texts(cls, texts: Iterable[str]) -> CtcAlphabet:
        labels = sorted({char for text in texts for char in text})
        return cls(tuple(labels))

    @property
    def class_count(self) -> int:
        return len(self.labels) + 1

    def label_to_index(self) -> dict[str, int]:
        return {label: index + 1 for index, label in enumerate(self.labels)}

    def index_to_label(self) -> dict[int, str]:
        return {index + 1: label for index, label in enumerate(self.labels)}

    def encode(self, text: str) -> tuple[int, ...]:
        label_to_index = self.label_to_index()
        indices: list[int] = []
        for char in text:
            index = label_to_index.get(char)
            if index is None:
                raise CtcAlignmentError(f"character {char!r} is not in the CTC alphabet")
            if index == self.blank_index:
                raise CtcAlignmentError("encoded CTC target must not include blank")
            indices.append(index)
        return tuple(indices)

    def encode_target(self, text: str) -> CtcEncodedTarget:
        indices = self.encode(text)
        return CtcEncodedTarget(
            text=text,
            indices=indices,
            target_length=len(indices),
            required_timesteps=ctc_required_timesteps(text),
        )

    def greedy_decode(
        self,
        indices: Sequence[int],
        confidences: Sequence[float] = (),
    ) -> tuple[str, tuple[float, ...]]:
        if confidences and len(confidences) != len(indices):
            raise CtcAlignmentError("CTC confidence path length must match index path")
        index_to_label = self.index_to_label()
        labels: list[str] = []
        emitted_confidences: list[float] = []
        previous = self.blank_index
        for step, index in enumerate(indices):
            index_int = int(index)
            if index_int == self.blank_index:
                previous = index_int
                continue
            if index_int == previous:
                continue
            label = index_to_label.get(index_int)
            if label is None:
                raise CtcAlignmentError(f"unknown CTC class index {index_int}")
            labels.append(label)
            if confidences:
                emitted_confidences.append(float(confidences[step]))
            previous = index_int
        return "".join(labels), tuple(emitted_confidences)


def ctc_required_timesteps(text: str) -> int:
    """Return the minimum CTC path length needed for a label sequence.

    Adjacent repeated labels require an intervening blank in at least one valid
    path, so ``1000`` needs six time steps: ``1 0 blank 0 blank 0``.
    """

    if not text:
        return 0
    adjacent_repeats = sum(
        1 for left, right in zip(text, text[1:], strict=False) if left == right
    )
    return len(text) + adjacent_repeats


def ctc_time_step_budget(
    text: str,
    *,
    input_timesteps: int,
    min_ratio: float = DEFAULT_CTC_MIN_TIMESTEP_RATIO,
) -> CtcTimeStepBudget:
    target_length = len(text)
    required_timesteps = ctc_required_timesteps(text)
    ratio = math.inf if required_timesteps == 0 else input_timesteps / required_timesteps
    if input_timesteps <= 0:
        return CtcTimeStepBudget(
            text=text,
            target_length=target_length,
            required_timesteps=required_timesteps,
            input_timesteps=input_timesteps,
            min_ratio=min_ratio,
            ratio=ratio,
            valid=False,
            reason="non_positive_input_timesteps",
        )
    if target_length == 0:
        return CtcTimeStepBudget(
            text=text,
            target_length=target_length,
            required_timesteps=required_timesteps,
            input_timesteps=input_timesteps,
            min_ratio=min_ratio,
            ratio=ratio,
            valid=False,
            reason="empty_target",
        )
    if input_timesteps < math.ceil(required_timesteps * min_ratio):
        return CtcTimeStepBudget(
            text=text,
            target_length=target_length,
            required_timesteps=required_timesteps,
            input_timesteps=input_timesteps,
            min_ratio=min_ratio,
            ratio=ratio,
            valid=False,
            reason="insufficient_timesteps",
        )
    return CtcTimeStepBudget(
        text=text,
        target_length=target_length,
        required_timesteps=required_timesteps,
        input_timesteps=input_timesteps,
        min_ratio=min_ratio,
        ratio=ratio,
        valid=True,
        reason="ok",
    )


def validate_ctc_time_step_budget(
    text: str,
    *,
    input_timesteps: int,
    min_ratio: float = DEFAULT_CTC_MIN_TIMESTEP_RATIO,
) -> CtcTimeStepBudget:
    budget = ctc_time_step_budget(
        text,
        input_timesteps=input_timesteps,
        min_ratio=min_ratio,
    )
    if not budget.valid:
        raise CtcAlignmentError(
            "CTC time-step budget failed for "
            f"{text!r}: T={budget.input_timesteps}, "
            f"target_len={budget.target_length}, "
            f"required={budget.required_timesteps}, "
            f"ratio={budget.ratio:.3f}, reason={budget.reason}"
        )
    return budget


def validate_ctc_log_probs_shape(
    shape: Sequence[int],
    *,
    batch_size: int,
    class_count: int,
) -> CtcLogProbShape:
    if len(shape) != 3:
        raise CtcAlignmentError(
            f"CTC log_probs must be shaped (T, N, C), got rank {len(shape)}"
        )
    timesteps, observed_batch, observed_classes = (int(value) for value in shape)
    if timesteps <= 0:
        raise CtcAlignmentError("CTC log_probs must have positive T")
    if observed_batch != batch_size:
        raise CtcAlignmentError(
            "CTC log_probs batch dimension must be N: "
            f"expected {batch_size}, got {observed_batch}"
        )
    if observed_classes != class_count:
        raise CtcAlignmentError(
            "CTC log_probs class dimension must include blank: "
            f"expected {class_count}, got {observed_classes}"
        )
    return CtcLogProbShape(
        timesteps=timesteps,
        batch_size=observed_batch,
        class_count=observed_classes,
    )
