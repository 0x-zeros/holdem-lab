"""Character-level number recognizer prototypes for Poker Legends crops."""

from __future__ import annotations

import argparse
import html
import importlib
import json
import math
import shutil
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, cast

import cv2
import numpy as np
from numpy.typing import NDArray

from holdem_bot.vision import poker_legends_numbers

RgbImage = NDArray[np.uint8]
GrayImage = NDArray[np.uint8]
FloatImage = NDArray[np.float32]
FloatVector = NDArray[np.float32]

DEFAULT_GLYPH_WIDTH = 24
DEFAULT_GLYPH_HEIGHT = 32
DEFAULT_TEMPLATE_MAX_DISTANCE = 0.060
DEFAULT_MLP_MIN_CONFIDENCE = 0.60
DEFAULT_CNN_MIN_CONFIDENCE = 0.80
DEFAULT_CNN_EPOCHS = 60
DEFAULT_CNN_BATCH_SIZE = 64
DEFAULT_CNN_SEED = 1729
DEFAULT_TEST_FRAME_MODULO = 5
NUMBER_CHAR_RECOGNIZER_SUMMARY = "number_char_recognizer_summary.json"
NUMBER_CHAR_RECOGNIZER_REPORT = "number_char_recognizer_report.md"
NUMBER_CHAR_RECOGNIZER_REVIEW_HTML = "number_char_recognizer_review.html"
NUMBER_CHAR_RECOGNIZER_REVIEW_MD = "number_char_recognizer_review.md"
ALLOWED_CHARS = frozenset("$0123456789+,.KM")
RecognitionMethod = Literal["template", "opencv_mlp", "cnn", "template_cnn", "tesseract"]


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
class NumberCharRow:
    row_id: str
    frame_id: str
    group: str
    name: str
    crop_variant: str
    crop_path: Path
    expected_text: str
    split: str


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
class _EvaluationRow:
    source: NumberCharRow
    segmentation_boxes: tuple[NumberCharBox, ...]
    segmentation_status: str
    template: StringPrediction
    opencv_mlp: StringPrediction
    cnn: StringPrediction
    template_cnn: StringPrediction
    tesseract: StringPrediction


class NumberTemplateRecognizer:
    def __init__(
        self,
        samples: Sequence[NumberGlyphSample],
        *,
        glyph_root: str | Path,
        max_distance: float = DEFAULT_TEMPLATE_MAX_DISTANCE,
    ) -> None:
        self.max_distance = max_distance
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
        best: tuple[str, float] | None = None
        for template in self._glyphs:
            if exclude_frame_id is not None and template.frame_id == exclude_frame_id:
                continue
            distance = float(np.mean((glyph - template.image) ** 2))
            if best is None or distance < best[1]:
                best = (template.label, distance)
        return best


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


def build_and_evaluate_poker_legends_number_char_recognizers(
    manifest_path: str | Path,
    *,
    output_dir: str | Path,
    field_name: str = "hero_stack",
    max_crops: int | None = None,
    test_frame_modulo: int = DEFAULT_TEST_FRAME_MODULO,
    template_max_distance: float = DEFAULT_TEMPLATE_MAX_DISTANCE,
    mlp_min_confidence: float = DEFAULT_MLP_MIN_CONFIDENCE,
    cnn_min_confidence: float = DEFAULT_CNN_MIN_CONFIDENCE,
    cnn_epochs: int = DEFAULT_CNN_EPOCHS,
    enable_cnn: bool = True,
) -> dict[str, object]:
    manifest_file = Path(manifest_path)
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
    row_glyphs: dict[str, tuple[FloatImage, ...]] = {}
    segmentation_by_row: dict[str, tuple[str, tuple[NumberCharBox, ...]]] = {}
    for row in source_rows:
        crop = _load_rgb_image(row.crop_path)
        boxes = segment_number_characters(crop)
        segmentation_status = "match" if len(boxes) == len(row.expected_text) else "mismatch"
        segmentation_by_row[row.row_id] = (segmentation_status, tuple(boxes))
        if segmentation_status != "match":
            continue
        glyphs = tuple(_normalize_glyph(_text_mask(crop), box) for box in boxes)
        row_glyphs[row.row_id] = glyphs
        for index, (label, glyph) in enumerate(zip(row.expected_text, glyphs, strict=True)):
            glyph_path = (
                Path("number_char_glyphs")
                / label_to_path_component(label)
                / f"{row.row_id}__{index:02d}.png"
            )
            target = output / glyph_path
            target.parent.mkdir(parents=True, exist_ok=True)
            _write_glyph(target, glyph)
            glyph_samples.append(
                NumberGlyphSample(
                    label=label,
                    frame_id=row.frame_id,
                    row_id=row.row_id,
                    crop_path=str(row.crop_path),
                    glyph_path=str(glyph_path),
                    box=boxes[index],
                )
            )

    train_frame_ids = {row.frame_id for row in source_rows if row.split == "train"}
    train_samples = [sample for sample in glyph_samples if sample.frame_id in train_frame_ids]
    template = NumberTemplateRecognizer(
        train_samples,
        glyph_root=output,
        max_distance=template_max_distance,
    )
    mlp = NumberOpenCvMlpRecognizer(
        train_samples,
        glyph_root=output,
        min_confidence=mlp_min_confidence,
    )
    cnn = (
        NumberTorchCnnRecognizer(
            train_samples,
            glyph_root=output,
            min_confidence=cnn_min_confidence,
            epochs=cnn_epochs,
        )
        if enable_cnn
        else None
    )

    evaluation_rows: list[_EvaluationRow] = []
    for row in source_rows:
        if row.split != "test":
            continue
        segmentation_status, boxes = segmentation_by_row[row.row_id]
        glyphs = row_glyphs.get(row.row_id, ())
        template_prediction = (
            template.recognize(glyphs)
            if segmentation_status == "match"
            else _segmentation_failed_prediction("template", segmentation_status)
        )
        mlp_prediction = (
            mlp.recognize(glyphs)
            if segmentation_status == "match"
            else _segmentation_failed_prediction("opencv_mlp", segmentation_status)
        )
        if segmentation_status != "match":
            cnn_prediction = _segmentation_failed_prediction("cnn", segmentation_status)
        elif cnn is None:
            cnn_prediction = _unavailable_prediction("cnn", "disabled")
        else:
            cnn_prediction = cnn.recognize(glyphs)
        template_cnn_prediction = _template_cnn_consensus_prediction(
            template_prediction,
            cnn_prediction,
        )
        tesseract_prediction = _tesseract_prediction(row.crop_path)
        evaluation_rows.append(
            _EvaluationRow(
                source=row,
                segmentation_boxes=boxes,
                segmentation_status=segmentation_status,
                template=template_prediction,
                opencv_mlp=mlp_prediction,
                cnn=cnn_prediction,
                template_cnn=template_cnn_prediction,
                tesseract=tesseract_prediction,
            )
        )
    cnn_summary = _method_summary(evaluation_rows, "cnn")
    if not enable_cnn:
        cnn_summary["status"] = "not_run"
        cnn_summary["reason"] = "disabled"
    elif cnn is None or not cnn.available:
        cnn_summary["status"] = "not_run"
        cnn_summary["reason"] = cnn.reason if cnn is not None else "disabled"
    else:
        cnn_summary["status"] = "trained"
        cnn_summary["epochs"] = cnn_epochs
        cnn_summary["min_confidence"] = cnn_min_confidence

    summary: dict[str, object] = {
        "schema_version": 1,
        "manifest": str(manifest_file),
        "field_name": field_name,
        "max_crops": max_crops,
        "test_frame_modulo": test_frame_modulo,
        "template_max_distance": template_max_distance,
        "mlp_min_confidence": mlp_min_confidence,
        "cnn_min_confidence": cnn_min_confidence,
        "cnn_epochs": cnn_epochs,
        "enable_cnn": enable_cnn,
        "rows": len(source_rows),
        "train_rows": len([row for row in source_rows if row.split == "train"]),
        "test_rows": len(evaluation_rows),
        "glyph_samples": len(glyph_samples),
        "train_glyph_samples": len(train_samples),
        "glyph_label_counts": dict(
            sorted(Counter(sample.label for sample in glyph_samples).items())
        ),
        "segmentation": _segmentation_summary(source_rows, segmentation_by_row),
        "template": _method_summary(evaluation_rows, "template"),
        "opencv_mlp": _method_summary(evaluation_rows, "opencv_mlp"),
        "template_cnn": _method_summary(evaluation_rows, "template_cnn"),
        "tesseract": _method_summary(evaluation_rows, "tesseract"),
        "cnn": cnn_summary,
        "evaluation_rows": [
            _evaluation_row_to_dict(row, output, preview_crop_dir) for row in evaluation_rows
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
    parser.add_argument("--disable-cnn", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    summary = build_and_evaluate_poker_legends_number_char_recognizers(
        args.manifest,
        output_dir=args.out,
        field_name=args.field_name,
        max_crops=args.max_crops,
        test_frame_modulo=args.test_frame_modulo,
        template_max_distance=args.template_max_distance,
        mlp_min_confidence=args.mlp_min_confidence,
        cnn_min_confidence=args.cnn_min_confidence,
        cnn_epochs=args.cnn_epochs,
        enable_cnn=not args.disable_cnn,
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
        and isinstance(row.get("truth_canonical_text"), str)
        and str(row.get("truth_canonical_text") or "").strip()
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
        expected = _expected_text(row)
        if expected is None:
            continue
        frame_id = str(row.get("frame_id") or "")
        split = "test" if frame_index.get(frame_id, 0) % modulo == 0 else "train"
        crop_path = _resolve_crop_path(manifest_path.parent, row.get("crop_path"))
        rows.append(
            NumberCharRow(
                row_id=f"{index:04d}",
                frame_id=frame_id,
                group=str(row.get("group") or ""),
                name=str(row.get("name") or ""),
                crop_variant=str(row.get("crop_variant") or "default"),
                crop_path=crop_path,
                expected_text=expected,
                split=split,
            )
        )
    return tuple(rows)


def _expected_text(row: Mapping[str, object]) -> str | None:
    role = str(row.get("role") or "")
    if role in {"hero_stack", "seat_stack"}:
        normalized_number = _optional_int(row.get("truth_normalized_number"))
        if normalized_number is not None:
            return f"${normalized_number:,}"
    raw = row.get("truth_canonical_text")
    if not isinstance(raw, str):
        return None
    text = _normalize_compare_text(raw)
    if not text:
        return None
    if role in {"hero_stack", "seat_stack"} and not text.startswith("$"):
        text = f"${text}"
    if any(char not in ALLOWED_CHARS for char in text):
        return None
    return text


def _text_mask(image: RgbImage) -> GrayImage:
    white = (image[:, :, 0] > 165) & (image[:, :, 1] > 165) & (image[:, :, 2] > 165)
    mask = white.astype(np.uint8) * 255
    return cast(GrayImage, cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8)))


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


def _tesseract_prediction(crop_path: Path) -> StringPrediction:
    image = _load_rgb_image(crop_path)
    raw = poker_legends_numbers._best_numeric_text(  # pyright: ignore[reportPrivateUsage]
        image,
        whitelist=poker_legends_numbers._NUMERIC_WHITELIST,  # pyright: ignore[reportPrivateUsage]
    )
    text = _normalize_compare_text(raw)
    return StringPrediction(
        method="tesseract",
        text=text or None,
        confidence=1.0 if text else 0.0,
        accepted=bool(text),
        reason="raw_text" if text else "missing",
    )


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
) -> StringPrediction:
    confidence = min(template.confidence, cnn.confidence)
    if not template.accepted:
        return StringPrediction(
            method="template_cnn",
            text=None,
            confidence=confidence,
            accepted=False,
            reason=f"template_{template.reason}",
        )
    if not cnn.accepted:
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
    return StringPrediction(
        method="template_cnn",
        text=template.text,
        confidence=confidence,
        accepted=True,
        reason="accepted",
    )


def _method_summary(rows: Sequence[_EvaluationRow], method: RecognitionMethod) -> dict[str, object]:
    evaluated = len(rows)
    exact = 0
    accepted = 0
    accepted_exact = 0
    accepted_wrong = 0
    missing = 0
    for row in rows:
        prediction = _prediction_for_method(row, method)
        if prediction.text is None:
            missing += 1
        is_exact = prediction.text == row.source.expected_text
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


def _prediction_for_method(row: _EvaluationRow, method: RecognitionMethod) -> StringPrediction:
    if method == "template":
        return row.template
    if method == "opencv_mlp":
        return row.opencv_mlp
    if method == "cnn":
        return row.cnn
    if method == "template_cnn":
        return row.template_cnn
    return row.tesseract


def _segmentation_summary(
    rows: Sequence[NumberCharRow],
    segmentation_by_row: Mapping[str, tuple[str, tuple[NumberCharBox, ...]]],
) -> dict[str, object]:
    counts = Counter(status for status, _boxes in segmentation_by_row.values())
    test_counts = Counter(
        segmentation_by_row[row.row_id][0]
        for row in rows
        if row.split == "test" and row.row_id in segmentation_by_row
    )
    return {
        "counts": dict(sorted(counts.items())),
        "test_counts": dict(sorted(test_counts.items())),
        "success_rate": counts["match"] / len(rows) if rows else None,
    }


def _evaluation_row_to_dict(
    row: _EvaluationRow,
    output_root: Path,
    preview_crop_dir: Path,
) -> dict[str, object]:
    preview_path = preview_crop_dir / f"{row.source.row_id}__{row.source.crop_path.name}"
    shutil.copy2(row.source.crop_path, preview_path)
    return {
        "row_id": row.source.row_id,
        "frame_id": row.source.frame_id,
        "field": f"{row.source.group}.{row.source.name}",
        "crop_variant": row.source.crop_variant,
        "crop_path": str(row.source.crop_path),
        "preview_crop": str(preview_path.relative_to(output_root)),
        "expected": row.source.expected_text,
        "split": row.source.split,
        "segmentation_status": row.segmentation_status,
        "segmentation_boxes": [box.to_dict() for box in row.segmentation_boxes],
        "template": row.template.to_dict(),
        "opencv_mlp": row.opencv_mlp.to_dict(),
        "cnn": row.cnn.to_dict(),
        "template_cnn": row.template_cnn.to_dict(),
        "tesseract": row.tesseract.to_dict(),
    }


def _write_report(path: Path, summary: Mapping[str, object]) -> None:
    lines = [
        "# Poker Legends Number Character Recognizers",
        "",
        "## Summary",
        f"- Manifest: `{summary['manifest']}`",
        f"- Field: `{summary['field_name']}`",
        f"- Rows: {summary['rows']}",
        f"- Train rows: {summary['train_rows']}",
        f"- Test rows: {summary['test_rows']}",
        f"- Glyph samples: {summary['glyph_samples']}",
        f"- Train glyph samples: {summary['train_glyph_samples']}",
        f"- Segmentation: {summary['segmentation']}",
        "",
        "## Methods",
        f"- Template: {_compact_method_summary(summary['template'])}",
        f"- CNN: {_compact_method_summary(summary['cnn'])}",
        f"- Template+CNN consensus: {_compact_method_summary(summary['template_cnn'])}",
        f"- OpenCV MLP: {_compact_method_summary(summary['opencv_mlp'])}",
        f"- Tesseract: {_compact_method_summary(summary['tesseract'])}",
        "",
        "## Notes",
        "- Template, CNN, and OpenCV MLP are offline prototypes and are not connected to runtime.",
        "- CNN uses PyTorch and trains on the generated train split only.",
        "- Accepted predictions are thresholded; accepted_wrong must stay at zero before "
        "any runtime use.",
    ]
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
        "| # | Crop | Expected | Template | CNN | Template+CNN | MLP | Tesseract | Segmentation |",
        "|---:|---|---|---|---|---|---|---|---|",
    ]
    html_rows: list[str] = []
    for index, row in enumerate(rows, start=1):
        crop = str(row["preview_crop"])
        expected = str(row["expected"])
        template = cast(Mapping[str, object], row["template"])
        cnn = cast(Mapping[str, object], row["cnn"])
        template_cnn = cast(Mapping[str, object], row["template_cnn"])
        mlp = cast(Mapping[str, object], row["opencv_mlp"])
        tesseract = cast(Mapping[str, object], row["tesseract"])
        segmentation = str(row["segmentation_status"])
        md_lines.append(
            "| "
            + " | ".join(
                [
                    str(index),
                    f'<img src="{crop}" width="180">',
                    f"`{expected}`",
                    _review_prediction_md(template),
                    _review_prediction_md(cnn),
                    _review_prediction_md(template_cnn),
                    _review_prediction_md(mlp),
                    _review_prediction_md(tesseract),
                    f"`{segmentation}`",
                ]
            )
            + " |"
        )
        html_rows.append(
            "<tr>"
            f"<td>{index}</td>"
            f'<td><img src="{html.escape(crop)}" alt="crop {index}"></td>'
            f"<td><code>{html.escape(expected)}</code></td>"
            f"<td>{_review_prediction_html(template)}</td>"
            f"<td>{_review_prediction_html(cnn)}</td>"
            f"<td>{_review_prediction_html(template_cnn)}</td>"
            f"<td>{_review_prediction_html(mlp)}</td>"
            f"<td>{_review_prediction_html(tesseract)}</td>"
            f"<td><code>{html.escape(segmentation)}</code></td>"
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
<ul>
<li>Template: {html.escape(_compact_method_summary(summary["template"]))}</li>
<li>CNN: {html.escape(_compact_method_summary(summary["cnn"]))}</li>
<li>Template+CNN consensus: {html.escape(_compact_method_summary(summary["template_cnn"]))}</li>
<li>OpenCV MLP: {html.escape(_compact_method_summary(summary["opencv_mlp"]))}</li>
<li>Tesseract: {html.escape(_compact_method_summary(summary["tesseract"]))}</li>
</ul>
<table><thead><tr><th>#</th><th>Crop</th><th>Expected</th><th>Template</th><th>CNN</th><th>Template+CNN</th><th>MLP</th><th>Tesseract</th><th>Segmentation</th></tr></thead>
<tbody>{''.join(html_rows)}</tbody></table>
</body></html>
"""
    (output / NUMBER_CHAR_RECOGNIZER_REVIEW_HTML).write_text(html_doc, encoding="utf-8")


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


def _stdout_summary(summary: Mapping[str, object]) -> dict[str, object]:
    return {
        "rows": summary["rows"],
        "train_rows": summary["train_rows"],
        "test_rows": summary["test_rows"],
        "glyph_samples": summary["glyph_samples"],
        "segmentation": summary["segmentation"],
        "template": summary["template"],
        "cnn": summary["cnn"],
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
