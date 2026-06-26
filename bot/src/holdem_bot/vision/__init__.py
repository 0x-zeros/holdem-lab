"""Vision annotation and recognition support."""

from holdem_bot.vision.annotations import (
    AnnotatedButton,
    AnnotatedCard,
    AnnotatedSeat,
    AnnotatedText,
    ScreenRect,
    TableAnnotation,
)
from holdem_bot.vision.evaluate import evaluate_fixture, score_to_dict
from holdem_bot.vision.recognition import (
    CategoryScore,
    RecognitionScore,
    RecognizedButton,
    RecognizedCard,
    RecognizedSeat,
    RecognizedTable,
    evaluate_recognition,
    recognize_from_annotation,
)
from holdem_bot.vision.roi_ocr import RoiOcrConfig, RoiOcrRecognizer
from holdem_bot.vision.selection import (
    KeyframeSelectionManifest,
    KeyframeSelectionRequest,
    SelectedKeyframe,
    select_keyframes,
)
from holdem_bot.vision.video import (
    ExtractedFrame,
    VideoIngestManifest,
    VideoMetadata,
    ingest_video,
)

__all__ = [
    "AnnotatedButton",
    "AnnotatedCard",
    "AnnotatedSeat",
    "AnnotatedText",
    "CategoryScore",
    "ExtractedFrame",
    "KeyframeSelectionManifest",
    "KeyframeSelectionRequest",
    "RecognitionScore",
    "RecognizedButton",
    "RecognizedCard",
    "RecognizedSeat",
    "RecognizedTable",
    "RoiOcrConfig",
    "RoiOcrRecognizer",
    "ScreenRect",
    "SelectedKeyframe",
    "TableAnnotation",
    "VideoIngestManifest",
    "VideoMetadata",
    "evaluate_fixture",
    "evaluate_recognition",
    "ingest_video",
    "recognize_from_annotation",
    "score_to_dict",
    "select_keyframes",
]
