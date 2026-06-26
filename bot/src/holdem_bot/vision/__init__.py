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

__all__ = [
    "AnnotatedButton",
    "AnnotatedCard",
    "AnnotatedSeat",
    "AnnotatedText",
    "CategoryScore",
    "RecognitionScore",
    "RecognizedButton",
    "RecognizedCard",
    "RecognizedSeat",
    "RecognizedTable",
    "RoiOcrConfig",
    "RoiOcrRecognizer",
    "ScreenRect",
    "TableAnnotation",
    "evaluate_fixture",
    "evaluate_recognition",
    "recognize_from_annotation",
    "score_to_dict",
]
