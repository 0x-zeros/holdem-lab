"""Vision annotation and recognition support."""

from holdem_bot.vision.annotations import (
    AnnotatedButton,
    AnnotatedCard,
    AnnotatedSeat,
    AnnotatedText,
    ScreenRect,
    TableAnnotation,
)
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
    "ScreenRect",
    "TableAnnotation",
    "evaluate_recognition",
    "recognize_from_annotation",
]
