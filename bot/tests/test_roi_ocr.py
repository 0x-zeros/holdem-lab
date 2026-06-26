import os
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from holdem_bot.vision import RoiOcrRecognizer, evaluate_recognition
from holdem_game.fixtures import write_pygame_fixture


def test_roi_ocr_recognizes_core_fields_from_pygame_fixture(tmp_path: Path) -> None:
    annotation = write_pygame_fixture(tmp_path, stem="roi", size=(1180, 760))
    recognized = RoiOcrRecognizer().recognize(tmp_path / "roi.png", annotation)
    score = evaluate_recognition(recognized, annotation)

    assert recognized.pot == 7
    assert recognized.current_seat == 0
    assert score.category("chips").accuracy == 1.0
    assert score.category("cards").accuracy >= 0.8
    assert score.category("buttons").correct >= 1
