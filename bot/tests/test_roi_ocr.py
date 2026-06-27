import os
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import pytest
from holdem_bot.vision import (
    RoiOcrRecognizer,
    evaluate_fixture,
    evaluate_recognition,
    score_to_dict,
)
from holdem_bot.vision.evaluate import main as evaluate_main
from holdem_game.fixtures import write_pygame_fixture


def test_roi_ocr_recognizes_core_fields_from_pygame_fixture(tmp_path: Path) -> None:
    annotation = write_pygame_fixture(tmp_path, stem="roi", size=(1180, 760))
    recognized = RoiOcrRecognizer().recognize(tmp_path / "roi.png", annotation)
    score = evaluate_recognition(recognized, annotation)

    assert recognized.pot == 35
    assert recognized.current_seat == 0
    assert score.accuracy == 1.0
    assert score.category("buttons").accuracy == 1.0
    assert score.category("chips").accuracy == 1.0
    assert score.category("cards").accuracy == 1.0


def test_evaluate_fixture_reports_scores(tmp_path: Path) -> None:
    write_pygame_fixture(tmp_path, stem="roi", size=(1180, 760))

    score = evaluate_fixture(tmp_path / "roi.png", tmp_path / "roi.json")
    report = score_to_dict(score)

    assert report["accuracy"] == 1.0
    assert score.category("cards").accuracy == 1.0


def test_evaluate_fixture_cli_outputs_json(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    write_pygame_fixture(tmp_path, stem="roi", size=(1180, 760))

    evaluate_main(
        [
            str(tmp_path / "roi.png"),
            str(tmp_path / "roi.json"),
            "--min-accuracy",
            "1.0",
        ]
    )
    output = capsys.readouterr().out

    assert '"accuracy": 1.0' in output
