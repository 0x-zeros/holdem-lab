import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from pathlib import Path

from holdem_bot.vision import TableAnnotation
from holdem_game.fixtures import write_pygame_fixture


def test_write_pygame_fixture_outputs_png_and_annotation(tmp_path: Path) -> None:
    annotation = write_pygame_fixture(tmp_path, stem="sample", size=(900, 620))

    image_path = tmp_path / "sample.png"
    json_path = tmp_path / "sample.json"
    restored = TableAnnotation.read_json(json_path)

    assert image_path.exists()
    assert image_path.stat().st_size > 0
    assert restored == annotation
    assert restored.image == "sample.png"
    assert restored.width == 900
    assert restored.height == 620
    assert len(restored.board) == 5
    assert len(restored.seats) == 3
    assert any(text.name == "pot" for text in restored.texts)
    assert any(button.action_type in {"call", "fold", "check"} for button in restored.buttons)
