import json
from pathlib import Path
from typing import cast

import cv2
import numpy as np
import pytest
from holdem_bot.vision import (
    LlmAnnotationManifest,
    annotation_output_schema,
    apply_poker_legends_layout,
    build_llm_annotation_package,
    execute_llm_annotation_package,
)
from holdem_bot.vision.llm_annotation import main as llm_main


def write_test_frame(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.zeros((491, 800, 3), dtype=np.uint8)
    image[216:272, 254:296] = (255, 255, 255)
    image[388:450, 555:610] = (0, 255, 0)
    if not cv2.imwrite(str(path), image):
        raise RuntimeError("could not write test frame")


def write_draft_annotation(path: Path, *, image: str) -> None:
    draft = {
        "schema_version": 1,
        "source": "poker_legends_video",
        "image": image,
        "video": "sample.mov",
        "width": 800,
        "height": 491,
        "source_width": 1600,
        "source_height": 982,
        "frame_index": 0,
        "timestamp_seconds": 0.0,
        "todo": ["hero_hole_cards", "board", "buttons"],
        "regions": {"board": [], "buttons": [], "seats": [], "texts": []},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(draft), encoding="utf-8")


def test_build_llm_annotation_package_writes_requests_and_crops(tmp_path: Path) -> None:
    image_root = tmp_path / "selection"
    frame_path = image_root / "frames" / "keyframe_000001.png"
    annotation_path = image_root / "annotations" / "keyframe_000001.json"
    output_dir = tmp_path / "llm"
    write_test_frame(frame_path)
    write_draft_annotation(annotation_path, image="frames/keyframe_000001.png")
    apply_poker_legends_layout(annotation_path)

    manifest = build_llm_annotation_package(
        [annotation_path],
        output_dir,
        image_root=image_root,
        provider="gemini",
        model="gemini-3.1-flash-lite",
        detail="original",
        crop_scale=2.0,
    )
    restored = LlmAnnotationManifest.read_json(output_dir / "manifest.json")

    assert restored == manifest
    assert len(manifest.frames) == 1
    assert manifest.image_format == "jpg"
    assert manifest.full_max_width == 1280
    assert manifest.crop_max_edge == 640
    assert manifest.crop_groups == ("board", "cards", "buttons", "texts")
    frame = manifest.frames[0]
    assert (output_dir / frame.image).exists()
    assert len(frame.crops) == 19
    assert any(crop.id == "texts.hero_current_bet" for crop in frame.crops)
    assert all((output_dir / crop.image).exists() for crop in frame.crops)
    requests = (output_dir / "requests.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(requests) == 1
    request = json.loads(requests[0])
    assert request["provider"] == "gemini"
    assert request["model"] == "gemini-3.1-flash-lite"
    assert request["detail"] == "original"
    assert request["output_schema"]["properties"]["table_state"]
    assert (output_dir / "package_report.md").exists()


def test_annotation_output_schema_requires_frame_id() -> None:
    schema = annotation_output_schema()
    required = cast(list[str], schema["required"])
    properties = cast(dict[str, object], schema["properties"])
    schema_version = cast(dict[str, object], properties["schema_version"])

    assert "frame_id" in required
    assert schema_version["const"] == 1


def test_llm_annotation_cli_builds_package_without_execute(tmp_path: Path) -> None:
    image_root = tmp_path / "selection"
    frame_path = image_root / "frames" / "keyframe_000001.png"
    annotation_path = image_root / "annotations" / "keyframe_000001.json"
    output_dir = tmp_path / "llm"
    write_test_frame(frame_path)
    write_draft_annotation(annotation_path, image="frames/keyframe_000001.png")
    apply_poker_legends_layout(annotation_path)

    llm_main(
        [
            str(annotation_path),
            "--image-root",
            str(image_root),
            "--out",
            str(output_dir),
            "--limit",
            "1",
        ]
    )

    assert (output_dir / "manifest.json").exists()
    assert not (output_dir / "responses").exists()


def test_llm_annotation_cli_loads_dotenv_defaults(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image_root = tmp_path / "selection"
    frame_path = image_root / "frames" / "keyframe_000001.png"
    annotation_path = image_root / "annotations" / "keyframe_000001.json"
    output_dir = tmp_path / "llm"
    write_test_frame(frame_path)
    write_draft_annotation(annotation_path, image="frames/keyframe_000001.png")
    apply_poker_legends_layout(annotation_path)
    (tmp_path / ".env").write_text(
        "HOLDEM_LLM_PROVIDER=openai\nHOLDEM_LLM_MODEL=gpt-5.5\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("HOLDEM_LLM_PROVIDER", raising=False)
    monkeypatch.delenv("HOLDEM_LLM_MODEL", raising=False)
    monkeypatch.chdir(tmp_path)

    llm_main(
        [
            str(annotation_path),
            "--image-root",
            str(image_root),
            "--out",
            str(output_dir),
            "--limit",
            "1",
        ]
    )
    manifest = LlmAnnotationManifest.read_json(output_dir / "manifest.json")

    assert manifest.provider == "openai"
    assert manifest.model == "gpt-5.5"


def test_execute_llm_annotation_package_skips_existing_candidates(tmp_path: Path) -> None:
    image_root = tmp_path / "selection"
    frame_path = image_root / "frames" / "keyframe_000001.png"
    annotation_path = image_root / "annotations" / "keyframe_000001.json"
    output_dir = tmp_path / "llm"
    write_test_frame(frame_path)
    write_draft_annotation(annotation_path, image="frames/keyframe_000001.png")
    apply_poker_legends_layout(annotation_path)
    build_llm_annotation_package([annotation_path], output_dir, image_root=image_root)
    candidate_dir = output_dir / "candidate_annotations"
    response_dir = output_dir / "responses"
    candidate_dir.mkdir()
    response_dir.mkdir()
    (candidate_dir / "keyframe_000001.json").write_text(
        json.dumps({"frame_id": "keyframe_000001", "uncertain": []}),
        encoding="utf-8",
    )
    (response_dir / "keyframe_000001.json").write_text(
        '{"frame_id":"keyframe_000001"}\n',
        encoding="utf-8",
    )

    execute_llm_annotation_package(output_dir / "manifest.json")

    assert "keyframe_000001" in (output_dir / "candidate_report.md").read_text(encoding="utf-8")
