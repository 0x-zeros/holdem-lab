import json
from collections.abc import Callable
from pathlib import Path
from typing import cast

import cv2
import numpy as np
import pytest
from holdem_bot.vision import VideoIngestManifest, ingest_video
from holdem_bot.vision.video import main as ingest_main


def write_synthetic_video(path: Path) -> None:
    fourcc = cast(
        Callable[[str, str, str, str], int],
        cv2.VideoWriter_fourcc,  # type: ignore[attr-defined]
    )
    writer = cv2.VideoWriter(
        str(path),
        fourcc("M", "J", "P", "G"),
        10.0,
        (160, 100),
    )
    if not writer.isOpened():
        raise RuntimeError("OpenCV could not create synthetic test video")
    try:
        for index in range(30):
            frame = np.zeros((100, 160, 3), dtype=np.uint8)
            if index >= 10:
                frame[:, :] = (40, 80, 120)
            if index >= 20:
                frame[20:80, 50:110] = (220, 220, 220)
            writer.write(frame)
    finally:
        writer.release()


def test_ingest_video_extracts_keyframes_and_process_artifacts(tmp_path: Path) -> None:
    video_path = tmp_path / "sample.avi"
    output_dir = tmp_path / "ingested"
    write_synthetic_video(video_path)

    manifest = ingest_video(
        video_path,
        output_dir,
        sample_fps=5.0,
        diff_threshold=2.0,
        max_gap_seconds=1.0,
    )
    restored = VideoIngestManifest.read_json(output_dir / "manifest.json")

    assert restored == manifest
    assert len(manifest.frames) >= 3
    assert (output_dir / manifest.process_report).exists()
    assert manifest.contact_sheet is not None
    assert (output_dir / manifest.contact_sheet).exists()
    for frame in manifest.frames:
        assert (output_dir / frame.image).exists()
        annotation_path = output_dir / frame.annotation
        assert annotation_path.exists()
        annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
        assert annotation["source"] == "poker_legends_video"
        assert annotation["frame_index"] == frame.frame_index


def test_ingest_video_cli_outputs_manifest_path(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    video_path = tmp_path / "sample.avi"
    output_dir = tmp_path / "ingested"
    write_synthetic_video(video_path)

    ingest_main(
        [
            str(video_path),
            "--out",
            str(output_dir),
            "--sample-fps",
            "5",
            "--max-frames",
            "2",
        ]
    )
    output = capsys.readouterr().out

    assert '"frames": 2' in output
    assert (output_dir / "manifest.json").exists()
