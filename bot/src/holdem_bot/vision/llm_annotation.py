"""LLM-assisted Poker Legends annotation package generation and execution."""

from __future__ import annotations

import argparse
import base64
import json
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Self, cast

import cv2
from numpy.typing import NDArray
from openai import OpenAI

from holdem_bot.vision.annotations import ScreenRect

RgbImage = NDArray[Any]

DEFAULT_MODEL = "gpt-5.5"
DEFAULT_DETAIL = "original"
REQUESTS_JSONL = "requests.jsonl"


@dataclass(frozen=True, slots=True)
class LlmRoiCrop:
    id: str
    group: str
    name: str
    kind: str
    image: str
    rect: ScreenRect


@dataclass(frozen=True, slots=True)
class LlmFrameRequest:
    frame_id: str
    annotation: str
    image: str
    prompt: str
    crops: tuple[LlmRoiCrop, ...]


@dataclass(frozen=True, slots=True)
class LlmAnnotationManifest:
    schema_version: int
    model: str
    detail: str
    requests_jsonl: str
    frames: tuple[LlmFrameRequest, ...]

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("unsupported LLM annotation manifest schema version")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)

    def write_json(self, path: str | Path) -> None:
        Path(path).write_text(self.to_json() + "\n", encoding="utf-8")

    @classmethod
    def read_json(cls, path: str | Path) -> Self:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            schema_version=int(data["schema_version"]),
            model=str(data["model"]),
            detail=str(data["detail"]),
            requests_jsonl=str(data["requests_jsonl"]),
            frames=tuple(_frame_request_from_dict(frame) for frame in data["frames"]),
        )


def build_llm_annotation_package(
    annotation_paths: Sequence[str | Path],
    output_dir: str | Path,
    *,
    image_root: str | Path,
    model: str = DEFAULT_MODEL,
    detail: str = DEFAULT_DETAIL,
    crop_scale: float = 2.0,
    limit: int | None = None,
) -> LlmAnnotationManifest:
    if crop_scale <= 0:
        raise ValueError("crop_scale must be positive")
    selected_paths = [Path(path) for path in annotation_paths]
    if limit is not None:
        selected_paths = selected_paths[:limit]
    if not selected_paths:
        raise ValueError("at least one annotation is required")

    output = Path(output_dir)
    images_dir = output / "images"
    crops_dir = output / "crops"
    images_dir.mkdir(parents=True, exist_ok=True)
    crops_dir.mkdir(parents=True, exist_ok=True)

    frames: list[LlmFrameRequest] = []
    for annotation_path in selected_paths:
        annotation = _read_json_object(annotation_path)
        frame_id = annotation_path.stem
        source_image = Path(image_root) / str(annotation["image"])
        image_path = images_dir / f"{frame_id}{source_image.suffix or '.png'}"
        shutil.copy2(source_image, image_path)
        image = _read_image(image_path)
        crops = _write_crops(
            image,
            annotation,
            output,
            crops_dir / frame_id,
            frame_id=frame_id,
            crop_scale=crop_scale,
        )
        prompt = _build_frame_prompt(annotation, crops)
        frames.append(
            LlmFrameRequest(
                frame_id=frame_id,
                annotation=str(annotation_path),
                image=str(image_path.relative_to(output)),
                prompt=prompt,
                crops=tuple(crops),
            )
        )

    manifest = LlmAnnotationManifest(
        schema_version=1,
        model=model,
        detail=detail,
        requests_jsonl=REQUESTS_JSONL,
        frames=tuple(frames),
    )
    manifest.write_json(output / "manifest.json")
    _write_requests_jsonl(output / REQUESTS_JSONL, manifest)
    _write_package_report(output / "package_report.md", manifest)
    return manifest


def execute_llm_annotation_package(
    manifest_path: str | Path,
    *,
    output_dir: str | Path | None = None,
    model: str | None = None,
    detail: str | None = None,
    limit: int | None = None,
) -> None:
    manifest_file = Path(manifest_path)
    package_dir = manifest_file.parent
    manifest = LlmAnnotationManifest.read_json(manifest_file)
    target_dir = package_dir if output_dir is None else Path(output_dir)
    responses_dir = target_dir / "responses"
    candidate_dir = target_dir / "candidate_annotations"
    responses_dir.mkdir(parents=True, exist_ok=True)
    candidate_dir.mkdir(parents=True, exist_ok=True)

    client = OpenAI()
    selected_frames = manifest.frames if limit is None else manifest.frames[:limit]
    outputs: list[dict[str, object]] = []
    for frame in selected_frames:
        response = cast(Any, client.responses).create(
            model=model or manifest.model,
            input=[
                {
                    "role": "user",
                    "content": _response_content(package_dir, frame, detail or manifest.detail),
                }
            ],
            text={
                "format": {
                    "type": "json_schema",
                    "name": "poker_legends_frame_annotation",
                    "schema": annotation_output_schema(),
                    "strict": True,
                }
            },
            temperature=0,
            max_output_tokens=4000,
        )
        raw_text = str(getattr(response, "output_text", ""))
        response_path = responses_dir / f"{frame.frame_id}.json"
        response_path.write_text(raw_text + "\n", encoding="utf-8")
        candidate = _parse_candidate(raw_text, frame.frame_id)
        candidate_path = candidate_dir / f"{frame.frame_id}.json"
        candidate_path.write_text(
            json.dumps(candidate, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        uncertain = candidate.get("uncertain", [])
        uncertain_count = len(uncertain) if isinstance(uncertain, list) else 0
        outputs.append(
            {
                "frame_id": frame.frame_id,
                "candidate": str(candidate_path.relative_to(target_dir)),
                "response": str(response_path.relative_to(target_dir)),
                "uncertain_count": uncertain_count,
            }
        )
    _write_candidate_report(target_dir / "candidate_report.md", outputs)
    _write_uncertain_report(target_dir / "uncertain_report.md", candidate_dir)


def annotation_output_schema() -> dict[str, object]:
    card_value = {
        "type": ["string", "null"],
        "pattern": "^(A|K|Q|J|T|9|8|7|6|5|4|3|2)(S|H|D|C)$",
    }
    confidence = {"type": "number", "minimum": 0, "maximum": 1}
    return {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "schema_version",
            "frame_id",
            "table_state",
            "hero_hole_cards",
            "board",
            "buttons",
            "texts",
            "seats",
            "uncertain",
        ],
        "properties": {
            "schema_version": {"type": "integer", "const": 1},
            "frame_id": {"type": "string"},
            "table_state": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "is_table",
                    "is_actionable",
                    "street",
                    "blocking_reason",
                    "summary",
                    "confidence",
                ],
                "properties": {
                    "is_table": {"type": "boolean"},
                    "is_actionable": {"type": "boolean"},
                    "street": {
                        "type": ["string", "null"],
                        "enum": [None, "preflop", "flop", "turn", "river", "showdown", "unknown"],
                    },
                    "blocking_reason": {
                        "type": ["string", "null"],
                        "enum": [
                            None,
                            "none",
                            "buy_in_modal",
                            "reward_overlay",
                            "challenge_overlay",
                            "leave_table_modal",
                            "lobby_overlay",
                            "other_overlay",
                        ],
                    },
                    "summary": {"type": "string"},
                    "confidence": confidence,
                },
            },
            "hero_hole_cards": {
                "type": "array",
                "items": _card_schema(card_value, confidence),
            },
            "board": {
                "type": "array",
                "items": _card_schema(card_value, confidence),
            },
            "buttons": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["name", "visible", "label", "action_type", "confidence"],
                    "properties": {
                        "name": {"type": "string"},
                        "visible": {"type": "boolean"},
                        "label": {"type": ["string", "null"]},
                        "action_type": {
                            "type": ["string", "null"],
                            "enum": [
                                None,
                                "fold",
                                "check",
                                "call",
                                "raise",
                                "bet",
                                "all_in",
                                "confirm",
                                "cancel",
                                "other",
                            ],
                        },
                        "confidence": confidence,
                    },
                },
            },
            "texts": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["name", "visible", "value", "normalized_number", "confidence"],
                    "properties": {
                        "name": {"type": "string"},
                        "visible": {"type": "boolean"},
                        "value": {"type": ["string", "null"]},
                        "normalized_number": {"type": ["integer", "null"]},
                        "confidence": confidence,
                    },
                },
            },
            "seats": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": [
                        "name",
                        "visible",
                        "stack",
                        "committed",
                        "current",
                        "active",
                        "confidence",
                    ],
                    "properties": {
                        "name": {"type": "string"},
                        "visible": {"type": "boolean"},
                        "stack": {"type": ["integer", "null"]},
                        "committed": {"type": ["integer", "null"]},
                        "current": {"type": ["boolean", "null"]},
                        "active": {"type": ["boolean", "null"]},
                        "confidence": confidence,
                    },
                },
            },
            "uncertain": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["field", "reason"],
                    "properties": {
                        "field": {"type": "string"},
                        "reason": {"type": "string"},
                    },
                },
            },
        },
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build or execute LLM annotation requests for Poker Legends frames."
    )
    parser.add_argument("annotations", nargs="*", help="Draft annotation JSON files.")
    parser.add_argument("--image-root", help="Root for annotation image paths.")
    parser.add_argument("--out", required=True, help="Output directory for the LLM package.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="OpenAI model for execution.")
    parser.add_argument("--detail", default=DEFAULT_DETAIL, help="Image detail mode.")
    parser.add_argument("--crop-scale", type=float, default=2.0, help="Scale ROI crops before use.")
    parser.add_argument("--limit", type=int, default=None, help="Optional frame limit.")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Call OpenAI after building the package. Requires OPENAI_API_KEY.",
    )
    parser.add_argument(
        "--manifest",
        help="Existing LLM package manifest to execute. Skips package generation.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    if args.manifest is None:
        if args.image_root is None:
            raise SystemExit("--image-root is required when building a new LLM annotation package")
        manifest = build_llm_annotation_package(
            args.annotations,
            args.out,
            image_root=args.image_root,
            model=args.model,
            detail=args.detail,
            crop_scale=args.crop_scale,
            limit=args.limit,
        )
        manifest_path = Path(args.out) / "manifest.json"
    else:
        manifest = LlmAnnotationManifest.read_json(args.manifest)
        manifest_path = Path(args.manifest)
    if args.execute:
        execute_llm_annotation_package(
            manifest_path,
            output_dir=args.out,
            model=args.model,
            detail=args.detail,
            limit=args.limit,
        )
    print(
        json.dumps(
            {
                "execute": bool(args.execute),
                "frames": len(
                    manifest.frames if args.limit is None else manifest.frames[: args.limit]
                ),
                "manifest": str(manifest_path),
                "requests_jsonl": str(Path(args.out) / manifest.requests_jsonl),
            },
            indent=2,
            sort_keys=True,
        )
    )


def _write_crops(
    image: RgbImage,
    annotation: Mapping[str, object],
    output: Path,
    crop_dir: Path,
    *,
    frame_id: str,
    crop_scale: float,
) -> list[LlmRoiCrop]:
    crop_dir.mkdir(parents=True, exist_ok=True)
    crops: list[LlmRoiCrop] = []
    for group, regions in _region_groups(annotation).items():
        for region in regions:
            name = str(region["name"])
            kind = str(region["kind"])
            rect = _rect_from_region(region)
            crop_image = _crop(image, rect, pad=8)
            if crop_scale != 1.0:
                crop_image = _resize_by_scale(crop_image, crop_scale)
            crop_id = f"{group}.{name}"
            path = crop_dir / f"{group}__{name}.png"
            if not cv2.imwrite(str(path), crop_image):
                raise RuntimeError(f"could not write crop: {path}")
            crops.append(
                LlmRoiCrop(
                    id=crop_id,
                    group=group,
                    name=name,
                    kind=kind,
                    image=str(path.relative_to(output)),
                    rect=rect,
                )
            )
    return crops


def _write_requests_jsonl(path: Path, manifest: LlmAnnotationManifest) -> None:
    lines = []
    for frame in manifest.frames:
        lines.append(
            json.dumps(
                {
                    "frame_id": frame.frame_id,
                    "model": manifest.model,
                    "detail": manifest.detail,
                    "prompt": frame.prompt,
                    "image": frame.image,
                    "crops": [asdict(crop) for crop in frame.crops],
                    "output_schema": annotation_output_schema(),
                },
                sort_keys=True,
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_package_report(path: Path, manifest: LlmAnnotationManifest) -> None:
    total_crops = sum(len(frame.crops) for frame in manifest.frames)
    lines = [
        "# Poker Legends LLM Annotation Package",
        "",
        "## Configuration",
        f"- Model: `{manifest.model}`",
        f"- Image detail: `{manifest.detail}`",
        f"- Frames: {len(manifest.frames)}",
        f"- ROI crops: {total_crops}",
        f"- Requests JSONL: `{manifest.requests_jsonl}`",
        "",
        "## Process",
        "- Each frame includes the full image plus enlarged ROI crops.",
        "- The model is constrained to a JSON schema with explicit confidence fields.",
        "- Candidate outputs should not be treated as final truth until conflict checks pass.",
        "",
        "## Execute",
        "Run the same command with `--execute` in an environment with `OPENAI_API_KEY` set.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_candidate_report(path: Path, outputs: Sequence[Mapping[str, object]]) -> None:
    lines = [
        "# Poker Legends LLM Candidate Report",
        "",
        "| Frame | Candidate | Raw response | Uncertain fields |",
        "| --- | --- | --- | ---: |",
    ]
    for output in outputs:
        lines.append(
            f"| `{output['frame_id']}` | `{output['candidate']}` | "
            f"`{output['response']}` | {output['uncertain_count']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_uncertain_report(path: Path, candidate_dir: Path) -> None:
    lines = [
        "# Poker Legends LLM Uncertain Fields",
        "",
    ]
    for candidate_path in sorted(candidate_dir.glob("*.json")):
        candidate = _read_json_object(candidate_path)
        uncertain = candidate.get("uncertain", [])
        if not isinstance(uncertain, list) or not uncertain:
            continue
        lines.append(f"## {candidate_path.stem}")
        for item in uncertain:
            if isinstance(item, dict):
                lines.append(f"- `{item.get('field', '')}`: {item.get('reason', '')}")
        lines.append("")
    if len(lines) == 2:
        lines.append("No uncertain fields reported.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _build_frame_prompt(
    annotation: Mapping[str, object],
    crops: Sequence[LlmRoiCrop],
) -> str:
    frame_id = str(Path(str(annotation["image"])).stem)
    crop_lines = "\n".join(
        f"- {index + 1}: id={crop.id}, group={crop.group}, kind={crop.kind}, "
        f"rect=({crop.rect.x},{crop.rect.y},{crop.rect.width},{crop.rect.height})"
        for index, crop in enumerate(crops)
    )
    return (
        "You are annotating Poker Legends poker-table screenshots for a CV bot. "
        "Return only JSON matching the provided schema. Do not guess hidden cards or unclear text; "
        "use null and add an uncertain entry when unsure. Card codes must be rank+suit using "
        "S,H,D,C, for example AS, TD, 7H. Normalize chip numbers by removing commas, dollar signs, "
        "and plus deltas when a single stack/pot value is clear.\n\n"
        f"Frame id: {frame_id}\n"
        "Image 0 is the full frame. Additional images are enlarged ROI crops in this order:\n"
        f"{crop_lines}\n\n"
        "For table_state.is_actionable, return true only when the player can choose an in-hand "
        "poker action now. Modal dialogs, reward panels, challenge panels, buddy/lobby panels, "
        "and leave-table confirmations are not actionable poker states."
    )


def _response_content(
    package_dir: Path, frame: LlmFrameRequest, detail: str
) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = [
        {"type": "input_text", "text": frame.prompt},
        {
            "type": "input_image",
            "image_url": _image_data_uri(package_dir / frame.image),
            "detail": detail,
        },
    ]
    for crop in frame.crops:
        content.append({"type": "input_text", "text": f"ROI crop: {crop.id} ({crop.kind})"})
        content.append(
            {
                "type": "input_image",
                "image_url": _image_data_uri(package_dir / crop.image),
                "detail": detail,
            }
        )
    return content


def _card_schema(
    card_value: Mapping[str, object], confidence: Mapping[str, object]
) -> dict[str, object]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["slot", "visible", "card", "confidence"],
        "properties": {
            "slot": {"type": "string"},
            "visible": {"type": "boolean"},
            "card": card_value,
            "confidence": confidence,
        },
    }


def _parse_candidate(raw_text: str, expected_frame_id: str) -> dict[str, object]:
    try:
        candidate = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        return {
            "schema_version": 1,
            "frame_id": expected_frame_id,
            "parse_error": str(exc),
            "raw_text": raw_text,
            "uncertain": [{"field": "response", "reason": "model response was not valid JSON"}],
        }
    if isinstance(candidate, dict):
        return cast(dict[str, object], candidate)
    return {
        "schema_version": 1,
        "frame_id": expected_frame_id,
        "parse_error": "top-level response was not an object",
        "raw_text": raw_text,
        "uncertain": [{"field": "response", "reason": "model response was not a JSON object"}],
    }


def _region_groups(annotation: Mapping[str, object]) -> dict[str, list[dict[str, object]]]:
    raw_regions = annotation.get("regions", {})
    if not isinstance(raw_regions, dict):
        return {}
    groups: dict[str, list[dict[str, object]]] = {}
    for group, regions in raw_regions.items():
        if not isinstance(group, str) or not isinstance(regions, list):
            continue
        groups[group] = [dict(region) for region in regions if isinstance(region, dict)]
    return groups


def _rect_from_region(region: Mapping[str, object]) -> ScreenRect:
    raw_rect = region["rect"]
    if not isinstance(raw_rect, dict):
        raise TypeError(f"expected rect object: {raw_rect!r}")
    return ScreenRect(
        x=_to_int(raw_rect["x"]),
        y=_to_int(raw_rect["y"]),
        width=_to_int(raw_rect["width"]),
        height=_to_int(raw_rect["height"]),
    )


def _crop(image: RgbImage, rect: ScreenRect, *, pad: int) -> RgbImage:
    height, width = image.shape[:2]
    x = max(0, rect.x - pad)
    y = max(0, rect.y - pad)
    right = min(width, rect.x + rect.width + pad)
    bottom = min(height, rect.y + rect.height + pad)
    return image[y:bottom, x:right]


def _resize_by_scale(image: RgbImage, scale: float) -> RgbImage:
    height, width = image.shape[:2]
    return cast(
        RgbImage,
        cv2.resize(
            image,
            (max(1, round(width * scale)), max(1, round(height * scale))),
            interpolation=cv2.INTER_CUBIC,
        ),
    )


def _image_data_uri(path: Path) -> str:
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    suffix = path.suffix.lower()
    mime = "image/jpeg" if suffix in {".jpg", ".jpeg"} else "image/png"
    return f"data:{mime};base64,{data}"


def _read_image(path: Path) -> RgbImage:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"could not read image: {path}")
    return cast(RgbImage, image)


def _read_json_object(path: Path) -> dict[str, object]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise TypeError(f"expected JSON object: {path}")
    return cast(dict[str, object], data)


def _frame_request_from_dict(data: object) -> LlmFrameRequest:
    raw = cast(dict[str, object], data)
    raw_crops = raw["crops"]
    if not isinstance(raw_crops, list):
        raise TypeError(f"expected crop list: {raw_crops!r}")
    return LlmFrameRequest(
        frame_id=str(raw["frame_id"]),
        annotation=str(raw["annotation"]),
        image=str(raw["image"]),
        prompt=str(raw["prompt"]),
        crops=tuple(_crop_from_dict(crop) for crop in raw_crops),
    )


def _crop_from_dict(data: object) -> LlmRoiCrop:
    raw = cast(dict[str, object], data)
    return LlmRoiCrop(
        id=str(raw["id"]),
        group=str(raw["group"]),
        name=str(raw["name"]),
        kind=str(raw["kind"]),
        image=str(raw["image"]),
        rect=_screen_rect_from_dict(raw["rect"]),
    )


def _screen_rect_from_dict(data: object) -> ScreenRect:
    raw = cast(dict[str, object], data)
    return ScreenRect(
        x=_to_int(raw["x"]),
        y=_to_int(raw["y"]),
        width=_to_int(raw["width"]),
        height=_to_int(raw["height"]),
    )


def _to_int(value: object) -> int:
    if type(value) is int:
        return value
    if isinstance(value, str):
        return int(value)
    raise TypeError(f"expected int or string: {value!r}")
