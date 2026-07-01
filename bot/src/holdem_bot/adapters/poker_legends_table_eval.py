"""Evaluate Poker Legends table recognizer assembly over reviewed datasets."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

from holdem_common import GameState

from holdem_bot.adapters.poker_legends import PokerLegendsTableRecognizer
from holdem_bot.capture import CapturedFrame
from holdem_bot.recognize import RecognitionResult
from holdem_bot.screen_state import ScreenKind


def evaluate_poker_legends_table_recognizer(
    *,
    dataset_manifest_path: str | Path,
    card_part_manifest: str | Path,
    card_classifier_manifest: str | Path,
    button_manifest: str | Path,
    output_dir: str | Path,
    card_template_manifest: str | Path | None = None,
    controlled_seat: int = 0,
    actionable_only: bool = True,
    limit: int | None = None,
) -> dict[str, object]:
    recognizer = PokerLegendsTableRecognizer.from_manifests(
        card_part_manifest=card_part_manifest,
        card_classifier_manifest=card_classifier_manifest,
        button_manifest=button_manifest,
        card_template_manifest=card_template_manifest,
        controlled_seat=controlled_seat,
    )
    manifest_path = Path(dataset_manifest_path)
    manifest = _read_json_object(manifest_path)
    annotation_dir = _annotation_dir(manifest, manifest_path=manifest_path)
    rows: list[dict[str, object]] = []
    result_counts: dict[str, int] = {}
    issue_counts: dict[str, int] = {}
    action_panel_flag_counts: dict[str, int] = {}
    blocking_action_panel_flag_counts: dict[str, int] = {}
    examples: dict[str, list[str]] = {}

    frames = _manifest_frames(manifest)
    if limit is not None:
        frames = frames[:limit]
    for frame in frames:
        frame_id = str(frame.get("frame_id") or "")
        truth_path = _resolve_path(frame.get("truth_path"), base=manifest_path.parent)
        if truth_path is None:
            continue
        truth = _read_json_object(truth_path)
        if actionable_only and _screen_kind(truth) != ScreenKind.ACTIONABLE_TABLE.value:
            continue
        image_path = _resolve_path(frame.get("image"), base=manifest_path.parent)
        annotation_path = annotation_dir / f"{frame_id}.json"
        if image_path is None or not annotation_path.exists():
            continue
        result = recognizer.recognize(
            CapturedFrame(
                payload=image_path,
                source="poker_legends_table_eval",
                metadata={
                    "poker_legends_image_path": str(image_path),
                    "poker_legends_layout_annotation_path": str(annotation_path),
                    "poker_legends_annotation_path": str(truth_path),
                    "coordinate_space": "image",
                },
            )
        )
        row = _row_from_result(frame_id, image_path=image_path, result=result)
        rows.append(row)
        outcome = str(row["result"])
        result_counts[outcome] = result_counts.get(outcome, 0) + 1
        examples.setdefault(outcome, [])
        if len(examples[outcome]) < 8:
            examples[outcome].append(frame_id)
        for issue_code in _string_list(row.get("issue_codes")):
            issue_counts[issue_code] = issue_counts.get(issue_code, 0) + 1
        for flag in _action_panel_flags(row.get("action_panels")):
            action_panel_flag_counts[flag] = action_panel_flag_counts.get(flag, 0) + 1
            if outcome != "state":
                blocking_action_panel_flag_counts[flag] = (
                    blocking_action_panel_flag_counts.get(flag, 0) + 1
                )

    summary: dict[str, object] = {
        "schema_version": 1,
        "frames": len(rows),
        "actionable_only": actionable_only,
        "result_counts": dict(sorted(result_counts.items())),
        "issue_counts": dict(sorted(issue_counts.items())),
        "action_panel_flag_counts": dict(sorted(action_panel_flag_counts.items())),
        "blocking_action_panel_flag_counts": dict(
            sorted(blocking_action_panel_flag_counts.items())
        ),
        "examples": dict(sorted(examples.items())),
        "rows": rows,
    }
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "table_recognizer_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_report(output / "table_recognizer_report.md", summary)
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate Poker Legends table recognizer assembly over a dataset manifest."
    )
    parser.add_argument("--dataset-manifest", required=True)
    parser.add_argument("--card-part-manifest", required=True)
    parser.add_argument("--card-classifier-manifest", required=True)
    parser.add_argument("--button-manifest", required=True)
    parser.add_argument("--card-template-manifest")
    parser.add_argument("--out", required=True)
    parser.add_argument("--seat", type=int, default=0)
    parser.add_argument(
        "--include-non-actionable",
        action="store_true",
        help="Also scan truth frames whose screen.kind is not actionable_table.",
    )
    parser.add_argument("--limit", type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    summary = evaluate_poker_legends_table_recognizer(
        dataset_manifest_path=args.dataset_manifest,
        card_part_manifest=args.card_part_manifest,
        card_classifier_manifest=args.card_classifier_manifest,
        button_manifest=args.button_manifest,
        card_template_manifest=args.card_template_manifest,
        output_dir=args.out,
        controlled_seat=args.seat,
        actionable_only=not args.include_non_actionable,
        limit=args.limit,
    )
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=2, sort_keys=True))


def _row_from_result(
    frame_id: str,
    *,
    image_path: Path,
    result: RecognitionResult,
) -> dict[str, object]:
    block_reason = result.metadata.get("state_block_reason")
    outcome = "state" if result.state is not None else str(block_reason or "no_state")
    table = result.metadata.get("recognized_table")
    table_dict = table if isinstance(table, Mapping) else {}
    assembly = result.assembly_result
    return {
        "frame_id": frame_id,
        "image": str(image_path),
        "result": outcome,
        "screen_kind": result.screen.kind.value,
        "assembly_status": assembly.status.value if assembly is not None else None,
        "validity_scope": assembly.validity_scope.value if assembly is not None else None,
        "issue_codes": [issue.reason_code for issue in assembly.issues]
        if assembly is not None
        else [],
        "confidence": result.confidence,
        "state": _state_summary(result.state),
        "recognized_table": _jsonable(table_dict),
        "action_panels": [
            panel.to_dict() for panel in result.visual_observation.action_panels
        ]
        if result.visual_observation is not None
        else [],
        "accepted_number_predictions": result.metadata.get("accepted_number_predictions", []),
    }


def _state_summary(state: GameState | None) -> dict[str, object] | None:
    if state is None:
        return None
    return {
        "street": state.street.value,
        "current_seat": state.current_seat,
        "button_seat": state.button_seat,
        "to_call": state.to_call,
        "pot_total": state.pot_total,
        "players": [
            {
                "seat": player.seat,
                "stack": player.stack,
                "committed": player.committed,
                "active": player.active,
                "hole_cards": [card.code for card in player.hole_cards],
            }
            for player in state.players
        ],
        "board": [card.code for card in state.board],
        "legal_actions": [
            {
                "type": action.action_type.value,
                "amount": action.amount,
                "min_amount": action.min_amount,
                "max_amount": action.max_amount,
            }
            for action in state.legal_actions
        ],
    }


def _write_report(path: Path, summary: Mapping[str, object]) -> None:
    counts = summary.get("result_counts")
    issue_counts = summary.get("issue_counts")
    action_panel_flag_counts = summary.get("action_panel_flag_counts")
    blocking_action_panel_flag_counts = summary.get("blocking_action_panel_flag_counts")
    examples = summary.get("examples")
    rows = _row_mappings(summary.get("rows"))
    blockers = [row for row in rows if row.get("result") != "state"]
    lines = [
        "# Poker Legends Table Recognizer Report",
        "",
        "## Summary",
        "",
        f"- Frames scanned: {summary.get('frames', 0)}",
        f"- Actionable only: {summary.get('actionable_only', True)}",
        "",
        "## Result Counts",
        "",
        *_count_lines(counts),
        "",
        "## Issue Counts",
        "",
        *_count_lines(issue_counts),
        "",
        "## Action Panel Flag Counts",
        "",
        *_count_lines(action_panel_flag_counts),
        "",
        "## Blocking Action Panel Flag Counts",
        "",
        *_count_lines(blocking_action_panel_flag_counts),
        "",
        "## Examples",
        "",
    ]
    if isinstance(examples, Mapping):
        for key, value in sorted(examples.items()):
            if isinstance(value, list):
                joined = ", ".join(f"`{item}`" for item in value)
                lines.append(f"- {key}: {joined}")
    else:
        lines.append("- none")
    lines.extend(
        [
            "",
            "## Blocking Details",
            "",
        ]
    )
    if blockers:
        lines.append(
            "| Frame | Result | Issues | Action Panels | Street | Pot | Buttons | Seats | "
            "Accepted Numbers |"
        )
        lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
        for row in blockers:
            table = row.get("recognized_table")
            table_dict = table if isinstance(table, Mapping) else {}
            lines.append(
                "| "
                f"`{row.get('frame_id')}` | "
                f"`{row.get('result')}` | "
                f"{_inline_codes(row.get('issue_codes'))} | "
                f"{_action_panel_summary(row.get('action_panels'))} | "
                f"`{table_dict.get('street')}` | "
                f"`{table_dict.get('pot')}` | "
                f"{_button_summary(table_dict.get('buttons'))} | "
                f"{_seat_summary(table_dict.get('seats'))} | "
                f"{_number_summary(row.get('accepted_number_predictions'))} |"
            )
    else:
        lines.append("No blocking rows.")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _count_lines(value: object) -> list[str]:
    if not isinstance(value, Mapping) or not value:
        return ["- none"]
    return [f"- {key}: {count}" for key, count in sorted(value.items())]


def _row_mappings(value: object) -> list[Mapping[str, object]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, Mapping)]


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str)]


def _inline_codes(value: object) -> str:
    values = _string_list(value)
    if not values:
        return "`none`"
    return ", ".join(f"`{item}`" for item in values)


def _button_summary(value: object) -> str:
    buttons = _row_mappings(value)
    if not buttons:
        return "`none`"
    return "<br>".join(
        f"`{button.get('command')}:{button.get('action_type')}:{button.get('label')}`"
        for button in buttons
    )


def _action_panel_summary(value: object) -> str:
    panels = _row_mappings(value)
    if not panels:
        return "`none`"
    parts: list[str] = []
    for panel in panels:
        flags = _string_list(panel.get("ambiguity_flags"))
        suffix = f":{'+'.join(flags)}" if flags else ""
        parts.append(f"`{panel.get('panel_kind')}:visible={panel.get('visible')}{suffix}`")
    return "<br>".join(parts)


def _action_panel_flags(value: object) -> list[str]:
    flags: list[str] = []
    for panel in _row_mappings(value):
        flags.extend(_string_list(panel.get("ambiguity_flags")))
    return flags


def _seat_summary(value: object) -> str:
    seats = _row_mappings(value)
    if not seats:
        return "`none`"
    return "<br>".join(
        "`seat={seat} stack={stack} committed={committed} current={current}`".format(
            seat=seat.get("seat"),
            stack=seat.get("stack"),
            committed=seat.get("committed"),
            current=seat.get("current"),
        )
        for seat in seats
    )


def _number_summary(value: object) -> str:
    numbers = _row_mappings(value)
    if not numbers:
        return "`none`"
    return "<br>".join(
        f"`{number.get('group')}:{number.get('name')}={number.get('normalized_number')}`"
        for number in numbers
    )


def _annotation_dir(manifest: Mapping[str, object], *, manifest_path: Path) -> Path:
    value = manifest.get("annotation_dir")
    if not isinstance(value, str) or not value:
        raise ValueError("dataset manifest missing annotation_dir")
    path = _resolve_path(value, base=manifest_path.parent)
    if path is None:
        raise ValueError("dataset manifest missing annotation_dir")
    return path


def _manifest_frames(manifest: Mapping[str, object]) -> list[Mapping[str, object]]:
    frames = manifest.get("frames")
    if not isinstance(frames, list):
        return []
    return [cast(Mapping[str, object], item) for item in frames if isinstance(item, Mapping)]


def _screen_kind(truth: Mapping[str, object]) -> str | None:
    screen = truth.get("screen")
    if not isinstance(screen, Mapping):
        return None
    kind = screen.get("kind")
    return kind if isinstance(kind, str) else None


def _resolve_path(value: object, *, base: Path) -> Path | None:
    if not isinstance(value, str) or not value:
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    if path.exists():
        return path.resolve()
    return (base / path).resolve()


def _read_json_object(path: str | Path) -> dict[str, object]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object: {path}")
    return cast(dict[str, object], data)


def _jsonable(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(item) for item in value]
    if hasattr(value, "__dataclass_fields__"):
        return _jsonable(asdict(cast(Any, value)))
    return value


if __name__ == "__main__":
    main()
