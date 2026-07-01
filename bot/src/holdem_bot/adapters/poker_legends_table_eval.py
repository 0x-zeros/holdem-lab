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
from holdem_bot.recognize import (
    AcceptedCriticalFieldEvaluation,
    RecognitionMode,
    RecognitionResult,
    evaluate_accepted_critical_fields,
)
from holdem_bot.screen_state import ScreenKind
from holdem_bot.vision.poker_legends_numbers import parse_poker_legends_chip_amount


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
    image_only_replay: bool = False,
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
    review_tag_counts: dict[str, int] = {}
    screen_kind_counts: dict[str, int] = {}
    screen_confusion_counts: dict[str, dict[str, int]] = {}
    recognition_mode_counts: dict[str, int] = {}
    contract_counts: dict[str, int] = {}
    assembly_status_counts: dict[str, int] = {}
    table_readiness_flag_counts: dict[str, int] = {}
    number_readiness_flag_counts: dict[str, int] = {}
    number_truth_comparison_counts: dict[str, int] = {}
    number_component_truth_comparison_counts: dict[str, int] = {}
    false_actionable_examples: list[str] = []
    screen_false_actionable_examples: list[str] = []
    screen_missed_actionable_examples: list[str] = []
    accepted_critical_wrong_examples: list[dict[str, object]] = []
    source_policy_violation_examples: list[dict[str, object]] = []
    number_truth_mismatch_examples: list[dict[str, object]] = []
    authorization_events = 0
    unsafe_authorization_events = 0
    stale_authorization_events = 0
    truth_assisted_authorization_events = 0
    source_policy_violation_count = 0
    accepted_critical_wrong_count = 0
    non_actionable_frames = 0
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
        truth_screen_kind = _screen_kind(truth) or "unknown"
        if actionable_only and truth_screen_kind != ScreenKind.ACTIONABLE_TABLE.value:
            continue
        screen_kind_counts[truth_screen_kind] = screen_kind_counts.get(truth_screen_kind, 0) + 1
        image_path = _resolve_path(frame.get("image"), base=manifest_path.parent)
        annotation_path = annotation_dir / f"{frame_id}.json"
        if image_path is None or not annotation_path.exists():
            continue
        result = recognizer.recognize(
            CapturedFrame(
                payload=image_path,
                source="poker_legends_table_eval",
                metadata=_frame_metadata(
                    image_path=image_path,
                    layout_annotation_path=annotation_path,
                    truth_path=truth_path,
                    image_only_replay=image_only_replay,
                ),
            )
        )

        critical_eval = evaluate_accepted_critical_fields(
            result,
            expected_values=_expected_critical_values(
                truth,
                controlled_seat=controlled_seat,
            ),
        )
        row = _row_from_result(
            frame_id,
            image_path=image_path,
            truth_path=truth_path,
            layout_annotation_path=annotation_path,
            result=result,
            critical_eval=critical_eval,
            truth=truth,
            truth_screen_kind=truth_screen_kind,
        )
        rows.append(row)
        outcome = str(row["result"])
        recognized_screen_kind = result.screen.kind.value
        screen_confusion_counts.setdefault(truth_screen_kind, {})
        screen_confusion_counts[truth_screen_kind][recognized_screen_kind] = (
            screen_confusion_counts[truth_screen_kind].get(recognized_screen_kind, 0) + 1
        )
        if (
            recognized_screen_kind == ScreenKind.ACTIONABLE_TABLE.value
            and truth_screen_kind != ScreenKind.ACTIONABLE_TABLE.value
        ):
            screen_false_actionable_examples.append(frame_id)
        if (
            truth_screen_kind == ScreenKind.ACTIONABLE_TABLE.value
            and recognized_screen_kind != ScreenKind.ACTIONABLE_TABLE.value
        ):
            screen_missed_actionable_examples.append(frame_id)
        recognition_mode_counts[result.recognition_mode.value] = (
            recognition_mode_counts.get(result.recognition_mode.value, 0) + 1
        )
        contract_counts[result.safety_contract.value] = (
            contract_counts.get(result.safety_contract.value, 0) + 1
        )
        assembly_key = (
            result.assembly_result.status.value
            if result.assembly_result is not None
            else "missing_assembly_result"
        )
        assembly_status_counts[assembly_key] = assembly_status_counts.get(assembly_key, 0) + 1
        false_authorization = False
        if result.state is not None:
            authorization_events += 1
            if truth_screen_kind != ScreenKind.ACTIONABLE_TABLE.value:
                false_authorization = True
                false_actionable_examples.append(frame_id)
            if result.recognition_mode is RecognitionMode.TRUTH_ASSISTED_REPLAY:
                truth_assisted_authorization_events += 1
            if (
                result.assembly_result is not None
                and not result.assembly_result.freshness.current_frame_revalidated
            ):
                stale_authorization_events += 1
            if critical_eval.unsafe_authorization_events or false_authorization:
                unsafe_authorization_events += 1
        source_policy_violation_count += len(critical_eval.source_policy_violations)
        accepted_critical_wrong_count += len(critical_eval.accepted_critical_wrong_cases)
        for violation in critical_eval.source_policy_violations:
            if len(source_policy_violation_examples) < 8:
                source_policy_violation_examples.append(
                    {"frame_id": frame_id, **violation.to_dict()}
                )
        for mismatch in critical_eval.accepted_critical_wrong_cases:
            if len(accepted_critical_wrong_examples) < 8:
                accepted_critical_wrong_examples.append(
                    {"frame_id": frame_id, **mismatch.to_dict()}
                )
        if truth_screen_kind != ScreenKind.ACTIONABLE_TABLE.value:
            non_actionable_frames += 1
        result_counts[outcome] = result_counts.get(outcome, 0) + 1
        examples.setdefault(outcome, [])
        if len(examples[outcome]) < 8:
            examples[outcome].append(frame_id)
        for issue_code in _string_list(row.get("issue_codes")):
            issue_counts[issue_code] = issue_counts.get(issue_code, 0) + 1
        for flag in _string_list(row.get("table_readiness_flags")):
            table_readiness_flag_counts[flag] = table_readiness_flag_counts.get(flag, 0) + 1
        for flag in _string_list(row.get("number_readiness_flags")):
            number_readiness_flag_counts[flag] = number_readiness_flag_counts.get(flag, 0) + 1
        for number_eval in _row_mappings(row.get("number_truth_evaluations")):
            key = _number_truth_comparison_key(number_eval)
            if key is None:
                continue
            number_truth_comparison_counts[key] = number_truth_comparison_counts.get(key, 0) + 1
            if (
                number_eval.get("status") != "match"
                and len(number_truth_mismatch_examples) < 12
            ):
                number_truth_mismatch_examples.append(
                    {"frame_id": frame_id, **dict(number_eval)}
                )
        for number_eval in _row_mappings(row.get("number_component_truth_evaluations")):
            key = _number_component_truth_comparison_key(number_eval)
            if key is None:
                continue
            number_component_truth_comparison_counts[key] = (
                number_component_truth_comparison_counts.get(key, 0) + 1
            )
        for flag in _action_panel_flags(row.get("action_panels")):
            action_panel_flag_counts[flag] = action_panel_flag_counts.get(flag, 0) + 1
            if outcome != "state":
                blocking_action_panel_flag_counts[flag] = (
                    blocking_action_panel_flag_counts.get(flag, 0) + 1
                )
        if outcome != "state":
            for tag in _string_list(row.get("review_tags")):
                review_tag_counts[tag] = review_tag_counts.get(tag, 0) + 1

    review_queue = _review_queue_rows(rows)
    number_readiness_rows = _number_readiness_rows(rows)
    summary: dict[str, object] = {
        "schema_version": 1,
        "frames": len(rows),
        "actionable_only": actionable_only,
        "image_only_replay": image_only_replay,
        "result_counts": dict(sorted(result_counts.items())),
        "issue_counts": dict(sorted(issue_counts.items())),
        "screen_kind_counts": dict(sorted(screen_kind_counts.items())),
        "screen_confusion_counts": _sorted_nested_counts(screen_confusion_counts),
        "screen_false_actionable_count": len(screen_false_actionable_examples),
        "screen_false_actionable_examples": screen_false_actionable_examples[:8],
        "screen_missed_actionable_count": len(screen_missed_actionable_examples),
        "screen_missed_actionable_examples": screen_missed_actionable_examples[:8],
        "recognition_mode_counts": dict(sorted(recognition_mode_counts.items())),
        "contract_counts": dict(sorted(contract_counts.items())),
        "assembly_status_counts": dict(sorted(assembly_status_counts.items())),
        "table_readiness_flag_counts": dict(sorted(table_readiness_flag_counts.items())),
        "number_readiness_flag_counts": dict(sorted(number_readiness_flag_counts.items())),
        "number_truth_comparison_counts": dict(sorted(number_truth_comparison_counts.items())),
        "number_component_truth_comparison_counts": dict(
            sorted(number_component_truth_comparison_counts.items())
        ),
        "number_truth_mismatch_examples": number_truth_mismatch_examples,
        "number_prediction_slot_counts": _number_prediction_slot_counts(
            rows,
            field_name="number_predictions",
        ),
        "accepted_number_prediction_slot_counts": _number_prediction_slot_counts(
            rows,
            field_name="accepted_number_predictions",
        ),
        "number_prediction_confidence_counts": _number_prediction_confidence_counts(
            rows,
            field_name="number_predictions",
        ),
        "accepted_number_prediction_confidence_counts": _number_prediction_confidence_counts(
            rows,
            field_name="accepted_number_predictions",
        ),
        "authorization_events": authorization_events,
        "unsafe_authorization_events": unsafe_authorization_events,
        "stale_authorization_events": stale_authorization_events,
        "truth_assisted_authorization_events": truth_assisted_authorization_events,
        "non_actionable_frames": non_actionable_frames,
        "false_actionable_count": len(false_actionable_examples),
        "false_actionable_examples": false_actionable_examples[:8],
        "source_policy_violation_count": source_policy_violation_count,
        "source_policy_violation_examples": source_policy_violation_examples,
        "accepted_critical_wrong_count": accepted_critical_wrong_count,
        "accepted_critical_wrong_examples": accepted_critical_wrong_examples,
        "review_queue_frames": len(review_queue),
        "review_queue_tag_counts": _review_queue_tag_counts(review_queue),
        "review_queue_by_tag": _review_queue_by_tag(review_queue),
        "number_readiness_rows_count": len(number_readiness_rows),
        "number_readiness_by_flag": _rows_by_string_field(
            rows,
            field_name="number_readiness_flags",
        ),
        "action_panel_flag_counts": dict(sorted(action_panel_flag_counts.items())),
        "blocking_action_panel_flag_counts": dict(
            sorted(blocking_action_panel_flag_counts.items())
        ),
        "review_tag_counts": dict(sorted(review_tag_counts.items())),
        "examples": dict(sorted(examples.items())),
        "rows": rows,
    }
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "table_recognizer_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output / "table_recognizer_review_queue.json").write_text(
        json.dumps(review_queue, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output / "table_recognizer_review_queue_by_tag.json").write_text(
        json.dumps(summary["review_queue_by_tag"], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output / "table_recognizer_number_readiness_by_flag.json").write_text(
        json.dumps(summary["number_readiness_by_flag"], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output / "table_recognizer_number_readiness_rows.json").write_text(
        json.dumps(number_readiness_rows, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_report(output / "table_recognizer_report.md", summary)
    return summary


def _frame_metadata(
    *,
    image_path: Path,
    layout_annotation_path: Path,
    truth_path: Path,
    image_only_replay: bool,
) -> dict[str, object]:
    metadata: dict[str, object] = {
        "poker_legends_image_path": str(image_path),
        "poker_legends_layout_annotation_path": str(layout_annotation_path),
        "coordinate_space": "image",
    }
    if image_only_replay:
        metadata["recognition_mode"] = RecognitionMode.IMAGE_ONLY_REPLAY.value
    else:
        metadata["poker_legends_annotation_path"] = str(truth_path)
    return metadata


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
    parser.add_argument(
        "--image-only-replay",
        action="store_true",
        help="Replay images without passing reviewed truth into the recognizer.",
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
        image_only_replay=args.image_only_replay,
        limit=args.limit,
    )
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=2, sort_keys=True))


def _row_from_result(
    frame_id: str,
    *,
    image_path: Path,
    truth_path: Path,
    layout_annotation_path: Path,
    result: RecognitionResult,
    critical_eval: AcceptedCriticalFieldEvaluation,
    truth: Mapping[str, object],
    truth_screen_kind: str,
) -> dict[str, object]:
    block_reason = result.metadata.get("state_block_reason")
    outcome = "state" if result.state is not None else str(block_reason or "no_state")
    table = result.metadata.get("recognized_table")
    table_dict = table if isinstance(table, Mapping) else {}
    truth_summary = _truth_summary(truth)
    assembly = result.assembly_result
    table_readiness_flags = (
        _table_readiness_flags(table_dict)
        if result.recognition_mode is RecognitionMode.IMAGE_ONLY_REPLAY
        and result.screen.kind is ScreenKind.ACTIONABLE_TABLE
        else []
    )
    number_predictions = result.metadata.get("number_predictions", [])
    accepted_number_predictions = result.metadata.get("accepted_number_predictions", [])
    number_readiness_flags = _number_readiness_flags(
        table_readiness_flags,
        number_predictions=number_predictions,
        accepted_number_predictions=accepted_number_predictions,
    )
    number_truth_evaluations = _number_truth_evaluations(
        number_predictions=number_predictions,
        accepted_number_predictions=accepted_number_predictions,
        truth=truth,
    )
    number_component_truth_evaluations = _number_component_truth_evaluations(
        number_predictions=number_predictions,
        accepted_number_predictions=accepted_number_predictions,
        truth=truth,
    )
    return {
        "frame_id": frame_id,
        "image": str(image_path),
        "truth_path": str(truth_path),
        "layout_annotation_path": str(layout_annotation_path),
        "result": outcome,
        "truth_screen_kind": truth_screen_kind,
        "truth": truth_summary,
        "screen_kind": result.screen.kind.value,
        "review_tags": _review_tags(
            outcome,
            truth_summary,
            table_dict,
            truth_screen_kind=truth_screen_kind,
            recognized_screen_kind=result.screen.kind.value,
        ),
        "recognition_mode": result.recognition_mode.value,
        "safety_contract": result.safety_contract.value,
        "assembly_status": assembly.status.value if assembly is not None else None,
        "validity_scope": assembly.validity_scope.value if assembly is not None else None,
        "issue_codes": [issue.reason_code for issue in assembly.issues]
        if assembly is not None
        else [],
        "table_readiness_flags": table_readiness_flags,
        "number_readiness_flags": number_readiness_flags,
        "confidence": result.confidence,
        "state": _state_summary(result.state),
        "recognized_table": _jsonable(table_dict),
        "action_panels": [
            panel.to_dict() for panel in result.visual_observation.action_panels
        ]
        if result.visual_observation is not None
        else [],
        "number_predictions": number_predictions,
        "accepted_number_predictions": accepted_number_predictions,
        "number_truth_evaluations": number_truth_evaluations,
        "number_component_truth_evaluations": number_component_truth_evaluations,
        "accepted_critical_fields": [
            field.to_dict() for field in result.accepted_critical_fields
        ],
        "source_policy_violations": [
            violation.to_dict() for violation in critical_eval.source_policy_violations
        ],
        "accepted_critical_wrong_cases": [
            mismatch.to_dict() for mismatch in critical_eval.accepted_critical_wrong_cases
        ],
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


def _truth_summary(truth: Mapping[str, object]) -> dict[str, object]:
    return {
        "buttons": [
            {
                "name": str(button.get("name") or ""),
                "visible": bool(button.get("visible", True)),
                "action_type": button.get("action_type"),
                "label": button.get("label"),
            }
            for button in _row_mappings(truth.get("buttons"))
        ],
        "seats": [
            {
                "name": str(seat.get("name") or ""),
                "visible": bool(seat.get("visible", True)),
                "stack": seat.get("stack"),
                "committed": seat.get("committed"),
                "active": seat.get("active"),
                "current": seat.get("current"),
            }
            for seat in _row_mappings(truth.get("seats"))
        ],
        "texts": [
            {
                "name": str(text.get("name") or ""),
                "visible": bool(text.get("visible", True)),
                "value": text.get("value"),
                "normalized_number": text.get("normalized_number"),
            }
            for text in _row_mappings(truth.get("texts"))
            if bool(text.get("visible", True))
        ],
    }


def _table_readiness_flags(table: Mapping[str, object]) -> list[str]:
    if not table:
        return ["readiness_missing_recognized_table"]
    flags: list[str] = []
    seats = _row_mappings(table.get("seats"))
    hero = _seat_by_index(seats, 0)
    if hero is None:
        flags.append("readiness_missing_hero_seat")
    else:
        hero_cards = [
            card
            for card in _row_mappings(hero.get("hole_cards"))
            if bool(card.get("visible", True)) and _optional_str(card.get("card"))
        ]
        if len(hero_cards) != 2:
            flags.append("readiness_missing_hero_hole_cards")
        hero_stack = _optional_int(hero.get("stack"))
        if hero_stack is None or hero_stack < 0:
            flags.append("readiness_missing_hero_stack")
    active_seats = [seat for seat in seats if bool(seat.get("active", True))]
    if len(active_seats) < 2:
        flags.append("readiness_not_enough_players")
    pot = _optional_int(table.get("pot"))
    if pot is None:
        flags.append("readiness_missing_pot")
    board_count = len(
        [
            card
            for card in _row_mappings(table.get("board"))
            if bool(card.get("visible", True)) and _optional_str(card.get("card"))
        ]
    )
    if board_count not in {0, 3, 4, 5}:
        flags.append("readiness_invalid_board_count")
    buttons = _row_mappings(table.get("buttons"))
    if not buttons:
        flags.append("readiness_missing_current_action_row")
    elif not any(
        _optional_str(button.get("action_type")) in {"check", "call", "bet"}
        for button in buttons
    ):
        flags.append("readiness_missing_passive_action")
    return flags


def _number_readiness_flags(
    table_readiness_flags: list[str],
    *,
    number_predictions: object,
    accepted_number_predictions: object,
) -> list[str]:
    flags: list[str] = []
    predictions = _row_mappings(number_predictions)
    accepted = _row_mappings(accepted_number_predictions)
    if "readiness_not_enough_players" in table_readiness_flags and not _has_number_prediction(
        accepted,
        group="texts",
        name="right_top_stack",
    ):
        right_top = _best_number_prediction(
            predictions,
            group="texts",
            name="right_top_stack",
        )
        if _prediction_has_value(right_top):
            flags.append("readiness_low_confidence_opponent_stack")
        else:
            flags.append("readiness_missing_opponent_stack_ocr")
    if "readiness_missing_hero_seat" in table_readiness_flags and not _has_number_prediction(
        accepted,
        group="texts",
        name="hero_stack",
    ):
        hero_stack = _best_number_prediction(
            predictions,
            group="texts",
            name="hero_stack",
        )
        if _prediction_has_value(hero_stack):
            flags.append("readiness_low_confidence_hero_stack")
        else:
            flags.append("readiness_missing_hero_stack_ocr")
    return flags


def _has_number_prediction(
    predictions: Sequence[Mapping[str, object]],
    *,
    group: str,
    name: str,
) -> bool:
    return any(
        prediction.get("group") == group
        and prediction.get("name") == name
        and _optional_int(prediction.get("normalized_number")) is not None
        for prediction in predictions
    )


def _best_number_prediction(
    predictions: Sequence[Mapping[str, object]],
    *,
    group: str,
    name: str,
) -> Mapping[str, object] | None:
    matches = [
        prediction
        for prediction in predictions
        if prediction.get("group") == group and prediction.get("name") == name
    ]
    if not matches:
        return None
    return max(matches, key=lambda prediction: _optional_float(prediction.get("confidence")) or 0.0)


def _prediction_has_value(prediction: Mapping[str, object] | None) -> bool:
    return prediction is not None and _optional_int(prediction.get("normalized_number")) is not None


def _seat_by_index(
    seats: Sequence[Mapping[str, object]],
    seat_index: int,
) -> Mapping[str, object] | None:
    for seat in seats:
        if _optional_int(seat.get("seat")) == seat_index:
            return seat
    return None


def _expected_critical_values(
    truth: Mapping[str, object],
    *,
    controlled_seat: int,
) -> dict[str, object]:
    expected: dict[str, object] = {}
    screen = truth.get("screen")
    if isinstance(screen, Mapping) and bool(screen.get("hero_turn")):
        expected["screen.actionability"] = controlled_seat
    hero_cards = _truth_cards(truth.get("hero_hole_cards"))
    if hero_cards:
        expected["cards.hero"] = tuple(hero_cards)
    if "board" in truth:
        expected["cards.board"] = tuple(_truth_cards(truth.get("board")))
    pot = _truth_text_number(truth, "pot")
    if pot is not None:
        expected["numbers.pot"] = pot
    hero_stack = _truth_text_number(truth, "hero_stack")
    if hero_stack is None:
        hero_stack = _truth_hero_stack_from_seats(truth)
    if hero_stack is not None:
        expected["numbers.hero_stack"] = hero_stack
    legal_labels = tuple(
        action_type
        for action_type in (
            _truth_button_action_type(button)
            for button in _visible_direct_truth_action_buttons(truth)
        )
        if action_type
    )
    if legal_labels:
        expected["actions.legal_labels"] = legal_labels
    call_amount = _truth_call_amount(truth)
    if call_amount is not None:
        expected["numbers.call_amount"] = call_amount
    return expected


def _truth_cards(value: object) -> list[str]:
    cards: list[str] = []
    for entry in sorted(
        _row_mappings(value),
        key=lambda item: str(item.get("slot") or ""),
    ):
        if not bool(entry.get("visible", True)):
            continue
        card = _canonical_card_code(entry.get("card"))
        if card is not None:
            cards.append(card)
    return cards


def _canonical_card_code(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    code = value.strip().replace(" ", "").upper()
    if len(code) < 2:
        return None
    suit = code[-1].lower()
    rank = code[:-1]
    if rank == "10":
        rank = "T"
    if rank not in {"2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", "A"}:
        return None
    if suit not in {"c", "d", "h", "s"}:
        return None
    return f"{rank}{suit}"


def _truth_text_number(truth: Mapping[str, object], name: str) -> int | None:
    for text in _row_mappings(truth.get("texts")):
        if str(text.get("name") or "") != name or not bool(text.get("visible", True)):
            continue
        number = _optional_int(text.get("normalized_number"))
        if number is not None:
            return number
        value = text.get("value")
        if isinstance(value, str):
            return parse_poker_legends_chip_amount(value)
    return None


def _truth_hero_stack_from_seats(truth: Mapping[str, object]) -> int | None:
    for seat in _visible_truth_seats(truth):
        if str(seat.get("name") or "") == "hero":
            return _optional_int(seat.get("stack"))
    return None


def _truth_call_amount(truth: Mapping[str, object]) -> int | None:
    for button in _visible_direct_truth_action_buttons(truth):
        if _truth_button_action_type(button) != "call":
            continue
        label = button.get("label")
        if isinstance(label, str):
            amount = parse_poker_legends_chip_amount(label)
            if amount is not None:
                return amount
    committed: list[int] = []
    for seat in _visible_truth_seats(truth):
        seat_committed = _optional_int(seat.get("committed"))
        if seat_committed is not None:
            committed.append(seat_committed)
    hero_committed = None
    for seat in _visible_truth_seats(truth):
        if str(seat.get("name") or "") == "hero":
            hero_committed = _optional_int(seat.get("committed"))
            break
    if hero_committed is None or not committed:
        return None
    to_call = max(committed) - hero_committed
    return to_call if to_call > 0 else None


def _truth_button_action_type(button: Mapping[str, object]) -> str | None:
    label = button.get("label")
    if isinstance(label, str):
        normalized = label.strip().lower()
        if "check" in normalized:
            return "check"
        if "call" in normalized:
            return "call"
        if "fold" in normalized:
            return "fold"
        if "all in" in normalized or "all-in" in normalized:
            return "all_in"
        if "bet" in normalized:
            return "bet"
        if "raise" in normalized or "blind" in normalized:
            return "raise"
    return _optional_str(button.get("action_type"))


def _review_tags(
    outcome: str,
    truth: Mapping[str, object],
    table: Mapping[str, object],
    *,
    truth_screen_kind: str,
    recognized_screen_kind: str,
) -> list[str]:
    if outcome == "state":
        return []
    if (
        truth_screen_kind != ScreenKind.ACTIONABLE_TABLE.value
        and recognized_screen_kind == ScreenKind.ACTIONABLE_TABLE.value
    ):
        return ["screen_false_actionable"]
    if (
        truth_screen_kind == ScreenKind.ACTIONABLE_TABLE.value
        and recognized_screen_kind != ScreenKind.ACTIONABLE_TABLE.value
    ):
        return ["screen_missed_actionable"]
    if outcome == "screen_not_actionable":
        return ["negative_screen_state"]
    if outcome == "hero_not_current":
        return ["hero_turn_not_confirmed"]
    if outcome == "preselect_ambiguous":
        return ["primary_preselect_shortcut"]
    if outcome == "not_enough_players":
        if len(_visible_truth_seats(truth)) < 2:
            return ["truth_missing_opponent_seat"]
        return ["seat_assembly_gap"]
    if outcome == "missing_current_action_row":
        if _visible_direct_truth_action_buttons(truth):
            return ["button_recognizer_missed_truth_action_row"]
        return ["truth_missing_current_action_row"]
    if outcome == "missing_passive_action":
        if _has_truth_action_type(truth, {"check", "call", "bet"}):
            return ["passive_action_assembly_gap"]
        if _has_recognized_action_type(table, {"check", "call", "bet"}):
            return ["passive_action_contract_gap"]
        return ["truth_missing_passive_action"]
    return [outcome]


def _review_queue_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    queue: list[dict[str, object]] = []
    for row in rows:
        if row.get("result") == "state":
            continue
        tags = _string_list(row.get("review_tags"))
        if "negative_screen_state" in tags:
            continue
        queue.append(row)
    return queue


def _number_readiness_rows(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    readiness_rows: list[dict[str, object]] = []
    for row in rows:
        flags = _string_list(row.get("number_readiness_flags"))
        if not flags:
            continue
        readiness_rows.append(
            {
                "frame_id": row.get("frame_id"),
                "result": row.get("result"),
                "truth_screen_kind": row.get("truth_screen_kind"),
                "screen_kind": row.get("screen_kind"),
                "truth_path": row.get("truth_path"),
                "layout_annotation_path": row.get("layout_annotation_path"),
                "table_readiness_flags": _string_list(row.get("table_readiness_flags")),
                "number_readiness_flags": flags,
                "number_predictions": _row_mappings(row.get("number_predictions")),
                "accepted_number_predictions": _row_mappings(
                    row.get("accepted_number_predictions")
                ),
            }
        )
    return readiness_rows


def _review_queue_tag_counts(rows: list[dict[str, object]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        for tag in _string_list(row.get("review_tags")):
            counts[tag] = counts.get(tag, 0) + 1
    return dict(sorted(counts.items()))


def _review_queue_by_tag(rows: list[dict[str, object]]) -> dict[str, list[str]]:
    return _rows_by_string_field(rows, field_name="review_tags")


def _rows_by_string_field(
    rows: Sequence[Mapping[str, object]],
    *,
    field_name: str,
) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {}
    for row in rows:
        frame_id = str(row.get("frame_id") or "")
        if not frame_id:
            continue
        for value in _string_list(row.get(field_name)):
            grouped.setdefault(value, []).append(frame_id)
    return {value: grouped[value] for value in sorted(grouped)}


def _number_prediction_slot_counts(
    rows: Sequence[Mapping[str, object]],
    *,
    field_name: str,
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        for prediction in _row_mappings(row.get(field_name)):
            key = _number_prediction_slot_key(prediction)
            if key is None:
                continue
            counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def _number_prediction_confidence_counts(
    rows: Sequence[Mapping[str, object]],
    *,
    field_name: str,
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        for prediction in _row_mappings(row.get(field_name)):
            slot_key = _number_prediction_slot_key(prediction)
            if slot_key is None:
                continue
            confidence = _optional_float(prediction.get("confidence"))
            label = "none" if confidence is None else f"{confidence:.2f}"
            key = f"{slot_key}:conf={label}"
            counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def _number_prediction_slot_key(prediction: Mapping[str, object]) -> str | None:
    group = _optional_str(prediction.get("group"))
    name = _optional_str(prediction.get("name"))
    if group is None or name is None:
        return None
    return f"{group}:{name}"


def _number_truth_evaluations(
    *,
    number_predictions: object,
    accepted_number_predictions: object,
    truth: Mapping[str, object],
) -> list[dict[str, object]]:
    expected_numbers = _truth_text_numbers_by_name(truth)
    evaluations: list[dict[str, object]] = []
    for prediction_set, predictions in (
        ("raw", number_predictions),
        ("accepted", accepted_number_predictions),
    ):
        for prediction in _row_mappings(predictions):
            if prediction.get("group") != "texts":
                continue
            name = _optional_str(prediction.get("name"))
            if name is None or name not in expected_numbers:
                continue
            predicted = _optional_int(prediction.get("normalized_number"))
            expected = expected_numbers[name]
            status = "match" if predicted == expected else "mismatch"
            evaluations.append(
                {
                    "prediction_set": prediction_set,
                    "field_path": f"numbers.{name}",
                    "expected": expected,
                    "predicted": predicted,
                    "status": status,
                    "confidence": prediction.get("confidence"),
                    "raw": prediction.get("raw"),
                }
            )
    return evaluations


def _number_component_truth_evaluations(
    *,
    number_predictions: object,
    accepted_number_predictions: object,
    truth: Mapping[str, object],
) -> list[dict[str, object]]:
    expected_numbers = _truth_text_numbers_by_name(truth)
    evaluations: list[dict[str, object]] = []
    for prediction_set, predictions in (
        ("raw", number_predictions),
        ("accepted", accepted_number_predictions),
    ):
        for prediction in _row_mappings(predictions):
            if prediction.get("group") != "texts":
                continue
            name = _optional_str(prediction.get("name"))
            if name is None or name not in expected_numbers:
                continue
            expected = expected_numbers[name]
            for component in ("base_number", "total_number"):
                predicted = _optional_int(prediction.get(component))
                if predicted is None:
                    continue
                status = "match" if predicted == expected else "mismatch"
                evaluations.append(
                    {
                        "prediction_set": prediction_set,
                        "field_path": f"numbers.{name}",
                        "component": component,
                        "expected": expected,
                        "predicted": predicted,
                        "status": status,
                        "confidence": prediction.get("confidence"),
                        "raw": prediction.get("raw"),
                    }
                )
    return evaluations


def _truth_text_numbers_by_name(truth: Mapping[str, object]) -> dict[str, int]:
    numbers: dict[str, int] = {}
    for text in _row_mappings(truth.get("texts")):
        if not bool(text.get("visible", True)):
            continue
        name = _optional_str(text.get("name"))
        if name is None:
            continue
        number = _optional_int(text.get("normalized_number"))
        if number is None and isinstance(text.get("value"), str):
            number = parse_poker_legends_chip_amount(str(text.get("value")))
        if number is not None:
            numbers[name] = number
    return numbers


def _number_truth_comparison_key(evaluation: Mapping[str, object]) -> str | None:
    prediction_set = _optional_str(evaluation.get("prediction_set"))
    field_path = _optional_str(evaluation.get("field_path"))
    status = _optional_str(evaluation.get("status"))
    if prediction_set is None or field_path is None or status is None:
        return None
    return f"{prediction_set}:{field_path}:{status}"


def _number_component_truth_comparison_key(evaluation: Mapping[str, object]) -> str | None:
    prediction_set = _optional_str(evaluation.get("prediction_set"))
    field_path = _optional_str(evaluation.get("field_path"))
    component = _optional_str(evaluation.get("component"))
    status = _optional_str(evaluation.get("status"))
    if prediction_set is None or field_path is None or component is None or status is None:
        return None
    return f"{prediction_set}:{field_path}:{component}:{status}"


def _visible_truth_seats(truth: Mapping[str, object]) -> list[Mapping[str, object]]:
    return [seat for seat in _row_mappings(truth.get("seats")) if bool(seat.get("visible", True))]


def _visible_direct_truth_action_buttons(truth: Mapping[str, object]) -> list[Mapping[str, object]]:
    direct_names = {
        "primary_left",
        "primary_middle",
        "primary_right",
        "check",
        "call",
        "raise",
        "fold",
    }
    buttons: list[Mapping[str, object]] = []
    for button in _row_mappings(truth.get("buttons")):
        if not bool(button.get("visible", True)):
            continue
        name = str(button.get("name") or "")
        if name in direct_names or name.endswith("_button"):
            buttons.append(button)
    return buttons


def _has_truth_action_type(truth: Mapping[str, object], action_types: set[str]) -> bool:
    for button in _visible_direct_truth_action_buttons(truth):
        action_type = _truth_button_action_type(button)
        if action_type in action_types:
            return True
    return False


def _has_recognized_action_type(table: Mapping[str, object], action_types: set[str]) -> bool:
    for button in _row_mappings(table.get("buttons")):
        action_type = button.get("action_type")
        if isinstance(action_type, str) and action_type in action_types:
            return True
    return False


def _write_report(path: Path, summary: Mapping[str, object]) -> None:
    counts = summary.get("result_counts")
    issue_counts = summary.get("issue_counts")
    action_panel_flag_counts = summary.get("action_panel_flag_counts")
    blocking_action_panel_flag_counts = summary.get("blocking_action_panel_flag_counts")
    review_tag_counts = summary.get("review_tag_counts")
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
        f"- Image-only replay: {summary.get('image_only_replay', False)}",
        f"- Authorization events: {summary.get('authorization_events', 0)}",
        f"- Unsafe authorization events: {summary.get('unsafe_authorization_events', 0)}",
        f"- Stale authorization events: {summary.get('stale_authorization_events', 0)}",
        "- Truth-assisted authorization events: "
        f"{summary.get('truth_assisted_authorization_events', 0)}",
        f"- Non-actionable frames: {summary.get('non_actionable_frames', 0)}",
        f"- False actionable count: {summary.get('false_actionable_count', 0)}",
        f"- Screen false actionable count: {summary.get('screen_false_actionable_count', 0)}",
        f"- Screen missed actionable count: {summary.get('screen_missed_actionable_count', 0)}",
        f"- Source policy violation count: {summary.get('source_policy_violation_count', 0)}",
        f"- Accepted critical wrong count: {summary.get('accepted_critical_wrong_count', 0)}",
        f"- Review queue frames: {summary.get('review_queue_frames', 0)}",
        f"- Number readiness rows: {summary.get('number_readiness_rows_count', 0)}",
        "",
        "## Recognition Mode Counts",
        "",
        *_count_lines(summary.get("recognition_mode_counts")),
        "",
        "## Contract Counts",
        "",
        *_count_lines(summary.get("contract_counts")),
        "",
        "## Assembly Status Counts",
        "",
        *_count_lines(summary.get("assembly_status_counts")),
        "",
        "## Table Readiness Flag Counts",
        "",
        *_count_lines(summary.get("table_readiness_flag_counts")),
        "",
        "## Number Readiness Flag Counts",
        "",
        *_count_lines(summary.get("number_readiness_flag_counts")),
        "",
        "## Number Readiness By Flag",
        "",
        *_frame_list_lines(summary.get("number_readiness_by_flag")),
        "",
        "## Number Truth Comparison Counts",
        "",
        *_count_lines(summary.get("number_truth_comparison_counts")),
        "",
        "## Number Component Truth Comparison Counts",
        "",
        *_count_lines(summary.get("number_component_truth_comparison_counts")),
        "",
        "## Number Truth Mismatch Examples",
        "",
        *_number_truth_mismatch_lines(summary.get("number_truth_mismatch_examples")),
        "",
        "## Number Prediction Slot Counts",
        "",
        *_count_lines(summary.get("number_prediction_slot_counts")),
        "",
        "## Accepted Number Prediction Slot Counts",
        "",
        *_count_lines(summary.get("accepted_number_prediction_slot_counts")),
        "",
        "## Number Prediction Confidence Counts",
        "",
        *_count_lines(summary.get("number_prediction_confidence_counts")),
        "",
        "## Accepted Number Prediction Confidence Counts",
        "",
        *_count_lines(summary.get("accepted_number_prediction_confidence_counts")),
        "",
        "## Number Readiness Details",
        "",
        *_number_readiness_detail_lines(rows),
        "",
        "## Screen Kind Counts",
        "",
        *_count_lines(summary.get("screen_kind_counts")),
        "",
        "## Screen Truth Confusion",
        "",
        *_nested_count_lines(summary.get("screen_confusion_counts")),
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
        "## Review Tag Counts",
        "",
        *_count_lines(review_tag_counts),
        "",
        "## Review Queue Tag Counts",
        "",
        *_count_lines(summary.get("review_queue_tag_counts")),
        "",
        "## Review Queue By Tag",
        "",
        *_frame_list_lines(summary.get("review_queue_by_tag")),
        "",
        "## Accepted Critical Wrong Examples",
        "",
        *_example_dict_lines(summary.get("accepted_critical_wrong_examples")),
        "",
        "## Source Policy Violation Examples",
        "",
        *_example_dict_lines(summary.get("source_policy_violation_examples")),
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
            "| Frame | Result | Review Tags | Issues | Action Panels | Truth Buttons | Buttons | "
            "Truth Seats | Seats | Truth Texts | Accepted Numbers |"
        )
        lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
        for row in blockers:
            table = row.get("recognized_table")
            table_dict = table if isinstance(table, Mapping) else {}
            truth = row.get("truth")
            truth_dict = truth if isinstance(truth, Mapping) else {}
            lines.append(
                "| "
                f"`{row.get('frame_id')}` | "
                f"`{row.get('result')}` | "
                f"{_inline_codes(row.get('review_tags'))} | "
                f"{_inline_codes(row.get('issue_codes'))} | "
                f"{_action_panel_summary(row.get('action_panels'))} | "
                f"{_truth_button_summary(truth_dict.get('buttons'))} | "
                f"{_button_summary(table_dict.get('buttons'))} | "
                f"{_truth_seat_summary(truth_dict.get('seats'))} | "
                f"{_seat_summary(table_dict.get('seats'))} | "
                f"{_truth_text_summary(truth_dict.get('texts'))} | "
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


def _nested_count_lines(value: object) -> list[str]:
    if not isinstance(value, Mapping) or not value:
        return ["- none"]
    lines: list[str] = []
    for outer, inner in sorted(value.items()):
        if not isinstance(inner, Mapping) or not inner:
            lines.append(f"- {outer}: none")
            continue
        parts = ", ".join(f"{key}={count}" for key, count in sorted(inner.items()))
        lines.append(f"- {outer}: {parts}")
    return lines


def _sorted_nested_counts(
    counts: Mapping[str, Mapping[str, int]],
) -> dict[str, dict[str, int]]:
    return {
        outer: dict(sorted(inner.items()))
        for outer, inner in sorted(counts.items())
    }


def _example_dict_lines(value: object) -> list[str]:
    if not isinstance(value, list) or not value:
        return ["- none"]
    lines: list[str] = []
    for item in value:
        if not isinstance(item, Mapping):
            continue
        frame_id = item.get("frame_id", "")
        field_path = item.get("field_path", "")
        expected = item.get("expected")
        accepted = item.get("accepted")
        source = item.get("source", "")
        mode = item.get("recognition_mode", "")
        if "expected" in item or "accepted" in item:
            lines.append(
                f"- `{frame_id}` `{field_path}` expected={expected!r} "
                f"accepted={accepted!r} source=`{source}`"
            )
        else:
            lines.append(f"- `{frame_id}` `{field_path}` source=`{source}` mode=`{mode}`")
    return lines or ["- none"]


def _frame_list_lines(value: object) -> list[str]:
    if not isinstance(value, Mapping) or not value:
        return ["- none"]
    lines: list[str] = []
    for key, frames in sorted(value.items()):
        if not isinstance(frames, list) or not frames:
            lines.append(f"- {key}: none")
            continue
        joined = ", ".join(f"`{frame}`" for frame in frames if isinstance(frame, str))
        lines.append(f"- {key}: {joined or 'none'}")
    return lines


def _number_readiness_detail_lines(rows: Sequence[Mapping[str, object]]) -> list[str]:
    readiness_rows = _number_readiness_rows(rows)
    if not readiness_rows:
        return ["- none"]
    lines = [
        "| Frame | Number Flags | Table Flags | Raw Numbers | Accepted Numbers |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in readiness_rows:
        lines.append(
            "| "
            f"`{row.get('frame_id')}` | "
            f"{_inline_codes(row.get('number_readiness_flags'))} | "
            f"{_inline_codes(row.get('table_readiness_flags'))} | "
            f"{_number_detail_summary(row.get('number_predictions'))} | "
            f"{_number_detail_summary(row.get('accepted_number_predictions'))} |"
        )
    return lines


def _number_truth_mismatch_lines(value: object) -> list[str]:
    if not isinstance(value, list) or not value:
        return ["- none"]
    lines: list[str] = []
    for item in value:
        if not isinstance(item, Mapping):
            continue
        lines.append(
            "- `{frame_id}` `{prediction_set}` `{field_path}` expected={expected!r} "
            "predicted={predicted!r} conf={confidence!r} raw=`{raw}`".format(
                frame_id=item.get("frame_id"),
                prediction_set=item.get("prediction_set"),
                field_path=item.get("field_path"),
                expected=item.get("expected"),
                predicted=item.get("predicted"),
                confidence=item.get("confidence"),
                raw=_compact_text(item.get("raw")),
            )
        )
    return lines or ["- none"]


def _row_mappings(value: object) -> list[Mapping[str, object]]:
    if not isinstance(value, list | tuple):
        return []
    return [item for item in value if isinstance(item, Mapping)]


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str)]


def _optional_str(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _optional_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


def _optional_float(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    return None


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


def _truth_button_summary(value: object) -> str:
    buttons = _row_mappings(value)
    if not buttons:
        return "`none`"
    return "<br>".join(
        "`{name}:visible={visible}:{action_type}:{label}`".format(
            name=button.get("name"),
            visible=button.get("visible"),
            action_type=button.get("action_type"),
            label=button.get("label"),
        )
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


def _truth_seat_summary(value: object) -> str:
    seats = _row_mappings(value)
    if not seats:
        return "`none`"
    return "<br>".join(
        "`{name}:visible={visible} stack={stack} committed={committed} current={current}`".format(
            name=seat.get("name"),
            visible=seat.get("visible"),
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


def _number_detail_summary(value: object) -> str:
    numbers = _row_mappings(value)
    if not numbers:
        return "`none`"
    return "<br>".join(
        "`{group}:{name}={number} conf={confidence} raw={raw}`".format(
            group=number.get("group"),
            name=number.get("name"),
            number=number.get("normalized_number"),
            confidence=number.get("confidence"),
            raw=_compact_text(number.get("raw")),
        )
        for number in numbers
    )


def _compact_text(value: object) -> str:
    if not isinstance(value, str):
        return ""
    compacted = " ".join(value.split())
    return compacted[:48] + "..." if len(compacted) > 51 else compacted


def _truth_text_summary(value: object) -> str:
    texts = _row_mappings(value)
    if not texts:
        return "`none`"
    return "<br>".join(
        "`{name}={number}:{value}`".format(
            name=text.get("name"),
            number=text.get("normalized_number"),
            value=text.get("value"),
        )
        for text in texts
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
