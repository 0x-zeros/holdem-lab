"""Build reviewable Poker Legends truth overlays from LLM candidates."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import cast

from holdem_bot.screen_state import ScreenKind, ScreenState

POKER_ACTIONS = frozenset({"fold", "check", "call", "bet", "raise", "all_in"})
KNOWN_BLOCKING_REASONS = frozenset(
    {
        "buy_in_modal",
        "reward_overlay",
        "challenge_overlay",
        "leave_table_modal",
        "lobby_overlay",
        "other_overlay",
    }
)
CARD_RANKS = frozenset("AKQJT98765432")
CARD_SUITS = frozenset("SHDC")


def build_poker_legends_truth_overlay(
    candidate_paths: Sequence[str | Path],
    *,
    review_decisions_path: str | Path,
    output_dir: str | Path,
) -> dict[str, object]:
    candidates = [Path(path) for path in candidate_paths]
    if not candidates:
        raise ValueError("at least one LLM candidate annotation is required")

    review = _read_json_object(review_decisions_path)
    decisions = _decisions_by_frame(review)
    output = Path(output_dir)
    truth_dir = output / "truth_overlays"
    truth_dir.mkdir(parents=True, exist_ok=True)

    frames: list[dict[str, object]] = []
    screen_counts: Counter[str] = Counter()
    status_counts: Counter[str] = Counter()
    warning_count = 0
    uncertain_count = 0
    human_reviewed = 0

    for candidate_path in sorted(candidates):
        candidate = _read_json_object(candidate_path)
        frame_id = str(candidate.get("frame_id") or candidate_path.stem)
        decision = decisions.get(frame_id)
        truth = merge_poker_legends_truth(candidate, decision=decision)
        (truth_dir / f"{frame_id}.json").write_text(
            json.dumps(truth, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        screen = _mapping_or_empty(truth["screen"])
        review_info = _mapping_or_empty(truth["review"])
        validation = _mapping_or_empty(truth["validation"])
        warnings = _sequence(validation.get("warnings"))
        uncertain = _sequence(truth.get("uncertain"))
        screen_kind = str(screen.get("kind", ScreenKind.UNKNOWN_OR_TRANSITION.value))
        review_status = str(review_info.get("status", "candidate_unreviewed"))

        screen_counts[screen_kind] += 1
        status_counts[review_status] += 1
        warning_count += len(warnings)
        uncertain_count += len(uncertain)
        if review_status != "candidate_unreviewed":
            human_reviewed += 1
        frames.append(_report_frame_row(truth))

    summary = {
        "schema_version": 1,
        "frames": len(frames),
        "human_reviewed_frames": human_reviewed,
        "screen_counts": dict(sorted(screen_counts.items())),
        "review_status_counts": dict(sorted(status_counts.items())),
        "warning_count": warning_count,
        "uncertain_count": uncertain_count,
        "truth_overlays": "truth_overlays",
        "report": "truth_overlay_report.md",
        "summary": "truth_overlay_summary.json",
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "truth_overlay_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_truth_report(output / "truth_overlay_report.md", summary=summary, frames=frames)
    return summary


def merge_poker_legends_truth(
    candidate: Mapping[str, object],
    *,
    decision: Mapping[str, object] | None = None,
) -> dict[str, object]:
    frame_id = str(candidate.get("frame_id", "unknown"))
    warnings: list[str] = []
    table_state = _normalize_table_state(_mapping_or_empty(candidate.get("table_state")))
    board = _normalize_card_items(candidate.get("board"), group="board", warnings=warnings)
    hero_hole_cards = _normalize_card_items(
        candidate.get("hero_hole_cards"),
        group="hero_hole_cards",
        warnings=warnings,
    )
    buttons = _normalize_buttons(candidate.get("buttons"))
    texts = _normalize_texts(candidate.get("texts"))
    seats = _normalize_seats(candidate.get("seats"))
    uncertain = _normalize_uncertain(candidate.get("uncertain"))
    ignored_fields: list[str] = []

    review_info = _normalize_review_info(decision)
    overrides = _mapping_or_empty(review_info.get("overrides"))
    if "is_actionable" in overrides:
        table_state["is_actionable"] = bool(overrides["is_actionable"])
    if "blocking_reason" in overrides:
        table_state["blocking_reason"] = _blocking_reason_or_none(overrides["blocking_reason"])
    if "hero_seat" in overrides:
        review_info["hero_seat"] = str(overrides["hero_seat"])

    ignored_fields.extend(str(item) for item in _sequence(overrides.get("ignore_fields")))
    if bool(overrides.get("ignore_poker_fields")):
        ignored_fields.extend(["board", "hero_hole_cards", "texts", "seats"])

    for field, value in overrides.items():
        if isinstance(field, str) and field.startswith("board."):
            _apply_board_override(board, field, value, warnings)

    buttons = _normalize_buttons_for_table_state(buttons, table_state)
    table_state, board, hero_hole_cards, texts, seats = _apply_ignored_fields(
        table_state,
        board,
        hero_hole_cards,
        texts,
        seats,
        ignored_fields,
    )
    seats = _apply_seat_overrides(
        seats,
        _mapping_or_empty(overrides.get("seat_overrides")),
        texts,
    )
    hero = _build_hero_info(seats, table_state, buttons, review_info)
    screen = _screen_dict_from_truth(table_state, hero=hero, buttons=buttons)
    _validate_truth(
        frame_id,
        table_state=table_state,
        screen=screen,
        board=board,
        hero_hole_cards=hero_hole_cards,
        buttons=buttons,
        warnings=warnings,
    )

    ignored = tuple(dict.fromkeys(ignored_fields))
    review_info["ignored_fields"] = list(ignored)
    return {
        "schema_version": 1,
        "frame_id": frame_id,
        "source": "llm_candidate_human_review_v1",
        "screen": screen,
        "table_state": table_state,
        "hero": hero,
        "hero_hole_cards": hero_hole_cards,
        "board": board,
        "buttons": buttons,
        "texts": texts,
        "seats": seats,
        "ignored_fields": list(ignored),
        "uncertain": uncertain,
        "review": review_info,
        "validation": {
            "status": "needs_review" if warnings else "ok",
            "warnings": warnings,
        },
    }


def screen_state_from_poker_legends_annotation(
    annotation: Mapping[str, object],
) -> ScreenState:
    screen = annotation.get("screen")
    if isinstance(screen, Mapping):
        return _screen_state_from_screen_mapping(screen)
    table_state = _normalize_table_state(_mapping_or_empty(annotation.get("table_state")))
    buttons = _normalize_buttons(annotation.get("buttons"))
    hero = _build_hero_info(
        _normalize_seats(annotation.get("seats")),
        table_state,
        buttons,
        {},
    )
    return _screen_state_from_screen_mapping(
        _screen_dict_from_truth(table_state, hero=hero, buttons=buttons)
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build reviewable Poker Legends truth overlays from LLM candidates."
    )
    parser.add_argument("candidates", nargs="+", help="LLM candidate annotation JSON files.")
    parser.add_argument(
        "--review-decisions",
        required=True,
        help="Human review decisions JSON file.",
    )
    parser.add_argument(
        "--out", required=True, help="Output directory for truth overlay artifacts."
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    summary = build_poker_legends_truth_overlay(
        args.candidates,
        review_decisions_path=args.review_decisions,
        output_dir=args.out,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


def _screen_state_from_screen_mapping(screen: Mapping[str, object]) -> ScreenState:
    kind = ScreenKind(str(screen.get("kind", ScreenKind.UNKNOWN_OR_TRANSITION.value)))
    confidence = _to_float(screen.get("confidence", 0.0))
    reason = str(screen.get("reason") or kind.value)
    blocking_reason = _string_or_none(screen.get("blocking_reason"))
    hero_turn = _optional_bool(screen.get("hero_turn"))
    if kind is ScreenKind.ACTIONABLE_TABLE:
        return ScreenState.actionable_table(
            confidence=confidence,
            hero_turn=hero_turn,
            reason=reason,
        )
    if kind is ScreenKind.TABLE_OBSERVE:
        return ScreenState.table_observe(confidence=confidence, reason=reason)
    if kind is ScreenKind.BLOCKED_OVERLAY:
        return ScreenState.blocked_overlay(
            blocking_reason=blocking_reason or "other_overlay",
            confidence=confidence,
            reason=reason,
        )
    if kind is ScreenKind.NON_TABLE_UI:
        return ScreenState.non_table_ui(confidence=confidence, reason=reason)
    return ScreenState.unknown_or_transition(confidence=confidence, reason=reason)


def _screen_dict_from_truth(
    table_state: Mapping[str, object],
    *,
    hero: Mapping[str, object],
    buttons: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    is_table = bool(table_state.get("is_table"))
    is_actionable = bool(table_state.get("is_actionable"))
    blocking_reason = _blocking_reason_or_none(table_state.get("blocking_reason"))
    confidence = _to_float(table_state.get("confidence", 0.0))
    summary = str(table_state.get("summary") or "")
    hero_turn = _optional_bool(hero.get("current"))
    if hero_turn is None and is_actionable:
        hero_turn = _has_visible_poker_action(buttons)
    if not is_table:
        kind = ScreenKind.NON_TABLE_UI
    elif blocking_reason is not None:
        kind = ScreenKind.BLOCKED_OVERLAY
        hero_turn = False
    elif is_actionable:
        kind = ScreenKind.ACTIONABLE_TABLE
    elif is_table:
        kind = ScreenKind.TABLE_OBSERVE
        hero_turn = False
    else:
        kind = ScreenKind.UNKNOWN_OR_TRANSITION

    return {
        "kind": kind.value,
        "confidence": confidence,
        "reason": summary or kind.value,
        "blocking_reason": blocking_reason if kind is ScreenKind.BLOCKED_OVERLAY else None,
        "hero_turn": hero_turn,
    }


def _normalize_table_state(table_state: Mapping[str, object]) -> dict[str, object]:
    return {
        "is_table": bool(table_state.get("is_table", False)),
        "is_actionable": bool(table_state.get("is_actionable", False)),
        "street": _street_or_none(table_state.get("street")),
        "blocking_reason": _blocking_reason_or_none(table_state.get("blocking_reason")),
        "summary": str(table_state.get("summary") or ""),
        "confidence": _to_float(table_state.get("confidence", 0.0)),
    }


def _normalize_card_items(
    value: object,
    *,
    group: str,
    warnings: list[str],
) -> list[dict[str, object]]:
    items: list[dict[str, object]] = []
    for item in _mapping_sequence(value):
        raw_slot = item.get("slot") or item.get("name")
        slot = _roi_name(raw_slot)
        raw_card = item.get("card")
        card = _card_or_none(raw_card)
        if bool(item.get("visible")) and raw_card is not None and card is None:
            warnings.append(f"{group}.{slot} has invalid card code {raw_card!r}")
        items.append(
            {
                "slot": slot,
                "visible": bool(item.get("visible")),
                "card": card,
                "confidence": _to_float(item.get("confidence", 0.0)),
            }
        )
    return items


def _normalize_buttons(value: object) -> list[dict[str, object]]:
    buttons: list[dict[str, object]] = []
    for item in _mapping_sequence(value):
        buttons.append(
            {
                "name": _roi_name(item.get("name")),
                "visible": bool(item.get("visible")),
                "label": _string_or_none(item.get("label")),
                "action_type": _string_or_none(item.get("action_type")),
                "confidence": _to_float(item.get("confidence", 0.0)),
            }
        )
    return buttons


def _normalize_buttons_for_table_state(
    buttons: Sequence[Mapping[str, object]],
    table_state: Mapping[str, object],
) -> list[dict[str, object]]:
    actionable = (
        bool(table_state.get("is_actionable"))
        and _blocking_reason_or_none(table_state.get("blocking_reason")) is None
    )
    return [
        _normalize_actionable_button(button) if actionable else _clear_poker_button_action(button)
        for button in buttons
    ]


def _normalize_actionable_button(button: Mapping[str, object]) -> dict[str, object]:
    normalized = dict(button)
    if not bool(button.get("visible")):
        normalized["action_type"] = None
        return normalized

    name = str(button.get("name") or "")
    if name == "primary_left":
        normalized["action_type"] = _primary_left_action(button)
    elif name == "primary_middle":
        normalized["action_type"] = "raise"
    elif name == "primary_right":
        normalized["action_type"] = "fold"
    elif name.startswith("raise_shortcut"):
        normalized["action_type"] = "raise"
    elif button.get("action_type") not in POKER_ACTIONS:
        normalized["action_type"] = None
    return normalized


def _primary_left_action(button: Mapping[str, object]) -> str | None:
    text = " ".join(
        value.lower()
        for value in (
            _string_or_none(button.get("label")),
            _string_or_none(button.get("action_type")),
        )
        if value
    )
    for action in ("check", "call"):
        if action in text:
            return action
    return None


def _clear_poker_button_action(button: Mapping[str, object]) -> dict[str, object]:
    normalized = dict(button)
    if normalized.get("action_type") in POKER_ACTIONS:
        normalized["action_type"] = None
    return normalized


def _normalize_texts(value: object) -> list[dict[str, object]]:
    texts: list[dict[str, object]] = []
    for item in _mapping_sequence(value):
        texts.append(
            {
                "name": _roi_name(item.get("name")),
                "visible": bool(item.get("visible")),
                "value": _string_or_none(item.get("value")),
                "normalized_number": _optional_int(item.get("normalized_number")),
                "confidence": _to_float(item.get("confidence", 0.0)),
            }
        )
    return texts


def _normalize_seats(value: object) -> list[dict[str, object]]:
    seats: list[dict[str, object]] = []
    for item in _mapping_sequence(value):
        name = str(item.get("name") or "")
        seats.append(
            {
                "name": name,
                "visible": bool(item.get("visible")),
                "stack": _optional_int(item.get("stack")),
                "committed": _optional_int(item.get("committed")),
                "current": _optional_bool(item.get("current")),
                "active": _optional_bool(item.get("active")),
                "confidence": _to_float(item.get("confidence", 0.0)),
            }
        )
    return seats


def _normalize_uncertain(value: object) -> list[dict[str, str]]:
    uncertain: list[dict[str, str]] = []
    for item in _mapping_sequence(value):
        uncertain.append(
            {
                "field": _roi_name(item.get("field")),
                "reason": str(item.get("reason") or ""),
            }
        )
    return uncertain


def _normalize_review_info(decision: Mapping[str, object] | None) -> dict[str, object]:
    if decision is None:
        return {
            "status": "candidate_unreviewed",
            "decision": None,
            "notes": [],
            "overrides": {},
            "ignored_fields": [],
        }
    return {
        "status": "human_reviewed",
        "decision": str(decision.get("decision") or "reviewed"),
        "notes": [str(item) for item in _sequence(decision.get("notes"))],
        "overrides": dict(_mapping_or_empty(decision.get("overrides"))),
        "ignored_fields": [],
    }


def _apply_board_override(
    board: list[dict[str, object]],
    field: str,
    value: object,
    warnings: list[str],
) -> None:
    slot = _roi_name(field)
    card = _card_or_none(value)
    if card is None:
        warnings.append(f"{field} override has invalid card code {value!r}")
        return
    for item in board:
        if item.get("slot") == slot:
            item["visible"] = True
            item["card"] = card
            item["confidence"] = 1.0
            return
    board.append({"slot": slot, "visible": True, "card": card, "confidence": 1.0})


def _apply_ignored_fields(
    table_state: dict[str, object],
    board: list[dict[str, object]],
    hero_hole_cards: list[dict[str, object]],
    texts: list[dict[str, object]],
    seats: list[dict[str, object]],
    ignored_fields: Sequence[str],
) -> tuple[
    dict[str, object],
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
]:
    ignored = {field.lower() for field in ignored_fields}
    if "board" in ignored:
        board = []
    if "hero_hole_cards" in ignored:
        hero_hole_cards = []
    if "texts" in ignored:
        texts = []
    if "seats" in ignored:
        seats = []
    if "pot" in ignored:
        texts = [text for text in texts if str(text.get("name")).lower() != "pot"]
    if "street" in ignored:
        table_state["street"] = None
    if "seats.bb.stack" in ignored:
        for seat in seats:
            if str(seat.get("name")).lower() == "bb":
                seat["stack"] = None
    return table_state, board, hero_hole_cards, texts, seats


def _apply_seat_overrides(
    seats: list[dict[str, object]],
    seat_overrides: Mapping[str, object],
    texts: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    for raw_name, raw_override in seat_overrides.items():
        if not isinstance(raw_name, str) or not isinstance(raw_override, Mapping):
            continue
        override = cast(Mapping[str, object], raw_override)
        seat = _find_or_create_seat(seats, raw_name)
        if "current" in override:
            seat["current"] = _optional_bool(override["current"])
        if "active" in override:
            seat["active"] = _optional_bool(override["active"])
        if "position" in override:
            seat["position"] = str(override["position"])
        if seat.get("stack") is None and raw_name.lower() == "hero":
            stack = _text_number(texts, "hero_stack")
            if stack is not None:
                seat["stack"] = stack
        seat["visible"] = True
        seat["confidence"] = max(_to_float(seat.get("confidence", 0.0)), 1.0)
    return seats


def _find_or_create_seat(seats: list[dict[str, object]], name: str) -> dict[str, object]:
    for seat in seats:
        if str(seat.get("name")).lower() == name.lower():
            return seat
    created: dict[str, object] = {
        "name": name,
        "visible": True,
        "stack": None,
        "committed": None,
        "current": None,
        "active": None,
        "confidence": 1.0,
    }
    seats.append(created)
    return created


def _build_hero_info(
    seats: Sequence[Mapping[str, object]],
    table_state: Mapping[str, object],
    buttons: Sequence[Mapping[str, object]],
    review_info: Mapping[str, object],
) -> dict[str, object]:
    hero_seat = _string_or_none(review_info.get("hero_seat"))
    hero_current = None
    for seat in seats:
        if str(seat.get("name")).lower() == "hero":
            hero_current = _optional_bool(seat.get("current"))
            if hero_seat is None:
                hero_seat = _string_or_none(seat.get("position")) or "hero"
            break
    if hero_current is None and bool(table_state.get("is_actionable")):
        hero_current = _has_visible_poker_action(buttons)
    return {
        "seat": hero_seat,
        "current": hero_current,
    }


def _validate_truth(
    frame_id: str,
    *,
    table_state: Mapping[str, object],
    screen: Mapping[str, object],
    board: Sequence[Mapping[str, object]],
    hero_hole_cards: Sequence[Mapping[str, object]],
    buttons: Sequence[Mapping[str, object]],
    warnings: list[str],
) -> None:
    visible_cards = [
        str(item.get("card"))
        for item in (*board, *hero_hole_cards)
        if bool(item.get("visible")) and item.get("card") is not None
    ]
    duplicates = sorted(card for card, count in Counter(visible_cards).items() if count > 1)
    if duplicates:
        warnings.append(f"{frame_id} has duplicate visible cards: {', '.join(duplicates)}")
    if screen.get("kind") == ScreenKind.ACTIONABLE_TABLE.value:
        if not _has_visible_poker_action(buttons):
            warnings.append(f"{frame_id} is actionable but has no visible poker action button")
        if not any(bool(item.get("visible")) and item.get("card") for item in hero_hole_cards):
            warnings.append(f"{frame_id} is actionable but has no visible hero hole cards")
        if _optional_bool(screen.get("hero_turn")) is not True:
            warnings.append(f"{frame_id} is actionable but hero_turn is not confirmed")
    if (
        bool(table_state.get("is_actionable"))
        and screen.get("kind") != ScreenKind.ACTIONABLE_TABLE.value
    ):
        warnings.append(f"{frame_id} table_state is actionable but screen is {screen.get('kind')}")


def _report_frame_row(truth: Mapping[str, object]) -> dict[str, object]:
    screen = _mapping_or_empty(truth["screen"])
    table_state = _mapping_or_empty(truth["table_state"])
    hero = _mapping_or_empty(truth["hero"])
    validation = _mapping_or_empty(truth["validation"])
    return {
        "frame_id": truth["frame_id"],
        "screen_kind": screen.get("kind"),
        "street": table_state.get("street"),
        "hero": _format_hero(hero),
        "cards": _format_cards(
            [
                *_mapping_sequence(truth.get("hero_hole_cards")),
                *_mapping_sequence(truth.get("board")),
            ]
        ),
        "buttons": _format_buttons(_mapping_sequence(truth.get("buttons"))),
        "ignored": ", ".join(str(item) for item in _sequence(truth.get("ignored_fields"))) or "-",
        "review": _mapping_or_empty(truth.get("review")).get("decision") or "-",
        "warnings": len(_sequence(validation.get("warnings"))),
        "uncertain": len(_sequence(truth.get("uncertain"))),
    }


def _write_truth_report(
    path: Path,
    *,
    summary: Mapping[str, object],
    frames: Sequence[Mapping[str, object]],
) -> None:
    screen_counts = cast(Mapping[str, object], summary["screen_counts"])
    review_counts = cast(Mapping[str, object], summary["review_status_counts"])
    lines = [
        "# Poker Legends Truth Overlay",
        "",
        "## Scope",
        f"- Frames: {summary['frames']}",
        f"- Human-reviewed frames: {summary['human_reviewed_frames']}",
        f"- Warnings: {summary['warning_count']}",
        f"- Candidate uncertain fields retained: {summary['uncertain_count']}",
        "",
        "## Screen Counts",
    ]
    for kind, count in screen_counts.items():
        lines.append(f"- `{kind}`: {count}")
    lines.extend(["", "## Review Status"])
    for status, count in review_counts.items():
        lines.append(f"- `{status}`: {count}")
    lines.extend(
        [
            "",
            "## Frames",
            "| Frame | Screen | Street | Hero | Cards | Buttons | Ignored | Review | "
            "Warnings | Uncertain |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | ---: | ---: |",
        ]
    )
    for frame in frames:
        lines.append(
            f"| `{frame['frame_id']}` | `{frame['screen_kind']}` | "
            f"{frame['street'] or '-'} | {frame['hero']} | {frame['cards']} | "
            f"{frame['buttons']} | {frame['ignored']} | {frame['review']} | "
            f"{frame['warnings']} | {frame['uncertain']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _format_hero(hero: Mapping[str, object]) -> str:
    seat = hero.get("seat") or "-"
    current = hero.get("current")
    if current is None:
        return str(seat)
    return f"{seat} ({'current' if bool(current) else 'not current'})"


def _format_cards(cards: Sequence[Mapping[str, object]]) -> str:
    values = [
        str(card.get("card"))
        for card in cards
        if bool(card.get("visible")) and card.get("card") is not None
    ]
    return " ".join(values) if values else "-"


def _format_buttons(buttons: Sequence[Mapping[str, object]]) -> str:
    values = [
        str(button.get("action_type"))
        for button in buttons
        if bool(button.get("visible")) and button.get("action_type") is not None
    ]
    return " ".join(values) if values else "-"


def _has_visible_poker_action(buttons: Sequence[Mapping[str, object]]) -> bool:
    return any(
        bool(button.get("visible")) and button.get("action_type") in POKER_ACTIONS
        for button in buttons
    )


def _text_number(texts: Sequence[Mapping[str, object]], name: str) -> int | None:
    for text in texts:
        if str(text.get("name")).lower() == name.lower():
            return _optional_int(text.get("normalized_number"))
    return None


def _decisions_by_frame(review: Mapping[str, object]) -> dict[str, Mapping[str, object]]:
    decisions: dict[str, Mapping[str, object]] = {}
    for decision in _mapping_sequence(review.get("decisions")):
        frame_id = _string_or_none(decision.get("frame_id"))
        if frame_id:
            decisions[frame_id] = decision
    return decisions


def _roi_name(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if "." in text:
        return text.rsplit(".", 1)[-1]
    return text


def _card_or_none(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip().upper()
    if len(text) != 2:
        return None
    if text[0] not in CARD_RANKS or text[1] not in CARD_SUITS:
        return None
    return text


def _street_or_none(value: object) -> str | None:
    text = _string_or_none(value)
    if text in {None, "none"}:
        return None
    if text in {"preflop", "flop", "turn", "river", "showdown", "unknown"}:
        return text
    return "unknown"


def _blocking_reason_or_none(value: object) -> str | None:
    text = _string_or_none(value)
    if text in {None, "", "none"}:
        return None
    if text in KNOWN_BLOCKING_REASONS:
        return text
    return "other_overlay"


def _read_json_object(path: str | Path) -> dict[str, object]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object: {path}")
    return cast(dict[str, object], data)


def _mapping_or_empty(value: object) -> Mapping[str, object]:
    if isinstance(value, Mapping):
        return cast(Mapping[str, object], value)
    return {}


def _mapping_sequence(value: object) -> list[Mapping[str, object]]:
    if not isinstance(value, list):
        return []
    return [cast(Mapping[str, object], item) for item in value if isinstance(item, Mapping)]


def _sequence(value: object) -> list[object]:
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return []


def _string_or_none(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    if type(value) is int:
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str) and value.strip():
        return int(value.replace(",", ""))
    return None


def _optional_bool(value: object) -> bool | None:
    if value is None:
        return None
    if type(value) is bool:
        return value
    return None


def _to_float(value: object) -> float:
    if isinstance(value, int | float):
        number = float(value)
    elif isinstance(value, str) and value.strip():
        number = float(value)
    else:
        number = 0.0
    return max(0.0, min(1.0, number))
