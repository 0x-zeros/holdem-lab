"""Temporal session tracking for Poker Legends recognition outputs."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Self, cast

from holdem_bot.screen_state import ScreenKind


@dataclass(frozen=True, slots=True)
class PokerLegendsFrameObservation:
    frame_id: str
    timestamp_seconds: float | None
    screen_kind: str
    street: str | None
    blocking_reason: str | None
    hero_turn: bool | None
    hero_cards: tuple[str, ...] = ()
    board_cards: tuple[str, ...] = ()
    action_types: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "frame_id": self.frame_id,
            "timestamp_seconds": self.timestamp_seconds,
            "screen_kind": self.screen_kind,
            "street": self.street,
            "blocking_reason": self.blocking_reason,
            "hero_turn": self.hero_turn,
            "hero_cards": list(self.hero_cards),
            "board_cards": list(self.board_cards),
            "action_types": list(self.action_types),
        }

    @classmethod
    def from_truth(
        cls,
        truth: Mapping[str, object],
        *,
        timestamp_seconds: float | None = None,
    ) -> Self:
        frame_id = str(truth.get("frame_id") or "")
        screen = _mapping_or_empty(truth.get("screen"))
        table_state = _mapping_or_empty(truth.get("table_state"))
        return cls(
            frame_id=frame_id,
            timestamp_seconds=timestamp_seconds,
            screen_kind=str(screen.get("kind") or ScreenKind.UNKNOWN_OR_TRANSITION.value),
            street=_string_or_none(table_state.get("street")),
            blocking_reason=_string_or_none(screen.get("blocking_reason")),
            hero_turn=_optional_bool(screen.get("hero_turn")),
            hero_cards=_visible_cards(truth.get("hero_hole_cards")),
            board_cards=_visible_cards(truth.get("board")),
            action_types=_visible_actions(truth.get("buttons")),
        )


@dataclass(frozen=True, slots=True)
class PokerLegendsTimelineEvent:
    event_type: str
    frame_id: str
    timestamp_seconds: float | None
    hand_index: int | None
    detail: str
    metadata: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class PokerLegendsHandSegment:
    hand_index: int
    start_frame_id: str
    end_frame_id: str
    start_timestamp_seconds: float | None
    end_timestamp_seconds: float | None
    hero_cards: tuple[str, ...]
    final_board_cards: tuple[str, ...]
    streets: tuple[str, ...]
    frame_count: int

    def to_dict(self) -> dict[str, object]:
        return {
            "hand_index": self.hand_index,
            "start_frame_id": self.start_frame_id,
            "end_frame_id": self.end_frame_id,
            "start_timestamp_seconds": self.start_timestamp_seconds,
            "end_timestamp_seconds": self.end_timestamp_seconds,
            "hero_cards": list(self.hero_cards),
            "final_board_cards": list(self.final_board_cards),
            "streets": list(self.streets),
            "frame_count": self.frame_count,
        }


@dataclass(frozen=True, slots=True)
class PokerLegendsSessionTimeline:
    schema_version: int
    frames: tuple[PokerLegendsFrameObservation, ...]
    events: tuple[PokerLegendsTimelineEvent, ...]
    hands: tuple[PokerLegendsHandSegment, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "frames": [frame.to_dict() for frame in self.frames],
            "events": [event.to_dict() for event in self.events],
            "hands": [hand.to_dict() for hand in self.hands],
        }


class PokerLegendsSessionTracker:
    def track(
        self,
        observations: Sequence[PokerLegendsFrameObservation],
    ) -> PokerLegendsSessionTimeline:
        ordered = tuple(
            sorted(
                observations,
                key=lambda item: (
                    float("inf") if item.timestamp_seconds is None else item.timestamp_seconds,
                    item.frame_id,
                ),
            )
        )
        events: list[PokerLegendsTimelineEvent] = []
        hands: list[PokerLegendsHandSegment] = []
        hand_state: _OpenHand | None = None
        previous: PokerLegendsFrameObservation | None = None
        next_hand_index = 0

        for observation in ordered:
            should_start_new = _should_start_new_hand(previous, observation, hand_state)
            if should_start_new and hand_state is not None:
                hands.append(hand_state.close(previous or observation))
                events.append(
                    _event(
                        "hand_ended",
                        previous or observation,
                        hand_state.hand_index,
                        "next sampled frame indicated new hand boundary",
                        boundary_frame_id=observation.frame_id,
                    )
                )
                hand_state = None

            if hand_state is None and _has_hand_signal(observation):
                hand_state = _OpenHand.open(next_hand_index, observation)
                next_hand_index += 1
                events.append(
                    _event(
                        "hand_started",
                        observation,
                        hand_state.hand_index,
                        "hand signal appeared",
                        hero_cards=list(observation.hero_cards),
                    )
                )

            if hand_state is not None:
                events.extend(_state_change_events(previous, observation, hand_state.hand_index))
                hand_state = hand_state.update(observation)
            else:
                events.extend(_screen_change_events(previous, observation, hand_index=None))

            previous = observation

        if hand_state is not None and previous is not None:
            hands.append(hand_state.close(previous))

        return PokerLegendsSessionTimeline(
            schema_version=1,
            frames=ordered,
            events=tuple(events),
            hands=tuple(hands),
        )


def build_poker_legends_session_timeline(
    truth_paths: Sequence[str | Path],
    *,
    output_dir: str | Path,
    selection_manifest_path: str | Path | None = None,
) -> dict[str, object]:
    timestamps = (
        _timestamps_by_frame(selection_manifest_path) if selection_manifest_path is not None else {}
    )
    observations = [
        PokerLegendsFrameObservation.from_truth(
            _read_json_object(path),
            timestamp_seconds=timestamps.get(Path(path).stem),
        )
        for path in truth_paths
    ]
    timeline = PokerLegendsSessionTracker().track(observations)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "session_timeline.json").write_text(
        json.dumps(timeline.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_timeline_report(output / "session_timeline.md", timeline)
    summary = {
        "schema_version": 1,
        "frames": len(timeline.frames),
        "events": len(timeline.events),
        "hands": len(timeline.hands),
        "report": "session_timeline.md",
        "timeline": "session_timeline.json",
    }
    (output / "session_timeline_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a temporal Poker Legends session timeline from truth overlays."
    )
    parser.add_argument("truth_overlays", nargs="+", help="Truth overlay JSON files.")
    parser.add_argument("--out", required=True, help="Output directory for timeline artifacts.")
    parser.add_argument(
        "--selection-manifest",
        help="Optional selected_manifest.json with frame timestamps.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    summary = build_poker_legends_session_timeline(
        args.truth_overlays,
        output_dir=args.out,
        selection_manifest_path=args.selection_manifest,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


@dataclass(frozen=True, slots=True)
class _OpenHand:
    hand_index: int
    start: PokerLegendsFrameObservation
    latest: PokerLegendsFrameObservation
    hero_cards: tuple[str, ...]
    board_cards: tuple[str, ...]
    streets: tuple[str, ...]
    frame_count: int

    @classmethod
    def open(cls, hand_index: int, observation: PokerLegendsFrameObservation) -> _OpenHand:
        street = _street_tuple((), observation.street)
        return cls(
            hand_index=hand_index,
            start=observation,
            latest=observation,
            hero_cards=observation.hero_cards,
            board_cards=observation.board_cards,
            streets=street,
            frame_count=0,
        )

    def update(self, observation: PokerLegendsFrameObservation) -> _OpenHand:
        update_hero_cards = observation.screen_kind != ScreenKind.BLOCKED_OVERLAY.value and bool(
            observation.hero_cards
        )
        return _OpenHand(
            hand_index=self.hand_index,
            start=self.start,
            latest=observation,
            hero_cards=observation.hero_cards if update_hero_cards else self.hero_cards,
            board_cards=observation.board_cards or self.board_cards,
            streets=_street_tuple(self.streets, observation.street),
            frame_count=self.frame_count + 1,
        )

    def close(self, end: PokerLegendsFrameObservation) -> PokerLegendsHandSegment:
        return PokerLegendsHandSegment(
            hand_index=self.hand_index,
            start_frame_id=self.start.frame_id,
            end_frame_id=end.frame_id,
            start_timestamp_seconds=self.start.timestamp_seconds,
            end_timestamp_seconds=end.timestamp_seconds,
            hero_cards=self.hero_cards,
            final_board_cards=self.board_cards,
            streets=self.streets,
            frame_count=self.frame_count,
        )


def _state_change_events(
    previous: PokerLegendsFrameObservation | None,
    current: PokerLegendsFrameObservation,
    hand_index: int,
) -> list[PokerLegendsTimelineEvent]:
    events = _screen_change_events(previous, current, hand_index=hand_index)
    if previous is None:
        if current.street is not None and current.street != "unknown":
            events.append(_event("street_changed", current, hand_index, f"street={current.street}"))
        if current.board_cards:
            events.append(
                _event(
                    "board_changed",
                    current,
                    hand_index,
                    "board visible",
                    board_cards=list(current.board_cards),
                )
            )
        return events

    if (
        current.street != previous.street
        and current.street is not None
        and current.street != "unknown"
    ):
        events.append(_event("street_changed", current, hand_index, f"street={current.street}"))
    if current.board_cards != previous.board_cards and current.board_cards:
        events.append(
            _event(
                "board_changed",
                current,
                hand_index,
                "board cards changed",
                board_cards=list(current.board_cards),
            )
        )
    if (
        current.screen_kind != ScreenKind.BLOCKED_OVERLAY.value
        and current.hero_cards != previous.hero_cards
        and current.hero_cards
    ):
        events.append(
            _event(
                "hero_cards_changed",
                current,
                hand_index,
                "hero cards changed",
                hero_cards=list(current.hero_cards),
            )
        )
    return events


def _screen_change_events(
    previous: PokerLegendsFrameObservation | None,
    current: PokerLegendsFrameObservation,
    *,
    hand_index: int | None,
) -> list[PokerLegendsTimelineEvent]:
    if previous is not None and current.screen_kind == previous.screen_kind:
        return []
    if current.screen_kind == ScreenKind.BLOCKED_OVERLAY.value:
        return [
            _event(
                "blocked_started",
                current,
                hand_index,
                current.blocking_reason or "blocked overlay",
            )
        ]
    if previous is not None and previous.screen_kind == ScreenKind.BLOCKED_OVERLAY.value:
        return [_event("blocked_ended", current, hand_index, "blocked overlay ended")]
    return [
        _event(
            "screen_changed",
            current,
            hand_index,
            f"screen={current.screen_kind}",
        )
    ]


def _should_start_new_hand(
    previous: PokerLegendsFrameObservation | None,
    current: PokerLegendsFrameObservation,
    hand_state: _OpenHand | None,
) -> bool:
    if previous is None or hand_state is None or not _has_hand_signal(current):
        return False
    if (
        current.screen_kind == ScreenKind.ACTIONABLE_TABLE.value
        and hand_state.hero_cards
        and current.hero_cards
        and current.hero_cards != hand_state.hero_cards
    ):
        return True
    if previous.board_cards and not current.board_cards and current.street == "preflop":
        return True
    if len(current.board_cards) < len(previous.board_cards) and current.street == "preflop":
        return True
    if previous.street in {"showdown", "river"} and current.street == "preflop":
        return True
    return False


def _has_hand_signal(observation: PokerLegendsFrameObservation) -> bool:
    return bool(
        observation.hero_cards
        or observation.board_cards
        or (observation.street is not None and observation.street != "unknown")
    )


def _event(
    event_type: str,
    observation: PokerLegendsFrameObservation,
    hand_index: int | None,
    detail: str,
    **metadata: object,
) -> PokerLegendsTimelineEvent:
    return PokerLegendsTimelineEvent(
        event_type=event_type,
        frame_id=observation.frame_id,
        timestamp_seconds=observation.timestamp_seconds,
        hand_index=hand_index,
        detail=detail,
        metadata=dict(metadata),
    )


def _street_tuple(streets: tuple[str, ...], street: str | None) -> tuple[str, ...]:
    if street is None or street == "unknown" or street in streets:
        return streets
    return (*streets, street)


def _visible_cards(value: object) -> tuple[str, ...]:
    cards: list[str] = []
    for item in _mapping_sequence(value):
        card = _string_or_none(item.get("card"))
        if bool(item.get("visible")) and card is not None:
            cards.append(card)
    return tuple(cards)


def _visible_actions(value: object) -> tuple[str, ...]:
    actions: list[str] = []
    for item in _mapping_sequence(value):
        action = _string_or_none(item.get("action_type"))
        if bool(item.get("visible")) and action is not None:
            actions.append(action)
    return tuple(actions)


def _timestamps_by_frame(path: str | Path) -> dict[str, float]:
    data = _read_json_object(path)
    timestamps: dict[str, float] = {}
    for frame in _mapping_sequence(data.get("frames")):
        frame_id = _string_or_none(frame.get("frame_id"))
        timestamp = frame.get("timestamp_seconds")
        if frame_id is not None and isinstance(timestamp, int | float):
            timestamps[frame_id] = float(timestamp)
    return timestamps


def _write_timeline_report(path: Path, timeline: PokerLegendsSessionTimeline) -> None:
    lines = [
        "# Poker Legends Session Timeline",
        "",
        "## Summary",
        f"- Frames: {len(timeline.frames)}",
        f"- Events: {len(timeline.events)}",
        f"- Hands: {len(timeline.hands)}",
        "",
        "## Hands",
        "| Hand | Start | End | Hero | Board | Streets | Frames |",
        "| ---: | --- | --- | --- | --- | --- | ---: |",
    ]
    for hand in timeline.hands:
        lines.append(
            f"| {hand.hand_index} | `{hand.start_frame_id}` | `{hand.end_frame_id}` | "
            f"{_format_cards(hand.hero_cards)} | {_format_cards(hand.final_board_cards)} | "
            f"{', '.join(hand.streets) or '-'} | {hand.frame_count} |"
        )
    lines.extend(
        [
            "",
            "## Events",
            "| Frame | Time | Hand | Event | Detail |",
            "| --- | ---: | ---: | --- | --- |",
        ]
    )
    for event in timeline.events:
        time_value = "-" if event.timestamp_seconds is None else f"{event.timestamp_seconds:.1f}"
        hand_value = "-" if event.hand_index is None else str(event.hand_index)
        lines.append(
            f"| `{event.frame_id}` | {time_value} | {hand_value} | "
            f"`{event.event_type}` | {event.detail} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _format_cards(cards: Sequence[str]) -> str:
    return " ".join(cards) if cards else "-"


def _read_json_object(path: str | Path) -> dict[str, object]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object: {path}")
    return cast(dict[str, object], data)


def _mapping_or_empty(value: object) -> Mapping[str, object]:
    return cast(Mapping[str, object], value) if isinstance(value, Mapping) else {}


def _mapping_sequence(value: object) -> list[Mapping[str, object]]:
    if not isinstance(value, list):
        return []
    return [cast(Mapping[str, object], item) for item in value if isinstance(item, Mapping)]


def _string_or_none(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        if not text or text.lower() == "none":
            return None
        return text
    return str(value)


def _optional_bool(value: object) -> bool | None:
    if type(value) is bool:
        return value
    return None
