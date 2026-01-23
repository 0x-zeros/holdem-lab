"""Event logging and hand replay for Texas Hold'em."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Iterator

from holdem_lab.cards import Card, format_cards, parse_cards


class EventType(Enum):
    """Types of events in a poker hand."""

    INIT_HAND = "init_hand"
    DEAL_HOLE = "deal_hole"
    DEAL_FLOP = "deal_flop"
    DEAL_TURN = "deal_turn"
    DEAL_RIVER = "deal_river"
    SHOWDOWN = "showdown"
    PAYOUT = "payout"
    EQUITY_SNAPSHOT = "equity_snapshot"


@dataclass
class Event:
    """
    A single event in a poker hand.

    Events are immutable records of state changes during a hand.
    """

    event_type: EventType
    timestamp: datetime
    data: dict[str, Any]
    sequence: int = 0  # Order within the hand

    def to_dict(self) -> dict[str, Any]:
        """Convert event to JSON-serializable dictionary."""
        return {
            "event_type": self.event_type.value,
            "timestamp": self.timestamp.isoformat(),
            "sequence": self.sequence,
            "data": self._serialize_data(self.data),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Event:
        """Create event from dictionary."""
        return cls(
            event_type=EventType(d["event_type"]),
            timestamp=datetime.fromisoformat(d["timestamp"]),
            sequence=d.get("sequence", 0),
            data=cls._deserialize_data(d["data"]),
        )

    @staticmethod
    def _serialize_data(data: dict[str, Any]) -> dict[str, Any]:
        """Serialize data, converting Cards to strings."""
        result: dict[str, Any] = {}
        for key, value in data.items():
            if isinstance(value, list) and value and isinstance(value[0], Card):
                result[key] = format_cards(value)
            elif isinstance(value, Card):
                result[key] = str(value)
            elif isinstance(value, dict):
                result[key] = Event._serialize_data(value)
            else:
                result[key] = value
        return result

    @staticmethod
    def _deserialize_data(data: dict[str, Any]) -> dict[str, Any]:
        """Deserialize data (cards remain as strings for simplicity)."""
        return data


@dataclass
class EventLog:
    """
    Collection of events for a poker hand.

    Provides append, iteration, and serialization capabilities.
    """

    hand_id: str = ""
    events: list[Event] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.hand_id:
            self.hand_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

    def append(self, event_type: EventType, **data: Any) -> Event:
        """
        Create and append a new event.

        Args:
            event_type: Type of the event.
            **data: Event-specific data.

        Returns:
            The created event.
        """
        event = Event(
            event_type=event_type,
            timestamp=datetime.now(),
            sequence=len(self.events),
            data=data,
        )
        self.events.append(event)
        return event

    def __iter__(self) -> Iterator[Event]:
        return iter(self.events)

    def __len__(self) -> int:
        return len(self.events)

    def __getitem__(self, index: int) -> Event:
        return self.events[index]

    def filter_by_type(self, event_type: EventType) -> list[Event]:
        """Get all events of a specific type."""
        return [e for e in self.events if e.event_type == event_type]

    def to_dict(self) -> dict[str, Any]:
        """Convert log to JSON-serializable dictionary."""
        return {
            "hand_id": self.hand_id,
            "metadata": self.metadata,
            "events": [e.to_dict() for e in self.events],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> EventLog:
        """Create log from dictionary."""
        log = cls(
            hand_id=d.get("hand_id", ""),
            metadata=d.get("metadata", {}),
        )
        log.events = [Event.from_dict(e) for e in d.get("events", [])]
        return log

    def save(self, path: str | Path) -> None:
        """
        Save log to JSON file.

        Args:
            path: File path to save to.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> EventLog:
        """
        Load log from JSON file.

        Args:
            path: File path to load from.

        Returns:
            Loaded EventLog.
        """
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    def to_json(self) -> str:
        """Convert log to JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> EventLog:
        """Create log from JSON string."""
        return cls.from_dict(json.loads(json_str))


@dataclass
class HandState:
    """State snapshot during hand replay."""

    step: int
    event: Event
    hole_cards: dict[int, list[Card]]  # player_index -> cards
    board: list[Card]
    equities: dict[int, float] | None  # player_index -> equity, if available
    winners: list[int] | None  # winning player indices, if resolved
    payouts: dict[int, float] | None  # player_index -> payout, if resolved


class HandReplayer:
    """
    Replays a recorded hand step by step.

    Provides a stateful iterator over hand events, reconstructing
    the game state at each step.
    """

    def __init__(self, log: EventLog):
        self.log = log
        self._step = 0
        self._hole_cards: dict[int, list[Card]] = {}
        self._board: list[Card] = []
        self._equities: dict[int, float] | None = None
        self._winners: list[int] | None = None
        self._payouts: dict[int, float] | None = None

    @property
    def current_step(self) -> int:
        """Current step index (0-based)."""
        return self._step

    @property
    def total_steps(self) -> int:
        """Total number of steps in the hand."""
        return len(self.log)

    def reset(self) -> None:
        """Reset replayer to beginning."""
        self._step = 0
        self._hole_cards = {}
        self._board = []
        self._equities = None
        self._winners = None
        self._payouts = None

    def step_forward(self) -> HandState | None:
        """
        Advance one step and return the new state.

        Returns:
            HandState at the new position, or None if at end.
        """
        if self._step >= len(self.log):
            return None

        event = self.log[self._step]
        self._process_event(event)
        state = self._get_current_state(event)
        self._step += 1
        return state

    def step_backward(self) -> HandState | None:
        """
        Go back one step. Requires replaying from start.

        Returns:
            HandState at the new position, or None if at start.
        """
        if self._step <= 1:
            # At step 0 or 1, can't go back further
            return None

        # Replay from beginning to step-2 (one before current)
        target_step = self._step - 2
        self.reset()

        result = None
        for _ in range(target_step + 1):
            result = self.step_forward()

        return result

    def goto_step(self, step: int) -> HandState | None:
        """
        Go to a specific step.

        Args:
            step: Target step index (0-based).

        Returns:
            HandState at the target step, or None if invalid.
        """
        if step < 0 or step >= len(self.log):
            return None

        self.reset()
        for _ in range(step + 1):
            result = self.step_forward()

        return result

    def _process_event(self, event: Event) -> None:
        """Process event and update internal state."""
        data = event.data

        match event.event_type:
            case EventType.INIT_HAND:
                self._hole_cards = {}
                self._board = []
                self._equities = None
                self._winners = None
                self._payouts = None

            case EventType.DEAL_HOLE:
                player = data.get("player", 0)
                cards_str = data.get("cards", "")
                if isinstance(cards_str, str):
                    self._hole_cards[player] = parse_cards(cards_str)
                else:
                    self._hole_cards[player] = cards_str

            case EventType.DEAL_FLOP:
                cards_str = data.get("cards", "")
                if isinstance(cards_str, str):
                    self._board = parse_cards(cards_str)
                else:
                    self._board = list(cards_str)

            case EventType.DEAL_TURN:
                card_str = data.get("card", "")
                if isinstance(card_str, str) and card_str:
                    self._board.extend(parse_cards(card_str))
                elif isinstance(card_str, Card):
                    self._board.append(card_str)

            case EventType.DEAL_RIVER:
                card_str = data.get("card", "")
                if isinstance(card_str, str) and card_str:
                    self._board.extend(parse_cards(card_str))
                elif isinstance(card_str, Card):
                    self._board.append(card_str)

            case EventType.EQUITY_SNAPSHOT:
                equities = data.get("equities", {})
                self._equities = {int(k): v for k, v in equities.items()}

            case EventType.SHOWDOWN:
                self._winners = data.get("winners", [])

            case EventType.PAYOUT:
                payouts = data.get("payouts", {})
                self._payouts = {int(k): v for k, v in payouts.items()}

    def _get_current_state(self, event: Event | None) -> HandState | None:
        """Get current state snapshot."""
        if event is None:
            return None

        return HandState(
            step=self._step,
            event=event,
            hole_cards=dict(self._hole_cards),
            board=list(self._board),
            equities=dict(self._equities) if self._equities else None,
            winners=list(self._winners) if self._winners else None,
            payouts=dict(self._payouts) if self._payouts else None,
        )

    def get_all_states(self) -> list[HandState]:
        """Get all states in the hand."""
        self.reset()
        states = []
        while True:
            state = self.step_forward()
            if state is None:
                break
            states.append(state)
        return states
