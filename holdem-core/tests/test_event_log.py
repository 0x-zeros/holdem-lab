"""Tests for event_log module."""

import json
import tempfile
from pathlib import Path

import pytest

from holdem_lab.cards import Card, Rank, Suit, parse_cards
from holdem_lab.event_log import (
    Event,
    EventLog,
    EventType,
    HandReplayer,
)


class TestEventType:
    """Tests for EventType enum."""

    def test_event_types_exist(self):
        assert EventType.INIT_HAND
        assert EventType.DEAL_HOLE
        assert EventType.DEAL_FLOP
        assert EventType.DEAL_TURN
        assert EventType.DEAL_RIVER
        assert EventType.SHOWDOWN
        assert EventType.PAYOUT
        assert EventType.EQUITY_SNAPSHOT


class TestEvent:
    """Tests for Event class."""

    def test_event_creation(self):
        from datetime import datetime

        event = Event(
            event_type=EventType.DEAL_HOLE,
            timestamp=datetime.now(),
            data={"player": 0, "cards": "Ah Kh"},
        )
        # Just test it doesn't raise
        assert event.event_type == EventType.DEAL_HOLE

    def test_event_to_dict(self):
        from datetime import datetime

        event = Event(
            event_type=EventType.DEAL_HOLE,
            timestamp=datetime(2024, 1, 1, 12, 0, 0),
            sequence=0,
            data={"player": 0, "cards": "Ah Kh"},
        )
        d = event.to_dict()

        assert d["event_type"] == "deal_hole"
        assert d["sequence"] == 0
        assert "timestamp" in d
        assert d["data"]["player"] == 0

    def test_event_from_dict(self):
        d = {
            "event_type": "deal_hole",
            "timestamp": "2024-01-01T12:00:00",
            "sequence": 0,
            "data": {"player": 0, "cards": "Ah Kh"},
        }
        event = Event.from_dict(d)

        assert event.event_type == EventType.DEAL_HOLE
        assert event.sequence == 0

    def test_event_serialize_cards(self):
        from datetime import datetime

        cards = parse_cards("Ah Kh")
        event = Event(
            event_type=EventType.DEAL_FLOP,
            timestamp=datetime.now(),
            data={"cards": cards},
        )
        d = event.to_dict()

        # Cards should be serialized as string
        assert d["data"]["cards"] == "Ah Kh"


class TestEventLog:
    """Tests for EventLog class."""

    def test_log_creation(self):
        log = EventLog()
        assert log.hand_id  # Should have auto-generated ID
        assert len(log) == 0

    def test_log_append(self):
        log = EventLog()
        event = log.append(EventType.INIT_HAND, num_players=2)

        assert len(log) == 1
        assert event.event_type == EventType.INIT_HAND
        assert event.sequence == 0

    def test_log_iteration(self):
        log = EventLog()
        log.append(EventType.INIT_HAND, num_players=2)
        log.append(EventType.DEAL_HOLE, player=0, cards="Ah Kh")

        events = list(log)
        assert len(events) == 2

    def test_log_getitem(self):
        log = EventLog()
        log.append(EventType.INIT_HAND, num_players=2)
        log.append(EventType.DEAL_HOLE, player=0, cards="Ah Kh")

        assert log[0].event_type == EventType.INIT_HAND
        assert log[1].event_type == EventType.DEAL_HOLE

    def test_log_filter_by_type(self):
        log = EventLog()
        log.append(EventType.INIT_HAND, num_players=2)
        log.append(EventType.DEAL_HOLE, player=0, cards="Ah Kh")
        log.append(EventType.DEAL_HOLE, player=1, cards="Qc Qd")

        deal_events = log.filter_by_type(EventType.DEAL_HOLE)
        assert len(deal_events) == 2

    def test_log_to_dict(self):
        log = EventLog(hand_id="test123")
        log.metadata["table"] = "Main"
        log.append(EventType.INIT_HAND, num_players=2)

        d = log.to_dict()
        assert d["hand_id"] == "test123"
        assert d["metadata"]["table"] == "Main"
        assert len(d["events"]) == 1

    def test_log_from_dict(self):
        d = {
            "hand_id": "test123",
            "metadata": {"table": "Main"},
            "events": [
                {
                    "event_type": "init_hand",
                    "timestamp": "2024-01-01T12:00:00",
                    "sequence": 0,
                    "data": {"num_players": 2},
                }
            ],
        }
        log = EventLog.from_dict(d)

        assert log.hand_id == "test123"
        assert log.metadata["table"] == "Main"
        assert len(log) == 1

    def test_log_save_load(self):
        log = EventLog(hand_id="test_save")
        log.append(EventType.INIT_HAND, num_players=2)
        log.append(EventType.DEAL_HOLE, player=0, cards="Ah Kh")

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_hand.json"
            log.save(path)

            loaded = EventLog.load(path)

            assert loaded.hand_id == "test_save"
            assert len(loaded) == 2

    def test_log_json_roundtrip(self):
        log = EventLog(hand_id="test_json")
        log.append(EventType.INIT_HAND, num_players=2)

        json_str = log.to_json()
        loaded = EventLog.from_json(json_str)

        assert loaded.hand_id == "test_json"
        assert len(loaded) == 1


class TestHandReplayer:
    """Tests for HandReplayer class."""

    def create_sample_log(self) -> EventLog:
        """Create a sample hand log for testing."""
        log = EventLog(hand_id="test_replay")
        log.append(EventType.INIT_HAND, num_players=2)
        log.append(EventType.DEAL_HOLE, player=0, cards="Ah Kh")
        log.append(EventType.DEAL_HOLE, player=1, cards="Qc Qd")
        log.append(EventType.DEAL_FLOP, cards="2c 3d 4s")
        log.append(EventType.DEAL_TURN, card="5h")
        log.append(EventType.DEAL_RIVER, card="6c")
        log.append(EventType.SHOWDOWN, winners=[0])
        log.append(EventType.PAYOUT, payouts={"0": 1.0})
        return log

    def test_replayer_creation(self):
        log = self.create_sample_log()
        replayer = HandReplayer(log)

        assert replayer.current_step == 0
        assert replayer.total_steps == 8

    def test_replayer_step_forward(self):
        log = self.create_sample_log()
        replayer = HandReplayer(log)

        # First step: INIT_HAND
        state = replayer.step_forward()
        assert state is not None
        assert state.event.event_type == EventType.INIT_HAND

        # Second step: DEAL_HOLE player 0
        state = replayer.step_forward()
        assert state is not None
        assert state.event.event_type == EventType.DEAL_HOLE
        assert 0 in state.hole_cards

    def test_replayer_full_playback(self):
        log = self.create_sample_log()
        replayer = HandReplayer(log)

        states = replayer.get_all_states()
        assert len(states) == 8

        # Check final state
        final = states[-1]
        assert final.event.event_type == EventType.PAYOUT
        assert len(final.board) == 5
        assert final.winners == [0]

    def test_replayer_goto_step(self):
        log = self.create_sample_log()
        replayer = HandReplayer(log)

        # Jump to flop
        state = replayer.goto_step(3)
        assert state is not None
        assert state.event.event_type == EventType.DEAL_FLOP
        assert len(state.board) == 3

    def test_replayer_reset(self):
        log = self.create_sample_log()
        replayer = HandReplayer(log)

        replayer.step_forward()
        replayer.step_forward()
        assert replayer.current_step == 2

        replayer.reset()
        assert replayer.current_step == 0

    def test_replayer_step_backward(self):
        log = self.create_sample_log()
        replayer = HandReplayer(log)

        # Go forward a few steps
        replayer.step_forward()
        replayer.step_forward()
        replayer.step_forward()
        assert replayer.current_step == 3

        # Go back
        state = replayer.step_backward()
        assert state is not None
        # After step_backward, we're at step 2
        assert replayer.current_step == 2

    def test_replayer_equity_snapshot(self):
        log = EventLog()
        log.append(EventType.INIT_HAND, num_players=2)
        log.append(EventType.DEAL_HOLE, player=0, cards="Ah Kh")
        log.append(EventType.DEAL_HOLE, player=1, cards="Qc Qd")
        log.append(EventType.EQUITY_SNAPSHOT, equities={"0": 0.65, "1": 0.35})

        replayer = HandReplayer(log)
        states = replayer.get_all_states()

        equity_state = states[-1]
        assert equity_state.equities is not None
        assert equity_state.equities[0] == 0.65
        assert equity_state.equities[1] == 0.35

    def test_replayer_at_end(self):
        log = EventLog()
        log.append(EventType.INIT_HAND, num_players=2)

        replayer = HandReplayer(log)
        replayer.step_forward()

        # Should return None at end
        state = replayer.step_forward()
        assert state is None

    def test_replayer_at_start(self):
        log = EventLog()
        log.append(EventType.INIT_HAND, num_players=2)

        replayer = HandReplayer(log)

        # Should return None at start
        state = replayer.step_backward()
        assert state is None
