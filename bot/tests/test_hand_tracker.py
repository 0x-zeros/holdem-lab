"""Tests for the per-hand id tracker (drives opponent-model accumulation in live play)."""

from __future__ import annotations

from holdem_bot.hand_tracker import HandTracker, hero_hole_signature
from holdem_common import Card, GameState, PlayerState, Street


def _state(*hole_codes: str, seat: int = 0) -> GameState:
    hero = PlayerState(
        seat=seat, stack=1000, hole_cards=tuple(Card.from_code(code) for code in hole_codes)
    )
    villain = PlayerState(seat=1, stack=1000)  # opponents are face-down (no hole cards)
    return GameState(
        hand_id="frame",  # the constant stem the live recogniser would stamp
        street=Street.PREFLOP,
        players=(hero, villain),
        board=(),
        pots=(),
        current_seat=seat,
        button_seat=0,
        small_blind=5,
        big_blind=10,
        min_raise=10,
        to_call=10,
    )


def test_signature_finds_hero_cards_order_independent() -> None:
    assert hero_hole_signature(_state("8s", "2d")) == ("2d", "8s")
    assert hero_hole_signature(_state("2d", "8s")) == ("2d", "8s")
    assert hero_hole_signature(_state()) is None


def test_new_id_only_when_hole_cards_change() -> None:
    tracker = HandTracker()
    assert tracker.hand_id(_state("8s", "2d")) == "h1"
    assert tracker.hand_id(_state("8s", "2d")) == "h1"  # same hand, re-read
    assert tracker.hand_id(_state("2d", "8s")) == "h1"  # reordered cards = same hand
    assert tracker.hand_id(_state("As", "Kh")) == "h2"  # new holding = new hand
    assert tracker.hand_id(_state("As", "Kh")) == "h2"


def test_unreadable_cards_keep_current_hand() -> None:
    tracker = HandTracker()
    assert tracker.hand_id(_state("8s", "2d")) == "h1"
    assert tracker.hand_id(_state()) == "h1"  # cards momentarily not visible -> hold the id
    assert tracker.hand_id(_state("8s", "2d")) == "h1"


def test_starts_before_any_hand() -> None:
    assert HandTracker().hand_id(_state()) == "h0"
