"""Assign a stable per-hand id to a stream of recognised states.

The live recogniser stamps ``hand_id`` from the (constant) temp frame filename, so the opponent
model -- which keys per-seat VPIP/PFR by ``hand_id`` and needs ~12 hands to classify -- never sees
a hand boundary and stays cold: every opponent reads as UNKNOWN, so the whole field-exploit layer
never fires and the bot plays pure ABC with zero adaptation.

A new hand is detected from the hero's hole cards changing. Those are read on every hero-turn
frame, are constant within a hand, and almost always differ between hands -- a more robust signal
than an empty board (which never changes if the hero keeps folding preflop) or the centre street
banner (which flashes and is easily missed when reads are gated to the hero's turn). An occasional
hole-card misread, or the rare back-to-back identical holding, only adds minor noise to rates taken
over a whole session.
"""

from __future__ import annotations

from holdem_common import GameState


def hero_hole_signature(state: GameState) -> tuple[str, ...] | None:
    """The hero's hole cards as an order-independent code tuple, or None if none are face up.

    Only the hero's cards are face up, so the first seat showing two cards is the hero.
    """
    for player in state.players:
        if len(player.hole_cards) >= 2:
            return tuple(sorted(card.code for card in player.hole_cards))
    return None


class HandTracker:
    """Map a stream of states to a stable, incrementing per-hand id (``h1``, ``h2``, ...)."""

    def __init__(self) -> None:
        self._count = 0
        self._signature: tuple[str, ...] | None = None

    def hand_id(self, state: GameState) -> str:
        """The current hand id, advancing it when the hero's hole cards change."""
        signature = hero_hole_signature(state)
        if signature is not None and signature != self._signature:
            self._count += 1
            self._signature = signature
        return f"h{self._count}"
