"""Side-pot construction and showdown settlement helpers."""

from __future__ import annotations

from collections.abc import Mapping, Set
from dataclasses import dataclass

from holdem_common import Card, Pot
from pokerkit import StandardHighHand
from pokerkit.hands import Hand


@dataclass(frozen=True, slots=True)
class ShowdownResult:
    pots: tuple[Pot, ...]
    payouts: Mapping[int, int]
    hands: Mapping[int, str]


def build_side_pots(
    contributions: Mapping[int, int],
    live_seats: Set[int],
) -> tuple[Pot, ...]:
    if not live_seats:
        raise ValueError("at least one live seat is required")
    if any(amount < 0 for amount in contributions.values()):
        raise ValueError("contributions cannot be negative")

    pots: list[Pot] = []
    previous_level = 0

    for level in sorted({amount for amount in contributions.values() if amount > 0}):
        participants = {seat for seat, amount in contributions.items() if amount >= level}
        amount = (level - previous_level) * len(participants)
        eligible = frozenset(participants & live_seats)
        if amount and eligible:
            if pots and pots[-1].eligible_seats == eligible:
                previous = pots.pop()
                pots.append(Pot(previous.amount + amount, eligible))
            else:
                pots.append(Pot(amount, eligible))
        previous_level = level

    return tuple(pots)


def settle_showdown(
    hole_cards: Mapping[int, tuple[Card, Card]],
    board: tuple[Card, ...],
    contributions: Mapping[int, int],
    live_seats: Set[int],
) -> ShowdownResult:
    pots = build_side_pots(contributions, live_seats)
    evaluated = {seat: _evaluate_holdem_high(hole_cards[seat], board) for seat in live_seats}
    payouts = {seat: 0 for seat in contributions}

    for pot in pots:
        eligible = sorted(pot.eligible_seats & live_seats)
        if not eligible:
            continue
        winners = _best_seats(eligible, evaluated)
        share, odd_chips = divmod(pot.amount, len(winners))
        for seat in winners:
            payouts[seat] += share
        for seat in winners[:odd_chips]:
            payouts[seat] += 1

    return ShowdownResult(
        pots=pots,
        payouts=payouts,
        hands={seat: str(hand) for seat, hand in evaluated.items()},
    )


def _best_seats(seats: list[int], hands: Mapping[int, Hand]) -> list[int]:
    best_hand: Hand | None = None
    winners: list[int] = []

    for seat in seats:
        hand = hands[seat]
        if best_hand is None or best_hand < hand:
            best_hand = hand
            winners = [seat]
        elif hand == best_hand:
            winners.append(seat)

    return winners


def _evaluate_holdem_high(hole_cards: tuple[Card, Card], board: tuple[Card, ...]) -> Hand:
    return StandardHighHand.from_game(
        _cards_to_pokerkit(hole_cards),
        _cards_to_pokerkit(board),
    )


def _cards_to_pokerkit(cards: tuple[Card, ...]) -> str:
    return "".join(card.code for card in cards)
