"""Lightweight local hand evaluation and equity estimation."""

from __future__ import annotations

import hashlib
import random
from collections import Counter
from collections.abc import Iterable, Sequence
from itertools import combinations

from holdem_common import Card, GameState, Rank, Suit

_RANK_VALUES = {
    Rank.TWO: 2,
    Rank.THREE: 3,
    Rank.FOUR: 4,
    Rank.FIVE: 5,
    Rank.SIX: 6,
    Rank.SEVEN: 7,
    Rank.EIGHT: 8,
    Rank.NINE: 9,
    Rank.TEN: 10,
    Rank.JACK: 11,
    Rank.QUEEN: 12,
    Rank.KING: 13,
    Rank.ACE: 14,
}

HandRank = tuple[int, tuple[int, ...]]


def evaluate_best_hand(cards: Iterable[Card]) -> HandRank:
    """Return a comparable high-hand rank for 5-7 cards."""

    card_tuple = tuple(cards)
    if len(card_tuple) < 5:
        raise ValueError("at least five cards are required")
    return max(_evaluate_five(combo) for combo in combinations(card_tuple, 5))


def estimate_showdown_equity(state: GameState, *, samples: int = 200) -> float:
    """Estimate current player's share of the pot at showdown.

    Hidden opponent cards and future board cards are sampled deterministically from
    the public ``GameState``. Known visible opponent cards are respected when
    present, which is useful for tests and terminal-like fixtures.
    """

    if samples <= 0:
        raise ValueError("samples must be positive")
    if state.current_seat is None:
        raise ValueError("cannot estimate equity for a terminal state")
    if len(state.board) > 5:
        raise ValueError("board cannot contain more than five cards")

    hero = state.player(state.current_seat)
    if len(hero.hole_cards) < 2:
        raise ValueError("current player must have two visible hole cards")

    opponents = tuple(
        player for player in state.active_players if player.seat != state.current_seat
    )
    if not opponents:
        return 1.0

    fixed_opponent_holes = {
        player.seat: tuple(player.hole_cards[:2])
        for player in opponents
        if len(player.hole_cards) >= 2
    }
    hidden_opponents = tuple(
        player for player in opponents if player.seat not in fixed_opponent_holes
    )
    known_cards = (
        tuple(hero.hole_cards[:2])
        + state.board
        + tuple(card for cards in fixed_opponent_holes.values() for card in cards)
    )
    deck = tuple(card for card in _full_deck() if card not in set(known_cards))
    draw_count = 2 * len(hidden_opponents) + (5 - len(state.board))
    if draw_count > len(deck):
        raise ValueError("not enough unseen cards to sample")

    rng = random.Random(_equity_seed(state, samples=samples))
    equity = 0.0
    for _ in range(samples):
        draw = rng.sample(deck, draw_count)
        cursor = 0
        sampled_holes: dict[int, tuple[Card, Card]] = {}
        for player in hidden_opponents:
            sampled_holes[player.seat] = (draw[cursor], draw[cursor + 1])
            cursor += 2
        runout = tuple(draw[cursor:])
        final_board = state.board + runout
        hero_rank = evaluate_best_hand(tuple(hero.hole_cards[:2]) + final_board)
        ranks = {state.current_seat: hero_rank}
        for player in opponents:
            hole_cards = fixed_opponent_holes.get(player.seat) or sampled_holes[player.seat]
            ranks[player.seat] = evaluate_best_hand(hole_cards + final_board)
        best_rank = max(ranks.values())
        winners = tuple(seat for seat, rank in ranks.items() if rank == best_rank)
        if state.current_seat in winners:
            equity += 1.0 / len(winners)

    return equity / samples


def _evaluate_five(cards: Sequence[Card]) -> HandRank:
    values = sorted((_RANK_VALUES[card.rank] for card in cards), reverse=True)
    counts = Counter(values)
    flush = len({card.suit for card in cards}) == 1
    straight_high = _straight_high(values)

    if flush and straight_high is not None:
        return (8, (straight_high,))

    quads = _ranks_with_count(counts, 4)
    if quads:
        quad = quads[0]
        return (7, (quad, _kickers(values, {quad})[0]))

    trips = _ranks_with_count(counts, 3)
    pairs = _ranks_with_count(counts, 2)
    if trips and (pairs or len(trips) > 1):
        trip = trips[0]
        pair = pairs[0] if pairs else trips[1]
        return (6, (trip, pair))

    if flush:
        return (5, tuple(values))
    if straight_high is not None:
        return (4, (straight_high,))
    if trips:
        trip = trips[0]
        return (3, (trip, *_kickers(values, {trip})[:2]))
    if len(pairs) >= 2:
        high_pair, low_pair = pairs[:2]
        return (2, (high_pair, low_pair, _kickers(values, {high_pair, low_pair})[0]))
    if pairs:
        pair = pairs[0]
        return (1, (pair, *_kickers(values, {pair})[:3]))
    return (0, tuple(values))


def _straight_high(values: Sequence[int]) -> int | None:
    unique = set(values)
    if 14 in unique:
        unique.add(1)
    for high in range(14, 4, -1):
        if set(range(high - 4, high + 1)).issubset(unique):
            return high
    return None


def _ranks_with_count(counts: Counter[int], count: int) -> list[int]:
    return sorted((rank for rank, amount in counts.items() if amount == count), reverse=True)


def _kickers(values: Sequence[int], excluded: set[int]) -> tuple[int, ...]:
    return tuple(value for value in values if value not in excluded)


def _full_deck() -> tuple[Card, ...]:
    return tuple(Card(rank=rank, suit=suit) for rank in Rank for suit in Suit)


def _equity_seed(state: GameState, *, samples: int) -> int:
    assert state.current_seat is not None
    payload = "|".join(
        (
            state.hand_id,
            str(state.current_seat),
            str(samples),
            ",".join(card.code for card in state.board),
            ",".join(card.code for card in state.player(state.current_seat).hole_cards),
            ",".join(str(player.seat) for player in state.active_players),
        )
    )
    return int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:8], "big")
