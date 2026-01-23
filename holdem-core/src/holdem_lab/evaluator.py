"""Hand evaluation for Texas Hold'em (5-card and 7-card hands)."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from enum import IntEnum
from itertools import combinations
from typing import Sequence

from holdem_lab.cards import Card, Rank


class HandType(IntEnum):
    """
    Poker hand types ranked from lowest to highest.

    Higher value = stronger hand.
    """

    HIGH_CARD = 0
    ONE_PAIR = 1
    TWO_PAIR = 2
    THREE_OF_A_KIND = 3
    STRAIGHT = 4
    FLUSH = 5
    FULL_HOUSE = 6
    FOUR_OF_A_KIND = 7
    STRAIGHT_FLUSH = 8
    ROYAL_FLUSH = 9

    def __str__(self) -> str:
        return self.name.replace("_", " ").title()


@dataclass(frozen=True, slots=True, order=True)
class HandRank:
    """
    Comparable hand ranking.

    Hands are compared by:
    1. hand_type (higher is better)
    2. primary_ranks (e.g., pair rank, trips rank)
    3. kickers (remaining cards for tiebreakers)

    The comparison is lexicographic: hand_type first, then primary_ranks, then kickers.
    """

    hand_type: HandType
    primary_ranks: tuple[int, ...]
    kickers: tuple[int, ...]

    def __str__(self) -> str:
        return f"{self.hand_type}"

    def describe(self) -> str:
        """Return a detailed description of the hand."""
        rank_names = {
            14: "Ace",
            13: "King",
            12: "Queen",
            11: "Jack",
            10: "Ten",
            9: "Nine",
            8: "Eight",
            7: "Seven",
            6: "Six",
            5: "Five",
            4: "Four",
            3: "Three",
            2: "Two",
        }

        def rank_str(r: int) -> str:
            return rank_names.get(r, str(r))

        match self.hand_type:
            case HandType.ROYAL_FLUSH:
                return "Royal Flush"
            case HandType.STRAIGHT_FLUSH:
                return f"Straight Flush, {rank_str(self.primary_ranks[0])} high"
            case HandType.FOUR_OF_A_KIND:
                return f"Four of a Kind, {rank_str(self.primary_ranks[0])}s"
            case HandType.FULL_HOUSE:
                return f"Full House, {rank_str(self.primary_ranks[0])}s full of {rank_str(self.primary_ranks[1])}s"
            case HandType.FLUSH:
                return f"Flush, {rank_str(self.primary_ranks[0])} high"
            case HandType.STRAIGHT:
                return f"Straight, {rank_str(self.primary_ranks[0])} high"
            case HandType.THREE_OF_A_KIND:
                return f"Three of a Kind, {rank_str(self.primary_ranks[0])}s"
            case HandType.TWO_PAIR:
                return f"Two Pair, {rank_str(self.primary_ranks[0])}s and {rank_str(self.primary_ranks[1])}s"
            case HandType.ONE_PAIR:
                return f"Pair of {rank_str(self.primary_ranks[0])}s"
            case HandType.HIGH_CARD:
                return f"High Card, {rank_str(self.primary_ranks[0])}"
            case _:
                return str(self.hand_type)


def _get_ranks(cards: Sequence[Card]) -> list[int]:
    """Extract rank values from cards, sorted descending."""
    return sorted([c.rank.value for c in cards], reverse=True)


def _is_flush(cards: Sequence[Card]) -> bool:
    """Check if all 5 cards are the same suit."""
    suits = [c.suit for c in cards]
    return len(set(suits)) == 1


def _get_straight_high(ranks: list[int]) -> int | None:
    """
    Check if ranks form a straight and return the high card.

    Returns None if not a straight.
    Handles wheel (A-2-3-4-5) where high card is 5.
    """
    unique = sorted(set(ranks), reverse=True)
    if len(unique) != 5:
        return None

    # Normal straight check
    if unique[0] - unique[4] == 4:
        return unique[0]

    # Wheel: A-5-4-3-2
    if unique == [14, 5, 4, 3, 2]:
        return 5  # High card is 5 in wheel

    return None


def evaluate_five(cards: Sequence[Card]) -> HandRank:
    """
    Evaluate exactly 5 cards and return their hand rank.

    Args:
        cards: Exactly 5 cards to evaluate.

    Returns:
        HandRank representing the hand strength.

    Raises:
        ValueError: If not exactly 5 cards provided.
    """
    if len(cards) != 5:
        raise ValueError(f"Must provide exactly 5 cards, got {len(cards)}")

    ranks = _get_ranks(cards)
    is_flush = _is_flush(cards)
    straight_high = _get_straight_high(ranks)
    is_straight = straight_high is not None

    # Count rank occurrences
    rank_counts = Counter(ranks)
    counts = sorted(rank_counts.values(), reverse=True)

    # Get ranks grouped by count (for determining primary ranks and kickers)
    # Sort by (count descending, rank descending) to get pairs/trips before kickers
    ranks_by_count = sorted(rank_counts.items(), key=lambda x: (x[1], x[0]), reverse=True)

    # Straight flush / Royal flush
    if is_flush and is_straight:
        if straight_high == 14:  # A-high straight flush
            return HandRank(HandType.ROYAL_FLUSH, (14,), ())
        return HandRank(HandType.STRAIGHT_FLUSH, (straight_high,), ())

    # Four of a kind
    if counts == [4, 1]:
        quad_rank = ranks_by_count[0][0]
        kicker = ranks_by_count[1][0]
        return HandRank(HandType.FOUR_OF_A_KIND, (quad_rank,), (kicker,))

    # Full house
    if counts == [3, 2]:
        trips_rank = ranks_by_count[0][0]
        pair_rank = ranks_by_count[1][0]
        return HandRank(HandType.FULL_HOUSE, (trips_rank, pair_rank), ())

    # Flush
    if is_flush:
        return HandRank(HandType.FLUSH, tuple(ranks), ())

    # Straight
    if is_straight:
        return HandRank(HandType.STRAIGHT, (straight_high,), ())

    # Three of a kind
    if counts == [3, 1, 1]:
        trips_rank = ranks_by_count[0][0]
        kickers = tuple(sorted([ranks_by_count[1][0], ranks_by_count[2][0]], reverse=True))
        return HandRank(HandType.THREE_OF_A_KIND, (trips_rank,), kickers)

    # Two pair
    if counts == [2, 2, 1]:
        pair_ranks = sorted([ranks_by_count[0][0], ranks_by_count[1][0]], reverse=True)
        kicker = ranks_by_count[2][0]
        return HandRank(HandType.TWO_PAIR, tuple(pair_ranks), (kicker,))

    # One pair
    if counts == [2, 1, 1, 1]:
        pair_rank = ranks_by_count[0][0]
        kickers = tuple(
            sorted([ranks_by_count[1][0], ranks_by_count[2][0], ranks_by_count[3][0]], reverse=True)
        )
        return HandRank(HandType.ONE_PAIR, (pair_rank,), kickers)

    # High card
    return HandRank(HandType.HIGH_CARD, (ranks[0],), tuple(ranks[1:]))


def evaluate_hand(cards: Sequence[Card]) -> HandRank:
    """
    Evaluate 5-7 cards and return the best possible 5-card hand rank.

    For 7 cards (hole cards + board), evaluates all C(7,5)=21 combinations.

    Args:
        cards: 5-7 cards to evaluate.

    Returns:
        HandRank of the best 5-card hand.

    Raises:
        ValueError: If fewer than 5 or more than 7 cards provided.
    """
    if len(cards) < 5:
        raise ValueError(f"Need at least 5 cards, got {len(cards)}")
    if len(cards) > 7:
        raise ValueError(f"Maximum 7 cards allowed, got {len(cards)}")

    if len(cards) == 5:
        return evaluate_five(cards)

    # Evaluate all 5-card combinations and return the best
    best_rank: HandRank | None = None
    for combo in combinations(cards, 5):
        rank = evaluate_five(combo)
        if best_rank is None or rank > best_rank:
            best_rank = rank

    assert best_rank is not None
    return best_rank


def find_winners(hands: list[Sequence[Card]]) -> list[int]:
    """
    Find the winner(s) among multiple hands.

    Args:
        hands: List of hands, each containing 5-7 cards.

    Returns:
        List of indices of winning players (multiple for ties).

    Raises:
        ValueError: If no hands provided.
    """
    if not hands:
        raise ValueError("Must provide at least one hand")

    # Evaluate all hands
    ranks = [evaluate_hand(hand) for hand in hands]

    # Find the best rank
    best_rank = max(ranks)

    # Return all players with the best rank (handles ties)
    return [i for i, rank in enumerate(ranks) if rank == best_rank]


def compare_hands(hand1: Sequence[Card], hand2: Sequence[Card]) -> int:
    """
    Compare two hands.

    Args:
        hand1: First hand (5-7 cards).
        hand2: Second hand (5-7 cards).

    Returns:
        1 if hand1 wins, -1 if hand2 wins, 0 if tie.
    """
    rank1 = evaluate_hand(hand1)
    rank2 = evaluate_hand(hand2)

    if rank1 > rank2:
        return 1
    elif rank1 < rank2:
        return -1
    return 0
