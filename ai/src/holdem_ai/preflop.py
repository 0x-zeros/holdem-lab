"""Preflop hand-class abstraction and heads-up all-in equity table.

The 169 strategically distinct starting hands (13 pairs, 78 suited, 78 offsuit)
are the standard abstraction used by poker solvers. :data:`PREFLOP_ALLIN_EQUITY`
holds each class's heads-up equity in an all-in *preflop* pot against a uniformly
random hand, precomputed with this project's own evaluator (deterministic Monte
Carlo, no external data). It is a real, auditable replacement for the hand-tuned
preflop strength formula and the foundation for later push/fold and CFR work.

Regenerate / audit a class with :func:`all_in_equity_vs_random` (same seeds), and
look a hole pair up at runtime with :func:`preflop_equity`.
"""

from __future__ import annotations

import hashlib
import random
from collections.abc import Iterable

from holdem_common import Card, Rank, Suit

from holdem_ai.equity import evaluate_best_hand

__all__ = [
    "BUCKET_EQUITY",
    "PREFLOP_ALLIN_EQUITY",
    "PREFLOP_BUCKET_COUNT",
    "all_in_equity_vs_random",
    "bucket_equity",
    "bucket_of",
    "bucket_weights",
    "hand_class",
    "preflop_bucket",
    "preflop_equity",
    "representative_cards",
]

#: Ranks ordered high to low; index 0 (ace) is the strongest.
RANK_ORDER = "AKQJT98765432"
_RANK_INDEX = {char: index for index, char in enumerate(RANK_ORDER)}
_DEFAULT_SAMPLES = 12000


def hand_class(first: Card, second: Card) -> str:
    """Canonical 169-class label, e.g. ``"AA"``, ``"AKs"``, ``"72o"``."""
    first_index = _RANK_INDEX[first.rank.value]
    second_index = _RANK_INDEX[second.rank.value]
    if first_index == second_index:
        return first.rank.value + second.rank.value
    high, low = (first, second) if first_index < second_index else (second, first)
    suffix = "s" if first.suit == second.suit else "o"
    return high.rank.value + low.rank.value + suffix


def representative_cards(label: str) -> tuple[Card, Card]:
    """A concrete two-card representative for a hand-class label."""
    if len(label) == 2:
        return (
            Card(rank=Rank(label[0]), suit=Suit.HEARTS),
            Card(rank=Rank(label[1]), suit=Suit.DIAMONDS),
        )
    high, low, kind = label[0], label[1], label[2]
    second_suit = Suit.HEARTS if kind == "s" else Suit.DIAMONDS
    return (
        Card(rank=Rank(high), suit=Suit.HEARTS),
        Card(rank=Rank(low), suit=second_suit),
    )


def preflop_equity(hole: Iterable[Card]) -> float:
    """Heads-up all-in equity vs a random hand for the given hole cards."""
    cards = tuple(hole)
    if len(cards) < 2:
        return 0.5
    return PREFLOP_ALLIN_EQUITY.get(hand_class(cards[0], cards[1]), 0.5)


def all_in_equity_vs_random(
    label: str, *, samples: int = _DEFAULT_SAMPLES, seed: int | None = None
) -> float:
    """Monte-Carlo heads-up all-in equity for a class vs one random hand.

    Deterministic: with the default per-class seed and ``samples`` this exactly
    reproduces the embedded :data:`PREFLOP_ALLIN_EQUITY` value.
    """
    hole = representative_cards(label)
    if seed is None:
        seed = int.from_bytes(hashlib.sha256(label.encode("utf-8")).digest()[:8], "big")
    rng = random.Random(seed)
    deck = [card for card in _full_deck() if card not in set(hole)]
    base = tuple(hole)
    won = 0.0
    for _ in range(samples):
        draw = rng.sample(deck, 7)
        board = tuple(draw[2:])
        hero = evaluate_best_hand(base + board)
        villain = evaluate_best_hand((draw[0], draw[1]) + board)
        won += 1.0 if hero > villain else (0.5 if hero == villain else 0.0)
    return won / samples


def _full_deck() -> tuple[Card, ...]:
    return tuple(Card(rank=rank, suit=suit) for rank in Rank for suit in Suit)


#: Precomputed heads-up all-in equity vs a random hand (12000 samples / class).
#: Sorted by equity descending. Regenerate with ``all_in_equity_vs_random``.
PREFLOP_ALLIN_EQUITY: dict[str, float] = {
    "AA": 0.8552,
    "KK": 0.8213,
    "QQ": 0.7959,
    "JJ": 0.7735,
    "TT": 0.7457,
    "99": 0.7235,
    "88": 0.6916,
    "AKs": 0.6698,
    "AQs": 0.6579,
    "AJs": 0.6577,
    "77": 0.6541,
    "AQo": 0.6523,
    "AKo": 0.6521,
    "ATs": 0.6485,
    "AJo": 0.6359,
    "66": 0.633,
    "KQs": 0.6328,
    "KJs": 0.6253,
    "A9s": 0.6247,
    "KTs": 0.6217,
    "ATo": 0.6179,
    "KQo": 0.6158,
    "A8s": 0.6143,
    "A7s": 0.6071,
    "A9o": 0.6062,
    "QJs": 0.6053,
    "KJo": 0.6029,
    "A6s": 0.6016,
    "55": 0.5995,
    "A5s": 0.5991,
    "A7o": 0.5971,
    "KTo": 0.5958,
    "A8o": 0.594,
    "QTs": 0.5935,
    "A4s": 0.5922,
    "K9s": 0.5907,
    "A3s": 0.5859,
    "Q9s": 0.585,
    "A6o": 0.5814,
    "JTs": 0.5814,
    "K8s": 0.5809,
    "QJo": 0.5807,
    "A5o": 0.5778,
    "K9o": 0.5757,
    "A4o": 0.5747,
    "A2s": 0.5738,
    "QTo": 0.5725,
    "K7s": 0.5703,
    "44": 0.5689,
    "K6s": 0.5679,
    "Q9o": 0.5619,
    "K5s": 0.5607,
    "A3o": 0.559,
    "K8o": 0.5576,
    "J9s": 0.5566,
    "Q8s": 0.5532,
    "JTo": 0.5526,
    "K7o": 0.5503,
    "K4s": 0.5478,
    "A2o": 0.546,
    "33": 0.5459,
    "J9o": 0.5455,
    "Q7s": 0.5434,
    "K3s": 0.5414,
    "J8s": 0.5412,
    "K6o": 0.54,
    "K5o": 0.538,
    "Q6s": 0.5359,
    "T9s": 0.5359,
    "Q8o": 0.5347,
    "K2s": 0.5282,
    "K3o": 0.5238,
    "Q5s": 0.5232,
    "T8s": 0.5228,
    "J7s": 0.5227,
    "K4o": 0.5203,
    "Q4s": 0.5167,
    "Q6o": 0.5164,
    "Q7o": 0.515,
    "T9o": 0.5116,
    "T7s": 0.5085,
    "J8o": 0.5082,
    "Q5o": 0.5076,
    "Q3s": 0.5053,
    "K2o": 0.504,
    "98s": 0.5032,
    "J6s": 0.5024,
    "T6s": 0.4995,
    "22": 0.498,
    "T8o": 0.4973,
    "Q2s": 0.4972,
    "J5s": 0.4944,
    "Q4o": 0.4921,
    "J7o": 0.492,
    "J4s": 0.4919,
    "96s": 0.4856,
    "97s": 0.4846,
    "Q3o": 0.4826,
    "98o": 0.4826,
    "J6o": 0.4823,
    "T7o": 0.4809,
    "J3s": 0.4804,
    "J5o": 0.4788,
    "87s": 0.4783,
    "J2s": 0.477,
    "T5s": 0.472,
    "Q2o": 0.4674,
    "97o": 0.4651,
    "T4s": 0.4613,
    "95s": 0.4594,
    "86s": 0.4588,
    "J4o": 0.4569,
    "T3s": 0.4534,
    "85s": 0.4519,
    "76s": 0.4509,
    "J3o": 0.4508,
    "T2s": 0.4504,
    "87o": 0.4501,
    "T6o": 0.4496,
    "96o": 0.4477,
    "J2o": 0.4427,
    "T5o": 0.4418,
    "75s": 0.44,
    "94s": 0.435,
    "T4o": 0.4338,
    "93s": 0.4331,
    "65s": 0.4327,
    "84s": 0.4324,
    "95o": 0.4308,
    "86o": 0.4292,
    "92s": 0.429,
    "76o": 0.4224,
    "74s": 0.4218,
    "T3o": 0.4205,
    "T2o": 0.4193,
    "54s": 0.4148,
    "64s": 0.4111,
    "94o": 0.4087,
    "83s": 0.4086,
    "85o": 0.4076,
    "73s": 0.4055,
    "75o": 0.4025,
    "84o": 0.4006,
    "53s": 0.3994,
    "93o": 0.3991,
    "65o": 0.3985,
    "63s": 0.3943,
    "92o": 0.3926,
    "82s": 0.3892,
    "74o": 0.389,
    "43s": 0.3873,
    "62s": 0.3858,
    "72s": 0.3849,
    "64o": 0.3842,
    "83o": 0.3818,
    "54o": 0.381,
    "73o": 0.3743,
    "52s": 0.3734,
    "82o": 0.3728,
    "32s": 0.3694,
    "42s": 0.3678,
    "53o": 0.3609,
    "63o": 0.3569,
    "43o": 0.3501,
    "72o": 0.3426,
    "62o": 0.3419,
    "52o": 0.34,
    "42o": 0.333,
    "32o": 0.3235,
}


#: Number of equity-quantile preflop buckets (a coarse card abstraction for CFR).
PREFLOP_BUCKET_COUNT = 8
_TOTAL_COMBOS = 1326


def _class_combos(label: str) -> int:
    """Number of card combinations a hand class covers (pair 6 / suited 4 / offsuit 12)."""
    if len(label) == 2:
        return 6
    return 4 if label.endswith("s") else 12


def _build_bucket_map(bucket_count: int) -> dict[str, int]:
    # Bucket 0 = strongest. Walk classes from strongest equity to weakest, filling
    # buckets to roughly equal combination mass so each bucket is equally likely.
    ranked = sorted(
        PREFLOP_ALLIN_EQUITY, key=lambda label: PREFLOP_ALLIN_EQUITY[label], reverse=True
    )
    per_bucket = _TOTAL_COMBOS / bucket_count
    mapping: dict[str, int] = {}
    cumulative = 0
    for label in ranked:
        mapping[label] = min(bucket_count - 1, int(cumulative / per_bucket))
        cumulative += _class_combos(label)
    return mapping


_BUCKET_MAP = _build_bucket_map(PREFLOP_BUCKET_COUNT)


def bucket_of(label: str) -> int:
    """Equity-quantile bucket index for a hand class (0 = strongest)."""
    return _BUCKET_MAP[label]


def preflop_bucket(hole: Iterable[Card]) -> int:
    """Equity-quantile bucket for hole cards (0 = strongest, default mid bucket)."""
    cards = tuple(hole)
    if len(cards) < 2:
        return PREFLOP_BUCKET_COUNT // 2
    return _BUCKET_MAP[hand_class(cards[0], cards[1])]


def bucket_weights() -> tuple[float, ...]:
    """Probability mass (by combinations) of each bucket; sums to 1.0."""
    totals = [0] * PREFLOP_BUCKET_COUNT
    for label, bucket in _BUCKET_MAP.items():
        totals[bucket] += _class_combos(label)
    return tuple(total / _TOTAL_COMBOS for total in totals)


#: ``BUCKET_EQUITY[i][j]`` = P(bucket i beats bucket j) at an all-in preflop
#: showdown (heads-up, full board, card removal), Monte Carlo 6000 samples/pair,
#: bucket 0 = strongest. Symmetric: ``[i][j] + [j][i] == 1``; diagonal ~0.5.
BUCKET_EQUITY: tuple[tuple[float, ...], ...] = (
    (0.5046, 0.6617, 0.6721, 0.6931, 0.7037, 0.7037, 0.7082, 0.7240),
    (0.3383, 0.4927, 0.5894, 0.6234, 0.6518, 0.6595, 0.6572, 0.6760),
    (0.3279, 0.4106, 0.5075, 0.5866, 0.6329, 0.6392, 0.6482, 0.6690),
    (0.3069, 0.3766, 0.4134, 0.5052, 0.5807, 0.6259, 0.6522, 0.6681),
    (0.2963, 0.3482, 0.3671, 0.4193, 0.5042, 0.5947, 0.6388, 0.6496),
    (0.2963, 0.3405, 0.3608, 0.3741, 0.4053, 0.4926, 0.5862, 0.6464),
    (0.2918, 0.3428, 0.3518, 0.3478, 0.3612, 0.4138, 0.5038, 0.5972),
    (0.2760, 0.3240, 0.3310, 0.3319, 0.3504, 0.3536, 0.4028, 0.4890),
)


def bucket_equity(hero_bucket: int, villain_bucket: int) -> float:
    """All-in preflop showdown equity of ``hero_bucket`` vs ``villain_bucket``."""
    return BUCKET_EQUITY[hero_bucket][villain_bucket]
