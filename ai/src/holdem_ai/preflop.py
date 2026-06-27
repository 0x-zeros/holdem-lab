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
    "BUCKET_PAIR_WEIGHTS",
    "PREFLOP_ALLIN_EQUITY",
    "PREFLOP_BUCKET_COUNT",
    "all_in_equity_vs_random",
    "bucket_deal_conditional",
    "bucket_deal_marginals",
    "bucket_equity",
    "bucket_of",
    "bucket_pair_weights",
    "bucket_weights",
    "compute_bucket_pair_weights",
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


#: ``BUCKET_PAIR_WEIGHTS[i][j]`` = exact joint probability that two hands dealt from
#: one shared 52-card deck (disjoint cards) fall in buckets (i, j). Computed by full
#: enumeration of all 1326*1225 ordered disjoint two-card combo pairs — no Monte
#: Carlo, so it is exactly symmetric and its row sums equal the single-hand bucket
#: marginals. Unlike ``bucket_weights()`` squared, it captures **card removal**: two
#: strong hands co-occur LESS than independence predicts (they compete for the same
#: high cards), while strong-vs-weak co-occurs slightly MORE. Regenerate / audit
#: with :func:`compute_bucket_pair_weights`.
BUCKET_PAIR_WEIGHTS: tuple[tuple[float, ...], ...] = (
    (0.014276, 0.015046, 0.015174, 0.015699, 0.016134, 0.015999, 0.017080, 0.015780),
    (0.015046, 0.015771, 0.015671, 0.016115, 0.016452, 0.016196, 0.017159, 0.015795),
    (0.015174, 0.015671, 0.015009, 0.015396, 0.015770, 0.015502, 0.016309, 0.014849),
    (0.015699, 0.016115, 0.015396, 0.015306, 0.015807, 0.015571, 0.016410, 0.014886),
    (0.016134, 0.016452, 0.015770, 0.015807, 0.015573, 0.015514, 0.016413, 0.015034),
    (0.015999, 0.016196, 0.015502, 0.015571, 0.015514, 0.014822, 0.015647, 0.014430),
    (0.017080, 0.017159, 0.016309, 0.016410, 0.016413, 0.015647, 0.016166, 0.014529),
    (0.015780, 0.015795, 0.014849, 0.014886, 0.015034, 0.014430, 0.014529, 0.012345),
)


def bucket_pair_weights() -> tuple[tuple[float, ...], ...]:
    """Joint P(bucket i, bucket j) for two hands dealt from one shared deck."""
    return BUCKET_PAIR_WEIGHTS


def bucket_deal_marginals() -> tuple[float, ...]:
    """Marginal P(a dealt hand is in each bucket); the joint's normalized row sums."""
    rows = [sum(row) for row in BUCKET_PAIR_WEIGHTS]
    total = sum(rows)
    return tuple(value / total for value in rows)


def bucket_deal_conditional(hero_bucket: int) -> tuple[float, ...]:
    """Card-removal-aware P(villain bucket | hero bucket = ``hero_bucket``); sums to 1."""
    row = BUCKET_PAIR_WEIGHTS[hero_bucket]
    total = sum(row)
    return tuple(value / total for value in row)


def compute_bucket_pair_weights() -> tuple[tuple[float, ...], ...]:
    """Exact regeneration of :data:`BUCKET_PAIR_WEIGHTS` by full enumeration.

    Counts every ordered pair of disjoint two-card combos (1326 * 1225 of them),
    bucketed and normalized — deterministic and exact, no Monte Carlo. Embedded as
    a constant because the double loop is ~1.7M iterations; this reproduces it.
    """
    deck = _full_deck()
    combos = [(i, j) for i in range(len(deck)) for j in range(i + 1, len(deck))]
    masks = [(1 << i) | (1 << j) for i, j in combos]
    buckets = [_BUCKET_MAP[hand_class(deck[i], deck[j])] for i, j in combos]
    counts = [[0] * PREFLOP_BUCKET_COUNT for _ in range(PREFLOP_BUCKET_COUNT)]
    for x in range(len(combos)):
        mask_x = masks[x]
        row = counts[buckets[x]]
        for y in range(len(combos)):
            if not (mask_x & masks[y]):
                row[buckets[y]] += 1
    total = sum(sum(row) for row in counts)
    return tuple(tuple(round(value / total, 6) for value in row) for row in counts)


#: ``BUCKET_EQUITY[i][j]`` = P(bucket i beats bucket j) at an all-in preflop
#: showdown (heads-up, full board), bucket 0 = strongest. Monte Carlo 150000
#: samples/pair, sampled **consistently with the joint deal**: hero uniform in
#: bucket i, villain uniform in bucket j, rejection-resampled until disjoint, then
#: a random board — so the same card-removal universe as BUCKET_PAIR_WEIGHTS. At
#: 150k the per-entry std is ~0.0013 (was ~0.0065 at 6000). Forced exactly
#: symmetric: ``[i][j] + [j][i] == 1`` and a bucket vs itself is exactly 0.5.
BUCKET_EQUITY: tuple[tuple[float, ...], ...] = (
    (0.5000, 0.6530, 0.6695, 0.6860, 0.6957, 0.7009, 0.7131, 0.7233),
    (0.3470, 0.5000, 0.5794, 0.6233, 0.6465, 0.6455, 0.6575, 0.6706),
    (0.3305, 0.4206, 0.5000, 0.5760, 0.6166, 0.6337, 0.6480, 0.6683),
    (0.3140, 0.3767, 0.4240, 0.5000, 0.5767, 0.6194, 0.6406, 0.6570),
    (0.3043, 0.3535, 0.3834, 0.4233, 0.5000, 0.5854, 0.6282, 0.6602),
    (0.2991, 0.3545, 0.3663, 0.3806, 0.4146, 0.5000, 0.5810, 0.6456),
    (0.2869, 0.3425, 0.3520, 0.3594, 0.3718, 0.4190, 0.5000, 0.5994),
    (0.2767, 0.3294, 0.3317, 0.3430, 0.3398, 0.3544, 0.4006, 0.5000),
)


def bucket_equity(hero_bucket: int, villain_bucket: int) -> float:
    """All-in preflop showdown equity of ``hero_bucket`` vs ``villain_bucket``."""
    return BUCKET_EQUITY[hero_bucket][villain_bucket]
