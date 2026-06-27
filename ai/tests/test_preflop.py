import pytest
from holdem_ai.preflop import (
    PREFLOP_ALLIN_EQUITY,
    all_in_equity_vs_random,
    hand_class,
    preflop_equity,
    representative_cards,
)
from holdem_common import Card


def card(code: str) -> Card:
    return Card.from_code(code)


def test_hand_class_canonicalizes_pairs_suited_and_offsuit() -> None:
    assert hand_class(card("Ah"), card("Ad")) == "AA"
    assert hand_class(card("Ah"), card("Kh")) == "AKs"
    assert hand_class(card("Ah"), card("Kd")) == "AKo"
    # Order of the two cards must not matter.
    assert hand_class(card("Kh"), card("Ah")) == "AKs"
    assert hand_class(card("2c"), card("7d")) == "72o"


def test_table_covers_all_169_classes_and_is_ranked() -> None:
    assert len(PREFLOP_ALLIN_EQUITY) == 169
    assert PREFLOP_ALLIN_EQUITY["AA"] == max(PREFLOP_ALLIN_EQUITY.values())
    assert PREFLOP_ALLIN_EQUITY["32o"] == min(PREFLOP_ALLIN_EQUITY.values())
    assert all(0.30 < value < 0.90 for value in PREFLOP_ALLIN_EQUITY.values())


def test_representative_cards_round_trip_to_same_class() -> None:
    for label in PREFLOP_ALLIN_EQUITY:
        first, second = representative_cards(label)
        assert hand_class(first, second) == label


def test_preflop_equity_ranks_and_defaults() -> None:
    assert preflop_equity((card("Ah"), card("Ad"))) > preflop_equity((card("7c"), card("2d")))
    assert preflop_equity((card("Ah"),)) == 0.5  # not enough cards


@pytest.mark.parametrize("label", ["AA", "72o"])
def test_table_value_is_reproducible_from_compute(label: str) -> None:
    # Same default seed + sample count, rounded as the table was, reproduces it.
    assert round(all_in_equity_vs_random(label), 4) == PREFLOP_ALLIN_EQUITY[label]


def test_compute_matches_table_within_tolerance_at_low_samples() -> None:
    estimate = all_in_equity_vs_random("AKs", samples=3000, seed=99)
    assert estimate == pytest.approx(PREFLOP_ALLIN_EQUITY["AKs"], abs=0.03)
