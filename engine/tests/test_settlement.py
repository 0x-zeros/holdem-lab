from holdem_common import Card
from holdem_engine import build_side_pots, settle_showdown


def cards(*codes: str) -> tuple[Card, ...]:
    return tuple(Card.from_code(code) for code in codes)


def hole(first: str, second: str) -> tuple[Card, Card]:
    return (Card.from_code(first), Card.from_code(second))


def test_side_pots_split_by_all_in_contribution_levels() -> None:
    pots = build_side_pots(
        contributions={0: 50, 1: 100, 2: 100},
        live_seats={0, 1, 2},
    )

    assert [(pot.amount, pot.eligible_seats) for pot in pots] == [
        (150, frozenset({0, 1, 2})),
        (100, frozenset({1, 2})),
    ]


def test_showdown_splits_board_only_best_hand() -> None:
    result = settle_showdown(
        hole_cards={
            0: hole("2c", "3d"),
            1: hole("4c", "5d"),
        },
        board=cards("Ah", "Kh", "Qh", "Jh", "Th"),
        contributions={0: 100, 1: 100},
        live_seats={0, 1},
    )

    assert result.payouts == {0: 100, 1: 100}
    assert sum(result.payouts.values()) == 200


def test_showdown_handles_side_pot_and_conserves_chips() -> None:
    result = settle_showdown(
        hole_cards={
            0: hole("As", "Ad"),
            1: hole("Kc", "Kd"),
            2: hole("Qs", "Qd"),
        },
        board=cards("2c", "3d", "4h", "5s", "9c"),
        contributions={0: 50, 1: 100, 2: 100},
        live_seats={0, 1, 2},
    )

    assert result.payouts == {0: 150, 1: 100, 2: 0}
    assert sum(result.payouts.values()) == sum({0: 50, 1: 100, 2: 100}.values())
