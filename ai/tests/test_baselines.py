import pytest
from holdem_ai import (
    AggressivePolicy,
    CallStationPolicy,
    Policy,
    RandomPolicy,
    RockPolicy,
    TAGPolicy,
    ThreeBetJammerPolicy,
    profile_from_name,
)
from holdem_ai.profiles import REFERENCE_PROFILE_NAMES
from holdem_common import Action, ActionType, Card, GameState, PlayerState, Pot, Street


def cards(*codes: str) -> tuple[Card, ...]:
    return tuple(Card.from_code(code) for code in codes)


def state_with(
    *,
    legal_actions: tuple[Action, ...],
    to_call: int = 0,
    pot: int = 12,
    committed: int = 0,
    board: tuple[Card, ...] = (),
    current_seat: int | None = 0,
) -> GameState:
    return GameState(
        hand_id="test-hand",
        street=Street.PREFLOP if not board else Street.FLOP,
        players=(
            PlayerState(
                seat=0, stack=100 - committed, committed=committed, hole_cards=cards("As", "Kd")
            ),
            PlayerState(seat=1, stack=100, committed=0, hole_cards=()),
        ),
        board=board,
        pots=(Pot(amount=pot, eligible_seats=frozenset({0, 1})),),
        current_seat=current_seat,
        button_seat=0,
        small_blind=1,
        big_blind=2,
        min_raise=4,
        to_call=to_call,
        legal_actions=legal_actions,
    )


FACING_BET = (
    Action(ActionType.FOLD),
    Action(ActionType.CALL, amount=20),
    Action(ActionType.RAISE, amount=40, min_amount=40, max_amount=100),
    Action(ActionType.ALL_IN, amount=100, min_amount=100, max_amount=100),
)
FREE = (
    Action(ActionType.CHECK),
    Action(ActionType.BET, amount=6, min_amount=6, max_amount=100),
)


def test_reference_profiles_satisfy_policy_protocol() -> None:
    for name in REFERENCE_PROFILE_NAMES:
        profile = profile_from_name(name)
        assert profile.name == name
        assert isinstance(profile.policy, Policy)


def test_random_policy_is_deterministic_and_legal() -> None:
    policy = RandomPolicy(seed=123)
    state = state_with(legal_actions=FACING_BET, to_call=20, pot=40)

    first = policy.decide(state)
    second = policy.decide(state)

    assert first == second
    legal_types = {action.action_type for action in FACING_BET}
    assert first.action_type in legal_types


def test_random_policy_sizes_within_bounds() -> None:
    policy = RandomPolicy(seed=7)
    state = state_with(legal_actions=FREE, pot=40)

    for _ in range(20):
        action = policy.decide(state)
        if action.action_type is ActionType.BET:
            assert 6 <= action.amount <= 100


def test_call_station_checks_when_free_and_calls_a_bet() -> None:
    policy = CallStationPolicy()

    assert policy.decide(state_with(legal_actions=FREE)).action_type is ActionType.CHECK

    facing = policy.decide(state_with(legal_actions=FACING_BET, to_call=20, pot=40))
    assert facing.action_type is ActionType.CALL
    assert facing.amount == 20


def test_rock_checks_when_free_and_folds_to_a_bet() -> None:
    policy = RockPolicy()

    assert policy.decide(state_with(legal_actions=FREE)).action_type is ActionType.CHECK
    assert (
        policy.decide(state_with(legal_actions=FACING_BET, to_call=20, pot=40)).action_type
        is ActionType.FOLD
    )


def test_maniac_bets_and_never_folds() -> None:
    policy = AggressivePolicy()

    free_action = policy.decide(state_with(legal_actions=FREE, pot=40))
    assert free_action.action_type is ActionType.BET
    assert 6 <= free_action.amount <= 100

    facing = policy.decide(state_with(legal_actions=FACING_BET, to_call=20, pot=40, committed=20))
    assert facing.action_type is ActionType.RAISE
    assert facing.amount >= 40

    capped = policy.decide(
        state_with(
            legal_actions=(Action(ActionType.FOLD), Action(ActionType.CALL, amount=20)),
            to_call=20,
            pot=40,
        )
    )
    assert capped.action_type is ActionType.CALL


@pytest.mark.parametrize(
    "policy",
    [
        RandomPolicy(),
        CallStationPolicy(),
        RockPolicy(),
        AggressivePolicy(),
        TAGPolicy(),
        ThreeBetJammerPolicy(),
    ],
)
def test_reference_policies_reject_terminal_state(policy: Policy) -> None:
    state = state_with(legal_actions=(), current_seat=None)
    with pytest.raises(ValueError, match="terminal"):
        policy.decide(state)


def _hu_preflop(
    hole: tuple[Card, ...],
    *,
    current_seat: int,
    hero_committed: int,
    villain_committed: int,
    to_call: int,
    effective: int,
    legal_actions: tuple[Action, ...],
) -> GameState:
    seats = {current_seat: hero_committed, 1 - current_seat: villain_committed}
    players = tuple(
        PlayerState(
            seat=seat,
            stack=effective - seats[seat],
            committed=seats[seat],
            hole_cards=hole if seat == current_seat else (),
        )
        for seat in (0, 1)
    )
    pot = hero_committed + villain_committed
    return GameState(
        hand_id="hu",
        street=Street.PREFLOP,
        players=players,
        board=(),
        pots=(Pot(amount=pot, eligible_seats=frozenset({0, 1})),),
        current_seat=current_seat,
        button_seat=0,
        small_blind=1,
        big_blind=2,
        min_raise=4,
        to_call=to_call,
        legal_actions=legal_actions,
    )


def _open_actions(effective: int) -> tuple[Action, ...]:
    return (
        Action(ActionType.FOLD),
        Action(ActionType.CALL, amount=1),
        Action(ActionType.RAISE, amount=4, min_amount=4, max_amount=effective - 1),
        Action(
            ActionType.ALL_IN,
            amount=effective - 1,
            min_amount=effective - 1,
            max_amount=effective - 1,
        ),
    )


def _facing_raise_actions(effective: int, call_amount: int) -> tuple[Action, ...]:
    return (
        Action(ActionType.FOLD),
        Action(ActionType.CALL, amount=call_amount),
        Action(
            ActionType.RAISE,
            amount=call_amount * 3,
            min_amount=call_amount * 3,
            max_amount=effective - 2,
        ),
        Action(
            ActionType.ALL_IN,
            amount=effective - 2,
            min_amount=effective - 2,
            max_amount=effective - 2,
        ),
    )


def _flop(hole: tuple[Card, ...], board: tuple[Card, ...], *, to_call: int) -> GameState:
    free = to_call == 0
    legal = (
        (Action(ActionType.CHECK), Action(ActionType.BET, amount=8, min_amount=8, max_amount=98))
        if free
        else (
            Action(ActionType.FOLD),
            Action(ActionType.CALL, amount=to_call),
            Action(ActionType.RAISE, amount=to_call * 3, min_amount=to_call * 3, max_amount=98),
        )
    )
    return GameState(
        hand_id="flop",
        street=Street.FLOP,
        players=(
            PlayerState(seat=0, stack=90, committed=2, hole_cards=hole),
            PlayerState(seat=1, stack=90, committed=2 + to_call, hole_cards=()),
        ),
        board=board,
        pots=(Pot(amount=20, eligible_seats=frozenset({0, 1})),),
        current_seat=0,
        button_seat=1,
        small_blind=1,
        big_blind=2,
        min_raise=4,
        to_call=to_call,
        legal_actions=legal,
    )


def test_tag_opens_premium_and_folds_trash_preflop() -> None:
    policy = TAGPolicy()
    premium = policy.explain(
        _hu_preflop(
            cards("As", "Ah"),
            current_seat=0,
            hero_committed=1,
            villain_committed=2,
            to_call=1,
            effective=200,
            legal_actions=_open_actions(200),
        )
    )
    assert premium.action.action_type is ActionType.RAISE
    assert premium.reason == "tag_open"

    trash = policy.explain(
        _hu_preflop(
            cards("7c", "2d"),
            current_seat=0,
            hero_committed=1,
            villain_committed=2,
            to_call=1,
            effective=200,
            legal_actions=_open_actions(200),
        )
    )
    assert trash.action.action_type is ActionType.FOLD
    assert trash.reason == "tag_open_fold"


def test_tag_value_bets_strong_and_folds_air_postflop() -> None:
    policy = TAGPolicy(samples=200)
    # A set on a dry board: clear value, bet when checked to.
    value = policy.explain(_flop(cards("9s", "9h"), cards("9d", "4c", "2s"), to_call=0))
    assert value.action.action_type is ActionType.BET
    assert value.reason == "tag_value_bet"

    # Total air facing a big bet: fold.
    air = policy.explain(_flop(cards("7c", "2d"), cards("As", "Ks", "Qh"), to_call=16))
    assert air.action.action_type is ActionType.FOLD
    assert air.reason == "tag_fold"


def test_three_bet_jammer_jams_premium_over_open_when_short() -> None:
    policy = ThreeBetJammerPolicy()
    jam = policy.explain(
        _hu_preflop(
            cards("As", "Ks"),
            current_seat=1,
            hero_committed=2,
            villain_committed=6,
            to_call=4,
            effective=24,  # 12bb effective -> jam range
            legal_actions=_facing_raise_actions(24, 4),
        )
    )
    assert jam.action.action_type is ActionType.ALL_IN
    assert jam.reason == "jammer_3bet_jam"

    fold = policy.explain(
        _hu_preflop(
            cards("9c", "4d"),
            current_seat=1,
            hero_committed=2,
            villain_committed=6,
            to_call=4,
            effective=24,
            legal_actions=_facing_raise_actions(24, 4),
        )
    )
    assert fold.action.action_type is ActionType.FOLD
    assert fold.reason == "jammer_fold"


def test_three_bet_jammer_does_not_open_jam_when_deep() -> None:
    # 100bb deep: no open-jam; premium 3bets by raising, never shoves 100bb.
    decision = ThreeBetJammerPolicy().explain(
        _hu_preflop(
            cards("As", "Ah"),
            current_seat=1,
            hero_committed=2,
            villain_committed=6,
            to_call=4,
            effective=200,
            legal_actions=_facing_raise_actions(200, 4),
        )
    )
    assert decision.action.action_type is ActionType.RAISE
    assert decision.reason == "jammer_3bet"
