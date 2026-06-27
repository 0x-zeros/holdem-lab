import pytest
from holdem_ai import (
    AggressivePolicy,
    CallStationPolicy,
    Policy,
    RandomPolicy,
    RockPolicy,
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
    "policy", [RandomPolicy(), CallStationPolicy(), RockPolicy(), AggressivePolicy()]
)
def test_reference_policies_reject_terminal_state(policy: Policy) -> None:
    state = state_with(legal_actions=(), current_seat=None)
    with pytest.raises(ValueError, match="terminal"):
        policy.decide(state)
