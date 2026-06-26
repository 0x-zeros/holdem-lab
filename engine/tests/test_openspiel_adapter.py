from holdem_common import Action, ActionType, Card
from holdem_engine import HoldemConfig, HoldemEnv
from holdem_engine.adapters import (
    OpenSpielAction,
    openspiel_action_to_action,
    openspiel_legal_action_ids,
    registered_poker_games,
    to_openspiel_observation,
)


def cards(*codes: str) -> tuple[Card, ...]:
    return tuple(Card.from_code(code) for code in codes)


def test_openspiel_backend_exposes_universal_poker() -> None:
    assert "universal_poker" in registered_poker_games()


def test_openspiel_observation_encodes_public_and_private_tensors() -> None:
    env = HoldemEnv(
        HoldemConfig(
            starting_stacks=(100, 100, 100),
            deck=cards("As", "Ks", "Qs", "Ah", "Kh", "Qh"),
        ),
    )
    state = env.reset()

    observation = to_openspiel_observation(state, seat=0)

    assert observation.current_player == state.current_seat
    assert len(observation.observation_tensor) == 136
    assert len(observation.information_state_tensor) == 136
    assert observation.observation_tensor[52] == 0.0
    assert observation.information_state_tensor[52] == 1.0
    assert observation.raw_state["viewer_seat"] == 0


def test_openspiel_legal_action_ids_and_decode_call_all_in() -> None:
    env = HoldemEnv(HoldemConfig(starting_stacks=(100, 100, 100)))
    state = env.reset(seed=1)

    legal_ids = openspiel_legal_action_ids(state)

    assert int(OpenSpielAction.FOLD) in legal_ids
    assert int(OpenSpielAction.CHECK_CALL) in legal_ids
    assert int(OpenSpielAction.ALL_IN) in legal_ids
    assert openspiel_action_to_action(OpenSpielAction.CHECK_CALL, state).action_type is (
        ActionType.CALL
    )
    assert openspiel_action_to_action(OpenSpielAction.CHECK_CALL, state).amount == state.to_call
    assert openspiel_action_to_action(OpenSpielAction.ALL_IN, state).action_type is (
        ActionType.ALL_IN
    )


def test_openspiel_raise_action_decodes_to_total_commitment_amount() -> None:
    env = HoldemEnv(HoldemConfig(starting_stacks=(100, 100, 100)))
    state = env.reset(seed=1)

    action = openspiel_action_to_action(OpenSpielAction.MIN_RAISE, state)

    assert action.action_type is ActionType.RAISE
    assert action.amount == action.min_amount
    assert action.amount >= state.min_raise


def test_openspiel_check_call_decodes_to_check_when_no_call_required() -> None:
    env = HoldemEnv(
        HoldemConfig(
            starting_stacks=(100, 100, 100),
            deck=cards("As", "Ks", "Qs", "Ah", "Kh", "Qh", "2c", "7d", "9s", "Jc"),
        ),
    )
    state = env.reset()

    for action_type in (ActionType.CALL, ActionType.CALL, ActionType.CHECK):
        action = next(action for action in state.legal_actions if action.action_type == action_type)
        state = env.step(action).observation

    assert state.to_call == 0
    assert openspiel_action_to_action(OpenSpielAction.CHECK_CALL, state).action_type is (
        ActionType.CHECK
    )


def test_openspiel_terminal_observation_uses_terminal_player_and_rewards() -> None:
    env = HoldemEnv(
        HoldemConfig(
            starting_stacks=(100, 100, 100),
            deck=cards("As", "Ks", "Qs", "Ah", "Kh", "Qh"),
        ),
    )
    state = env.reset()

    while state.current_seat is not None and len(state.active_players) > 1:
        state = env.step(Action(ActionType.FOLD)).observation

    observation = to_openspiel_observation(state)

    assert observation.current_player == -4
    assert observation.legal_actions == ()
    assert sum(observation.rewards) == 0
