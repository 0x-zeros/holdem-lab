from holdem_common import Action, ActionType
from holdem_engine import HoldemConfig, HoldemEnv


def test_env_reset_exposes_game_state_and_legal_actions() -> None:
    env = HoldemEnv(HoldemConfig(starting_stacks=(100, 100, 100)))

    state = env.reset(seed=1)

    assert state.current_seat is not None
    assert state.pot_total == 3
    assert {action.action_type for action in state.legal_actions} >= {
        ActionType.FOLD,
        ActionType.CALL,
    }


def test_env_step_returns_zero_rewards_before_terminal_state() -> None:
    env = HoldemEnv(HoldemConfig(starting_stacks=(100, 100, 100)))
    state = env.reset(seed=1)
    call = next(action for action in state.legal_actions if action.action_type == ActionType.CALL)

    result = env.step(Action(ActionType.CALL, amount=call.amount))

    assert result.observation.hand_id == "hand-1"
    if not result.terminated:
        assert set(result.rewards.values()) == {0}
