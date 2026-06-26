from holdem_common import Action, ActionType, Card, Street
from holdem_engine import HoldemConfig, HoldemEnv


def cards(*codes: str) -> tuple[Card, ...]:
    return tuple(Card.from_code(code) for code in codes)


def test_env_reset_exposes_game_state_and_legal_actions() -> None:
    env = HoldemEnv(HoldemConfig(starting_stacks=(100, 100, 100)))

    state = env.reset(seed=1)

    assert state.current_seat is not None
    assert state.pot_total == 3
    assert {action.action_type for action in state.legal_actions} >= {
        ActionType.FOLD,
        ActionType.CALL,
    }


def test_blinds_are_posted_and_button_roles_are_exposed() -> None:
    env = HoldemEnv(HoldemConfig(starting_stacks=(100, 100, 100)))

    state = env.reset(seed=1)

    assert state.player(0).dealer
    assert state.player(0).small_blind
    assert state.player(0).committed == 1
    assert state.player(1).big_blind
    assert state.player(1).committed == 2
    assert state.player(2).committed == 0
    assert state.pot_total == 3


def test_reset_can_use_fixed_deck_before_hole_dealing() -> None:
    env = HoldemEnv(
        HoldemConfig(
            starting_stacks=(100, 100, 100),
            deck=cards("As", "Ks", "Qs", "Ah", "Kh", "Qh", "2c", "3c", "4c"),
        ),
    )

    state = env.reset()

    assert state.player(0).hole_cards == cards("As", "Ah")
    assert state.player(1).hole_cards == cards("Ks", "Kh")
    assert state.player(2).hole_cards == cards("Qs", "Qh")


def test_observe_for_seat_hides_other_players_hole_cards() -> None:
    env = HoldemEnv(
        HoldemConfig(
            starting_stacks=(100, 100, 100),
            deck=cards("As", "Ks", "Qs", "Ah", "Kh", "Qh"),
        ),
    )
    env.reset()

    observation = env.observe(seat=0)

    assert observation.player(0).hole_cards == cards("As", "Ah")
    assert observation.player(1).hole_cards == ()
    assert observation.player(2).hole_cards == ()


def test_calling_round_advances_to_flop() -> None:
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

    assert state.street is Street.FLOP
    assert state.board == cards("7d", "9s", "Jc")
    assert state.pot_total == 6
    assert all(player.committed == 2 for player in state.players)


def test_env_step_returns_zero_rewards_before_terminal_state() -> None:
    env = HoldemEnv(HoldemConfig(starting_stacks=(100, 100, 100)))
    state = env.reset(seed=1)
    call = next(action for action in state.legal_actions if action.action_type == ActionType.CALL)

    result = env.step(Action(ActionType.CALL, amount=call.amount))

    assert result.observation.hand_id == "hand-1"
    if not result.terminated:
        assert set(result.rewards.values()) == {0}


def test_raise_amount_is_total_commitment_not_increment() -> None:
    env = HoldemEnv(
        HoldemConfig(
            starting_stacks=(100, 100, 100),
            deck=cards("As", "Ks", "Qs", "Ah", "Kh", "Qh"),
        ),
    )
    state = env.reset()
    raise_action = next(
        action for action in state.legal_actions if action.action_type == ActionType.RAISE
    )
    assert state.current_seat is not None

    result = env.step(Action(ActionType.RAISE, amount=raise_action.min_amount or 0))

    raiser = result.observation.player(state.current_seat)
    assert raiser.committed == raise_action.min_amount


def test_check_is_rejected_when_call_is_required() -> None:
    env = HoldemEnv(
        HoldemConfig(
            starting_stacks=(100, 100, 100),
            deck=cards("As", "Ks", "Qs", "Ah", "Kh", "Qh"),
        ),
    )
    env.reset()

    try:
        env.step(Action(ActionType.CHECK))
    except ValueError as exc:
        assert "use CALL" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("CHECK should not be accepted when chips are required")


def test_folded_hand_terminates_with_chip_conservation() -> None:
    env = HoldemEnv(
        HoldemConfig(
            starting_stacks=(100, 100, 100),
            deck=cards("As", "Ks", "Qs", "Ah", "Kh", "Qh"),
        ),
    )
    state = env.reset()

    while state.current_seat is not None and len(state.active_players) > 1:
        result = env.step(Action(ActionType.FOLD))
        state = result.observation

    assert result.terminated
    assert sum(result.rewards.values()) == 0
    assert sum(player.stack for player in state.players) == 300


def test_heads_up_all_in_runs_board_and_preserves_total_chips() -> None:
    env = HoldemEnv(
        HoldemConfig(
            starting_stacks=(10, 10),
            deck=cards("As", "Ks", "Ah", "Kh", "2c", "7d", "9s", "Jc", "3h", "4d", "5s", "6h"),
        ),
    )
    state = env.reset()

    shove = next(
        action for action in state.legal_actions if action.action_type == ActionType.ALL_IN
    )
    state = env.step(shove).observation
    call = next(action for action in state.legal_actions if action.action_type == ActionType.CALL)
    result = env.step(call)

    assert result.terminated
    assert result.rewards == {0: 10, 1: -10}
    assert result.observation.board == cards("7d", "9s", "Jc", "4d", "6h")
    assert sum(player.stack for player in result.observation.players) == 20
