import pyspiel  # type: ignore[import-not-found]
import pytest
from holdem_ai.preflop import PREFLOP_BUCKET_COUNT, bucket_equity
from holdem_ai.preflop_game import (
    CALL,
    FOLD,
    JAM,
    PushFoldGame,
    solve_push_fold,
)


def test_game_shape_is_two_player_zero_sum() -> None:
    game = PushFoldGame(stack=10.0)
    assert game.num_players() == 2
    assert game.num_distinct_actions() == 3


def test_terminal_returns_match_pushfold_rules() -> None:
    game = PushFoldGame(stack=10.0)

    # Button folds preflop -> loses its 0.5 blind.
    state = game.new_initial_state()
    state.apply_action(0)  # deal button bucket
    state.apply_action(3)  # deal big-blind bucket
    state.apply_action(int(FOLD))
    assert state.is_terminal()
    assert state.returns() == [-0.5, 0.5]

    # Button jams, big blind folds -> button wins the 1.0 blind.
    state = game.new_initial_state()
    state.apply_action(0)
    state.apply_action(3)
    state.apply_action(int(JAM))
    state.apply_action(int(FOLD))
    assert state.returns() == [1.0, -1.0]

    # Button jams, big blind calls -> showdown for the stack via bucket equity.
    state = game.new_initial_state()
    state.apply_action(1)  # button bucket 1
    state.apply_action(5)  # big-blind bucket 5
    state.apply_action(int(JAM))
    state.apply_action(int(CALL))
    expected = (2.0 * bucket_equity(1, 5) - 1.0) * 10.0
    assert state.returns()[0] == pytest.approx(expected)
    assert state.returns()[0] + state.returns()[1] == pytest.approx(0.0)


def test_information_state_hides_opponent_bucket() -> None:
    game = PushFoldGame(stack=10.0)
    state = game.new_initial_state()
    state.apply_action(2)  # button bucket
    state.apply_action(6)  # big-blind bucket
    # Button to act: its info state shows its own bucket, not the opponent's.
    assert state.information_state_string(0) == "p0|b2|h"
    state.apply_action(int(JAM))
    assert state.information_state_string(1) == "p1|b6|hj"


def test_cfr_solves_pushfold_to_near_nash() -> None:
    blueprint = solve_push_fold(stack=10.0, iterations=300)

    assert blueprint.exploitability < 0.01
    assert len(blueprint.sb_jam) == PREFLOP_BUCKET_COUNT
    # Always jam / call the strongest bucket; never with the weakest at 10bb.
    assert blueprint.sb_jam[0] > 0.9
    assert blueprint.bb_call[0] > 0.9
    assert blueprint.sb_jam[-1] < 0.1
    # The button jams at least as wide as the big blind calls.
    assert sum(blueprint.sb_jam) >= sum(blueprint.bb_call)


def test_shorter_stacks_jam_wider() -> None:
    short = solve_push_fold(stack=4.0, iterations=300)
    deep = solve_push_fold(stack=16.0, iterations=300)

    assert sum(short.sb_jam) >= sum(deep.sb_jam)


def test_game_tree_is_traversable_by_openspiel() -> None:
    # Sanity: OpenSpiel can enumerate the whole tree (needed for CFR / exploitability).
    game = PushFoldGame(stack=8.0)
    seen = 0

    def walk(state: pyspiel.State) -> None:
        nonlocal seen
        seen += 1
        if state.is_terminal():
            return
        for action, _prob in (
            state.chance_outcomes()
            if state.is_chance_node()
            else [(a, 1.0) for a in state.legal_actions()]
        ):
            walk(state.child(action))

    walk(game.new_initial_state())
    assert seen > PREFLOP_BUCKET_COUNT
