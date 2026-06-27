import pyspiel  # type: ignore[import-not-found]
import pytest
from holdem_ai.preflop import PREFLOP_BUCKET_COUNT
from holdem_ai.preflop_game_v2 import (
    CALL,
    FOLD,
    JAM,
    MINRAISE,
    ShortStackBlueprint,
    ShortStackPreflopGame,
    solve_short_stack_preflop,
)


@pytest.fixture(scope="module")
def blueprint_10bb() -> ShortStackBlueprint:
    # One solve shared across the normalisation / calling-range assertions.
    return solve_short_stack_preflop(stack=10.0, iterations=300)


def test_game_shape_is_two_player_zero_sum() -> None:
    game = ShortStackPreflopGame(stack=8.0)
    assert game.num_players() == 2
    assert game.num_distinct_actions() == 7


def test_tree_is_finite_traversable_and_zero_sum() -> None:
    game = ShortStackPreflopGame(stack=8.0)
    seen = 0

    def walk(state: pyspiel.State) -> None:
        nonlocal seen
        seen += 1
        if state.is_terminal():
            payoffs = state.returns()
            assert payoffs[0] + payoffs[1] == pytest.approx(0.0)
            return
        outcomes = (
            state.chance_outcomes()
            if state.is_chance_node()
            else [(a, 1.0) for a in state.legal_actions()]
        )
        for action, _prob in outcomes:
            walk(state.child(action))

    walk(game.new_initial_state())
    assert seen > 100  # a real multi-level tree, and it terminates


def test_key_terminal_returns_match_the_rules() -> None:
    game = ShortStackPreflopGame(stack=8.0)

    # Button folds the open -> loses its 0.5 small blind.
    state = game.new_initial_state()
    state.apply_action(0)
    state.apply_action(3)
    state.apply_action(int(FOLD))
    assert state.returns() == [-0.5, 0.5]

    # Button jams, big blind folds -> button wins the 1.0 big blind.
    state = game.new_initial_state()
    state.apply_action(0)
    state.apply_action(3)
    state.apply_action(int(JAM))
    state.apply_action(int(FOLD))
    assert state.returns() == [1.0, -1.0]

    # Big blind cannot re-jam over an all-in: only fold / call are legal.
    state = game.new_initial_state()
    state.apply_action(0)
    state.apply_action(3)
    state.apply_action(int(JAM))
    assert set(state.legal_actions()) == {int(FOLD), int(CALL)}


def test_information_state_hides_opponent_bucket() -> None:
    game = ShortStackPreflopGame(stack=8.0)
    state = game.new_initial_state()
    state.apply_action(2)  # button bucket
    state.apply_action(6)  # big-blind bucket
    assert state.information_state_string(0) == "p0|b2|"
    state.apply_action(int(MINRAISE))
    assert state.information_state_string(1) == "p1|b6|m"


def test_solve_is_near_nash_and_distributions_are_normalised(
    blueprint_10bb: ShortStackBlueprint,
) -> None:
    assert blueprint_10bb.exploitability < 0.01
    assert len(blueprint_10bb.button_open) == PREFLOP_BUCKET_COUNT
    for mix in blueprint_10bb.button_open:
        assert sum(mix) == pytest.approx(1.0, abs=1e-3)
    for mix in blueprint_10bb.bb_vs_jam:
        assert sum(mix) == pytest.approx(1.0, abs=1e-3)
    for mix in blueprint_10bb.bb_vs_minraise:
        assert sum(mix) == pytest.approx(1.0, abs=1e-3)


def test_bb_calls_a_jam_with_the_nuts_and_folds_the_worst(
    blueprint_10bb: ShortStackBlueprint,
) -> None:
    assert blueprint_10bb.bb_vs_jam[0][1] > 0.9  # strongest bucket calls
    assert blueprint_10bb.bb_vs_jam[-1][0] > 0.9  # weakest bucket folds


def test_shorter_stacks_jam_more() -> None:
    short = solve_short_stack_preflop(stack=6.0, iterations=300)
    deep = solve_short_stack_preflop(stack=16.0, iterations=300)
    short_jam = sum(mix[4] for mix in short.button_open)  # index 4 == JAM frequency
    deep_jam = sum(mix[4] for mix in deep.button_open)
    assert short_jam > deep_jam


def test_realization_of_one_collapses_to_limp_or_jam() -> None:
    # With no positional edge the intermediate sizes are strictly dominated.
    bp = solve_short_stack_preflop(stack=12.0, oop_realization=1.0, iterations=400)
    intermediate = max(mix[2] + mix[3] for mix in bp.button_open)  # minraise + 2.5x
    assert intermediate < 0.05


def test_sizing_emerges_with_a_positional_edge() -> None:
    # With R < 1 a deeper stack actually uses a non-limp, non-jam raise.
    bp = solve_short_stack_preflop(stack=16.0, oop_realization=0.85, iterations=800)
    raised = max(mix[2] + mix[3] for mix in bp.button_open)  # minraise + 2.5x
    assert raised > 0.1
