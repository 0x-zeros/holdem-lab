"""A richer heads-up short-stack preflop game for OpenSpiel CFR (S2c-2).

Where ``preflop_game`` only models button jam / fold, this adds real **sizing**:
the button may fold, limp, min-raise (to 2bb), raise 2.5x (to 2.5bb) or jam, and
the big blind answers each (check / fold / call / jam), with the button getting a
final fold-or-call decision when the big blind jams over a non-all-in open. It
covers exactly the spots the review asked for — BB-vs-minraise, the SB's small
value raises, and jam-over-open.

Cards use the same card-removal-aware bucket deal and ``BUCKET_EQUITY`` showdown
matrix as ``preflop_game`` (see S2c-1). There is **no postflop betting**: a line
called without going all-in is scored as a showdown of the committed pot (stacks
behind returned). To keep sizing meaningful without modelling streets, a single
``oop_realization`` knob (R, default 0.85) gives the in-position button an
edge in those non-all-in pots — the out-of-position big blind realizes only R of
its equity and the button captures the rest. ``R = 1`` recovers the pure
no-postflop model, which collapses to limp-or-jam (intermediate sizes dominated);
``R < 1`` makes the fold-equity-vs-position-vs-risk tradeoff real, so the solver
actually uses limp / min-raise / 2.5x. This is a deliberately crude proxy for
position/initiative, not real postflop play (that is S3).

Construct directly (``ShortStackPreflopGame(stack=8)``) and hand it to a CFR
solver, or call :func:`solve_short_stack_preflop` for the extracted blueprint.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass

import numpy as np
import pyspiel  # type: ignore[import-not-found]
from open_spiel.python.algorithms import cfr, exploitability  # type: ignore[import-untyped]

from holdem_ai.preflop import (
    PREFLOP_BUCKET_COUNT,
    bucket_deal_conditional,
    bucket_deal_marginals,
    bucket_equity,
)

__all__ = [
    "DEFAULT_STACK",
    "ShortStackBlueprint",
    "ShortStackPreflopGame",
    "ShortStackPreflopState",
    "solve_short_stack_preflop",
]

DEFAULT_STACK = 8.0
_SB = 0.5
_BB = 1.0
_MINRAISE_TO = 2.0
_RAISE25_TO = 2.5


class _Action(enum.IntEnum):
    FOLD = 0
    CHECK = 1
    CALL = 2
    LIMP = 3
    MINRAISE = 4
    RAISE25 = 5
    JAM = 6


FOLD, CHECK, CALL, LIMP, MINRAISE, RAISE25, JAM = (
    _Action.FOLD,
    _Action.CHECK,
    _Action.CALL,
    _Action.LIMP,
    _Action.MINRAISE,
    _Action.RAISE25,
    _Action.JAM,
)

_LABEL = {FOLD: "F", CHECK: "k", CALL: "c", LIMP: "l", MINRAISE: "m", RAISE25: "r", JAM: "j"}
#: Committed-so-far (in bb) by the button after each non-all-in open.
_OPEN_COMMIT = {LIMP: _BB, MINRAISE: _MINRAISE_TO, RAISE25: _RAISE25_TO}
#: The button-vs-jam context label keyed by the opening action.
_OPEN_CONTEXT = {LIMP: "limp", MINRAISE: "minraise", RAISE25: "raise25"}

_GAME_TYPE = pyspiel.GameType(
    short_name="python_holdem_shortstack_preflop",
    long_name="Heads-up Short-stack Preflop (bucketed, with sizing)",
    dynamics=pyspiel.GameType.Dynamics.SEQUENTIAL,
    chance_mode=pyspiel.GameType.ChanceMode.EXPLICIT_STOCHASTIC,
    information=pyspiel.GameType.Information.IMPERFECT_INFORMATION,
    utility=pyspiel.GameType.Utility.ZERO_SUM,
    reward_model=pyspiel.GameType.RewardModel.TERMINAL,
    max_num_players=2,
    min_num_players=2,
    provides_information_state_string=True,
    provides_information_state_tensor=False,
    provides_observation_string=True,
    provides_observation_tensor=False,
    provides_factored_observation_string=False,
)


class ShortStackPreflopGame(pyspiel.Game):  # type: ignore[misc]
    """Heads-up bucketed short-stack preflop game with sizing, for a chosen stack."""

    def __init__(self, stack: float = DEFAULT_STACK, oop_realization: float = 0.85) -> None:
        self.stack = float(stack)
        self.oop_realization = float(oop_realization)
        game_info = pyspiel.GameInfo(
            num_distinct_actions=len(_Action),
            max_chance_outcomes=PREFLOP_BUCKET_COUNT,
            num_players=2,
            min_utility=-self.stack,
            max_utility=self.stack,
            utility_sum=0.0,
            max_game_length=3,  # player decisions only (open / response / vs-jam); chance excluded
        )
        super().__init__(_GAME_TYPE, game_info, {})

    def new_initial_state(self) -> ShortStackPreflopState:
        return ShortStackPreflopState(self)

    def make_py_observer(
        self, iig_obs_type: object = None, params: object = None
    ) -> _ShortStackObserver:
        return _ShortStackObserver()


class ShortStackPreflopState(pyspiel.State):  # type: ignore[misc]
    """State: two chance deals, then button open / BB response / button vs jam."""

    def __init__(self, game: ShortStackPreflopGame) -> None:
        super().__init__(game)
        self._stack = game.stack
        self._realize = game.oop_realization
        self.buckets: list[int] = []
        self.history_actions: list[int] = []
        self._game_over = False
        self._returns = [0.0, 0.0]

    # -- structure ----------------------------------------------------------
    def current_player(self) -> int:
        if self._game_over:
            return int(pyspiel.PlayerId.TERMINAL)
        if len(self.buckets) < 2:
            return int(pyspiel.PlayerId.CHANCE)
        if not self.history_actions:
            return 0  # button opens
        if len(self.history_actions) == 1:
            return 1  # big blind responds
        return 0  # button answers a big-blind jam

    def _legal_actions(self, player: int) -> list[int]:
        if not self.history_actions:
            return self._open_actions()
        opening = self.history_actions[0]
        if len(self.history_actions) == 1:
            if opening == LIMP:
                return [int(CHECK), int(JAM)]
            if opening == JAM:
                return [int(FOLD), int(CALL)]  # cannot re-jam over an all-in
            return [int(FOLD), int(CALL), int(JAM)]  # facing a min-raise / 2.5x
        return [int(FOLD), int(CALL)]  # button answers the big-blind jam

    def _open_actions(self) -> list[int]:
        actions = [int(FOLD), int(LIMP)]
        if self._stack > _MINRAISE_TO:
            actions.append(int(MINRAISE))
        if self._stack > _RAISE25_TO:
            actions.append(int(RAISE25))
        actions.append(int(JAM))
        return actions

    def chance_outcomes(self) -> list[tuple[int, float]]:
        assert self.is_chance_node()
        if not self.buckets:
            return list(enumerate(bucket_deal_marginals()))
        return list(enumerate(bucket_deal_conditional(self.buckets[0])))

    # -- transitions --------------------------------------------------------
    def _apply_action(self, action: int) -> None:
        if self.is_chance_node():
            self.buckets.append(int(action))
            return
        self.history_actions.append(int(action))
        button_net = self._terminal_button_net()
        if button_net is not None:
            self._returns = [button_net, -button_net]
            self._game_over = True

    def _showdown(self, committed_each: float, *, all_in: bool) -> float:
        """Button's net from a showdown of ``committed_each`` (per player).

        All-in pots realize raw bucket equity. In a non-all-in pot there is no
        modelled postflop play, so the out-of-position big blind realizes only
        ``oop_realization`` of its equity and the in-position button captures the
        rest — a single-parameter proxy for position/initiative. ``R = 1`` recovers
        the pure no-postflop model (which collapses to limp-or-jam).
        """
        equity = bucket_equity(self.buckets[0], self.buckets[1])
        if all_in:
            return (2.0 * equity - 1.0) * committed_each
        button_share = 1.0 - self._realize * (1.0 - equity)
        return (2.0 * button_share - 1.0) * committed_each

    def _terminal_button_net(self) -> float | None:
        history = self.history_actions
        opening = _Action(history[0])
        if opening == FOLD:
            return -_SB
        if len(history) == 1:
            return None  # big blind still to act
        response = _Action(history[1])

        if opening == LIMP:
            if response == CHECK:
                return self._showdown(_BB, all_in=False)
            # response is JAM (big blind jams over the limp); button must answer
            if len(history) == 2:
                return None
            return -_BB if history[2] == FOLD else self._showdown(self._stack, all_in=True)

        if opening == JAM:
            return _BB if response == FOLD else self._showdown(self._stack, all_in=True)

        # opening is MINRAISE or RAISE25
        if response == FOLD:
            return _BB  # button wins the big blind
        if response == CALL:
            return self._showdown(_OPEN_COMMIT[opening], all_in=False)
        # response is JAM (3bet jam over the open); button must answer
        if len(history) == 2:
            return None
        return (
            -_OPEN_COMMIT[opening]
            if history[2] == FOLD
            else self._showdown(self._stack, all_in=True)
        )

    # -- pyspiel plumbing ---------------------------------------------------
    def _action_to_string(self, player: int, action: int) -> str:
        if player == pyspiel.PlayerId.CHANCE:
            return f"Deal:b{action}"
        return _Action(action).name.lower()

    def is_terminal(self) -> bool:
        return self._game_over

    def returns(self) -> list[float]:
        return list(self._returns)

    def information_state_string(self, player: int | None = None) -> str:
        if player is None:
            player = self.current_player()
        return _info_string(self.buckets, self.history_actions, player)

    def __str__(self) -> str:
        deal = "".join(f"b{b}" for b in self.buckets)
        acts = "".join(_LABEL[_Action(a)] for a in self.history_actions)
        return f"{deal}|{acts}"


def _info_string(buckets: list[int], history: list[int], player: int) -> str:
    bucket = buckets[player] if 0 <= player < len(buckets) else "?"
    acts = "".join(_LABEL[_Action(a)] for a in history)
    return f"p{player}|b{bucket}|{acts}"


class _ShortStackObserver:
    """Minimal string observer (no tensor); enough for CFR information states."""

    def __init__(self) -> None:
        self.tensor = np.zeros(0, np.float32)
        self.dict: dict[str, np.ndarray] = {}

    def set_from(self, state: ShortStackPreflopState, player: int) -> None:
        return None

    def string_from(self, state: ShortStackPreflopState, player: int) -> str:
        return _info_string(state.buckets, state.history_actions, player)


@dataclass(frozen=True, slots=True)
class ShortStackBlueprint:
    """A solved short-stack preflop strategy, per bucket (0 = strongest)."""

    stack: float
    iterations: int
    exploitability: float
    #: button open mix per bucket: (fold, limp, minraise, raise25, jam)
    button_open: tuple[tuple[float, ...], ...]
    #: big blind vs a limp: (check, jam)
    bb_vs_limp: tuple[tuple[float, ...], ...]
    #: big blind vs a min-raise: (fold, call, jam)
    bb_vs_minraise: tuple[tuple[float, ...], ...]
    #: big blind vs a 2.5x: (fold, call, jam)
    bb_vs_raise25: tuple[tuple[float, ...], ...]
    #: big blind vs a jam: (fold, call)
    bb_vs_jam: tuple[tuple[float, ...], ...]
    #: button facing a big-blind jam, keyed by the open context: (fold, call)
    button_vs_jam: dict[str, tuple[tuple[float, ...], ...]]


def _probs(policy: object, state: object, actions: tuple[_Action, ...]) -> tuple[float, ...]:
    table = dict(policy.action_probabilities(state))  # type: ignore[attr-defined]
    return tuple(float(table.get(int(action), 0.0)) for action in actions)


def solve_short_stack_preflop(
    *, stack: float = DEFAULT_STACK, oop_realization: float = 0.85, iterations: int = 600
) -> ShortStackBlueprint:
    """Solve the bucketed short-stack preflop game with CFR+ and extract the blueprint."""
    game = ShortStackPreflopGame(stack=stack, oop_realization=oop_realization)
    solver = cfr.CFRPlusSolver(game)
    for _ in range(iterations):
        solver.evaluate_and_update_policy()
    average_policy = solver.average_policy()
    expl = float(exploitability.exploitability(game, average_policy))

    button_open: list[tuple[float, ...]] = []
    bb_vs_limp: list[tuple[float, ...]] = []
    bb_vs_minraise: list[tuple[float, ...]] = []
    bb_vs_raise25: list[tuple[float, ...]] = []
    bb_vs_jam: list[tuple[float, ...]] = []
    button_vs_jam: dict[str, list[tuple[float, ...]]] = {
        "limp": [],
        "minraise": [],
        "raise25": [],
    }

    # A min-raise / 2.5x open is only legal (and only in the tree) above its size,
    # so those info sets do not exist at very short stacks. Extracting them anyway
    # navigates off-tree and KeyErrors. Gate on legality and fill an inert fold
    # default for the contexts that cannot occur.
    minraise_legal = stack > _MINRAISE_TO
    raise25_legal = stack > _RAISE25_TO
    for bucket in range(PREFLOP_BUCKET_COUNT):
        open_state = _deal(game, button=bucket, big_blind=0)
        button_open.append(_probs(average_policy, open_state, (FOLD, LIMP, MINRAISE, RAISE25, JAM)))

        bb_vs_limp.append(_probs(average_policy, _deal_then(game, 0, bucket, LIMP), (CHECK, JAM)))
        bb_vs_minraise.append(
            _probs(average_policy, _deal_then(game, 0, bucket, MINRAISE), (FOLD, CALL, JAM))
            if minraise_legal
            else (1.0, 0.0, 0.0)
        )
        bb_vs_raise25.append(
            _probs(average_policy, _deal_then(game, 0, bucket, RAISE25), (FOLD, CALL, JAM))
            if raise25_legal
            else (1.0, 0.0, 0.0)
        )
        bb_vs_jam.append(_probs(average_policy, _deal_then(game, 0, bucket, JAM), (FOLD, CALL)))

        for opening, context in _OPEN_CONTEXT.items():
            if (context == "minraise" and not minraise_legal) or (
                context == "raise25" and not raise25_legal
            ):
                button_vs_jam[context].append((1.0, 0.0))
                continue
            state = _deal_then(game, bucket, 0, opening)
            state.apply_action(int(JAM))
            button_vs_jam[context].append(_probs(average_policy, state, (FOLD, CALL)))

    return ShortStackBlueprint(
        stack=stack,
        iterations=iterations,
        exploitability=expl,
        button_open=tuple(button_open),
        bb_vs_limp=tuple(bb_vs_limp),
        bb_vs_minraise=tuple(bb_vs_minraise),
        bb_vs_raise25=tuple(bb_vs_raise25),
        bb_vs_jam=tuple(bb_vs_jam),
        button_vs_jam={key: tuple(value) for key, value in button_vs_jam.items()},
    )


def _deal(game: ShortStackPreflopGame, *, button: int, big_blind: int) -> ShortStackPreflopState:
    state = game.new_initial_state()
    state.apply_action(button)
    state.apply_action(big_blind)
    return state


def _deal_then(
    game: ShortStackPreflopGame, button: int, big_blind: int, opening: _Action
) -> ShortStackPreflopState:
    state = _deal(game, button=button, big_blind=big_blind)
    state.apply_action(int(opening))
    return state
