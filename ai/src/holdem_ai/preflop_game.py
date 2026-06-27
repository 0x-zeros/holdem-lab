"""A self-contained heads-up push/fold preflop game for OpenSpiel CFR.

This is the abstract game we *own*: the information-state string is defined here
and is therefore reproducible from our canonical ``GameState`` (see the S2b
blueprint bridge). Hands are abstracted into :data:`preflop.PREFLOP_BUCKET_COUNT`
equity buckets and showdowns are resolved by the precomputed
``preflop.BUCKET_EQUITY`` matrix, so the tree is tiny and CFR+ solves it to a
near-Nash push/fold strategy in milliseconds.

Game: button/small blind (player 0) posts 0.5, big blind (player 1) posts 1.0,
both with an effective ``stack`` (in big blinds). Player 0 folds or jams all-in;
facing a jam, player 1 folds or calls. Showdown equity comes from the bucket
matrix. Construct directly (``PushFoldGame(stack=10)``) and hand it to a CFR
solver — it does not need registration or game parameters.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass

import numpy as np
import pyspiel  # type: ignore[import-not-found]
from open_spiel.python.algorithms import cfr, exploitability  # type: ignore[import-untyped]

from holdem_ai.preflop import PREFLOP_BUCKET_COUNT, bucket_equity, bucket_weights

__all__ = [
    "CALL",
    "DEFAULT_STACK",
    "FOLD",
    "JAM",
    "PushFoldBlueprint",
    "PushFoldGame",
    "PushFoldState",
    "solve_push_fold",
]

DEFAULT_STACK = 10.0
_SB = 0.5
_BB = 1.0


class _Action(enum.IntEnum):
    FOLD = 0
    CALL = 1
    JAM = 2


FOLD, CALL, JAM = _Action.FOLD, _Action.CALL, _Action.JAM

_GAME_TYPE = pyspiel.GameType(
    short_name="python_holdem_pushfold",
    long_name="Heads-up Preflop Push/Fold (bucketed)",
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


class PushFoldGame(pyspiel.Game):  # type: ignore[misc]
    """Heads-up bucketed push/fold game with a chosen effective stack."""

    def __init__(self, stack: float = DEFAULT_STACK) -> None:
        self.stack = float(stack)
        game_info = pyspiel.GameInfo(
            num_distinct_actions=len(_Action),
            max_chance_outcomes=PREFLOP_BUCKET_COUNT,
            num_players=2,
            min_utility=-self.stack,
            max_utility=self.stack,
            utility_sum=0.0,
            max_game_length=2,
        )
        super().__init__(_GAME_TYPE, game_info, {})

    def new_initial_state(self) -> PushFoldState:
        return PushFoldState(self)

    def make_py_observer(
        self, iig_obs_type: object = None, params: object = None
    ) -> _PushFoldObserver:
        return _PushFoldObserver()


class PushFoldState(pyspiel.State):  # type: ignore[misc]
    """State for the push/fold game; chance deals two buckets, then two decisions."""

    def __init__(self, game: PushFoldGame) -> None:
        super().__init__(game)
        self._stack = game.stack
        self.buckets: list[int] = []
        self.history_actions: list[int] = []
        self._game_over = False
        self._returns = [0.0, 0.0]

    def current_player(self) -> int:
        if self._game_over:
            return int(pyspiel.PlayerId.TERMINAL)
        if len(self.buckets) < 2:
            return int(pyspiel.PlayerId.CHANCE)
        return len(self.history_actions)  # 0 -> SB acts, 1 -> BB acts

    def _legal_actions(self, player: int) -> list[int]:
        if len(self.history_actions) == 0:
            return [int(FOLD), int(JAM)]
        return [int(FOLD), int(CALL)]

    def chance_outcomes(self) -> list[tuple[int, float]]:
        assert self.is_chance_node()
        return [(bucket, weight) for bucket, weight in enumerate(bucket_weights())]

    def _apply_action(self, action: int) -> None:
        if self.is_chance_node():
            self.buckets.append(int(action))
            return
        self.history_actions.append(int(action))
        if len(self.history_actions) == 1:
            if action == FOLD:  # SB folds, loses its blind
                self._returns = [-_SB, _SB]
                self._game_over = True
            # SB jams -> BB to act
        else:
            self._game_over = True
            if action == FOLD:  # BB folds to the jam, loses its blind
                self._returns = [_BB, -_BB]
            else:  # BB calls: all-in showdown for the effective stack
                equity = bucket_equity(self.buckets[0], self.buckets[1])
                value = (2.0 * equity - 1.0) * self._stack
                self._returns = [value, -value]

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
        acts = "".join(_Action(a).name[0].lower() for a in self.history_actions)
        return f"{deal}|{acts}"


def _info_string(buckets: list[int], history: list[int], player: int) -> str:
    bucket = buckets[player] if 0 <= player < len(buckets) else "?"
    acts = "".join(_Action(a).name[0].lower() for a in history)
    return f"p{player}|b{bucket}|h{acts}"


class _PushFoldObserver:
    """Minimal string observer (no tensor); enough for CFR information states."""

    def __init__(self) -> None:
        self.tensor = np.zeros(0, np.float32)
        self.dict: dict[str, np.ndarray] = {}

    def set_from(self, state: PushFoldState, player: int) -> None:
        return None

    def string_from(self, state: PushFoldState, player: int) -> str:
        return _info_string(state.buckets, state.history_actions, player)


@dataclass(frozen=True, slots=True)
class PushFoldBlueprint:
    """A solved push/fold strategy: per-bucket jam/call frequencies."""

    stack: float
    iterations: int
    exploitability: float
    sb_jam: tuple[float, ...]  # P(button jams) per bucket (0 = strongest)
    bb_call: tuple[float, ...]  # P(big blind calls a jam) per bucket


def solve_push_fold(*, stack: float = DEFAULT_STACK, iterations: int = 400) -> PushFoldBlueprint:
    """Solve the bucketed push/fold game with CFR+ and extract the blueprint."""
    game = PushFoldGame(stack=stack)
    solver = cfr.CFRPlusSolver(game)
    for _ in range(iterations):
        solver.evaluate_and_update_policy()
    average_policy = solver.average_policy()
    expl = float(exploitability.exploitability(game, average_policy))

    sb_jam: list[float] = []
    bb_call: list[float] = []
    for bucket in range(PREFLOP_BUCKET_COUNT):
        sb_state = game.new_initial_state()
        sb_state.apply_action(bucket)  # button's bucket
        sb_state.apply_action(0)  # big blind's bucket (irrelevant to this info set)
        sb_jam.append(float(dict(average_policy.action_probabilities(sb_state)).get(int(JAM), 0.0)))

        bb_state = game.new_initial_state()
        bb_state.apply_action(0)  # button's bucket (irrelevant to this info set)
        bb_state.apply_action(bucket)  # big blind's bucket
        bb_state.apply_action(int(JAM))
        bb_call.append(
            float(dict(average_policy.action_probabilities(bb_state)).get(int(CALL), 0.0))
        )

    return PushFoldBlueprint(
        stack=stack,
        iterations=iterations,
        exploitability=expl,
        sb_jam=tuple(sb_jam),
        bb_call=tuple(bb_call),
    )
