"""Gym/PettingZoo-style environment wrapper for holdem-lab AI code."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from random import seed as set_random_seed

from holdem_common import Action, GameState

from holdem_engine.config import HoldemConfig
from holdem_engine.facade import PokerKitFacade


@dataclass(frozen=True, slots=True)
class StepResult:
    observation: GameState
    rewards: Mapping[int, int]
    terminated: bool
    truncated: bool = False
    info: Mapping[str, object] = field(default_factory=dict)


class HoldemEnv:
    def __init__(self, config: HoldemConfig | None = None) -> None:
        self.facade = PokerKitFacade(config)

    def reset(self, *, seed: int | None = None) -> GameState:
        if seed is not None:
            set_random_seed(seed)
        return self.facade.reset()

    def observe(self, seat: int | None = None) -> GameState:
        return self.facade.observe(seat)

    def legal_actions(self) -> tuple[Action, ...]:
        return self.facade.legal_actions()

    def step(self, action: Action) -> StepResult:
        observation = self.facade.step(action)
        terminated = observation.current_seat is None
        rewards = (
            self.facade.payoffs
            if terminated
            else {player.seat: 0 for player in observation.players}
        )
        return StepResult(
            observation=observation,
            rewards=rewards,
            terminated=terminated,
            info={"hand_id": observation.hand_id},
        )
