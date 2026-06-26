"""Configuration for local no-limit Texas Hold'em games."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class HoldemConfig:
    starting_stacks: tuple[int, ...] = (200, 200)
    small_blind: int = 1
    big_blind: int = 2
    ante: int = 0
    hand_id: str = "hand-1"
    button_seat: int = 0
    small_blind_seat: int = 0
    big_blind_seat: int = 1

    def __post_init__(self) -> None:
        if len(self.starting_stacks) < 2:
            raise ValueError("at least two players are required")
        if any(stack <= 0 for stack in self.starting_stacks):
            raise ValueError("starting stacks must be positive")
        if self.small_blind <= 0:
            raise ValueError("small blind must be positive")
        if self.big_blind < self.small_blind:
            raise ValueError("big blind must be at least the small blind")
        if self.ante < 0:
            raise ValueError("ante cannot be negative")
        for seat in (self.button_seat, self.small_blind_seat, self.big_blind_seat):
            if seat < 0 or seat >= len(self.starting_stacks):
                raise ValueError(f"seat {seat} is outside the table")
