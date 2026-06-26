"""Canonical state interface shared by the game, AI, engine, and bot."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Self


class Suit(StrEnum):
    CLUBS = "c"
    DIAMONDS = "d"
    HEARTS = "h"
    SPADES = "s"


class Rank(StrEnum):
    TWO = "2"
    THREE = "3"
    FOUR = "4"
    FIVE = "5"
    SIX = "6"
    SEVEN = "7"
    EIGHT = "8"
    NINE = "9"
    TEN = "T"
    JACK = "J"
    QUEEN = "Q"
    KING = "K"
    ACE = "A"


class Street(StrEnum):
    PREFLOP = "preflop"
    FLOP = "flop"
    TURN = "turn"
    RIVER = "river"
    SHOWDOWN = "showdown"


class ActionType(StrEnum):
    FOLD = "fold"
    CHECK = "check"
    CALL = "call"
    BET = "bet"
    RAISE = "raise"
    ALL_IN = "all_in"


@dataclass(frozen=True, slots=True)
class Card:
    rank: Rank
    suit: Suit

    @property
    def code(self) -> str:
        return f"{self.rank.value}{self.suit.value}"

    @classmethod
    def from_code(cls, code: str) -> Self:
        normalized = code.strip()
        if len(normalized) != 2:
            raise ValueError(f"card code must have exactly two characters: {code!r}")
        rank_code, suit_code = normalized[0].upper(), normalized[1].lower()
        return cls(rank=Rank(rank_code), suit=Suit(suit_code))


@dataclass(frozen=True, slots=True)
class Action:
    """A public action request or legal-action descriptor.

    For no-limit ``BET``/``RAISE``/``ALL_IN``, ``amount`` is the total street
    commitment after the action, matching PokerKit's "bet/raise to" semantics.
    For ``CALL``, it is the chips required to continue. ``min_amount`` and
    ``max_amount`` describe the legal total-commitment range when applicable.
    """

    action_type: ActionType
    amount: int = 0
    min_amount: int | None = None
    max_amount: int | None = None

    def __post_init__(self) -> None:
        if self.amount < 0:
            raise ValueError("action amount cannot be negative")
        if self.min_amount is not None and self.min_amount < 0:
            raise ValueError("minimum action amount cannot be negative")
        if self.max_amount is not None and self.max_amount < 0:
            raise ValueError("maximum action amount cannot be negative")
        if (
            self.min_amount is not None
            and self.max_amount is not None
            and self.min_amount > self.max_amount
        ):
            raise ValueError("minimum action amount cannot exceed maximum action amount")


@dataclass(frozen=True, slots=True)
class PlayerState:
    seat: int
    stack: int
    committed: int = 0
    hole_cards: tuple[Card, ...] = ()
    active: bool = True
    all_in: bool = False
    dealer: bool = False
    small_blind: bool = False
    big_blind: bool = False

    def __post_init__(self) -> None:
        if self.seat < 0:
            raise ValueError("seat cannot be negative")
        if self.stack < 0:
            raise ValueError("stack cannot be negative")
        if self.committed < 0:
            raise ValueError("committed chips cannot be negative")


@dataclass(frozen=True, slots=True)
class Pot:
    amount: int
    eligible_seats: frozenset[int]

    def __post_init__(self) -> None:
        if self.amount < 0:
            raise ValueError("pot amount cannot be negative")
        if not self.eligible_seats and self.amount > 0:
            raise ValueError("non-empty pot must have at least one eligible seat")


@dataclass(frozen=True, slots=True)
class GameState:
    hand_id: str
    street: Street
    players: tuple[PlayerState, ...]
    board: tuple[Card, ...]
    pots: tuple[Pot, ...]
    current_seat: int | None
    button_seat: int
    small_blind: int
    big_blind: int
    min_raise: int
    to_call: int
    legal_actions: tuple[Action, ...] = ()
    last_aggressor: int | None = None
    deck_remaining: int | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)

    @property
    def pot_total(self) -> int:
        return sum(pot.amount for pot in self.pots)

    @property
    def active_players(self) -> tuple[PlayerState, ...]:
        return tuple(player for player in self.players if player.active)

    def player(self, seat: int) -> PlayerState:
        for player in self.players:
            if player.seat == seat:
                return player
        raise KeyError(f"unknown seat: {seat}")
