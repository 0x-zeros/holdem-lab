"""Card representation, parsing, and deck management."""

from __future__ import annotations

import random
from dataclasses import dataclass
from enum import IntEnum
from typing import Sequence


class Rank(IntEnum):
    """Card rank with numeric values (2-14, where A=14)."""

    TWO = 2
    THREE = 3
    FOUR = 4
    FIVE = 5
    SIX = 6
    SEVEN = 7
    EIGHT = 8
    NINE = 9
    TEN = 10
    JACK = 11
    QUEEN = 12
    KING = 13
    ACE = 14

    def __str__(self) -> str:
        """Return single-character rank representation."""
        if self.value <= 9:
            return str(self.value)
        return {10: "T", 11: "J", 12: "Q", 13: "K", 14: "A"}[self.value]

    @classmethod
    def from_char(cls, c: str) -> Rank:
        """Parse rank from single character."""
        c = c.upper()
        char_map = {
            "2": cls.TWO,
            "3": cls.THREE,
            "4": cls.FOUR,
            "5": cls.FIVE,
            "6": cls.SIX,
            "7": cls.SEVEN,
            "8": cls.EIGHT,
            "9": cls.NINE,
            "T": cls.TEN,
            "10": cls.TEN,
            "J": cls.JACK,
            "Q": cls.QUEEN,
            "K": cls.KING,
            "A": cls.ACE,
        }
        if c not in char_map:
            raise ValueError(f"Invalid rank character: {c!r}")
        return char_map[c]


class Suit(IntEnum):
    """Card suit."""

    CLUBS = 0
    DIAMONDS = 1
    HEARTS = 2
    SPADES = 3

    def __str__(self) -> str:
        """Return single-character suit representation."""
        return {0: "c", 1: "d", 2: "h", 3: "s"}[self.value]

    @property
    def symbol(self) -> str:
        """Return Unicode symbol for the suit."""
        return {0: "\u2663", 1: "\u2666", 2: "\u2665", 3: "\u2660"}[self.value]

    @classmethod
    def from_char(cls, c: str) -> Suit:
        """Parse suit from single character."""
        c = c.lower()
        char_map = {
            "c": cls.CLUBS,
            "d": cls.DIAMONDS,
            "h": cls.HEARTS,
            "s": cls.SPADES,
            "\u2663": cls.CLUBS,
            "\u2666": cls.DIAMONDS,
            "\u2665": cls.HEARTS,
            "\u2660": cls.SPADES,
        }
        if c not in char_map:
            raise ValueError(f"Invalid suit character: {c!r}")
        return char_map[c]


@dataclass(frozen=True, slots=True)
class Card:
    """
    Immutable card representation.

    Cards are hashable and can be used in sets/dicts.
    """

    rank: Rank
    suit: Suit

    def __str__(self) -> str:
        """Return two-character card representation (e.g., 'Ah')."""
        return f"{self.rank}{self.suit}"

    def __repr__(self) -> str:
        return f"Card({self.rank!r}, {self.suit!r})"

    def to_index(self) -> int:
        """
        Convert card to unique index 0-51.

        Index = rank_offset * 4 + suit, where rank_offset = rank - 2.
        """
        return (self.rank.value - 2) * 4 + self.suit.value

    @classmethod
    def from_index(cls, index: int) -> Card:
        """Create card from index 0-51."""
        if not 0 <= index < 52:
            raise ValueError(f"Card index must be 0-51, got {index}")
        rank_value = (index // 4) + 2
        suit_value = index % 4
        return cls(Rank(rank_value), Suit(suit_value))

    def pretty(self) -> str:
        """Return card with Unicode suit symbol."""
        return f"{self.rank}{self.suit.symbol}"


def parse_card(s: str) -> Card:
    """
    Parse a single card from string.

    Accepts formats: "Ah", "AH", "ah", "A♥", etc.
    """
    s = s.strip()
    if len(s) < 2:
        raise ValueError(f"Card string too short: {s!r}")

    # Handle "10" as a rank
    if s.startswith("10"):
        rank = Rank.TEN
        suit = Suit.from_char(s[2:])
    else:
        rank = Rank.from_char(s[0])
        suit = Suit.from_char(s[1:])

    return Card(rank, suit)


def parse_cards(s: str) -> list[Card]:
    """
    Parse multiple cards from string.

    Accepts space-separated or concatenated cards:
    - "Ah Kh" -> [Card(A, h), Card(K, h)]
    - "AhKh" -> [Card(A, h), Card(K, h)]
    - "Ah, Kh" -> [Card(A, h), Card(K, h)]
    """
    s = s.strip()
    if not s:
        return []

    # Try space/comma separated first
    parts = s.replace(",", " ").split()
    if len(parts) > 1:
        return [parse_card(p) for p in parts]

    # Try concatenated cards (2 chars each, except "10" which is 3)
    cards = []
    i = 0
    while i < len(s):
        # Check for "10" as rank
        if s[i : i + 2] == "10":
            if i + 3 > len(s):
                raise ValueError(f"Invalid card string at position {i}: {s!r}")
            cards.append(parse_card(s[i : i + 3]))
            i += 3
        else:
            if i + 2 > len(s):
                raise ValueError(f"Invalid card string at position {i}: {s!r}")
            cards.append(parse_card(s[i : i + 2]))
            i += 2

    return cards


def format_card(card: Card) -> str:
    """Format card as two-character string."""
    return str(card)


def format_cards(cards: Sequence[Card]) -> str:
    """Format multiple cards as space-separated string."""
    return " ".join(str(c) for c in cards)


class Deck:
    """
    Standard 52-card deck with shuffle and deal operations.

    Supports reproducible shuffles via seed parameter.
    """

    def __init__(self, seed: int | None = None):
        """
        Initialize a new shuffled deck.

        Args:
            seed: Random seed for reproducible shuffles.
        """
        self._rng = random.Random(seed)
        self._cards: list[Card] = []
        self._removed: set[Card] = set()
        self.reset()

    def reset(self) -> None:
        """Reset and shuffle the deck, keeping previously removed cards excluded."""
        self._cards = [Card.from_index(i) for i in range(52)]
        # Keep previously removed cards excluded
        for card in self._removed:
            if card in self._cards:
                self._cards.remove(card)
        self._rng.shuffle(self._cards)

    def shuffle(self) -> None:
        """Shuffle remaining cards in the deck."""
        self._rng.shuffle(self._cards)

    def deal(self, n: int = 1) -> list[Card]:
        """
        Deal n cards from the deck.

        Args:
            n: Number of cards to deal.

        Returns:
            List of dealt cards.

        Raises:
            ValueError: If not enough cards remain.
        """
        if n > len(self._cards):
            raise ValueError(f"Cannot deal {n} cards, only {len(self._cards)} remain")
        dealt = self._cards[:n]
        self._cards = self._cards[n:]
        return dealt

    def deal_one(self) -> Card:
        """Deal a single card from the deck."""
        return self.deal(1)[0]

    def remove(self, cards: Sequence[Card]) -> None:
        """
        Remove specific cards from the deck.

        Used when cards are already known (e.g., hole cards in equity calculation).

        Args:
            cards: Cards to remove.

        Raises:
            ValueError: If a card is not in the deck.
        """
        for card in cards:
            if card not in self._cards:
                if card in self._removed:
                    raise ValueError(f"Card {card} was already removed")
                raise ValueError(f"Card {card} not in deck")
            self._cards.remove(card)
            self._removed.add(card)

    def peek(self, n: int = 1) -> list[Card]:
        """Peek at the top n cards without removing them."""
        if n > len(self._cards):
            raise ValueError(f"Cannot peek {n} cards, only {len(self._cards)} remain")
        return self._cards[:n]

    def remaining(self) -> int:
        """Return the number of cards remaining in the deck."""
        return len(self._cards)

    def __len__(self) -> int:
        """Return the number of cards remaining in the deck."""
        return len(self._cards)

    def __contains__(self, card: Card) -> bool:
        """Check if a card is still in the deck."""
        return card in self._cards


# Pre-computed full deck for convenience
FULL_DECK: tuple[Card, ...] = tuple(Card.from_index(i) for i in range(52))
