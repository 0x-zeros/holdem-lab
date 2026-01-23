"""Tests for cards module."""

import pytest

from holdem_lab.cards import (
    Card,
    Deck,
    Rank,
    Suit,
    format_card,
    format_cards,
    parse_card,
    parse_cards,
)


class TestRank:
    """Tests for Rank enum."""

    def test_rank_values(self):
        assert Rank.TWO.value == 2
        assert Rank.ACE.value == 14

    def test_rank_str(self):
        assert str(Rank.TWO) == "2"
        assert str(Rank.TEN) == "T"
        assert str(Rank.JACK) == "J"
        assert str(Rank.QUEEN) == "Q"
        assert str(Rank.KING) == "K"
        assert str(Rank.ACE) == "A"

    def test_rank_from_char(self):
        assert Rank.from_char("2") == Rank.TWO
        assert Rank.from_char("T") == Rank.TEN
        assert Rank.from_char("t") == Rank.TEN
        assert Rank.from_char("A") == Rank.ACE
        assert Rank.from_char("a") == Rank.ACE

    def test_rank_from_char_invalid(self):
        with pytest.raises(ValueError):
            Rank.from_char("X")


class TestSuit:
    """Tests for Suit enum."""

    def test_suit_values(self):
        assert Suit.CLUBS.value == 0
        assert Suit.SPADES.value == 3

    def test_suit_str(self):
        assert str(Suit.CLUBS) == "c"
        assert str(Suit.DIAMONDS) == "d"
        assert str(Suit.HEARTS) == "h"
        assert str(Suit.SPADES) == "s"

    def test_suit_symbol(self):
        assert Suit.CLUBS.symbol == "\u2663"
        assert Suit.HEARTS.symbol == "\u2665"

    def test_suit_from_char(self):
        assert Suit.from_char("c") == Suit.CLUBS
        assert Suit.from_char("C") == Suit.CLUBS
        assert Suit.from_char("h") == Suit.HEARTS
        assert Suit.from_char("\u2665") == Suit.HEARTS

    def test_suit_from_char_invalid(self):
        with pytest.raises(ValueError):
            Suit.from_char("x")


class TestCard:
    """Tests for Card class."""

    def test_card_creation(self):
        card = Card(Rank.ACE, Suit.HEARTS)
        assert card.rank == Rank.ACE
        assert card.suit == Suit.HEARTS

    def test_card_str(self):
        assert str(Card(Rank.ACE, Suit.HEARTS)) == "Ah"
        assert str(Card(Rank.TEN, Suit.SPADES)) == "Ts"

    def test_card_pretty(self):
        card = Card(Rank.ACE, Suit.HEARTS)
        assert card.pretty() == "A\u2665"

    def test_card_hashable(self):
        card1 = Card(Rank.ACE, Suit.HEARTS)
        card2 = Card(Rank.ACE, Suit.HEARTS)
        assert card1 == card2
        assert hash(card1) == hash(card2)

        card_set = {card1, card2}
        assert len(card_set) == 1

    def test_card_frozen(self):
        card = Card(Rank.ACE, Suit.HEARTS)
        with pytest.raises(AttributeError):
            card.rank = Rank.KING  # type: ignore

    def test_card_to_index(self):
        assert Card(Rank.TWO, Suit.CLUBS).to_index() == 0
        assert Card(Rank.TWO, Suit.SPADES).to_index() == 3
        assert Card(Rank.ACE, Suit.SPADES).to_index() == 51

    def test_card_from_index(self):
        assert Card.from_index(0) == Card(Rank.TWO, Suit.CLUBS)
        assert Card.from_index(51) == Card(Rank.ACE, Suit.SPADES)

    def test_card_from_index_invalid(self):
        with pytest.raises(ValueError):
            Card.from_index(-1)
        with pytest.raises(ValueError):
            Card.from_index(52)


class TestParseCard:
    """Tests for parse_card function."""

    def test_parse_basic(self):
        assert parse_card("Ah") == Card(Rank.ACE, Suit.HEARTS)
        assert parse_card("2c") == Card(Rank.TWO, Suit.CLUBS)
        assert parse_card("Ts") == Card(Rank.TEN, Suit.SPADES)

    def test_parse_case_insensitive(self):
        assert parse_card("AH") == Card(Rank.ACE, Suit.HEARTS)
        assert parse_card("ah") == Card(Rank.ACE, Suit.HEARTS)

    def test_parse_with_whitespace(self):
        assert parse_card("  Ah  ") == Card(Rank.ACE, Suit.HEARTS)

    def test_parse_ten_format(self):
        assert parse_card("10h") == Card(Rank.TEN, Suit.HEARTS)

    def test_parse_invalid(self):
        with pytest.raises(ValueError):
            parse_card("")
        with pytest.raises(ValueError):
            parse_card("A")


class TestParseCards:
    """Tests for parse_cards function."""

    def test_parse_space_separated(self):
        cards = parse_cards("Ah Kh")
        assert len(cards) == 2
        assert cards[0] == Card(Rank.ACE, Suit.HEARTS)
        assert cards[1] == Card(Rank.KING, Suit.HEARTS)

    def test_parse_comma_separated(self):
        cards = parse_cards("Ah, Kh")
        assert len(cards) == 2

    def test_parse_concatenated(self):
        cards = parse_cards("AhKh")
        assert len(cards) == 2
        assert cards[0] == Card(Rank.ACE, Suit.HEARTS)
        assert cards[1] == Card(Rank.KING, Suit.HEARTS)

    def test_parse_empty(self):
        assert parse_cards("") == []
        assert parse_cards("  ") == []


class TestFormatCards:
    """Tests for format functions."""

    def test_format_card(self):
        card = Card(Rank.ACE, Suit.HEARTS)
        assert format_card(card) == "Ah"

    def test_format_cards(self):
        cards = [Card(Rank.ACE, Suit.HEARTS), Card(Rank.KING, Suit.HEARTS)]
        assert format_cards(cards) == "Ah Kh"


class TestDeck:
    """Tests for Deck class."""

    def test_deck_size(self):
        deck = Deck()
        assert len(deck) == 52

    def test_deck_deal(self):
        deck = Deck()
        cards = deck.deal(2)
        assert len(cards) == 2
        assert len(deck) == 50

    def test_deck_deal_one(self):
        deck = Deck()
        card = deck.deal_one()
        assert isinstance(card, Card)
        assert len(deck) == 51

    def test_deck_deal_too_many(self):
        deck = Deck()
        with pytest.raises(ValueError):
            deck.deal(53)

    def test_deck_reproducible(self):
        deck1 = Deck(seed=42)
        deck2 = Deck(seed=42)

        cards1 = deck1.deal(5)
        cards2 = deck2.deal(5)
        assert cards1 == cards2

    def test_deck_remove(self):
        deck = Deck()
        ah = Card(Rank.ACE, Suit.HEARTS)
        deck.remove([ah])

        assert len(deck) == 51
        assert ah not in deck

    def test_deck_remove_not_in_deck(self):
        deck = Deck()
        ah = Card(Rank.ACE, Suit.HEARTS)
        deck.remove([ah])

        with pytest.raises(ValueError):
            deck.remove([ah])

    def test_deck_peek(self):
        deck = Deck(seed=42)
        peeked = deck.peek(3)
        assert len(peeked) == 3
        assert len(deck) == 52  # Deck unchanged

        dealt = deck.deal(3)
        assert dealt == peeked

    def test_deck_contains(self):
        deck = Deck()
        ah = Card(Rank.ACE, Suit.HEARTS)
        assert ah in deck

        deck.remove([ah])
        assert ah not in deck

    def test_deck_reset(self):
        deck = Deck(seed=42)
        deck.deal(10)
        assert len(deck) == 42

        # Remove a card permanently
        ah = Card(Rank.ACE, Suit.HEARTS)
        if ah in deck:
            deck.remove([ah])

        deck.reset()
        # After reset, removed cards stay removed
        assert ah not in deck
