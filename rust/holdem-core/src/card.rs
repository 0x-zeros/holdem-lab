//! Card representation and deck management.

use crate::error::{HoldemError, HoldemResult};
use rand::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::fmt;
use std::str::FromStr;
use thiserror::Error;

/// Card rank (2-14, where Ace = 14)
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum Rank {
    Two = 2,
    Three = 3,
    Four = 4,
    Five = 5,
    Six = 6,
    Seven = 7,
    Eight = 8,
    Nine = 9,
    Ten = 10,
    Jack = 11,
    Queen = 12,
    King = 13,
    Ace = 14,
}

impl Rank {
    /// All ranks in ascending order
    pub const ALL: [Rank; 13] = [
        Rank::Two,
        Rank::Three,
        Rank::Four,
        Rank::Five,
        Rank::Six,
        Rank::Seven,
        Rank::Eight,
        Rank::Nine,
        Rank::Ten,
        Rank::Jack,
        Rank::Queen,
        Rank::King,
        Rank::Ace,
    ];

    /// Create rank from numeric value (2-14)
    #[must_use]
    pub fn from_value(v: u8) -> Option<Self> {
        match v {
            2 => Some(Rank::Two),
            3 => Some(Rank::Three),
            4 => Some(Rank::Four),
            5 => Some(Rank::Five),
            6 => Some(Rank::Six),
            7 => Some(Rank::Seven),
            8 => Some(Rank::Eight),
            9 => Some(Rank::Nine),
            10 => Some(Rank::Ten),
            11 => Some(Rank::Jack),
            12 => Some(Rank::Queen),
            13 => Some(Rank::King),
            14 => Some(Rank::Ace),
            _ => None,
        }
    }

    /// Get numeric value (2-14)
    #[must_use]
    pub const fn value(self) -> u8 {
        self as u8
    }

    /// Parse from character ('2'-'9', 'T', 'J', 'Q', 'K', 'A')
    #[must_use]
    pub fn from_char(c: char) -> Option<Self> {
        match c.to_ascii_uppercase() {
            '2' => Some(Rank::Two),
            '3' => Some(Rank::Three),
            '4' => Some(Rank::Four),
            '5' => Some(Rank::Five),
            '6' => Some(Rank::Six),
            '7' => Some(Rank::Seven),
            '8' => Some(Rank::Eight),
            '9' => Some(Rank::Nine),
            'T' => Some(Rank::Ten),
            'J' => Some(Rank::Jack),
            'Q' => Some(Rank::Queen),
            'K' => Some(Rank::King),
            'A' => Some(Rank::Ace),
            _ => None,
        }
    }

    /// Convert to character
    #[must_use]
    pub const fn to_char(self) -> char {
        match self {
            Rank::Two => '2',
            Rank::Three => '3',
            Rank::Four => '4',
            Rank::Five => '5',
            Rank::Six => '6',
            Rank::Seven => '7',
            Rank::Eight => '8',
            Rank::Nine => '9',
            Rank::Ten => 'T',
            Rank::Jack => 'J',
            Rank::Queen => 'Q',
            Rank::King => 'K',
            Rank::Ace => 'A',
        }
    }
}

impl fmt::Display for Rank {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_char())
    }
}

/// Card suit
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum Suit {
    Clubs = 0,
    Diamonds = 1,
    Hearts = 2,
    Spades = 3,
}

impl Suit {
    /// All suits
    pub const ALL: [Suit; 4] = [Suit::Clubs, Suit::Diamonds, Suit::Hearts, Suit::Spades];

    /// Parse from character ('c', 'd', 'h', 's' or Unicode symbols)
    #[must_use]
    pub fn from_char(c: char) -> Option<Self> {
        match c.to_ascii_lowercase() {
            'c' | '♣' => Some(Suit::Clubs),
            'd' | '♦' => Some(Suit::Diamonds),
            'h' | '♥' => Some(Suit::Hearts),
            's' | '♠' => Some(Suit::Spades),
            _ => None,
        }
    }

    /// Convert to character
    #[must_use]
    pub const fn to_char(self) -> char {
        match self {
            Suit::Clubs => 'c',
            Suit::Diamonds => 'd',
            Suit::Hearts => 'h',
            Suit::Spades => 's',
        }
    }

    /// Convert to Unicode symbol
    #[must_use]
    pub const fn to_symbol(self) -> char {
        match self {
            Suit::Clubs => '♣',
            Suit::Diamonds => '♦',
            Suit::Hearts => '♥',
            Suit::Spades => '♠',
        }
    }
}

impl fmt::Display for Suit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_char())
    }
}

/// A playing card with rank and suit
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Card {
    pub rank: Rank,
    pub suit: Suit,
}

impl Card {
    /// Create a new card
    #[must_use]
    pub const fn new(rank: Rank, suit: Suit) -> Self {
        Self { rank, suit }
    }

    /// Convert to 0-51 index
    /// Formula: (rank - 2) * 4 + suit
    #[must_use]
    pub const fn to_index(self) -> u8 {
        (self.rank as u8 - 2) * 4 + self.suit as u8
    }

    /// Create from 0-51 index
    #[must_use]
    pub fn from_index(index: u8) -> Option<Self> {
        if index >= 52 {
            return None;
        }
        let rank_value = index / 4 + 2;
        let suit_value = index % 4;

        Some(Self {
            rank: Rank::from_value(rank_value)?,
            suit: match suit_value {
                0 => Suit::Clubs,
                1 => Suit::Diamonds,
                2 => Suit::Hearts,
                3 => Suit::Spades,
                _ => return None,
            },
        })
    }

    /// Parse from string (e.g., "Ah", "KS", "10c")
    pub fn parse(s: &str) -> Result<Self, ParseError> {
        let s = s.trim();
        if s.is_empty() {
            return Err(ParseError::Empty);
        }

        let chars: Vec<char> = s.chars().collect();

        // Handle "10x" format
        if chars.len() == 3 && chars[0] == '1' && chars[1] == '0' {
            let suit = Suit::from_char(chars[2]).ok_or(ParseError::InvalidSuit(chars[2]))?;
            return Ok(Self::new(Rank::Ten, suit));
        }

        if chars.len() != 2 {
            return Err(ParseError::InvalidFormat(s.to_string()));
        }

        let rank = Rank::from_char(chars[0]).ok_or(ParseError::InvalidRank(chars[0]))?;
        let suit = Suit::from_char(chars[1]).ok_or(ParseError::InvalidSuit(chars[1]))?;

        Ok(Self::new(rank, suit))
    }

    /// Format with Unicode suit symbol
    #[must_use]
    pub fn pretty(self) -> String {
        format!("{}{}", self.rank.to_char(), self.suit.to_symbol())
    }
}

impl fmt::Display for Card {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}{}", self.rank.to_char(), self.suit.to_char())
    }
}

impl FromStr for Card {
    type Err = ParseError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::parse(s)
    }
}

impl PartialOrd for Card {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Card {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.to_index().cmp(&other.to_index())
    }
}

/// Error parsing a card string
#[derive(Debug, Error, PartialEq, Eq)]
pub enum ParseError {
    #[error("empty card string")]
    Empty,
    #[error("invalid card format: {0}")]
    InvalidFormat(String),
    #[error("invalid rank character: {0}")]
    InvalidRank(char),
    #[error("invalid suit character: {0}")]
    InvalidSuit(char),
}

/// Parse multiple cards from a string
/// Supports formats: "Ah Kh", "AhKh", "Ah, Kh"
pub fn parse_cards(s: &str) -> Result<Vec<Card>, ParseError> {
    let s = s.trim();
    if s.is_empty() {
        return Ok(Vec::new());
    }

    // Try splitting by whitespace or comma first
    let parts: Vec<&str> = s.split([' ', ',', '\t']).filter(|p| !p.is_empty()).collect();

    if parts.len() > 1 {
        // Multiple parts separated by delimiter
        parts.into_iter().map(Card::parse).collect()
    } else {
        // Try parsing as concatenated cards (e.g., "AhKh")
        let chars: Vec<char> = s.chars().collect();
        let mut cards = Vec::new();
        let mut i = 0;

        while i < chars.len() {
            // Check for "10x" format
            if i + 2 < chars.len() && chars[i] == '1' && chars[i + 1] == '0' {
                let suit = Suit::from_char(chars[i + 2]).ok_or(ParseError::InvalidSuit(chars[i + 2]))?;
                cards.push(Card::new(Rank::Ten, suit));
                i += 3;
            } else if i + 1 < chars.len() {
                let rank = Rank::from_char(chars[i]).ok_or(ParseError::InvalidRank(chars[i]))?;
                let suit = Suit::from_char(chars[i + 1]).ok_or(ParseError::InvalidSuit(chars[i + 1]))?;
                cards.push(Card::new(rank, suit));
                i += 2;
            } else {
                return Err(ParseError::InvalidFormat(s.to_string()));
            }
        }

        Ok(cards)
    }
}

/// Format cards as string
#[must_use]
pub fn format_cards(cards: &[Card]) -> String {
    cards.iter().map(ToString::to_string).collect::<Vec<_>>().join(" ")
}

/// A deck of 52 playing cards
pub struct Deck {
    cards: Vec<Card>,
    removed: HashSet<Card>,
    rng: StdRng,
}

impl Deck {
    /// Create a new deck with optional seed for reproducible shuffles
    #[must_use]
    pub fn new(seed: Option<u64>) -> Self {
        let rng = match seed {
            Some(s) => StdRng::seed_from_u64(s),
            None => StdRng::from_os_rng(),
        };
        let mut deck = Self {
            cards: Self::full_deck(),
            removed: HashSet::new(),
            rng,
        };
        deck.shuffle();
        deck
    }

    /// Get all 52 cards in order
    #[must_use]
    pub fn full_deck() -> Vec<Card> {
        let mut cards = Vec::with_capacity(52);
        for rank in Rank::ALL {
            for suit in Suit::ALL {
                cards.push(Card::new(rank, suit));
            }
        }
        cards
    }

    /// Reset deck to full 52 cards
    pub fn reset(&mut self) {
        self.cards = Self::full_deck();
        if !self.removed.is_empty() {
            self.cards.retain(|c| !self.removed.contains(c));
        }
        self.shuffle();
    }

    /// Shuffle the remaining cards
    pub fn shuffle(&mut self) {
        self.cards.shuffle(&mut self.rng);
    }

    /// Deal n cards from the deck
    ///
    /// # Errors
    /// Returns an error if there are not enough cards remaining.
    pub fn deal(&mut self, n: usize) -> HoldemResult<Vec<Card>> {
        if n > self.cards.len() {
            return Err(HoldemError::InsufficientCards {
                requested: n,
                available: self.cards.len(),
            });
        }
        Ok(self.cards.drain(..n).collect())
    }

    /// Deal one card
    ///
    /// # Errors
    /// Returns an error if the deck is empty.
    pub fn deal_one(&mut self) -> HoldemResult<Card> {
        self.deal(1).map(|mut v| v.pop().unwrap())
    }

    /// Remove specific cards from the deck
    ///
    /// # Errors
    /// Returns an error if a card is not in the deck or was already removed.
    pub fn remove(&mut self, cards: &[Card]) -> HoldemResult<()> {
        for card in cards {
            if !self.cards.contains(card) {
                if self.removed.contains(card) {
                    return Err(HoldemError::CardAlreadyRemoved(card.to_string()));
                }
                return Err(HoldemError::CardNotInDeck(card.to_string()));
            }

            if let Some(index) = self.cards.iter().position(|c| c == card) {
                self.cards.remove(index);
                self.removed.insert(*card);
            }
        }
        Ok(())
    }

    /// Check if a card is in the deck
    #[must_use]
    pub fn contains(&self, card: Card) -> bool {
        self.cards.contains(&card)
    }

    /// Get remaining card count
    #[must_use]
    pub fn len(&self) -> usize {
        self.cards.len()
    }

    /// Check if deck is empty
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.cards.is_empty()
    }

    /// Get remaining cards (without removing)
    #[must_use]
    pub fn remaining(&self) -> &[Card] {
        &self.cards
    }

    /// Peek at the top n cards
    ///
    /// # Errors
    /// Returns an error if there are not enough cards remaining.
    pub fn peek(&self, n: usize) -> HoldemResult<&[Card]> {
        if n > self.cards.len() {
            return Err(HoldemError::InsufficientCards {
                requested: n,
                available: self.cards.len(),
            });
        }
        Ok(&self.cards[..n])
    }
}

impl Default for Deck {
    fn default() -> Self {
        Self::new(None)
    }
}

/// Pre-computed full deck as constant array
pub const FULL_DECK: [Card; 52] = {
    let mut cards = [Card { rank: Rank::Two, suit: Suit::Clubs }; 52];
    let mut i = 0;
    let ranks = [
        Rank::Two,
        Rank::Three,
        Rank::Four,
        Rank::Five,
        Rank::Six,
        Rank::Seven,
        Rank::Eight,
        Rank::Nine,
        Rank::Ten,
        Rank::Jack,
        Rank::Queen,
        Rank::King,
        Rank::Ace,
    ];
    let suits = [Suit::Clubs, Suit::Diamonds, Suit::Hearts, Suit::Spades];

    let mut r = 0;
    while r < 13 {
        let mut s = 0;
        while s < 4 {
            cards[i] = Card { rank: ranks[r], suit: suits[s] };
            i += 1;
            s += 1;
        }
        r += 1;
    }
    cards
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rank_value() {
        assert_eq!(Rank::Two.value(), 2);
        assert_eq!(Rank::Ace.value(), 14);
    }

    #[test]
    fn test_rank_from_char() {
        assert_eq!(Rank::from_char('A'), Some(Rank::Ace));
        assert_eq!(Rank::from_char('a'), Some(Rank::Ace));
        assert_eq!(Rank::from_char('T'), Some(Rank::Ten));
        assert_eq!(Rank::from_char('2'), Some(Rank::Two));
        assert_eq!(Rank::from_char('X'), None);
    }

    #[test]
    fn test_suit_from_char() {
        assert_eq!(Suit::from_char('h'), Some(Suit::Hearts));
        assert_eq!(Suit::from_char('H'), Some(Suit::Hearts));
        assert_eq!(Suit::from_char('♥'), Some(Suit::Hearts));
        assert_eq!(Suit::from_char('x'), None);
    }

    #[test]
    fn test_card_index_roundtrip() {
        for i in 0..52 {
            let card = Card::from_index(i).unwrap();
            assert_eq!(card.to_index(), i);
        }
    }

    #[test]
    fn test_card_parse() {
        assert_eq!(Card::parse("Ah"), Ok(Card::new(Rank::Ace, Suit::Hearts)));
        assert_eq!(Card::parse("ah"), Ok(Card::new(Rank::Ace, Suit::Hearts)));
        assert_eq!(Card::parse("AH"), Ok(Card::new(Rank::Ace, Suit::Hearts)));
        assert_eq!(Card::parse("10c"), Ok(Card::new(Rank::Ten, Suit::Clubs)));
        assert_eq!(Card::parse("Tc"), Ok(Card::new(Rank::Ten, Suit::Clubs)));
    }

    #[test]
    fn test_parse_cards() {
        let cards = parse_cards("Ah Kh").unwrap();
        assert_eq!(cards.len(), 2);
        assert_eq!(cards[0], Card::new(Rank::Ace, Suit::Hearts));
        assert_eq!(cards[1], Card::new(Rank::King, Suit::Hearts));

        let cards = parse_cards("AhKh").unwrap();
        assert_eq!(cards.len(), 2);

        let cards = parse_cards("Ah, Kh").unwrap();
        assert_eq!(cards.len(), 2);
    }

    #[test]
    fn test_deck_basics() {
        let mut deck = Deck::new(Some(42));
        assert_eq!(deck.len(), 52);

        deck.shuffle();
        let dealt = deck.deal(5).unwrap();
        assert_eq!(dealt.len(), 5);
        assert_eq!(deck.len(), 47);

        deck.reset();
        assert_eq!(deck.len(), 52);
    }

    #[test]
    fn test_deck_remove() {
        let mut deck = Deck::new(Some(42));
        let ah = Card::new(Rank::Ace, Suit::Hearts);
        let kh = Card::new(Rank::King, Suit::Hearts);

        deck.remove(&[ah, kh]).unwrap();
        assert_eq!(deck.len(), 50);
        assert!(!deck.contains(ah));
        assert!(!deck.contains(kh));
    }

    #[test]
    fn test_full_deck_const() {
        assert_eq!(FULL_DECK.len(), 52);
        // Check first and last cards
        assert_eq!(FULL_DECK[0], Card::new(Rank::Two, Suit::Clubs));
        assert_eq!(FULL_DECK[51], Card::new(Rank::Ace, Suit::Spades));
    }
}
