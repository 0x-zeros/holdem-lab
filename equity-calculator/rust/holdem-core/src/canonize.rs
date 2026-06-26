//! Canonical hand representation for the 169 strategically distinct starting hands.
//!
//! In Hold'em, while there are C(52,2) = 1,326 possible hole card combinations,
//! they can be grouped into 169 strategically equivalent categories:
//! - 13 pairs (AA, KK, ..., 22)
//! - 78 suited hands (AKs, AQs, ..., 32s)
//! - 78 offsuit hands (AKo, AQo, ..., 32o)

use crate::card::{Card, Rank, Suit};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::fmt;
use std::str::FromStr;
use thiserror::Error;

/// A canonical (strategically equivalent) starting hand.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct CanonicalHand {
    /// The higher rank (or equal for pairs)
    pub high_rank: Rank,
    /// The lower rank (or equal for pairs)
    pub low_rank: Rank,
    /// Whether the hand is suited (false for pairs)
    pub suited: bool,
}

impl CanonicalHand {
    /// Create a new canonical hand
    ///
    /// # Panics
    /// Panics if high_rank < low_rank or if pair is marked as suited
    #[must_use]
    pub fn new(high_rank: Rank, low_rank: Rank, suited: bool) -> Self {
        assert!(
            high_rank >= low_rank,
            "high_rank must be >= low_rank"
        );
        assert!(
            !(high_rank == low_rank && suited),
            "pairs cannot be suited"
        );

        Self { high_rank, low_rank, suited }
    }

    /// Try to create a canonical hand, returning None if invalid
    #[must_use]
    pub fn try_new(high_rank: Rank, low_rank: Rank, suited: bool) -> Option<Self> {
        if high_rank < low_rank {
            return None;
        }
        if high_rank == low_rank && suited {
            return None;
        }
        Some(Self { high_rank, low_rank, suited })
    }

    /// Check if this is a pocket pair
    #[must_use]
    pub fn is_pair(&self) -> bool {
        self.high_rank == self.low_rank
    }

    /// Get the number of combinations for this hand type
    /// - Pairs: 6 combinations (4 choose 2)
    /// - Suited: 4 combinations (one per suit)
    /// - Offsuit: 12 combinations (4 * 3)
    #[must_use]
    pub fn num_combos(&self) -> usize {
        if self.is_pair() {
            6
        } else if self.suited {
            4
        } else {
            12
        }
    }

    /// Get the gap between ranks (0 for pairs, 1 for connectors like AK)
    #[must_use]
    pub fn gap(&self) -> u8 {
        self.high_rank.value() - self.low_rank.value()
    }

    /// Get notation string (e.g., "AKs", "QQ", "72o")
    #[must_use]
    pub fn notation(&self) -> String {
        if self.is_pair() {
            format!("{}{}", self.high_rank.to_char(), self.low_rank.to_char())
        } else {
            format!(
                "{}{}{}",
                self.high_rank.to_char(),
                self.low_rank.to_char(),
                if self.suited { 's' } else { 'o' }
            )
        }
    }

    /// Parse from notation string
    pub fn parse(s: &str) -> Result<Self, CanonizeError> {
        let s = s.trim();

        if s.len() < 2 || s.len() > 3 {
            return Err(CanonizeError::InvalidFormat(s.to_string()));
        }

        let chars: Vec<char> = s.chars().collect();
        let rank1 = Rank::from_char(chars[0]).ok_or(CanonizeError::InvalidRank(chars[0]))?;
        let rank2 = Rank::from_char(chars[1]).ok_or(CanonizeError::InvalidRank(chars[1]))?;

        // Normalize so high >= low
        let (high_rank, low_rank) = if rank1 >= rank2 {
            (rank1, rank2)
        } else {
            (rank2, rank1)
        };

        // Determine suitedness
        let suited = if chars.len() == 3 {
            match chars[2].to_ascii_lowercase() {
                's' => true,
                'o' => false,
                c => return Err(CanonizeError::InvalidSuited(c)),
            }
        } else {
            // 2-char format: must be a pair
            if high_rank != low_rank {
                return Err(CanonizeError::MissingSuited);
            }
            false
        };

        // Validate pair cannot be suited
        if high_rank == low_rank && suited {
            return Err(CanonizeError::PairCannotBeSuited);
        }

        Ok(Self { high_rank, low_rank, suited })
    }

    /// Get row index for 13x13 matrix display (0 = AA row)
    /// - Pairs: row = high_rank index (diagonal)
    /// - Suited: row = high_rank index (upper right triangle)
    /// - Offsuit: row = low_rank index (lower left triangle)
    #[must_use]
    pub fn matrix_row(&self) -> usize {
        if self.is_pair() || self.suited {
            14 - self.high_rank.value() as usize
        } else {
            // Offsuit: swap row/col to place in lower left
            14 - self.low_rank.value() as usize
        }
    }

    /// Get column index for 13x13 matrix display (0 = AA column)
    /// - Pairs: col = low_rank index (diagonal)
    /// - Suited: col = low_rank index (upper right triangle)
    /// - Offsuit: col = high_rank index (lower left triangle)
    #[must_use]
    pub fn matrix_col(&self) -> usize {
        if self.is_pair() || self.suited {
            14 - self.low_rank.value() as usize
        } else {
            // Offsuit: swap row/col to place in lower left
            14 - self.high_rank.value() as usize
        }
    }
}

impl fmt::Display for CanonicalHand {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.notation())
    }
}

impl FromStr for CanonicalHand {
    type Err = CanonizeError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::parse(s)
    }
}

/// Error when parsing or creating canonical hands
#[derive(Debug, Clone, Error, PartialEq, Eq)]
pub enum CanonizeError {
    #[error("invalid format: {0}")]
    InvalidFormat(String),
    #[error("invalid rank character: {0}")]
    InvalidRank(char),
    #[error("invalid suited character: {0}")]
    InvalidSuited(char),
    #[error("non-pair hands must specify suited (s) or offsuit (o)")]
    MissingSuited,
    #[error("pairs cannot be suited")]
    PairCannotBeSuited,
    #[error("invalid hole cards count")]
    InvalidCardCount,
}

/// Convert two hole cards to their canonical form
#[must_use]
pub fn canonize_hole_cards(cards: &[Card; 2]) -> CanonicalHand {
    let (high, low) = if cards[0].rank >= cards[1].rank {
        (cards[0], cards[1])
    } else {
        (cards[1], cards[0])
    };

    let suited = high.suit == low.suit && high.rank != low.rank;

    CanonicalHand {
        high_rank: high.rank,
        low_rank: low.rank,
        suited,
    }
}

/// Get all actual card combinations for a canonical hand
#[must_use]
pub fn get_all_combos(hand: &CanonicalHand) -> Vec<(Card, Card)> {
    let mut combos = Vec::new();

    if hand.is_pair() {
        // Pairs: all C(4,2) = 6 suit combinations
        for i in 0..4 {
            for j in (i + 1)..4 {
                combos.push((
                    Card::new(hand.high_rank, Suit::ALL[i]),
                    Card::new(hand.low_rank, Suit::ALL[j]),
                ));
            }
        }
    } else if hand.suited {
        // Suited: 4 combinations (same suit)
        for suit in Suit::ALL {
            combos.push((
                Card::new(hand.high_rank, suit),
                Card::new(hand.low_rank, suit),
            ));
        }
    } else {
        // Offsuit: 12 combinations (different suits)
        for suit1 in Suit::ALL {
            for suit2 in Suit::ALL {
                if suit1 != suit2 {
                    combos.push((
                        Card::new(hand.high_rank, suit1),
                        Card::new(hand.low_rank, suit2),
                    ));
                }
            }
        }
    }

    combos
}

/// Get combinations excluding dead cards
#[must_use]
pub fn get_combos_excluding(hand: &CanonicalHand, dead_cards: &[Card]) -> Vec<(Card, Card)> {
    let dead_set: HashSet<Card> = dead_cards.iter().copied().collect();

    get_all_combos(hand)
        .into_iter()
        .filter(|(c1, c2)| !dead_set.contains(c1) && !dead_set.contains(c2))
        .collect()
}

/// Get all 169 canonical starting hands
#[must_use]
pub fn get_all_canonical_hands() -> Vec<CanonicalHand> {
    let mut hands = Vec::with_capacity(169);

    // All ranks in descending order
    let ranks: Vec<Rank> = Rank::ALL.iter().copied().rev().collect();

    // Pairs (13)
    for &rank in &ranks {
        hands.push(CanonicalHand::new(rank, rank, false));
    }

    // Suited non-pairs (78)
    for (i, &high) in ranks.iter().enumerate() {
        for &low in &ranks[(i + 1)..] {
            hands.push(CanonicalHand::new(high, low, true));
        }
    }

    // Offsuit non-pairs (78)
    for (i, &high) in ranks.iter().enumerate() {
        for &low in &ranks[(i + 1)..] {
            hands.push(CanonicalHand::new(high, low, false));
        }
    }

    hands
}

/// Check if two specific hole cards are strategically equivalent
#[must_use]
pub fn are_strategically_equivalent(hand1: &[Card; 2], hand2: &[Card; 2]) -> bool {
    canonize_hole_cards(hand1) == canonize_hole_cards(hand2)
}

/// Extended info for UI display
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CanonicalHandInfo {
    pub notation: String,
    pub is_pair: bool,
    pub suited: bool,
    pub num_combos: usize,
    pub matrix_row: usize,
    pub matrix_col: usize,
    pub gap: u8,
}

impl From<&CanonicalHand> for CanonicalHandInfo {
    fn from(hand: &CanonicalHand) -> Self {
        Self {
            notation: hand.notation(),
            is_pair: hand.is_pair(),
            suited: hand.suited,
            num_combos: hand.num_combos(),
            matrix_row: hand.matrix_row(),
            matrix_col: hand.matrix_col(),
            gap: hand.gap(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::card::parse_cards;

    #[test]
    fn test_canonical_hand_pair() {
        let hand = CanonicalHand::new(Rank::Ace, Rank::Ace, false);
        assert!(hand.is_pair());
        assert_eq!(hand.num_combos(), 6);
        assert_eq!(hand.notation(), "AA");
        assert_eq!(hand.gap(), 0);
    }

    #[test]
    fn test_canonical_hand_suited() {
        let hand = CanonicalHand::new(Rank::Ace, Rank::King, true);
        assert!(!hand.is_pair());
        assert!(hand.suited);
        assert_eq!(hand.num_combos(), 4);
        assert_eq!(hand.notation(), "AKs");
        assert_eq!(hand.gap(), 1);
    }

    #[test]
    fn test_canonical_hand_offsuit() {
        let hand = CanonicalHand::new(Rank::Ace, Rank::King, false);
        assert!(!hand.is_pair());
        assert!(!hand.suited);
        assert_eq!(hand.num_combos(), 12);
        assert_eq!(hand.notation(), "AKo");
    }

    #[test]
    fn test_parse_canonical() {
        assert_eq!(
            CanonicalHand::parse("AA").unwrap(),
            CanonicalHand::new(Rank::Ace, Rank::Ace, false)
        );
        assert_eq!(
            CanonicalHand::parse("AKs").unwrap(),
            CanonicalHand::new(Rank::Ace, Rank::King, true)
        );
        assert_eq!(
            CanonicalHand::parse("KAo").unwrap(), // reversed
            CanonicalHand::new(Rank::Ace, Rank::King, false)
        );
        assert_eq!(
            CanonicalHand::parse("72o").unwrap(),
            CanonicalHand::new(Rank::Seven, Rank::Two, false)
        );
    }

    #[test]
    fn test_parse_errors() {
        assert!(matches!(
            CanonicalHand::parse("AAs"),
            Err(CanonizeError::PairCannotBeSuited)
        ));
        assert!(matches!(
            CanonicalHand::parse("AK"),
            Err(CanonizeError::MissingSuited)
        ));
        assert!(matches!(
            CanonicalHand::parse("XK"),
            Err(CanonizeError::InvalidRank('X'))
        ));
    }

    #[test]
    fn test_canonize_hole_cards() {
        let cards: [Card; 2] = parse_cards("Ah Kh").unwrap().try_into().unwrap();
        let hand = canonize_hole_cards(&cards);
        assert_eq!(hand.notation(), "AKs");

        let cards: [Card; 2] = parse_cards("Kd Ac").unwrap().try_into().unwrap();
        let hand = canonize_hole_cards(&cards);
        assert_eq!(hand.notation(), "AKo");

        let cards: [Card; 2] = parse_cards("Qs Qh").unwrap().try_into().unwrap();
        let hand = canonize_hole_cards(&cards);
        assert_eq!(hand.notation(), "QQ");
    }

    #[test]
    fn test_get_all_combos_pair() {
        let hand = CanonicalHand::new(Rank::Ace, Rank::Ace, false);
        let combos = get_all_combos(&hand);
        assert_eq!(combos.len(), 6);
    }

    #[test]
    fn test_get_all_combos_suited() {
        let hand = CanonicalHand::new(Rank::Ace, Rank::King, true);
        let combos = get_all_combos(&hand);
        assert_eq!(combos.len(), 4);
        // All combos should be same suit
        for (c1, c2) in &combos {
            assert_eq!(c1.suit, c2.suit);
        }
    }

    #[test]
    fn test_get_all_combos_offsuit() {
        let hand = CanonicalHand::new(Rank::Ace, Rank::King, false);
        let combos = get_all_combos(&hand);
        assert_eq!(combos.len(), 12);
        // All combos should be different suits
        for (c1, c2) in &combos {
            assert_ne!(c1.suit, c2.suit);
        }
    }

    #[test]
    fn test_get_combos_excluding() {
        let hand = CanonicalHand::new(Rank::Ace, Rank::Ace, false);
        let dead = parse_cards("Ah").unwrap();
        let combos = get_combos_excluding(&hand, &dead);
        // Originally 6, now 3 (all combos with Ah removed)
        assert_eq!(combos.len(), 3);
    }

    #[test]
    fn test_get_all_canonical_hands() {
        let hands = get_all_canonical_hands();
        assert_eq!(hands.len(), 169);

        // Count by type
        let pairs = hands.iter().filter(|h| h.is_pair()).count();
        let suited = hands.iter().filter(|h| h.suited).count();
        let offsuit = hands.iter().filter(|h| !h.is_pair() && !h.suited).count();

        assert_eq!(pairs, 13);
        assert_eq!(suited, 78);
        assert_eq!(offsuit, 78);
    }

    #[test]
    fn test_matrix_positions() {
        // Pairs on diagonal
        let aa = CanonicalHand::new(Rank::Ace, Rank::Ace, false);
        assert_eq!(aa.matrix_row(), 0);
        assert_eq!(aa.matrix_col(), 0);

        let two_two = CanonicalHand::new(Rank::Two, Rank::Two, false);
        assert_eq!(two_two.matrix_row(), 12);
        assert_eq!(two_two.matrix_col(), 12);

        // Suited hands in upper right triangle
        let aks = CanonicalHand::new(Rank::Ace, Rank::King, true);
        assert_eq!(aks.matrix_row(), 0);
        assert_eq!(aks.matrix_col(), 1);

        // Offsuit hands in lower left triangle (row/col swapped)
        let ako = CanonicalHand::new(Rank::Ace, Rank::King, false);
        assert_eq!(ako.matrix_row(), 1);  // low_rank (K) determines row
        assert_eq!(ako.matrix_col(), 0);  // high_rank (A) determines col

        // Another offsuit test
        let t9o = CanonicalHand::new(Rank::Ten, Rank::Nine, false);
        assert_eq!(t9o.matrix_row(), 5);  // 9 is at index 5
        assert_eq!(t9o.matrix_col(), 4);  // T is at index 4
    }

    #[test]
    fn test_strategically_equivalent() {
        let hand1: [Card; 2] = parse_cards("Ah Kh").unwrap().try_into().unwrap();
        let hand2: [Card; 2] = parse_cards("As Ks").unwrap().try_into().unwrap();
        assert!(are_strategically_equivalent(&hand1, &hand2));

        let hand3: [Card; 2] = parse_cards("Ah Kc").unwrap().try_into().unwrap();
        assert!(!are_strategically_equivalent(&hand1, &hand3)); // suited vs offsuit
    }

    #[test]
    fn test_total_combos() {
        let hands = get_all_canonical_hands();
        let total: usize = hands.iter().map(|h| h.num_combos()).sum();
        assert_eq!(total, 1326); // C(52, 2) = 1326
    }
}
