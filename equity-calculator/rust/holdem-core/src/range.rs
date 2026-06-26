//! Range distribution for equity calculation.
//!
//! Implements pokerstove-style range enumeration for accurate equity calculation
//! when players have range-based hands rather than specific cards.

use crate::canonize::{get_combos_excluding, CanonicalHand, CanonizeError};
use crate::card::Card;
use std::collections::HashSet;

/// A player's hand distribution representing all possible hole card combinations.
///
/// Similar to pokerstove's CardDistribution, this allows calculating equity
/// against ranges by enumerating all valid combinations.
#[derive(Clone, Debug)]
pub struct CardDistribution {
    /// All possible 2-card combinations in this distribution
    hands: Vec<(Card, Card)>,
    /// Weight for each combination (defaults to 1.0)
    weights: Vec<f64>,
}

impl CardDistribution {
    /// Create a new empty distribution
    #[must_use]
    pub fn new() -> Self {
        Self {
            hands: Vec::new(),
            weights: Vec::new(),
        }
    }

    /// Create a distribution from a single specific hand
    #[must_use]
    pub fn from_hand(c1: Card, c2: Card) -> Self {
        Self {
            hands: vec![(c1, c2)],
            weights: vec![1.0],
        }
    }

    /// Parse a range from canonical hand strings (e.g., ["AA", "AKs", "QQ"])
    ///
    /// Excludes any combos that use cards in the `excluded` set.
    pub fn from_range(range: &[String], excluded: &[Card]) -> Result<Self, RangeError> {
        if range.is_empty() {
            return Err(RangeError::EmptyRange);
        }

        let mut hands = Vec::new();
        let mut weights = Vec::new();

        for notation in range {
            let canonical = CanonicalHand::parse(notation)
                .map_err(|e| RangeError::InvalidHand(notation.clone(), e))?;

            let combos = get_combos_excluding(&canonical, excluded);
            for combo in combos {
                hands.push(combo);
                weights.push(1.0);
            }
        }

        if hands.is_empty() {
            return Err(RangeError::NoCombosAvailable);
        }

        Ok(Self { hands, weights })
    }

    /// Get all hands in this distribution
    #[must_use]
    pub fn hands(&self) -> &[(Card, Card)] {
        &self.hands
    }

    /// Get the number of combinations in this distribution
    #[must_use]
    pub fn len(&self) -> usize {
        self.hands.len()
    }

    /// Check if the distribution is empty
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.hands.is_empty()
    }

    /// Get the weight for a specific hand index
    #[must_use]
    pub fn weight(&self, index: usize) -> f64 {
        self.weights.get(index).copied().unwrap_or(1.0)
    }

    /// Get hand at index
    #[must_use]
    pub fn get(&self, index: usize) -> Option<(Card, Card)> {
        self.hands.get(index).copied()
    }

    /// Filter out hands that conflict with the given cards
    #[must_use]
    pub fn filter_excluding(&self, excluded: &HashSet<Card>) -> Self {
        let mut hands = Vec::new();
        let mut weights = Vec::new();

        for (i, &(c1, c2)) in self.hands.iter().enumerate() {
            if !excluded.contains(&c1) && !excluded.contains(&c2) {
                hands.push((c1, c2));
                weights.push(self.weights[i]);
            }
        }

        Self { hands, weights }
    }
}

impl Default for CardDistribution {
    fn default() -> Self {
        Self::new()
    }
}

/// Odometer iterator for Cartesian product of multiple ranges.
///
/// Iterates through all combinations of indices, one from each range.
/// Similar to pokerstove's Odometer class.
#[derive(Clone, Debug)]
pub struct Odometer {
    /// Size of each range
    extents: Vec<usize>,
    /// Current indices
    current: Vec<usize>,
    /// Whether we've started iterating
    started: bool,
    /// Whether we've exhausted all combinations
    exhausted: bool,
}

impl Odometer {
    /// Create a new odometer for ranges with the given sizes
    ///
    /// # Arguments
    /// * `extents` - The size of each range. Empty extents or any zero-size extent
    ///               will result in an odometer that yields no values.
    #[must_use]
    pub fn new(extents: Vec<usize>) -> Self {
        let exhausted = extents.is_empty() || extents.iter().any(|&e| e == 0);
        Self {
            current: vec![0; extents.len()],
            extents,
            started: false,
            exhausted,
        }
    }

    /// Get the current indices
    #[must_use]
    pub fn indices(&self) -> &[usize] {
        &self.current
    }

    /// Get the total number of combinations
    #[must_use]
    pub fn total_combinations(&self) -> usize {
        if self.extents.is_empty() {
            return 0;
        }
        self.extents.iter().product()
    }
}

impl Iterator for Odometer {
    type Item = Vec<usize>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.exhausted {
            return None;
        }

        if !self.started {
            self.started = true;
            return Some(self.current.clone());
        }

        // Increment from the rightmost position
        for i in (0..self.extents.len()).rev() {
            self.current[i] += 1;
            if self.current[i] < self.extents[i] {
                return Some(self.current.clone());
            }
            self.current[i] = 0;
        }

        // We've wrapped around completely
        self.exhausted = true;
        None
    }
}

/// Check if a set of hands has any card conflicts
#[must_use]
pub fn hands_are_disjoint(hands: &[(Card, Card)]) -> bool {
    let mut seen = HashSet::new();
    for &(c1, c2) in hands {
        if !seen.insert(c1) || !seen.insert(c2) {
            return false;
        }
    }
    true
}

/// Collect all cards from a set of hands
#[must_use]
pub fn collect_cards(hands: &[(Card, Card)]) -> HashSet<Card> {
    let mut cards = HashSet::new();
    for &(c1, c2) in hands {
        cards.insert(c1);
        cards.insert(c2);
    }
    cards
}

/// Error type for range operations
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RangeError {
    /// The range array is empty
    EmptyRange,
    /// Invalid hand notation in range
    InvalidHand(String, CanonizeError),
    /// No valid combos after excluding dead cards
    NoCombosAvailable,
}

impl std::fmt::Display for RangeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RangeError::EmptyRange => write!(f, "empty range"),
            RangeError::InvalidHand(notation, e) => {
                write!(f, "invalid hand '{}': {}", notation, e)
            }
            RangeError::NoCombosAvailable => {
                write!(f, "no valid combos available after excluding dead cards")
            }
        }
    }
}

impl std::error::Error for RangeError {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::card::{Card, Rank, Suit};

    #[test]
    fn test_card_distribution_from_range() {
        let dist = CardDistribution::from_range(&["AA".to_string()], &[]).unwrap();
        assert_eq!(dist.len(), 6); // 6 combos for AA

        let dist = CardDistribution::from_range(&["AKs".to_string()], &[]).unwrap();
        assert_eq!(dist.len(), 4); // 4 combos for AKs

        let dist = CardDistribution::from_range(&["AKo".to_string()], &[]).unwrap();
        assert_eq!(dist.len(), 12); // 12 combos for AKo
    }

    #[test]
    fn test_card_distribution_with_exclusions() {
        let excluded = vec![Card::new(Rank::Ace, Suit::Hearts)];
        let dist = CardDistribution::from_range(&["AA".to_string()], &excluded).unwrap();
        // AA has 6 combos, but 3 use Ah, so 3 remain
        assert_eq!(dist.len(), 3);
    }

    #[test]
    fn test_card_distribution_multiple_hands() {
        let dist =
            CardDistribution::from_range(&["AA".to_string(), "KK".to_string()], &[]).unwrap();
        assert_eq!(dist.len(), 12); // 6 + 6
    }

    #[test]
    fn test_odometer_basic() {
        let odom = Odometer::new(vec![2, 3]);
        let combos: Vec<_> = odom.collect();
        assert_eq!(combos.len(), 6);
        assert_eq!(combos[0], vec![0, 0]);
        assert_eq!(combos[1], vec![0, 1]);
        assert_eq!(combos[2], vec![0, 2]);
        assert_eq!(combos[3], vec![1, 0]);
        assert_eq!(combos[4], vec![1, 1]);
        assert_eq!(combos[5], vec![1, 2]);
    }

    #[test]
    fn test_odometer_single() {
        let odom = Odometer::new(vec![3]);
        let combos: Vec<_> = odom.collect();
        assert_eq!(combos.len(), 3);
    }

    #[test]
    fn test_odometer_empty() {
        let odom = Odometer::new(vec![]);
        let combos: Vec<_> = odom.collect();
        assert!(combos.is_empty());
    }

    #[test]
    fn test_odometer_zero_extent() {
        let odom = Odometer::new(vec![2, 0, 3]);
        let combos: Vec<_> = odom.collect();
        assert!(combos.is_empty());
    }

    #[test]
    fn test_hands_are_disjoint() {
        let ah = Card::new(Rank::Ace, Suit::Hearts);
        let as_ = Card::new(Rank::Ace, Suit::Spades);
        let kh = Card::new(Rank::King, Suit::Hearts);
        let ks = Card::new(Rank::King, Suit::Spades);

        // No conflict
        assert!(hands_are_disjoint(&[(ah, as_), (kh, ks)]));

        // Conflict - Ah used twice
        assert!(!hands_are_disjoint(&[(ah, as_), (ah, kh)]));
    }

    #[test]
    fn test_filter_excluding() {
        let dist = CardDistribution::from_range(&["AA".to_string()], &[]).unwrap();
        assert_eq!(dist.len(), 6);

        let ah = Card::new(Rank::Ace, Suit::Hearts);
        let excluded: HashSet<Card> = [ah].into_iter().collect();
        let filtered = dist.filter_excluding(&excluded);
        assert_eq!(filtered.len(), 3);
    }
}
