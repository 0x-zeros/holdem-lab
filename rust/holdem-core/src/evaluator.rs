//! Hand evaluation for Texas Hold'em poker.
//!
//! Evaluates 5-7 card hands and determines the best 5-card combination.

use crate::card::{Card, Rank};
use itertools::Itertools;
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::HashMap;
use std::fmt;

/// Poker hand types in ascending strength order
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum HandType {
    HighCard = 0,
    OnePair = 1,
    TwoPair = 2,
    ThreeOfAKind = 3,
    Straight = 4,
    Flush = 5,
    FullHouse = 6,
    FourOfAKind = 7,
    StraightFlush = 8,
    RoyalFlush = 9,
}

impl HandType {
    /// Get human-readable name
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            HandType::HighCard => "High Card",
            HandType::OnePair => "One Pair",
            HandType::TwoPair => "Two Pair",
            HandType::ThreeOfAKind => "Three of a Kind",
            HandType::Straight => "Straight",
            HandType::Flush => "Flush",
            HandType::FullHouse => "Full House",
            HandType::FourOfAKind => "Four of a Kind",
            HandType::StraightFlush => "Straight Flush",
            HandType::RoyalFlush => "Royal Flush",
        }
    }
}

impl fmt::Display for HandType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name())
    }
}

/// A hand ranking that can be compared to determine winners.
///
/// Comparison uses lexicographic order: hand_type -> primary_ranks -> kickers
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct HandRank {
    pub hand_type: HandType,
    /// Primary ranks that define the hand (e.g., pair rank, trips rank)
    pub primary_ranks: Vec<u8>,
    /// Kicker cards for tiebreaking
    pub kickers: Vec<u8>,
}

impl HandRank {
    /// Create a new hand rank
    #[must_use]
    pub fn new(hand_type: HandType, primary_ranks: Vec<u8>, kickers: Vec<u8>) -> Self {
        Self { hand_type, primary_ranks, kickers }
    }

    /// Generate human-readable description
    #[must_use]
    pub fn describe(&self) -> String {
        let rank_name = |r: u8| -> &str {
            match Rank::from_value(r) {
                Some(Rank::Two) => "Twos",
                Some(Rank::Three) => "Threes",
                Some(Rank::Four) => "Fours",
                Some(Rank::Five) => "Fives",
                Some(Rank::Six) => "Sixes",
                Some(Rank::Seven) => "Sevens",
                Some(Rank::Eight) => "Eights",
                Some(Rank::Nine) => "Nines",
                Some(Rank::Ten) => "Tens",
                Some(Rank::Jack) => "Jacks",
                Some(Rank::Queen) => "Queens",
                Some(Rank::King) => "Kings",
                Some(Rank::Ace) => "Aces",
                None => "Unknown",
            }
        };

        let rank_single = |r: u8| -> String {
            Rank::from_value(r).map_or("?".to_string(), |rank| rank.to_char().to_string())
        };

        match self.hand_type {
            HandType::RoyalFlush => "Royal Flush".to_string(),
            HandType::StraightFlush => {
                format!("Straight Flush, {} high", rank_single(self.primary_ranks[0]))
            }
            HandType::FourOfAKind => {
                format!("Four of a Kind, {}", rank_name(self.primary_ranks[0]))
            }
            HandType::FullHouse => {
                format!(
                    "Full House, {} full of {}",
                    rank_name(self.primary_ranks[0]),
                    rank_name(self.primary_ranks[1])
                )
            }
            HandType::Flush => {
                format!("Flush, {} high", rank_single(self.primary_ranks[0]))
            }
            HandType::Straight => {
                format!("Straight, {} high", rank_single(self.primary_ranks[0]))
            }
            HandType::ThreeOfAKind => {
                format!("Three of a Kind, {}", rank_name(self.primary_ranks[0]))
            }
            HandType::TwoPair => {
                format!(
                    "Two Pair, {} and {}",
                    rank_name(self.primary_ranks[0]),
                    rank_name(self.primary_ranks[1])
                )
            }
            HandType::OnePair => {
                format!("Pair of {}", rank_name(self.primary_ranks[0]))
            }
            HandType::HighCard => {
                format!("{} high", rank_single(self.primary_ranks[0]))
            }
        }
    }
}

impl PartialOrd for HandRank {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for HandRank {
    fn cmp(&self, other: &Self) -> Ordering {
        // Compare hand type first
        match self.hand_type.cmp(&other.hand_type) {
            Ordering::Equal => {}
            ord => return ord,
        }

        // Then compare primary ranks
        match self.primary_ranks.cmp(&other.primary_ranks) {
            Ordering::Equal => {}
            ord => return ord,
        }

        // Finally compare kickers
        self.kickers.cmp(&other.kickers)
    }
}

impl fmt::Display for HandRank {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.describe())
    }
}

/// Check if all cards are the same suit
fn is_flush(cards: &[Card; 5]) -> bool {
    let suit = cards[0].suit;
    cards.iter().all(|c| c.suit == suit)
}

/// Check for straight and return high card (handles A-2-3-4-5 wheel)
fn check_straight(ranks: &[u8; 5]) -> Option<u8> {
    // Ranks should be sorted descending
    // Check for regular straight
    if ranks[0] - ranks[4] == 4
        && ranks[0] - ranks[1] == 1
        && ranks[1] - ranks[2] == 1
        && ranks[2] - ranks[3] == 1
        && ranks[3] - ranks[4] == 1
    {
        return Some(ranks[0]);
    }

    // Check for wheel (A-2-3-4-5) - Ace is 14, 5 is 5
    // Sorted descending: [14, 5, 4, 3, 2]
    if ranks == &[14, 5, 4, 3, 2] {
        return Some(5); // Wheel's high card is 5
    }

    None
}

/// Evaluate exactly 5 cards
#[must_use]
pub fn evaluate_five(cards: &[Card; 5]) -> HandRank {
    // Sort ranks descending
    let mut ranks: [u8; 5] = cards.map(|c| c.rank.value());
    ranks.sort_unstable_by(|a, b| b.cmp(a));

    let flush = is_flush(cards);
    let straight_high = check_straight(&ranks);

    // Straight flush / Royal flush
    if flush && straight_high.is_some() {
        let high = straight_high.unwrap();
        if high == 14 {
            return HandRank::new(HandType::RoyalFlush, vec![14], vec![]);
        }
        return HandRank::new(HandType::StraightFlush, vec![high], vec![]);
    }

    // Count rank frequencies
    let mut freq: HashMap<u8, usize> = HashMap::new();
    for &r in &ranks {
        *freq.entry(r).or_insert(0) += 1;
    }

    // Sort by (count desc, rank desc)
    let mut freq_vec: Vec<(u8, usize)> = freq.into_iter().collect();
    freq_vec.sort_by(|a, b| match b.1.cmp(&a.1) {
        Ordering::Equal => b.0.cmp(&a.0),
        ord => ord,
    });

    let counts: Vec<usize> = freq_vec.iter().map(|(_, c)| *c).collect();
    let rank_values: Vec<u8> = freq_vec.iter().map(|(r, _)| *r).collect();

    // Four of a kind: [4, 1]
    if counts == [4, 1] {
        return HandRank::new(
            HandType::FourOfAKind,
            vec![rank_values[0]],
            vec![rank_values[1]],
        );
    }

    // Full house: [3, 2]
    if counts == [3, 2] {
        return HandRank::new(
            HandType::FullHouse,
            vec![rank_values[0], rank_values[1]],
            vec![],
        );
    }

    // Flush
    if flush {
        return HandRank::new(HandType::Flush, ranks.to_vec(), vec![]);
    }

    // Straight
    if let Some(high) = straight_high {
        return HandRank::new(HandType::Straight, vec![high], vec![]);
    }

    // Three of a kind: [3, 1, 1]
    if counts == [3, 1, 1] {
        return HandRank::new(
            HandType::ThreeOfAKind,
            vec![rank_values[0]],
            vec![rank_values[1], rank_values[2]],
        );
    }

    // Two pair: [2, 2, 1]
    if counts == [2, 2, 1] {
        return HandRank::new(
            HandType::TwoPair,
            vec![rank_values[0], rank_values[1]],
            vec![rank_values[2]],
        );
    }

    // One pair: [2, 1, 1, 1]
    if counts == [2, 1, 1, 1] {
        return HandRank::new(
            HandType::OnePair,
            vec![rank_values[0]],
            vec![rank_values[1], rank_values[2], rank_values[3]],
        );
    }

    // High card: [1, 1, 1, 1, 1]
    HandRank::new(HandType::HighCard, ranks.to_vec(), vec![])
}

/// Evaluate 5-7 cards and return the best 5-card hand
#[must_use]
pub fn evaluate_hand(cards: &[Card]) -> HandRank {
    assert!(
        (5..=7).contains(&cards.len()),
        "evaluate_hand requires 5-7 cards, got {}",
        cards.len()
    );

    if cards.len() == 5 {
        let arr: [Card; 5] = cards.try_into().unwrap();
        return evaluate_five(&arr);
    }

    // Enumerate all C(n, 5) combinations and find the best
    cards
        .iter()
        .copied()
        .combinations(5)
        .map(|combo| {
            let arr: [Card; 5] = combo.try_into().unwrap();
            evaluate_five(&arr)
        })
        .max()
        .unwrap()
}

/// Find the indices of players with the best hand (handles ties)
#[must_use]
pub fn find_winners(hands: &[Vec<Card>]) -> Vec<usize> {
    if hands.is_empty() {
        return vec![];
    }

    let ranks: Vec<HandRank> = hands.iter().map(|h| evaluate_hand(h)).collect();

    let best = ranks.iter().max().unwrap();

    ranks
        .iter()
        .enumerate()
        .filter_map(|(i, r)| if r == best { Some(i) } else { None })
        .collect()
}

/// Compare two hands directly
/// Returns: 1 if hand1 wins, -1 if hand2 wins, 0 if tie
#[must_use]
pub fn compare_hands(hand1: &[Card], hand2: &[Card]) -> i8 {
    let rank1 = evaluate_hand(hand1);
    let rank2 = evaluate_hand(hand2);

    match rank1.cmp(&rank2) {
        Ordering::Greater => 1,
        Ordering::Less => -1,
        Ordering::Equal => 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::card::parse_cards;

    fn cards(s: &str) -> Vec<Card> {
        parse_cards(s).unwrap()
    }

    fn cards5(s: &str) -> [Card; 5] {
        cards(s).try_into().unwrap()
    }

    #[test]
    fn test_royal_flush() {
        let hand = cards5("Ah Kh Qh Jh Th");
        let rank = evaluate_five(&hand);
        assert_eq!(rank.hand_type, HandType::RoyalFlush);
    }

    #[test]
    fn test_straight_flush() {
        let hand = cards5("9h 8h 7h 6h 5h");
        let rank = evaluate_five(&hand);
        assert_eq!(rank.hand_type, HandType::StraightFlush);
        assert_eq!(rank.primary_ranks, vec![9]);
    }

    #[test]
    fn test_wheel_straight_flush() {
        let hand = cards5("5h 4h 3h 2h Ah");
        let rank = evaluate_five(&hand);
        assert_eq!(rank.hand_type, HandType::StraightFlush);
        assert_eq!(rank.primary_ranks, vec![5]); // Wheel high card is 5
    }

    #[test]
    fn test_four_of_a_kind() {
        let hand = cards5("Ks Kh Kd Kc 2h");
        let rank = evaluate_five(&hand);
        assert_eq!(rank.hand_type, HandType::FourOfAKind);
        assert_eq!(rank.primary_ranks, vec![13]); // Kings
    }

    #[test]
    fn test_full_house() {
        let hand = cards5("Ks Kh Kd 2c 2h");
        let rank = evaluate_five(&hand);
        assert_eq!(rank.hand_type, HandType::FullHouse);
        assert_eq!(rank.primary_ranks, vec![13, 2]);
    }

    #[test]
    fn test_flush() {
        let hand = cards5("Ah Kh 9h 5h 2h");
        let rank = evaluate_five(&hand);
        assert_eq!(rank.hand_type, HandType::Flush);
        assert_eq!(rank.primary_ranks, vec![14, 13, 9, 5, 2]);
    }

    #[test]
    fn test_straight() {
        let hand = cards5("9h 8c 7d 6s 5h");
        let rank = evaluate_five(&hand);
        assert_eq!(rank.hand_type, HandType::Straight);
        assert_eq!(rank.primary_ranks, vec![9]);
    }

    #[test]
    fn test_wheel_straight() {
        let hand = cards5("5h 4c 3d 2s Ah");
        let rank = evaluate_five(&hand);
        assert_eq!(rank.hand_type, HandType::Straight);
        assert_eq!(rank.primary_ranks, vec![5]); // Wheel high card is 5
    }

    #[test]
    fn test_three_of_a_kind() {
        let hand = cards5("Ks Kh Kd 7c 2h");
        let rank = evaluate_five(&hand);
        assert_eq!(rank.hand_type, HandType::ThreeOfAKind);
        assert_eq!(rank.primary_ranks, vec![13]);
        assert_eq!(rank.kickers, vec![7, 2]);
    }

    #[test]
    fn test_two_pair() {
        let hand = cards5("Ks Kh 7d 7c 2h");
        let rank = evaluate_five(&hand);
        assert_eq!(rank.hand_type, HandType::TwoPair);
        assert_eq!(rank.primary_ranks, vec![13, 7]);
        assert_eq!(rank.kickers, vec![2]);
    }

    #[test]
    fn test_one_pair() {
        let hand = cards5("Ks Kh 9d 7c 2h");
        let rank = evaluate_five(&hand);
        assert_eq!(rank.hand_type, HandType::OnePair);
        assert_eq!(rank.primary_ranks, vec![13]);
        assert_eq!(rank.kickers, vec![9, 7, 2]);
    }

    #[test]
    fn test_high_card() {
        let hand = cards5("Ah Kc 9d 7s 2h");
        let rank = evaluate_five(&hand);
        assert_eq!(rank.hand_type, HandType::HighCard);
        assert_eq!(rank.primary_ranks, vec![14, 13, 9, 7, 2]);
    }

    #[test]
    fn test_evaluate_seven_cards() {
        // Best 5 from 7 should be a flush
        let hand = cards("Ah Kh 9h 5h 2h 3c 4d");
        let rank = evaluate_hand(&hand);
        assert_eq!(rank.hand_type, HandType::Flush);
    }

    #[test]
    fn test_find_winners() {
        let hand1 = cards("Ah Kh Qh Jh Th"); // Royal flush
        let hand2 = cards("9h 8h 7h 6h 5h"); // Straight flush
        let hand3 = cards("Ks Kh Kd Kc 2h"); // Four of a kind

        let winners = find_winners(&[hand1, hand2, hand3]);
        assert_eq!(winners, vec![0]); // Royal flush wins
    }

    #[test]
    fn test_find_winners_tie() {
        let hand1 = cards("Ah Kd Qc Jh Ts"); // Broadway straight
        let hand2 = cards("Ac Ks Qh Jd Tc"); // Same broadway straight

        let winners = find_winners(&[hand1, hand2]);
        assert_eq!(winners, vec![0, 1]); // Tie
    }

    #[test]
    fn test_compare_hands() {
        let hand1 = cards("Ah Kh Qh Jh Th");
        let hand2 = cards("9h 8h 7h 6h 5h");

        assert_eq!(compare_hands(&hand1, &hand2), 1); // Royal > Straight flush
        assert_eq!(compare_hands(&hand2, &hand1), -1);
        assert_eq!(compare_hands(&hand1, &hand1), 0); // Tie
    }

    #[test]
    fn test_hand_rank_ordering() {
        let royal = evaluate_five(&cards5("Ah Kh Qh Jh Th"));
        let straight_flush = evaluate_five(&cards5("9h 8h 7h 6h 5h"));
        let four_kind = evaluate_five(&cards5("Ks Kh Kd Kc 2h"));
        let full_house = evaluate_five(&cards5("Ks Kh Kd 2c 2h"));
        let flush = evaluate_five(&cards5("Ah Kh 9h 5h 2h"));
        let straight = evaluate_five(&cards5("9h 8c 7d 6s 5h"));
        let trips = evaluate_five(&cards5("Ks Kh Kd 7c 2h"));
        let two_pair = evaluate_five(&cards5("Ks Kh 7d 7c 2h"));
        let pair = evaluate_five(&cards5("Ks Kh 9d 7c 2h"));
        let high = evaluate_five(&cards5("Ah Kc 9d 7s 2h"));

        assert!(royal > straight_flush);
        assert!(straight_flush > four_kind);
        assert!(four_kind > full_house);
        assert!(full_house > flush);
        assert!(flush > straight);
        assert!(straight > trips);
        assert!(trips > two_pair);
        assert!(two_pair > pair);
        assert!(pair > high);
    }

    #[test]
    fn test_kicker_comparison() {
        // Same pair, different kickers
        let pair_with_a = evaluate_five(&cards5("Ks Kh Ad 7c 2h"));
        let pair_with_q = evaluate_five(&cards5("Kd Kc Qd 7s 2d"));

        assert!(pair_with_a > pair_with_q);
    }
}
