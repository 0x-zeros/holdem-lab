//! Draw analysis for flush and straight draws.
//!
//! Analyzes hole cards + board to identify drawing hands and their outs.

use crate::card::{Card, Rank, Suit, FULL_DECK};
use crate::evaluator::{evaluate_hand, HandType};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

/// Types of draws
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DrawType {
    /// 4 cards to a flush (9 outs)
    FlushDraw,
    /// 3 cards to a flush (backdoor, need 2 running cards)
    BackdoorFlush,
    /// Open-ended straight draw (8 outs)
    OpenEnded,
    /// Gutshot straight draw (4 outs)
    Gutshot,
    /// Double gutshot (8 outs)
    DoubleGutshot,
    /// 3 connected cards (backdoor straight)
    BackdoorStraight,
}

impl DrawType {
    /// Get typical number of outs for this draw type
    #[must_use]
    pub const fn typical_outs(self) -> u8 {
        match self {
            DrawType::FlushDraw => 9,
            DrawType::BackdoorFlush => 0, // Needs 2 cards
            DrawType::OpenEnded => 8,
            DrawType::Gutshot => 4,
            DrawType::DoubleGutshot => 8,
            DrawType::BackdoorStraight => 0, // Needs 2 cards
        }
    }
}

/// A flush draw
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FlushDraw {
    /// The suit of the draw
    pub suit: Suit,
    /// Number of cards held in this suit
    pub cards_held: usize,
    /// Specific out cards
    pub outs: Vec<Card>,
    /// Whether hero holds the nut flush card (Ace of this suit)
    pub is_nut: bool,
}

impl FlushDraw {
    /// Get number of outs
    #[must_use]
    pub fn out_count(&self) -> usize {
        self.outs.len()
    }

    /// Get draw type
    #[must_use]
    pub fn draw_type(&self) -> DrawType {
        if self.cards_held >= 4 {
            DrawType::FlushDraw
        } else {
            DrawType::BackdoorFlush
        }
    }
}

/// A straight draw
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StraightDraw {
    /// Type of straight draw
    pub draw_type: DrawType,
    /// Ranks needed to complete
    pub needed_ranks: Vec<u8>,
    /// Specific out cards
    pub outs: Vec<Card>,
    /// High card of the completed straight
    pub high_card: u8,
    /// Whether this would make the nut straight
    pub is_nut: bool,
}

impl StraightDraw {
    /// Get number of outs
    #[must_use]
    pub fn out_count(&self) -> usize {
        self.outs.len()
    }
}

/// Complete draw analysis result
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DrawAnalysis {
    /// The hole cards analyzed
    pub hole_cards: Vec<Card>,
    /// The board cards
    pub board: Vec<Card>,
    /// Whether player already has a flush
    pub has_flush: bool,
    /// Whether player already has a straight
    pub has_straight: bool,
    /// Flush draws found
    pub flush_draws: Vec<FlushDraw>,
    /// Straight draws found
    pub straight_draws: Vec<StraightDraw>,
    /// Total unique outs (not double-counted)
    pub total_outs: usize,
    /// All out cards combined
    pub all_outs: Vec<Card>,
}

impl DrawAnalysis {
    /// Check if any draw exists
    #[must_use]
    pub fn has_draw(&self) -> bool {
        !self.flush_draws.is_empty() || !self.straight_draws.is_empty()
    }

    /// Check if this is a combo draw (flush + straight)
    #[must_use]
    pub fn is_combo_draw(&self) -> bool {
        let has_flush_draw = self.flush_draws.iter().any(|d| d.draw_type() == DrawType::FlushDraw);
        let has_straight_draw = self.straight_draws.iter().any(|d| {
            matches!(
                d.draw_type,
                DrawType::OpenEnded | DrawType::Gutshot | DrawType::DoubleGutshot
            )
        });
        has_flush_draw && has_straight_draw
    }
}

/// Build a 14-bit rank mask for straight detection
/// Bit 0 = Ace (low), Bits 1-13 = 2-A (high)
fn build_rank_mask(cards: &[Card]) -> u16 {
    let mut mask: u16 = 0;

    for card in cards {
        let rank = card.rank.value();
        // Set bit for rank (2=bit1, ..., A=bit13)
        mask |= 1 << (rank - 1);

        // Also set bit 0 for Ace (for wheel detection)
        if rank == 14 {
            mask |= 1;
        }
    }

    mask
}

/// Count bits set in a 16-bit value
#[allow(clippy::cast_possible_truncation)]
fn count_bits(n: u16) -> u8 {
    n.count_ones() as u8
}

/// Analyze flush draws
fn analyze_flush_draws(
    hole_cards: &[Card],
    board: &[Card],
    dead_cards: &HashSet<Card>,
) -> Vec<FlushDraw> {
    let mut draws = Vec::new();
    let all_cards: Vec<Card> = hole_cards.iter().chain(board.iter()).copied().collect();

    // Group by suit
    let mut by_suit: HashMap<Suit, Vec<Card>> = HashMap::new();
    for &card in &all_cards {
        by_suit.entry(card.suit).or_default().push(card);
    }

    // Check each suit
    for (suit, cards) in by_suit {
        let count = cards.len();

        // Need at least 3 for backdoor or 4 for regular flush draw
        if count < 3 {
            continue;
        }

        // Backdoor only valid on flop (3 board cards)
        if count == 3 && board.len() != 3 {
            continue;
        }

        // Find outs (remaining cards of this suit)
        let outs: Vec<Card> = FULL_DECK
            .iter()
            .filter(|c| c.suit == suit && !all_cards.contains(c) && !dead_cards.contains(c))
            .copied()
            .collect();

        // Check if hero has the Ace of this suit (nut flush card)
        let is_nut = hole_cards.iter().any(|c| c.suit == suit && c.rank == Rank::Ace);

        draws.push(FlushDraw {
            suit,
            cards_held: count,
            outs,
            is_nut,
        });
    }

    draws
}

/// Analyze straight draws using bitmask
fn analyze_straight_draws(
    hole_cards: &[Card],
    board: &[Card],
    dead_cards: &HashSet<Card>,
) -> Vec<StraightDraw> {
    let mut draws = Vec::new();
    let all_cards: Vec<Card> = hole_cards.iter().chain(board.iter()).copied().collect();
    let mask = build_rank_mask(&all_cards);

    // Check all possible 5-card windows
    // Window starting positions: 0 (A-5) through 9 (T-A)
    for start in 0..=9 {
        let window_mask: u16 = 0b11111 << start;
        let present = mask & window_mask;
        let present_count = count_bits(present);

        if present_count == 5 {
            // Already have a straight in this window, skip
            continue;
        }

        if present_count == 4 {
            // One gap - either OESD or gutshot
            let missing_mask = window_mask & !mask;
            let missing_bit = missing_mask.trailing_zeros() as u8;

            // Calculate high card of this straight
            let high_card = if start == 0 { 5 } else { start as u8 + 5 };

            // Check if it's nut straight (Broadway: T-A)
            let is_nut = high_card == 14;

            // Find needed rank
            let needed_rank = if missing_bit == 0 {
                14 // Ace (low position)
            } else {
                missing_bit + 1
            };

            // Get outs (all 4 suits of needed rank)
            let outs: Vec<Card> = FULL_DECK
                .iter()
                .filter(|c| {
                    c.rank.value() == needed_rank && !all_cards.contains(c) && !dead_cards.contains(c)
                })
                .copied()
                .collect();

            // Determine draw type
            let draw_type = if missing_bit == 0 || missing_bit == (start as u8 + 4) {
                // Gap at the edge - gutshot
                DrawType::Gutshot
            } else {
                // Check for OESD: need gaps at both edges of a 4-card run
                // Check if we can also complete a straight with another card
                let has_open_end_low = start > 0 && (mask & (1 << (start - 1))) == 0;
                let has_open_end_high = start < 9 && (mask & (1 << (start + 5))) == 0;

                if has_open_end_low || has_open_end_high {
                    // This is part of an OESD if we have 4 consecutive cards
                    // Check if the 4 present cards are consecutive
                    let mut consecutive = true;
                    let mut prev_bit: Option<u8> = None;
                    for bit in 0..5 {
                        if (present >> (start + bit)) & 1 == 1 {
                            if let Some(p) = prev_bit {
                                if bit != p + 1 {
                                    consecutive = false;
                                    break;
                                }
                            }
                            prev_bit = Some(bit);
                        }
                    }

                    if consecutive {
                        DrawType::OpenEnded
                    } else {
                        DrawType::Gutshot
                    }
                } else {
                    DrawType::Gutshot
                }
            };

            if !outs.is_empty() {
                draws.push(StraightDraw {
                    draw_type,
                    needed_ranks: vec![needed_rank],
                    outs,
                    high_card,
                    is_nut,
                });
            }
        }
    }

    // Check for double gutshot (6-card window with 4 cards, 2 internal gaps)
    for start in 0..=8 {
        let window_mask: u16 = 0b111111 << start;
        let present = mask & window_mask;
        let present_count = count_bits(present);

        if present_count == 4 {
            let missing_mask = window_mask & !mask;
            let gaps: Vec<u8> = (0..6)
                .filter(|&i| (missing_mask >> (start + i)) & 1 == 1)
                .collect();

            // Both gaps must be internal (not at position 0 or 5)
            if gaps.len() == 2 && gaps[0] > 0 && gaps[1] < 5 {
                let needed_ranks: Vec<u8> = gaps
                    .iter()
                    .map(|&g| {
                        let bit = start as u8 + g;
                        if bit == 0 {
                            14
                        } else {
                            bit + 1
                        }
                    })
                    .collect();

                // Find all outs
                let outs: Vec<Card> = FULL_DECK
                    .iter()
                    .filter(|c| {
                        needed_ranks.contains(&c.rank.value())
                            && !all_cards.contains(c)
                            && !dead_cards.contains(c)
                    })
                    .copied()
                    .collect();

                let high_card = if start == 0 { 6 } else { start as u8 + 6 };
                let is_nut = high_card == 14;

                if !outs.is_empty() {
                    draws.push(StraightDraw {
                        draw_type: DrawType::DoubleGutshot,
                        needed_ranks,
                        outs,
                        high_card,
                        is_nut,
                    });
                }
            }
        }
    }

    // Check for backdoor straights (only on flop)
    if board.len() == 3 {
        for start in 0..=9 {
            let window_mask: u16 = 0b11111 << start;
            let present = mask & window_mask;
            let present_count = count_bits(present);

            // 3 cards with 2 gaps (backdoor)
            if present_count == 3 {
                // Check if 3 cards are relatively connected (within 5 span)
                let missing_mask = window_mask & !mask;
                let gaps: Vec<u8> = (0..5)
                    .filter(|&i| (missing_mask >> (start + i)) & 1 == 1)
                    .collect();

                if gaps.len() == 2 {
                    let needed_ranks: Vec<u8> = gaps
                        .iter()
                        .map(|&g| {
                            let bit = start as u8 + g;
                            if bit == 0 {
                                14
                            } else {
                                bit + 1
                            }
                        })
                        .collect();

                    let high_card = if start == 0 { 5 } else { start as u8 + 5 };
                    let is_nut = high_card == 14;

                    // For backdoor, we don't count specific outs (need 2 running cards)
                    draws.push(StraightDraw {
                        draw_type: DrawType::BackdoorStraight,
                        needed_ranks,
                        outs: Vec::new(),
                        high_card,
                        is_nut,
                    });
                }
            }
        }
    }

    // Deduplicate: keep the best draw for each high_card
    let mut best_draws: HashMap<u8, StraightDraw> = HashMap::new();
    for draw in draws {
        let key = draw.high_card;
        if let Some(existing) = best_draws.get(&key) {
            // Prefer by: out count desc, then draw type priority
            if draw.outs.len() > existing.outs.len() {
                best_draws.insert(key, draw);
            }
        } else {
            best_draws.insert(key, draw);
        }
    }

    best_draws.into_values().collect()
}

/// Analyze draws for given hole cards and board
#[must_use]
pub fn analyze_draws(hole_cards: &[Card], board: &[Card], dead_cards: &[Card]) -> DrawAnalysis {
    assert!(hole_cards.len() == 2, "Must have exactly 2 hole cards");
    assert!(board.len() <= 5, "Board cannot exceed 5 cards");

    let dead_set: HashSet<Card> = dead_cards.iter().copied().collect();

    // Check if already has flush or straight
    let all_cards: Vec<Card> = hole_cards.iter().chain(board.iter()).copied().collect();
    let (has_flush, has_straight) = if all_cards.len() >= 5 {
        let rank = evaluate_hand(&all_cards);
        let flush = matches!(
            rank.hand_type,
            HandType::Flush | HandType::StraightFlush | HandType::RoyalFlush
        );
        let straight = matches!(
            rank.hand_type,
            HandType::Straight | HandType::StraightFlush | HandType::RoyalFlush
        );
        (flush, straight)
    } else {
        (false, false)
    };

    // Analyze draws (only if we don't already have the made hand)
    let flush_draws = if has_flush {
        Vec::new()
    } else {
        analyze_flush_draws(hole_cards, board, &dead_set)
    };

    let straight_draws = if has_straight {
        Vec::new()
    } else {
        analyze_straight_draws(hole_cards, board, &dead_set)
    };

    // Collect all unique outs
    let mut all_outs_set: HashSet<Card> = HashSet::new();
    for draw in &flush_draws {
        all_outs_set.extend(draw.outs.iter());
    }
    for draw in &straight_draws {
        all_outs_set.extend(draw.outs.iter());
    }
    let all_outs: Vec<Card> = all_outs_set.into_iter().collect();
    let total_outs = all_outs.len();

    DrawAnalysis {
        hole_cards: hole_cards.to_vec(),
        board: board.to_vec(),
        has_flush,
        has_straight,
        flush_draws,
        straight_draws,
        total_outs,
        all_outs,
    }
}

/// Count flush outs (convenience function)
#[must_use]
pub fn count_flush_outs(hole_cards: &[Card], board: &[Card]) -> usize {
    let analysis = analyze_draws(hole_cards, board, &[]);
    analysis
        .flush_draws
        .iter()
        .filter(|d| d.draw_type() == DrawType::FlushDraw)
        .map(FlushDraw::out_count)
        .max()
        .unwrap_or(0)
}

/// Count straight outs (convenience function)
#[must_use]
pub fn count_straight_outs(hole_cards: &[Card], board: &[Card]) -> usize {
    let analysis = analyze_draws(hole_cards, board, &[]);
    analysis
        .straight_draws
        .iter()
        .filter(|d| {
            matches!(
                d.draw_type,
                DrawType::OpenEnded | DrawType::Gutshot | DrawType::DoubleGutshot
            )
        })
        .map(StraightDraw::out_count)
        .sum()
}

/// Get the primary (strongest) draw type
#[must_use]
pub fn get_primary_draw(hole_cards: &[Card], board: &[Card]) -> Option<DrawType> {
    let analysis = analyze_draws(hole_cards, board, &[]);

    // Priority: Flush > OESD > Double Gutshot > Gutshot > Backdoor
    if analysis.flush_draws.iter().any(|d| d.draw_type() == DrawType::FlushDraw) {
        return Some(DrawType::FlushDraw);
    }
    if analysis.straight_draws.iter().any(|d| d.draw_type == DrawType::OpenEnded) {
        return Some(DrawType::OpenEnded);
    }
    if analysis.straight_draws.iter().any(|d| d.draw_type == DrawType::DoubleGutshot) {
        return Some(DrawType::DoubleGutshot);
    }
    if analysis.straight_draws.iter().any(|d| d.draw_type == DrawType::Gutshot) {
        return Some(DrawType::Gutshot);
    }
    if analysis.flush_draws.iter().any(|d| d.draw_type() == DrawType::BackdoorFlush) {
        return Some(DrawType::BackdoorFlush);
    }
    if analysis.straight_draws.iter().any(|d| d.draw_type == DrawType::BackdoorStraight) {
        return Some(DrawType::BackdoorStraight);
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::card::parse_cards;

    fn cards(s: &str) -> Vec<Card> {
        parse_cards(s).unwrap()
    }

    #[test]
    fn test_flush_draw() {
        let hole = cards("Ah 9h");
        let board = cards("Kh 5h 2c");

        let analysis = analyze_draws(&hole, &board, &[]);

        assert!(!analysis.has_flush);
        assert_eq!(analysis.flush_draws.len(), 1);
        assert_eq!(analysis.flush_draws[0].draw_type(), DrawType::FlushDraw);
        assert_eq!(analysis.flush_draws[0].out_count(), 9);
        assert!(analysis.flush_draws[0].is_nut); // Has Ace of hearts
    }

    #[test]
    fn test_backdoor_flush() {
        let hole = cards("Ah 9h");
        let board = cards("Kh 5c 2c");

        let analysis = analyze_draws(&hole, &board, &[]);

        assert!(!analysis.has_flush);
        assert_eq!(analysis.flush_draws.len(), 1);
        assert_eq!(analysis.flush_draws[0].draw_type(), DrawType::BackdoorFlush);
    }

    #[test]
    fn test_open_ended_straight() {
        let hole = cards("9h 8c");
        let board = cards("7d 6s 2h");

        let analysis = analyze_draws(&hole, &board, &[]);

        assert!(!analysis.has_straight);
        let oesd = analysis
            .straight_draws
            .iter()
            .filter(|d| d.draw_type == DrawType::OpenEnded)
            .collect::<Vec<_>>();
        assert!(!oesd.is_empty());
    }

    #[test]
    fn test_gutshot() {
        let hole = cards("Ah Kc");
        let board = cards("Qd Ts 2h"); // Need J for broadway

        let analysis = analyze_draws(&hole, &board, &[]);

        let gutshots = analysis
            .straight_draws
            .iter()
            .filter(|d| d.draw_type == DrawType::Gutshot)
            .collect::<Vec<_>>();
        assert!(!gutshots.is_empty());
    }

    #[test]
    fn test_combo_draw() {
        let hole = cards("9h 8h");
        let board = cards("7h 6c 2h"); // Flush draw + OESD

        let analysis = analyze_draws(&hole, &board, &[]);

        assert!(analysis.is_combo_draw());
        assert!(analysis.total_outs > 12); // Should be ~15 outs
    }

    #[test]
    fn test_already_has_flush() {
        let hole = cards("Ah 9h");
        let board = cards("Kh 5h 2h");

        let analysis = analyze_draws(&hole, &board, &[]);

        assert!(analysis.has_flush);
        assert!(analysis.flush_draws.is_empty());
    }

    #[test]
    fn test_already_has_straight() {
        let hole = cards("9h 8c");
        let board = cards("7d 6s 5h");

        let analysis = analyze_draws(&hole, &board, &[]);

        assert!(analysis.has_straight);
        assert!(analysis.straight_draws.is_empty());
    }

    #[test]
    fn test_wheel_straight_draw() {
        let hole = cards("Ah 2c");
        let board = cards("4d 3s Kh"); // Need 5 for wheel

        let analysis = analyze_draws(&hole, &board, &[]);

        let has_wheel_draw = analysis.straight_draws.iter().any(|d| d.high_card == 5);
        assert!(has_wheel_draw);
    }

    #[test]
    fn test_dead_cards_reduce_outs() {
        let hole = cards("Ah 9h");
        let board = cards("Kh 5h 2c");
        let dead = cards("Qh Jh"); // 2 hearts are dead

        let analysis_no_dead = analyze_draws(&hole, &board, &[]);
        let analysis_with_dead = analyze_draws(&hole, &board, &dead);

        assert!(analysis_with_dead.flush_draws[0].out_count() < analysis_no_dead.flush_draws[0].out_count());
    }

    #[test]
    fn test_get_primary_draw() {
        let hole = cards("9h 8h");
        let board = cards("7h 6c 2h");

        let primary = get_primary_draw(&hole, &board);
        assert_eq!(primary, Some(DrawType::FlushDraw)); // Flush > OESD
    }

    #[test]
    fn test_count_functions() {
        let hole = cards("Ah 9h");
        let board = cards("Kh 5h 2c");

        let flush_outs = count_flush_outs(&hole, &board);
        assert_eq!(flush_outs, 9);

        let straight_outs = count_straight_outs(&hole, &board);
        // straight_outs is usize, always >= 0
    }
}
