//! Equity calculation via Monte Carlo simulation.
//!
//! Calculates the probability of each player winning a hand by simulating
//! random runouts multiple times.

use crate::card::{Card, FULL_DECK};
use crate::evaluator::find_winners;
use rand::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;

// std::time::Instant is not available in WASM, so we skip timing there
// The WASM binding layer (holdem-wasm) handles timing with js_sys::Date
#[cfg(not(target_arch = "wasm32"))]
use std::time::Instant;

/// A player's hole cards
///
/// - If cards is Some: uses the specific 2 cards
/// - If is_random is true: random hand sampled each simulation
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PlayerHand {
    pub cards: Vec<Card>,
    #[serde(default)]
    pub is_random: bool,
}

impl PlayerHand {
    /// Create a new player hand with specific cards
    #[must_use]
    pub fn new(cards: Vec<Card>) -> Self {
        assert!(
            cards.len() == 2,
            "Player must have exactly 2 hole cards, got {}",
            cards.len()
        );
        Self {
            cards,
            is_random: false,
        }
    }

    /// Create a random player hand
    #[must_use]
    pub fn random() -> Self {
        Self {
            cards: Vec::new(),
            is_random: true,
        }
    }

    /// Parse from string notation (e.g., "Ah Kh")
    pub fn parse(s: &str) -> Result<Self, crate::card::ParseError> {
        let cards = crate::card::parse_cards(s)?;
        Ok(Self::new(cards))
    }
}

/// Equity result for a single player
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PlayerEquity {
    /// Player index (0-based)
    pub index: usize,
    /// Win count
    pub win_count: u64,
    /// Tie count
    pub tie_count: u64,
    /// Total simulations
    pub total_simulations: u64,
    /// Win rate (0.0 - 1.0)
    pub win_rate: f64,
    /// Tie rate (0.0 - 1.0)
    pub tie_rate: f64,
    /// Equity (win_rate + tie_rate / num_tied)
    pub equity: f64,
    /// Hand description
    pub hand_description: String,
    /// Number of combos (for range-based hands)
    pub combos: usize,
}

/// Request for equity calculation
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EquityRequest {
    /// Players with their hole cards
    pub players: Vec<PlayerHand>,
    /// Community cards (0-5)
    pub board: Vec<Card>,
    /// Dead cards (not available for runout)
    #[serde(default)]
    pub dead_cards: Vec<Card>,
    /// Number of Monte Carlo simulations
    #[serde(default = "default_simulations")]
    pub num_simulations: u32,
    /// Random seed for reproducibility
    pub seed: Option<u64>,
}

fn default_simulations() -> u32 {
    10_000
}

fn validate_equity_request(request: &EquityRequest) {
    assert!(
        request.players.len() >= 2,
        "Need at least 2 players"
    );
    assert!(
        request.board.len() <= 5,
        "Board cannot have more than 5 cards"
    );

    for (i, player) in request.players.iter().enumerate() {
        if player.is_random {
            assert!(
                player.cards.is_empty(),
                "Random player must not specify cards"
            );
        } else {
            assert!(
                player.cards.len() == 2,
                "Player {} must have exactly 2 hole cards, got {}",
                i,
                player.cards.len()
            );
        }
    }

    let mut known_cards: HashSet<Card> = HashSet::new();
    for &card in &request.board {
        if !known_cards.insert(card) {
            panic!("Duplicate cards detected in players/board");
        }
    }
    for &card in &request.dead_cards {
        if !known_cards.insert(card) {
            panic!("Duplicate cards detected in players/board");
        }
    }
    for player in &request.players {
        if !player.is_random {
            for &card in &player.cards {
                if !known_cards.insert(card) {
                    panic!("Duplicate cards detected in players/board");
                }
            }
        }
    }
}

impl EquityRequest {
    /// Create a new equity request
    #[must_use]
    pub fn new(players: Vec<PlayerHand>, board: Vec<Card>) -> Self {
        Self {
            players,
            board,
            dead_cards: Vec::new(),
            num_simulations: default_simulations(),
            seed: None,
        }
    }

    /// Set number of simulations
    #[must_use]
    pub fn with_simulations(mut self, n: u32) -> Self {
        self.num_simulations = n;
        self
    }

    /// Set random seed
    #[must_use]
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Set dead cards
    #[must_use]
    pub fn with_dead_cards(mut self, dead: Vec<Card>) -> Self {
        self.dead_cards = dead;
        self
    }
}

/// Result of equity calculation
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EquityResult {
    /// Equity for each player
    pub players: Vec<PlayerEquity>,
    /// Total simulations run
    pub total_simulations: u64,
    /// Elapsed time in milliseconds
    pub elapsed_ms: f64,
}

/// Internal accumulator for tracking equity during simulation
struct EquityAccumulator {
    num_players: usize,
    wins: Vec<u64>,
    ties: Vec<u64>,
    equity_sum: Vec<f64>,
    total: u64,
}

impl EquityAccumulator {
    fn new(num_players: usize) -> Self {
        Self {
            num_players,
            wins: vec![0; num_players],
            ties: vec![0; num_players],
            equity_sum: vec![0.0; num_players],
            total: 0,
        }
    }

    fn record(&mut self, winner_indices: &[usize]) {
        self.total += 1;

        if winner_indices.len() == 1 {
            // Single winner
            let winner = winner_indices[0];
            self.wins[winner] += 1;
            self.equity_sum[winner] += 1.0;
        } else {
            // Tie - split equity
            let share = 1.0 / winner_indices.len() as f64;
            for &idx in winner_indices {
                self.ties[idx] += 1;
                self.equity_sum[idx] += share;
            }
        }
    }

    fn into_results(self, hand_descriptions: Vec<String>, elapsed_ms: f64) -> EquityResult {
        let players: Vec<PlayerEquity> = (0..self.num_players)
            .map(|i| {
                let win_rate = if self.total > 0 {
                    self.wins[i] as f64 / self.total as f64
                } else {
                    0.0
                };
                let tie_rate = if self.total > 0 {
                    self.ties[i] as f64 / self.total as f64
                } else {
                    0.0
                };
                let equity = if self.total > 0 {
                    self.equity_sum[i] / self.total as f64
                } else {
                    0.0
                };

                PlayerEquity {
                    index: i,
                    win_count: self.wins[i],
                    tie_count: self.ties[i],
                    total_simulations: self.total,
                    win_rate,
                    tie_rate,
                    equity,
                    hand_description: hand_descriptions.get(i).cloned().unwrap_or_default(),
                    combos: 1, // Single hand, not range
                }
            })
            .collect();

        EquityResult {
            players,
            total_simulations: self.total,
            elapsed_ms,
        }
    }
}

/// Calculate equity for all players
///
/// Supports both known hands and random players. Random players have their
/// hole cards sampled from the remaining deck each simulation.
///
/// # Panics
/// Panics if fewer than 2 players or more than 5 board cards
#[must_use]
pub fn calculate_equity(request: &EquityRequest) -> EquityResult {
    validate_equity_request(request);

    #[cfg(not(target_arch = "wasm32"))]
    let start = Instant::now();

    // Identify random vs known players
    let random_player_indices: Vec<usize> = request
        .players
        .iter()
        .enumerate()
        .filter(|(_, p)| p.is_random)
        .map(|(i, _)| i)
        .collect();

    // Collect all known cards (board + known player hands + dead cards)
    let mut known_cards: HashSet<Card> = HashSet::new();
    for player in &request.players {
        if !player.is_random {
            for &card in &player.cards {
                known_cards.insert(card);
            }
        }
    }
    for &card in &request.board {
        known_cards.insert(card);
    }
    for &card in &request.dead_cards {
        known_cards.insert(card);
    }

    // Build remaining deck
    let remaining: Vec<Card> = FULL_DECK
        .iter()
        .filter(|c| !known_cards.contains(c))
        .copied()
        .collect();

    let cards_needed_board = 5 - request.board.len();
    let num_players = request.players.len();

    // Initialize RNG
    let mut rng = match request.seed {
        Some(seed) => StdRng::seed_from_u64(seed),
        None => StdRng::from_os_rng(),
    };

    // Initialize accumulator
    let mut acc = EquityAccumulator::new(num_players);

    // Hand descriptions
    let hand_descriptions: Vec<String> = request
        .players
        .iter()
        .map(|p| {
            if p.is_random {
                "(Random)".to_string()
            } else {
                p.cards
                    .iter()
                    .map(ToString::to_string)
                    .collect::<Vec<_>>()
                    .join(" ")
            }
        })
        .collect();

    // Run simulations
    let mut deck_remaining = remaining.clone();

    for _ in 0..request.num_simulations {
        // Shuffle remaining deck
        deck_remaining.shuffle(&mut rng);

        // Deal cards to random players first
        let mut deck_idx = 0;
        let mut sim_hole_cards: Vec<Vec<Card>> = Vec::with_capacity(num_players);

        for (i, player) in request.players.iter().enumerate() {
            if random_player_indices.contains(&i) {
                // Random player: deal from shuffled deck
                sim_hole_cards.push(vec![deck_remaining[deck_idx], deck_remaining[deck_idx + 1]]);
                deck_idx += 2;
            } else {
                // Known player: use their cards
                sim_hole_cards.push(player.cards.clone());
            }
        }

        // Deal community cards
        let runout: Vec<Card> = deck_remaining[deck_idx..deck_idx + cards_needed_board].to_vec();

        // Build complete board
        let mut full_board = request.board.clone();
        full_board.extend(runout);

        // Build complete hands for each player
        let hands: Vec<Vec<Card>> = sim_hole_cards
            .into_iter()
            .map(|mut hole| {
                hole.extend(full_board.iter().copied());
                hole
            })
            .collect();

        // Find winners
        let winners = find_winners(&hands);

        // Record result
        acc.record(&winners);
    }

    #[cfg(not(target_arch = "wasm32"))]
    let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
    #[cfg(target_arch = "wasm32")]
    let elapsed_ms = 0.0; // WASM timing handled by holdem-wasm with js_sys::Date

    acc.into_results(hand_descriptions, elapsed_ms)
}

/// Convenience function: calculate equity of hole cards vs random opponents
#[must_use]
pub fn equity_vs_random(
    hole_cards: &[Card],
    board: &[Card],
    num_opponents: usize,
    num_simulations: u32,
    seed: Option<u64>,
) -> f64 {
    assert!(
        hole_cards.len() == 2,
        "Hole cards must be exactly 2"
    );
    assert!(num_opponents >= 1, "Need at least 1 opponent");

    // Collect known cards
    let mut known_cards: HashSet<Card> = HashSet::new();
    for &card in hole_cards {
        known_cards.insert(card);
    }
    for &card in board {
        known_cards.insert(card);
    }

    // Build remaining deck
    let remaining: Vec<Card> = FULL_DECK
        .iter()
        .filter(|c| !known_cards.contains(c))
        .copied()
        .collect();

    let cards_needed_board = 5 - board.len();

    // Initialize RNG
    let mut rng = match seed {
        Some(s) => StdRng::seed_from_u64(s),
        None => StdRng::from_os_rng(),
    };

    let mut equity_sum = 0.0;
    let mut deck_remaining = remaining.clone();

    for _ in 0..num_simulations {
        deck_remaining.shuffle(&mut rng);

        let mut idx = 0;

        // Deal runout
        let runout: Vec<Card> = deck_remaining[idx..idx + cards_needed_board].to_vec();
        idx += cards_needed_board;

        // Deal opponent hands
        let mut opponent_hands: Vec<Vec<Card>> = Vec::with_capacity(num_opponents);
        for _ in 0..num_opponents {
            opponent_hands.push(deck_remaining[idx..idx + 2].to_vec());
            idx += 2;
        }

        // Build complete board
        let mut full_board = board.to_vec();
        full_board.extend(runout);

        // Build all hands
        let mut hands: Vec<Vec<Card>> = Vec::with_capacity(num_opponents + 1);

        // Hero's hand
        let mut hero_hand = hole_cards.to_vec();
        hero_hand.extend(full_board.iter().copied());
        hands.push(hero_hand);

        // Opponent hands
        for opp in opponent_hands {
            let mut hand = opp;
            hand.extend(full_board.iter().copied());
            hands.push(hand);
        }

        // Find winners
        let winners = find_winners(&hands);

        // Check if hero (index 0) won
        if winners.contains(&0) {
            equity_sum += 1.0 / winners.len() as f64;
        }
    }

    equity_sum / num_simulations as f64
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::card::parse_cards;

    fn cards(s: &str) -> Vec<Card> {
        parse_cards(s).unwrap()
    }

    #[test]
    fn test_equity_aa_vs_kk() {
        let request = EquityRequest::new(
            vec![
                PlayerHand::new(cards("Ah As")),
                PlayerHand::new(cards("Kh Ks")),
            ],
            vec![],
        )
        .with_simulations(10_000)
        .with_seed(42);

        let result = calculate_equity(&request);

        assert_eq!(result.players.len(), 2);
        // AA should have ~82% equity vs KK
        assert!(result.players[0].equity > 0.75);
        assert!(result.players[0].equity < 0.90);
        assert!(result.players[1].equity > 0.10);
        assert!(result.players[1].equity < 0.25);
    }

    #[test]
    fn test_equity_with_board() {
        let request = EquityRequest::new(
            vec![
                PlayerHand::new(cards("Ah Kh")),
                PlayerHand::new(cards("7h 2c")),
            ],
            cards("Qh Jh Th"), // AK has royal flush draw
        )
        .with_simulations(10_000)
        .with_seed(42);

        let result = calculate_equity(&request);

        // With the flush draw, AK should be heavily favored
        assert!(result.players[0].equity > 0.80);
    }

    #[test]
    fn test_equity_sums_to_one() {
        let request = EquityRequest::new(
            vec![
                PlayerHand::new(cards("Ah As")),
                PlayerHand::new(cards("Kh Ks")),
                PlayerHand::new(cards("Qh Qs")),
            ],
            vec![],
        )
        .with_simulations(5_000)
        .with_seed(42);

        let result = calculate_equity(&request);

        let total_equity: f64 = result.players.iter().map(|p| p.equity).sum();
        assert!((total_equity - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_equity_deterministic_with_seed() {
        let request1 = EquityRequest::new(
            vec![
                PlayerHand::new(cards("Ah Kh")),
                PlayerHand::new(cards("7s 2d")),
            ],
            vec![],
        )
        .with_simulations(1_000)
        .with_seed(12345);

        let request2 = request1.clone();

        let result1 = calculate_equity(&request1);
        let result2 = calculate_equity(&request2);

        assert_eq!(result1.players[0].equity, result2.players[0].equity);
    }

    #[test]
    fn test_equity_vs_random() {
        let hole = cards("Ah As");
        let equity = equity_vs_random(&hole, &[], 1, 10_000, Some(42));

        // AA vs 1 random should be ~85%
        assert!(equity > 0.80);
        assert!(equity < 0.90);
    }

    #[test]
    fn test_equity_vs_multiple_random() {
        let hole = cards("Ah As");
        let equity = equity_vs_random(&hole, &[], 5, 10_000, Some(42));

        // AA vs 5 random should be ~49%
        assert!(equity > 0.40);
        assert!(equity < 0.60);
    }

    #[test]
    fn test_player_hand_parse() {
        let hand = PlayerHand::parse("Ah Kh").unwrap();
        assert_eq!(hand.cards.len(), 2);
    }

    #[test]
    fn test_equity_with_random_player() {
        let request = EquityRequest::new(
            vec![
                PlayerHand::new(cards("As Kd")),
                PlayerHand::random(),
            ],
            vec![],
        )
        .with_simulations(5_000)
        .with_seed(42);

        let result = calculate_equity(&request);

        assert_eq!(result.players.len(), 2);
        // AK should have ~62-65% equity vs random
        assert!(result.players[0].equity > 0.55);
        assert!(result.players[0].equity < 0.70);
        // Random player hand description
        assert_eq!(result.players[1].hand_description, "(Random)");
    }

    #[test]
    fn test_equity_with_multiple_random_players() {
        let request = EquityRequest::new(
            vec![
                PlayerHand::new(cards("As Kd")),
                PlayerHand::random(),
                PlayerHand::random(),
            ],
            vec![],
        )
        .with_simulations(5_000)
        .with_seed(42);

        let result = calculate_equity(&request);

        assert_eq!(result.players.len(), 3);
        // AK vs 2 random should be ~47-50% equity
        assert!(result.players[0].equity > 0.40);
        assert!(result.players[0].equity < 0.55);
    }
}
