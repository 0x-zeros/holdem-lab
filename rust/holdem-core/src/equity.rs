//! Equity calculation via Monte Carlo simulation.
//!
//! Calculates the probability of each player winning a hand by simulating
//! random runouts multiple times.

use crate::card::{Card, FULL_DECK};
use crate::error::{HoldemError, HoldemResult};
use crate::evaluator::find_winners;
use crate::range::{hands_are_disjoint, CardDistribution, Odometer};
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
    ///
    /// # Panics
    /// Panics if cards.len() != 2. Use `try_new()` for a Result-returning version.
    #[must_use]
    pub fn new(cards: Vec<Card>) -> Self {
        Self::try_new(cards).expect("Player must have exactly 2 hole cards")
    }

    /// Try to create a new player hand with specific cards
    ///
    /// Returns an error if the number of cards is not exactly 2.
    pub fn try_new(cards: Vec<Card>) -> HoldemResult<Self> {
        if cards.len() != 2 {
            return Err(HoldemError::InvalidCardCount {
                expected: "2",
                got: cards.len(),
            });
        }
        Ok(Self {
            cards,
            is_random: false,
        })
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
        // Use new() here since parse_cards already validates
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

fn validate_equity_request(request: &EquityRequest) -> HoldemResult<()> {
    if request.players.len() < 2 {
        return Err(HoldemError::NotEnoughPlayers(2));
    }
    if request.board.len() > 5 {
        return Err(HoldemError::BoardTooLarge(request.board.len()));
    }

    for player in &request.players {
        if player.is_random {
            if !player.cards.is_empty() {
                return Err(HoldemError::InvalidCardCount {
                    expected: "0 (random player)",
                    got: player.cards.len(),
                });
            }
        } else if player.cards.len() != 2 {
            return Err(HoldemError::InvalidCardCount {
                expected: "2",
                got: player.cards.len(),
            });
        }
    }

    let mut known_cards: HashSet<Card> = HashSet::new();
    for &card in &request.board {
        if !known_cards.insert(card) {
            return Err(HoldemError::DuplicateCard(card.to_string()));
        }
    }
    for &card in &request.dead_cards {
        if !known_cards.insert(card) {
            return Err(HoldemError::DuplicateCard(card.to_string()));
        }
    }
    for player in &request.players {
        if !player.is_random {
            for &card in &player.cards {
                if !known_cards.insert(card) {
                    return Err(HoldemError::DuplicateCard(card.to_string()));
                }
            }
        }
    }
    Ok(())
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
/// # Errors
/// Returns an error if:
/// - Fewer than 2 players
/// - More than 5 board cards
/// - Duplicate cards detected
/// - Invalid player hand configuration
pub fn calculate_equity(request: &EquityRequest) -> HoldemResult<EquityResult> {
    validate_equity_request(request)?;

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

        // Find winners (unwrap is safe here - we always have 7-card hands)
        let winners = find_winners(&hands).unwrap();

        // Record result
        acc.record(&winners);
    }

    #[cfg(not(target_arch = "wasm32"))]
    let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
    #[cfg(target_arch = "wasm32")]
    let elapsed_ms = 0.0; // WASM timing handled by holdem-wasm with js_sys::Date

    Ok(acc.into_results(hand_descriptions, elapsed_ms))
}

/// Player input for range-based equity calculation
#[derive(Clone, Debug)]
pub enum RangePlayer {
    /// Specific cards (2 hole cards)
    Specific(Card, Card),
    /// Random cards from remaining deck
    Random,
    /// Range distribution
    Range(CardDistribution),
}

impl RangePlayer {
    /// Create from specific cards
    #[must_use]
    pub fn specific(c1: Card, c2: Card) -> Self {
        RangePlayer::Specific(c1, c2)
    }

    /// Create random player
    #[must_use]
    pub fn random() -> Self {
        RangePlayer::Random
    }

    /// Create from range distribution
    #[must_use]
    pub fn range(dist: CardDistribution) -> Self {
        RangePlayer::Range(dist)
    }
}

/// Request for range-based equity calculation
#[derive(Clone, Debug)]
pub struct RangeEquityRequest {
    /// Players with their hand distributions
    pub players: Vec<RangePlayer>,
    /// Community cards (0-5)
    pub board: Vec<Card>,
    /// Dead cards
    pub dead_cards: Vec<Card>,
    /// Number of Monte Carlo simulations per combination
    pub num_simulations: u32,
    /// Random seed
    pub seed: Option<u64>,
}

impl RangeEquityRequest {
    /// Create a new range equity request
    #[must_use]
    pub fn new(players: Vec<RangePlayer>, board: Vec<Card>) -> Self {
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

/// Result for range-based equity calculation
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RangeEquityResult {
    /// Equity for each player
    pub players: Vec<RangePlayerEquity>,
    /// Total valid combinations evaluated
    pub total_combinations: u64,
    /// Total simulations across all combinations
    pub total_simulations: u64,
    /// Elapsed time in milliseconds
    pub elapsed_ms: f64,
}

/// Equity result for a single player in range calculation
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RangePlayerEquity {
    /// Player index (0-based)
    pub index: usize,
    /// Overall equity (weighted average across combinations)
    pub equity: f64,
    /// Win rate
    pub win_rate: f64,
    /// Tie rate
    pub tie_rate: f64,
    /// Number of combos in the distribution
    pub combos: usize,
    /// Hand description
    pub hand_description: String,
}

// =============================================================================
// Adaptive Equity Calculation Strategy
// =============================================================================
//
// The calculation strategy is automatically selected based on total combo count:
//
// | Range Size | Combos    | Strategy   | Description                          |
// |------------|-----------|------------|--------------------------------------|
// | Small      | < 50      | Exhaustive | Enumerate all, more sims per combo   |
// | Medium     | 50-500    | Hybrid     | Enumerate all, fewer sims per combo  |
// | Large      | > 500     | Sampled    | Random sample up to MAX_SAMPLED      |
//
// This ensures reasonable performance across all range sizes while maintaining
// accuracy for smaller ranges where exhaustive enumeration is feasible.
// =============================================================================

/// Threshold for small ranges: enumerate all with full simulations
const SMALL_RANGE_THRESHOLD: usize = 50;

/// Threshold for medium ranges: enumerate all with reduced simulations
const MEDIUM_RANGE_THRESHOLD: usize = 500;

/// Threshold for huge ranges: use biased but fast sampling
const HUGE_RANGE_THRESHOLD: usize = 10_000;

/// Maximum combos to sample for large ranges
const MAX_SAMPLED_COMBOS: usize = 200;

/// Minimum simulations per combo to ensure statistical significance
const MIN_SIMS_PER_COMBO: u32 = 100;

/// Calculation strategy based on range size
#[derive(Debug, Clone, Copy)]
enum EquityStrategy {
    /// Enumerate all combinations with specified simulations per combo
    Exhaustive { sims_per_combo: u32 },
    /// Unbiased reservoir sampling - iterates all combos but only simulates sampled ones
    ReservoirSampled {
        max_combos: usize,
        sims_per_combo: u32,
    },
    /// Biased but fast sampling - breaks early once enough samples collected
    BiasedSampled {
        max_combos: usize,
        sims_per_combo: u32,
    },
}

/// Select optimal calculation strategy based on total combo count
fn select_strategy(total_combos: usize, requested_sims: u32) -> EquityStrategy {
    if total_combos <= SMALL_RANGE_THRESHOLD {
        // Small range: enumerate all, use full simulations (at least 1000)
        EquityStrategy::Exhaustive {
            sims_per_combo: requested_sims.max(1000),
        }
    } else if total_combos <= MEDIUM_RANGE_THRESHOLD {
        // Medium range: enumerate all, reduce sims to control total time
        // Target: roughly same total work as 50 combos × requested_sims
        let sims = ((requested_sims as usize * SMALL_RANGE_THRESHOLD) / total_combos)
            .max(MIN_SIMS_PER_COMBO as usize) as u32;
        EquityStrategy::Exhaustive { sims_per_combo: sims }
    } else if total_combos <= HUGE_RANGE_THRESHOLD {
        // Large range: unbiased reservoir sampling
        EquityStrategy::ReservoirSampled {
            max_combos: MAX_SAMPLED_COMBOS,
            sims_per_combo: requested_sims,
        }
    } else {
        // Huge range (>10k combos): biased but fast sampling
        EquityStrategy::BiasedSampled {
            max_combos: MAX_SAMPLED_COMBOS,
            sims_per_combo: requested_sims,
        }
    }
}

/// Calculate equity with range support using adaptive strategy.
///
/// # Performance Optimization
///
/// The function automatically selects the optimal calculation strategy based on
/// the total number of hand combinations:
///
/// | Range Size | Combos | Strategy | Description |
/// |-----------|--------|----------|-------------|
/// | Small | < 50 | Exhaustive | Enumerate all combos, more sims each |
/// | Medium | 50-500 | Exhaustive | Enumerate all, fewer sims to control time |
/// | Large | 500-10k | ReservoirSampled | Unbiased sampling, iterates all combos |
/// | Huge | > 10k | BiasedSampled | Fast but biased toward front of odometer |
///
/// # Algorithm
///
/// 1. Validate inputs (board/dead duplicates, player card conflicts)
/// 2. Build CardDistribution for each range player
/// 3. Use Odometer to iterate Cartesian product of all ranges
/// 4. Select strategy based on total combo count
/// 5. For each combination (or sampled subset):
///    - Skip if cards conflict (same card used twice)
///    - Run Monte Carlo simulation
///    - Weight and accumulate results
/// 6. Return weighted average equity
///
/// # Complexity
///
/// - Time: O(C × S × P) where C = combos (or MAX_SAMPLED), S = sims, P = players
/// - Space: O(P) for tracking equity per player
///
/// # Errors
/// Returns an error if fewer than 2 players, more than 5 board cards,
/// duplicate cards in board/dead, or no valid combinations exist.
pub fn calculate_equity_with_ranges(request: &RangeEquityRequest) -> HoldemResult<RangeEquityResult> {
    if request.players.len() < 2 {
        return Err(HoldemError::NotEnoughPlayers(2));
    }
    if request.board.len() > 5 {
        return Err(HoldemError::BoardTooLarge(request.board.len()));
    }

    #[cfg(not(target_arch = "wasm32"))]
    let start = Instant::now();

    let num_players = request.players.len();

    // Build base excluded cards (board + dead) with duplicate detection
    let mut base_excluded: HashSet<Card> = HashSet::new();
    for &card in &request.board {
        if !base_excluded.insert(card) {
            return Err(HoldemError::DuplicateCard(card.to_string()));
        }
    }
    for &card in &request.dead_cards {
        if !base_excluded.insert(card) {
            return Err(HoldemError::DuplicateCard(card.to_string()));
        }
    }

    // Build distributions for each player
    let mut distributions: Vec<Vec<(Card, Card)>> = Vec::with_capacity(num_players);
    let mut hand_descriptions: Vec<String> = Vec::with_capacity(num_players);
    let mut combo_counts: Vec<usize> = Vec::with_capacity(num_players);

    for player in &request.players {
        match player {
            RangePlayer::Specific(c1, c2) => {
                // Validate specific player cards don't conflict with board/dead
                if c1 == c2 {
                    return Err(HoldemError::DuplicateCard(c1.to_string()));
                }
                if base_excluded.contains(c1) {
                    return Err(HoldemError::DuplicateCard(c1.to_string()));
                }
                if base_excluded.contains(c2) {
                    return Err(HoldemError::DuplicateCard(c2.to_string()));
                }
                distributions.push(vec![(*c1, *c2)]);
                hand_descriptions.push(format!("{}{}", c1, c2));
                combo_counts.push(1);
            }
            RangePlayer::Random => {
                // Random will be handled specially during simulation
                distributions.push(vec![]); // Empty marker
                hand_descriptions.push("Random".to_string());
                combo_counts.push(1326);
            }
            RangePlayer::Range(dist) => {
                // Filter by base excluded cards
                let filtered = dist.filter_excluding(&base_excluded);
                hand_descriptions.push(format!("{} combos", filtered.len()));
                combo_counts.push(filtered.len());
                distributions.push(filtered.hands().to_vec());
            }
        }
    }

    // Check if any range player has no combos
    for (i, dist) in distributions.iter().enumerate() {
        if dist.is_empty() && !matches!(request.players[i], RangePlayer::Random) {
            return Err(HoldemError::InvalidCardCount {
                expected: "at least 1 combo",
                got: 0,
            });
        }
    }

    // Validate that multiple Specific players don't have conflicting cards
    let mut specific_cards: HashSet<Card> = HashSet::new();
    for player in &request.players {
        if let RangePlayer::Specific(c1, c2) = player {
            if !specific_cards.insert(*c1) {
                return Err(HoldemError::DuplicateCard(c1.to_string()));
            }
            if !specific_cards.insert(*c2) {
                return Err(HoldemError::DuplicateCard(c2.to_string()));
            }
        }
    }

    // Identify random players
    let random_player_indices: Vec<usize> = request
        .players
        .iter()
        .enumerate()
        .filter(|(_, p)| matches!(p, RangePlayer::Random))
        .map(|(i, _)| i)
        .collect();

    // Build odometer extents (use 1 for random players)
    let extents: Vec<usize> = distributions
        .iter()
        .enumerate()
        .map(|(i, d)| {
            if random_player_indices.contains(&i) {
                1 // Random players have single "virtual" combo
            } else {
                d.len()
            }
        })
        .collect();

    // Calculate total theoretical combinations and select strategy
    let odometer = Odometer::new(extents.clone());
    let total_theoretical_combos = odometer.total_combinations();
    let strategy = select_strategy(total_theoretical_combos, request.num_simulations);

    // Extract sims_per_combo (common to all strategies)
    let sims_per_combo = match strategy {
        EquityStrategy::Exhaustive { sims_per_combo } => sims_per_combo,
        EquityStrategy::ReservoirSampled { sims_per_combo, .. } => sims_per_combo,
        EquityStrategy::BiasedSampled { sims_per_combo, .. } => sims_per_combo,
    };

    // Initialize accumulators
    let mut total_equity: Vec<f64> = vec![0.0; num_players];
    let mut total_wins: Vec<f64> = vec![0.0; num_players];
    let mut total_ties: Vec<f64> = vec![0.0; num_players];
    let mut total_weight: f64 = 0.0;
    let mut total_combinations: u64 = 0;
    let mut total_simulations: u64 = 0;

    // Initialize RNG
    let mut rng = match request.seed {
        Some(seed) => StdRng::seed_from_u64(seed),
        None => StdRng::from_os_rng(),
    };

    let cards_needed_board = 5 - request.board.len();

    // Helper to check if a combination is valid (no card conflicts)
    let is_valid_combination = |indices: &[usize]| -> Option<(Vec<(Card, Card)>, Vec<Card>)> {
        let mut current_hands: Vec<(Card, Card)> = Vec::with_capacity(num_players);

        for (player_idx, &combo_idx) in indices.iter().enumerate() {
            if random_player_indices.contains(&player_idx) {
                // Random player - use placeholder
                let placeholder = Card::from_index(0).unwrap();
                current_hands.push((placeholder, placeholder));
            } else {
                current_hands.push(distributions[player_idx][combo_idx]);
            }
        }

        // Check for card conflicts (only for non-random players)
        let non_random_hands: Vec<(Card, Card)> = current_hands
            .iter()
            .enumerate()
            .filter(|(i, _)| !random_player_indices.contains(i))
            .map(|(_, h)| *h)
            .collect();

        if !hands_are_disjoint(&non_random_hands) {
            return None;
        }

        // Also check against board/dead cards
        let mut all_used = base_excluded.clone();
        for &(c1, c2) in &non_random_hands {
            if all_used.contains(&c1) || all_used.contains(&c2) {
                return None;
            }
            all_used.insert(c1);
            all_used.insert(c2);
        }

        // Build remaining deck for this combination
        let remaining: Vec<Card> = FULL_DECK
            .iter()
            .filter(|c| !all_used.contains(c))
            .copied()
            .collect();

        Some((current_hands, remaining))
    };

    // Helper to run simulation for a combination
    let run_simulation = |current_hands: &[(Card, Card)],
                          remaining: &[Card],
                          rng: &mut StdRng|
     -> (Vec<u64>, Vec<u64>, Vec<f64>) {
        let mut combo_wins = vec![0u64; num_players];
        let mut combo_ties = vec![0u64; num_players];
        let mut combo_equity = vec![0.0f64; num_players];
        let mut deck_remaining = remaining.to_vec();

        for _ in 0..sims_per_combo {
            deck_remaining.shuffle(rng);

            let mut deck_idx = 0;
            let mut sim_hole_cards: Vec<Vec<Card>> = Vec::with_capacity(num_players);

            for (i, &(c1, c2)) in current_hands.iter().enumerate() {
                if random_player_indices.contains(&i) {
                    // Deal random cards
                    sim_hole_cards.push(vec![deck_remaining[deck_idx], deck_remaining[deck_idx + 1]]);
                    deck_idx += 2;
                } else {
                    sim_hole_cards.push(vec![c1, c2]);
                }
            }

            // Deal community cards
            let runout: Vec<Card> = deck_remaining[deck_idx..deck_idx + cards_needed_board].to_vec();

            // Build complete board
            let mut full_board = request.board.clone();
            full_board.extend(runout);

            // Build complete hands
            let hands: Vec<Vec<Card>> = sim_hole_cards
                .into_iter()
                .map(|mut hole| {
                    hole.extend(full_board.iter().copied());
                    hole
                })
                .collect();

            // Find winners
            let winners = find_winners(&hands).unwrap();

            // Record results
            if winners.len() == 1 {
                let winner = winners[0];
                combo_wins[winner] += 1;
                combo_equity[winner] += 1.0;
            } else {
                let share = 1.0 / winners.len() as f64;
                for &idx in &winners {
                    combo_ties[idx] += 1;
                    combo_equity[idx] += share;
                }
            }
        }

        (combo_wins, combo_ties, combo_equity)
    };

    match strategy {
        EquityStrategy::Exhaustive { .. } => {
            // =================================================================
            // EXHAUSTIVE MODE: Process all combinations inline
            // =================================================================

            let odometer = Odometer::new(extents);
            for indices in odometer {
                if let Some((current_hands, remaining)) = is_valid_combination(&indices) {
                    total_combinations += 1;

                    let (combo_wins, combo_ties, combo_equity) =
                        run_simulation(&current_hands, &remaining, &mut rng);

                    total_simulations += sims_per_combo as u64;

                    let weight = 1.0;
                    total_weight += weight;

                    for i in 0..num_players {
                        let sim_count = sims_per_combo as f64;
                        total_equity[i] += (combo_equity[i] / sim_count) * weight;
                        total_wins[i] += (combo_wins[i] as f64 / sim_count) * weight;
                        total_ties[i] += (combo_ties[i] as f64 / sim_count) * weight;
                    }
                }
            }
        }

        EquityStrategy::ReservoirSampled { max_combos, .. } => {
            // =================================================================
            // RESERVOIR SAMPLING: Unbiased selection (iterates all combos)
            // =================================================================
            // This ensures each valid combination has equal probability of being
            // selected, regardless of its position in the odometer iteration.
            // Trade-off: Must iterate all combinations, slower for huge ranges.

            let mut reservoir: Vec<(Vec<(Card, Card)>, Vec<Card>)> =
                Vec::with_capacity(max_combos);
            let mut valid_count: usize = 0;

            // Phase 1: Collect samples using reservoir sampling (Algorithm R)
            let odometer = Odometer::new(extents.clone());
            for indices in odometer {
                if let Some((hands, remaining)) = is_valid_combination(&indices) {
                    valid_count += 1;

                    if reservoir.len() < max_combos {
                        // Fill the reservoir with first k valid combinations
                        reservoir.push((hands, remaining));
                    } else {
                        // Reservoir sampling: replace element j with probability k/n
                        let j = rng.random_range(0..valid_count);
                        if j < max_combos {
                            reservoir[j] = (hands, remaining);
                        }
                    }
                }
            }

            total_combinations = valid_count as u64;

            // Phase 2: Run simulations on reservoir samples
            for (hands, remaining) in &reservoir {
                let (combo_wins, combo_ties, combo_equity) =
                    run_simulation(hands, remaining, &mut rng);

                total_simulations += sims_per_combo as u64;

                let weight = 1.0;
                total_weight += weight;

                for i in 0..num_players {
                    let sim_count = sims_per_combo as f64;
                    total_equity[i] += (combo_equity[i] / sim_count) * weight;
                    total_wins[i] += (combo_wins[i] as f64 / sim_count) * weight;
                    total_ties[i] += (combo_ties[i] as f64 / sim_count) * weight;
                }
            }
        }

        EquityStrategy::BiasedSampled { max_combos, .. } => {
            // =================================================================
            // BIASED SAMPLING: Fast but biased toward front of odometer
            // =================================================================
            // Used for huge ranges (>10k combos) where reservoir sampling would
            // be too slow. Trade-off: Results may be biased toward combinations
            // that appear earlier in the odometer iteration order.

            let sample_rate = max_combos as f64 / total_theoretical_combos as f64;
            let mut sampled_count: usize = 0;

            let odometer = Odometer::new(extents);
            for indices in odometer {
                // Early exit once we have enough samples
                if sampled_count >= max_combos {
                    break;
                }

                // Probabilistic skip based on sample rate
                if rng.random::<f64>() > sample_rate {
                    continue;
                }

                if let Some((current_hands, remaining)) = is_valid_combination(&indices) {
                    total_combinations += 1;
                    sampled_count += 1;

                    let (combo_wins, combo_ties, combo_equity) =
                        run_simulation(&current_hands, &remaining, &mut rng);

                    total_simulations += sims_per_combo as u64;

                    let weight = 1.0;
                    total_weight += weight;

                    for i in 0..num_players {
                        let sim_count = sims_per_combo as f64;
                        total_equity[i] += (combo_equity[i] / sim_count) * weight;
                        total_wins[i] += (combo_wins[i] as f64 / sim_count) * weight;
                        total_ties[i] += (combo_ties[i] as f64 / sim_count) * weight;
                    }
                }
            }
        }
    }

    // Error if no valid combinations found (all combinations had card conflicts)
    if total_combinations == 0 {
        return Err(HoldemError::NoValidCombinations);
    }

    #[cfg(not(target_arch = "wasm32"))]
    let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
    #[cfg(target_arch = "wasm32")]
    let elapsed_ms = 0.0;

    // Normalize results
    let players: Vec<RangePlayerEquity> = (0..num_players)
        .map(|i| {
            let equity = if total_weight > 0.0 {
                total_equity[i] / total_weight
            } else {
                0.0
            };
            let win_rate = if total_weight > 0.0 {
                total_wins[i] / total_weight
            } else {
                0.0
            };
            let tie_rate = if total_weight > 0.0 {
                total_ties[i] / total_weight
            } else {
                0.0
            };

            RangePlayerEquity {
                index: i,
                equity,
                win_rate,
                tie_rate,
                combos: combo_counts[i],
                hand_description: hand_descriptions[i].clone(),
            }
        })
        .collect();

    Ok(RangeEquityResult {
        players,
        total_combinations,
        total_simulations,
        elapsed_ms,
    })
}

/// Convenience function: calculate equity of hole cards vs random opponents
///
/// # Errors
/// Returns an error if:
/// - `hole_cards.len() != 2`
/// - `num_opponents < 1`
pub fn equity_vs_random(
    hole_cards: &[Card],
    board: &[Card],
    num_opponents: usize,
    num_simulations: u32,
    seed: Option<u64>,
) -> HoldemResult<f64> {
    if hole_cards.len() != 2 {
        return Err(HoldemError::InvalidCardCount {
            expected: "2",
            got: hole_cards.len(),
        });
    }
    if num_opponents < 1 {
        return Err(HoldemError::NotEnoughOpponents(1));
    }

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

        // Find winners (unwrap is safe here - we always have 7-card hands)
        let winners = find_winners(&hands).unwrap();

        // Check if hero (index 0) won
        if winners.contains(&0) {
            equity_sum += 1.0 / winners.len() as f64;
        }
    }

    Ok(equity_sum / num_simulations as f64)
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

        let result = calculate_equity(&request).unwrap();

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

        let result = calculate_equity(&request).unwrap();

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

        let result = calculate_equity(&request).unwrap();

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

        let result1 = calculate_equity(&request1).unwrap();
        let result2 = calculate_equity(&request2).unwrap();

        assert_eq!(result1.players[0].equity, result2.players[0].equity);
    }

    #[test]
    fn test_equity_vs_random() {
        let hole = cards("Ah As");
        let equity = equity_vs_random(&hole, &[], 1, 10_000, Some(42)).unwrap();

        // AA vs 1 random should be ~85%
        assert!(equity > 0.80);
        assert!(equity < 0.90);
    }

    #[test]
    fn test_equity_vs_multiple_random() {
        let hole = cards("Ah As");
        let equity = equity_vs_random(&hole, &[], 5, 10_000, Some(42)).unwrap();

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

        let result = calculate_equity(&request).unwrap();

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

        let result = calculate_equity(&request).unwrap();

        assert_eq!(result.players.len(), 3);
        // AK vs 2 random should be ~47-50% equity
        assert!(result.players[0].equity > 0.40);
        assert!(result.players[0].equity < 0.55);
    }

    // =========================================================================
    // Range-based equity tests
    // =========================================================================

    #[test]
    fn test_range_vs_specific_aa_vs_kk() {
        use crate::CardDistribution;

        // AA (6 combos) vs specific KK
        let aa_dist = CardDistribution::from_range(&["AA".to_string()], &[]).unwrap();
        let kh = Card::parse("Kh").unwrap();
        let ks = Card::parse("Ks").unwrap();

        let request = RangeEquityRequest::new(
            vec![
                RangePlayer::range(aa_dist),
                RangePlayer::specific(kh, ks),
            ],
            vec![],
        )
        .with_simulations(1_000)
        .with_seed(42);

        let result = calculate_equity_with_ranges(&request).unwrap();

        assert_eq!(result.players.len(), 2);
        // AA should have ~82% equity vs KK
        assert!(result.players[0].equity > 0.75, "AA equity {} too low", result.players[0].equity);
        assert!(result.players[0].equity < 0.90, "AA equity {} too high", result.players[0].equity);
        // 6 combos for AA, 1 for specific KK
        assert_eq!(result.players[0].combos, 6);
        assert_eq!(result.players[1].combos, 1);
        // Should have evaluated 6 combinations (AA combos)
        assert_eq!(result.total_combinations, 6);
    }

    #[test]
    fn test_range_vs_range() {
        use crate::CardDistribution;

        // AA vs KK (both ranges)
        let aa_dist = CardDistribution::from_range(&["AA".to_string()], &[]).unwrap();
        let kk_dist = CardDistribution::from_range(&["KK".to_string()], &[]).unwrap();

        let request = RangeEquityRequest::new(
            vec![
                RangePlayer::range(aa_dist),
                RangePlayer::range(kk_dist),
            ],
            vec![],
        )
        .with_simulations(500)
        .with_seed(42);

        let result = calculate_equity_with_ranges(&request).unwrap();

        assert_eq!(result.players.len(), 2);
        // AA should have ~82% equity vs KK
        assert!(result.players[0].equity > 0.75, "AA equity {} too low", result.players[0].equity);
        assert!(result.players[0].equity < 0.90, "AA equity {} too high", result.players[0].equity);
        // 6 combos each
        assert_eq!(result.players[0].combos, 6);
        assert_eq!(result.players[1].combos, 6);
        // 6 * 6 = 36 total combinations, but some will conflict
        // AA and KK don't share cards, so all 36 should be valid
        assert_eq!(result.total_combinations, 36);
    }

    #[test]
    fn test_range_equity_sums_to_one() {
        use crate::CardDistribution;

        let aa_dist = CardDistribution::from_range(&["AA".to_string()], &[]).unwrap();
        let kk_dist = CardDistribution::from_range(&["KK".to_string()], &[]).unwrap();

        let request = RangeEquityRequest::new(
            vec![
                RangePlayer::range(aa_dist),
                RangePlayer::range(kk_dist),
            ],
            vec![],
        )
        .with_simulations(500)
        .with_seed(42);

        let result = calculate_equity_with_ranges(&request).unwrap();

        let total: f64 = result.players.iter().map(|p| p.equity).sum();
        assert!((total - 1.0).abs() < 0.02, "Total equity {} should be ~1.0", total);
    }

    #[test]
    fn test_range_with_card_conflicts() {
        use crate::CardDistribution;

        // AKs vs AQs - they share the Ace of each suit
        let aks_dist = CardDistribution::from_range(&["AKs".to_string()], &[]).unwrap();
        let aqs_dist = CardDistribution::from_range(&["AQs".to_string()], &[]).unwrap();

        let request = RangeEquityRequest::new(
            vec![
                RangePlayer::range(aks_dist),
                RangePlayer::range(aqs_dist),
            ],
            vec![],
        )
        .with_simulations(500)
        .with_seed(42);

        let result = calculate_equity_with_ranges(&request).unwrap();

        // 4 combos each, but each AKs combo conflicts with one AQs combo (same suit Ace)
        // So we expect 4 * 4 - 4 = 12 valid combinations
        assert_eq!(result.total_combinations, 12);
        // AKs should be favored over AQs
        assert!(result.players[0].equity > 0.65, "AKs equity {} too low", result.players[0].equity);
    }

    #[test]
    fn test_range_vs_random() {
        use crate::CardDistribution;

        let aa_dist = CardDistribution::from_range(&["AA".to_string()], &[]).unwrap();

        let request = RangeEquityRequest::new(
            vec![
                RangePlayer::range(aa_dist),
                RangePlayer::random(),
            ],
            vec![],
        )
        .with_simulations(500)
        .with_seed(42);

        let result = calculate_equity_with_ranges(&request).unwrap();

        // AA vs random should be ~85%
        assert!(result.players[0].equity > 0.80, "AA equity {} too low", result.players[0].equity);
        assert!(result.players[0].equity < 0.90, "AA equity {} too high", result.players[0].equity);
    }

    #[test]
    fn test_multiple_hands_in_range() {
        use crate::CardDistribution;

        // AA+KK (12 combos total) vs QQ (6 combos)
        let aa_kk = CardDistribution::from_range(
            &["AA".to_string(), "KK".to_string()],
            &[],
        ).unwrap();
        let qq = CardDistribution::from_range(&["QQ".to_string()], &[]).unwrap();

        assert_eq!(aa_kk.len(), 12);
        assert_eq!(qq.len(), 6);

        let request = RangeEquityRequest::new(
            vec![
                RangePlayer::range(aa_kk),
                RangePlayer::range(qq),
            ],
            vec![],
        )
        .with_simulations(200)
        .with_seed(42);

        let result = calculate_equity_with_ranges(&request).unwrap();

        assert_eq!(result.players[0].combos, 12);
        assert_eq!(result.players[1].combos, 6);
        // AA+KK vs QQ should be heavily favored
        assert!(result.players[0].equity > 0.70, "AA+KK equity {} too low", result.players[0].equity);
    }

    // =========================================================================
    // Strategy selection tests
    // =========================================================================

    #[test]
    fn test_strategy_selection_small_range() {
        // Small range (<50 combos) should use Exhaustive with high sims
        let strategy = select_strategy(30, 1000);
        match strategy {
            EquityStrategy::Exhaustive { sims_per_combo } => {
                assert!(sims_per_combo >= 1000, "Small range should have at least 1000 sims");
            }
            EquityStrategy::ReservoirSampled { .. } | EquityStrategy::BiasedSampled { .. } => {
                panic!("Small range should use Exhaustive strategy");
            }
        }
    }

    #[test]
    fn test_strategy_selection_medium_range() {
        // Medium range (50-500 combos) should use Exhaustive with reduced sims
        let strategy = select_strategy(200, 10000);
        match strategy {
            EquityStrategy::Exhaustive { sims_per_combo } => {
                // Should reduce sims to control time: 10000 * 50 / 200 = 2500
                assert!(sims_per_combo < 10000, "Medium range should reduce sims");
                assert!(sims_per_combo >= MIN_SIMS_PER_COMBO, "Should not go below minimum");
            }
            EquityStrategy::ReservoirSampled { .. } | EquityStrategy::BiasedSampled { .. } => {
                panic!("Medium range should use Exhaustive strategy");
            }
        }
    }

    #[test]
    fn test_strategy_selection_large_range() {
        // Large range (500-10000 combos) should use ReservoirSampled (unbiased)
        let strategy = select_strategy(1000, 5000);
        match strategy {
            EquityStrategy::Exhaustive { .. } => {
                panic!("Large range should use ReservoirSampled strategy");
            }
            EquityStrategy::ReservoirSampled { max_combos, sims_per_combo } => {
                assert_eq!(max_combos, MAX_SAMPLED_COMBOS);
                assert_eq!(sims_per_combo, 5000);
            }
            EquityStrategy::BiasedSampled { .. } => {
                panic!("Medium-large range should use ReservoirSampled, not BiasedSampled");
            }
        }
    }

    #[test]
    fn test_large_range_uses_sampling() {
        use crate::CardDistribution;

        // All pairs vs all pairs: 13 * 6 = 78 combos each
        // 78 * 78 = 6084 total combinations - should use sampling
        let pairs = [
            "AA", "KK", "QQ", "JJ", "TT", "99", "88", "77", "66", "55", "44", "33", "22",
        ];
        let pair_range: Vec<String> = pairs.iter().map(|s| s.to_string()).collect();

        let dist1 = CardDistribution::from_range(&pair_range, &[]).unwrap();
        let dist2 = CardDistribution::from_range(&pair_range, &[]).unwrap();

        assert_eq!(dist1.len(), 78);
        assert_eq!(dist2.len(), 78);

        let request = RangeEquityRequest::new(
            vec![
                RangePlayer::range(dist1),
                RangePlayer::range(dist2),
            ],
            vec![],
        )
        .with_simulations(100)
        .with_seed(42);

        let result = calculate_equity_with_ranges(&request).unwrap();

        // Should complete without timeout and return reasonable results
        assert_eq!(result.players.len(), 2);
        // With same ranges, equity should be close to 50/50
        assert!(result.players[0].equity > 0.40, "P1 equity {} too low", result.players[0].equity);
        assert!(result.players[0].equity < 0.60, "P1 equity {} too high", result.players[0].equity);
        // Due to sampling, total_combinations should be much less than 78*78
        // Actually evaluated combos depend on sampling and conflict
        assert!(result.total_combinations > 0);
    }

    // =========================================================================
    // Validation and error handling tests
    // =========================================================================

    #[test]
    fn test_specific_player_conflicts_with_board() {
        // Specific player's cards conflict with board - should error
        let ah = Card::parse("Ah").unwrap();
        let kh = Card::parse("Kh").unwrap();
        let qh = Card::parse("Qh").unwrap();
        let jh = Card::parse("Jh").unwrap();
        let th = Card::parse("Th").unwrap();

        // Player 1 has Ah, but Ah is also on the board
        let request = RangeEquityRequest::new(
            vec![
                RangePlayer::specific(ah, kh), // Ah conflicts with board
                RangePlayer::specific(qh, jh),
            ],
            vec![ah, th, Card::parse("2c").unwrap()], // board contains Ah
        );

        let result = calculate_equity_with_ranges(&request);
        assert!(result.is_err(), "Should error when specific player card conflicts with board");
        match result.unwrap_err() {
            HoldemError::DuplicateCard(card) => {
                assert_eq!(card, "Ah", "Error should mention the conflicting card");
            }
            e => panic!("Expected DuplicateCard error, got {:?}", e),
        }
    }

    #[test]
    fn test_specific_players_conflict_with_each_other() {
        // Two specific players have the same card
        let ah = Card::parse("Ah").unwrap();
        let kh = Card::parse("Kh").unwrap();
        let qh = Card::parse("Qh").unwrap();

        let request = RangeEquityRequest::new(
            vec![
                RangePlayer::specific(ah, kh),
                RangePlayer::specific(ah, qh), // Also uses Ah
            ],
            vec![],
        );

        let result = calculate_equity_with_ranges(&request);
        assert!(result.is_err(), "Should error when specific players share a card");
        assert!(matches!(result.unwrap_err(), HoldemError::DuplicateCard(_)));
    }

    #[test]
    fn test_all_combos_conflict_returns_error() {
        use crate::CardDistribution;

        // Create a scenario where all combinations conflict:
        // Player 1 has AKs (4 combos: AhKh, AdKd, AcKc, AsKs)
        // All Kings are dead except Kh
        // So Player 1's only valid combo is AhKh
        // Player 2 also has AKs, and with Kd, Kc, Ks dead, only valid is AhKh
        // But AhKh vs AhKh conflicts!

        let kd = Card::parse("Kd").unwrap();
        let kc = Card::parse("Kc").unwrap();
        let ks = Card::parse("Ks").unwrap();

        // Build AKs range with dead cards
        let dead = vec![kd, kc, ks];
        let aks1 = CardDistribution::from_range(&["AKs".to_string()], &dead).unwrap();
        let aks2 = CardDistribution::from_range(&["AKs".to_string()], &dead).unwrap();

        // Both should have only 1 combo (AhKh)
        assert_eq!(aks1.len(), 1, "With 3 kings dead, only AhKh remains");
        assert_eq!(aks2.len(), 1);

        let request = RangeEquityRequest::new(
            vec![RangePlayer::range(aks1), RangePlayer::range(aks2)],
            vec![], // board empty
        )
        .with_simulations(100)
        .with_seed(42)
        .with_dead_cards(dead);

        let result = calculate_equity_with_ranges(&request);
        assert!(
            result.is_err(),
            "Should error when all combinations have conflicts"
        );
        assert!(
            matches!(result.unwrap_err(), HoldemError::NoValidCombinations),
            "Should return NoValidCombinations error"
        );
    }

    #[test]
    fn test_sampling_covers_full_range() {
        use crate::CardDistribution;

        // Test that reservoir sampling produces valid results for large ranges
        // Must use ranges large enough to trigger sampling (>500 combos)
        //
        // Using: 10 pairs (60 combos) × 10 broadway (120 combos) = 7200 combos
        // This should trigger ReservoirSampled mode

        let pairs = ["AA", "KK", "QQ", "JJ", "TT", "99", "88", "77", "66", "55"];
        let broadway = [
            "AKs", "AQs", "AJs", "ATs", "KQs", "KJs", "KTs", "QJs", "QTs", "JTs",
        ];

        let range1: Vec<String> = pairs.iter().map(|s| s.to_string()).collect();
        let range2: Vec<String> = broadway.iter().map(|s| s.to_string()).collect();

        let dist1 = CardDistribution::from_range(&range1, &[]).unwrap();
        let dist2 = CardDistribution::from_range(&range2, &[]).unwrap();

        // Verify we exceed the sampling threshold
        let total_combos = dist1.len() * dist2.len();
        assert!(
            total_combos > MEDIUM_RANGE_THRESHOLD,
            "Test setup error: combos {} should exceed threshold {}",
            total_combos,
            MEDIUM_RANGE_THRESHOLD
        );

        // Run with seed 1
        let request1 = RangeEquityRequest::new(
            vec![
                RangePlayer::range(dist1.clone()),
                RangePlayer::range(dist2.clone()),
            ],
            vec![],
        )
        .with_simulations(100)
        .with_seed(1);

        let result1 = calculate_equity_with_ranges(&request1).unwrap();

        // Run with seed 2
        let dist1b = CardDistribution::from_range(&range1, &[]).unwrap();
        let dist2b = CardDistribution::from_range(&range2, &[]).unwrap();

        let request2 = RangeEquityRequest::new(
            vec![RangePlayer::range(dist1b), RangePlayer::range(dist2b)],
            vec![],
        )
        .with_simulations(100)
        .with_seed(2);

        let result2 = calculate_equity_with_ranges(&request2).unwrap();

        // Both should produce valid results
        assert!(result1.players[0].equity > 0.0);
        assert!(result2.players[0].equity > 0.0);

        // Pairs vs broadway suited - pairs should be favored (~55-65%)
        assert!(
            result1.players[0].equity > 0.45,
            "Pairs equity {} too low",
            result1.players[0].equity
        );
        assert!(
            result1.players[0].equity < 0.75,
            "Pairs equity {} too high",
            result1.players[0].equity
        );
    }

    // =========================================================================
    // Board/Dead duplicate validation tests
    // =========================================================================

    #[test]
    fn test_board_duplicate_cards_error() {
        // Board has duplicate cards - should error
        let ah = Card::parse("Ah").unwrap();
        let kh = Card::parse("Kh").unwrap();

        let request = RangeEquityRequest::new(
            vec![RangePlayer::random(), RangePlayer::random()],
            vec![ah, kh, ah], // Duplicate Ah in board
        );

        let result = calculate_equity_with_ranges(&request);
        assert!(result.is_err(), "Should error when board has duplicate cards");
        assert!(matches!(result.unwrap_err(), HoldemError::DuplicateCard(_)));
    }

    #[test]
    fn test_dead_cards_duplicate_error() {
        // Dead cards have duplicate - should error
        let ah = Card::parse("Ah").unwrap();

        let request = RangeEquityRequest::new(
            vec![RangePlayer::random(), RangePlayer::random()],
            vec![],
        )
        .with_dead_cards(vec![ah, ah]); // Duplicate Ah in dead cards

        let result = calculate_equity_with_ranges(&request);
        assert!(
            result.is_err(),
            "Should error when dead cards have duplicates"
        );
        assert!(matches!(result.unwrap_err(), HoldemError::DuplicateCard(_)));
    }

    #[test]
    fn test_board_and_dead_overlap_error() {
        // Board and dead cards share a card - should error
        let ah = Card::parse("Ah").unwrap();
        let kh = Card::parse("Kh").unwrap();
        let qh = Card::parse("Qh").unwrap();

        let request = RangeEquityRequest::new(
            vec![RangePlayer::random(), RangePlayer::random()],
            vec![ah, kh], // Board
        )
        .with_dead_cards(vec![ah, qh]); // Ah overlaps with board

        let result = calculate_equity_with_ranges(&request);
        assert!(
            result.is_err(),
            "Should error when board and dead cards overlap"
        );
        assert!(matches!(result.unwrap_err(), HoldemError::DuplicateCard(_)));
    }

    #[test]
    fn test_huge_range_uses_biased_sampling() {
        // Huge range (>10k combos) should use BiasedSampled strategy
        let strategy = select_strategy(15_000, 1000);
        assert!(
            matches!(strategy, EquityStrategy::BiasedSampled { .. }),
            "Huge range should use BiasedSampled, got {:?}",
            strategy
        );
    }

    #[test]
    fn test_medium_large_range_uses_reservoir_sampling() {
        // Large range (500-10k combos) should use ReservoirSampled strategy
        let strategy = select_strategy(5_000, 1000);
        assert!(
            matches!(strategy, EquityStrategy::ReservoirSampled { .. }),
            "Large range should use ReservoirSampled, got {:?}",
            strategy
        );
    }
}
