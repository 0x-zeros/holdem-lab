//! # Holdem Core
//!
//! Texas Hold'em poker library providing:
//! - Card representation and deck management
//! - Hand evaluation (7-card to 5-card best hand)
//! - Equity calculation via Monte Carlo simulation
//! - Draw analysis (flush draws, straight draws)
//! - Canonical hand representation (169 starting hands)

pub mod card;
pub mod canonize;
pub mod draws;
pub mod equity;
pub mod error;
pub mod evaluator;
pub mod range;

// Re-export commonly used types
pub use card::{Card, Deck, Rank, Suit};
pub use canonize::{CanonicalHand, get_all_canonical_hands};
pub use draws::{analyze_draws, DrawAnalysis, DrawType, FlushDraw, StraightDraw};
pub use equity::{
    calculate_equity, calculate_equity_with_ranges, EquityRequest, EquityResult, PlayerEquity,
    PlayerHand, RangeEquityRequest, RangeEquityResult, RangePlayer, RangePlayerEquity,
};
pub use error::{HoldemError, HoldemResult};
pub use evaluator::{evaluate_hand, find_winners, HandRank, HandType};
pub use range::{CardDistribution, Odometer, RangeError};
