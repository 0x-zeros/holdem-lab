//! Error types for holdem-core.
//!
//! This module defines the error types used throughout the library,
//! providing structured error handling instead of panics.

use thiserror::Error;

/// Main error type for holdem-core operations.
#[derive(Debug, Clone, Error)]
pub enum HoldemError {
    /// Invalid number of cards provided
    #[error("Invalid card count: expected {expected}, got {got}")]
    InvalidCardCount {
        /// Expected card count description (e.g., "2", "5-7")
        expected: &'static str,
        /// Actual card count received
        got: usize,
    },

    /// Duplicate card detected in input
    #[error("Duplicate card detected: {0}")]
    DuplicateCard(String),

    /// Not enough players for equity calculation
    #[error("Need at least {0} players")]
    NotEnoughPlayers(usize),

    /// Too many board cards
    #[error("Board cannot exceed 5 cards, got {0}")]
    BoardTooLarge(usize),

    /// Not enough cards in deck
    #[error("Cannot deal {requested} cards, only {available} remain")]
    InsufficientCards {
        /// Number of cards requested
        requested: usize,
        /// Number of cards available
        available: usize,
    },

    /// Card not found in deck
    #[error("Card {0} not in deck")]
    CardNotInDeck(String),

    /// Card was already removed from deck
    #[error("Card {0} already removed")]
    CardAlreadyRemoved(String),

    /// Empty hands provided to find_winners
    #[error("Empty hands provided")]
    EmptyHands,

    /// Need at least one opponent
    #[error("Need at least {0} opponent(s)")]
    NotEnoughOpponents(usize),
}

/// Result type alias for holdem-core operations.
pub type HoldemResult<T> = Result<T, HoldemError>;
