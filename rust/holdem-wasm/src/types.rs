//! Type definitions for WASM bindings.
//!
//! These types mirror the frontend TypeScript interfaces in `web/frontend/src/api/types.ts`
//! and match the Tauri commands in `rust/holdem-app/src-tauri/src/commands.rs`.

use holdem_core::{
    canonize::CanonicalHand,
    draws::DrawType,
    Card, Suit,
};
use serde::{Deserialize, Serialize};

// ============================================================================
// Equity Types
// ============================================================================

/// Player input for equity calculation (matches TypeScript `PlayerHandInput`)
#[derive(Debug, Deserialize)]
pub struct PlayerInput {
    /// Specific cards (e.g., ["Ah", "Kh"])
    pub cards: Option<Vec<String>>,
    /// Range notation (e.g., ["AA", "AKs"])
    pub range: Option<Vec<String>>,
    /// Random hand (sampled each simulation)
    #[serde(default)]
    pub random: bool,
}

/// Request for equity calculation (matches TypeScript `EquityRequest`)
#[derive(Debug, Deserialize)]
pub struct EquityRequestInput {
    pub players: Vec<PlayerInput>,
    #[serde(default)]
    pub board: Vec<String>,
    #[serde(default)]
    pub dead_cards: Vec<String>,
    #[serde(default = "default_simulations")]
    pub num_simulations: u32,
}

fn default_simulations() -> u32 {
    10_000
}

/// Equity result output (matches TypeScript `EquityResponse`)
#[derive(Debug, Serialize)]
pub struct EquityResultOutput {
    pub players: Vec<PlayerEquityOutput>,
    pub total_simulations: u64,
    pub elapsed_ms: f64,
}

/// Per-player equity result (matches TypeScript `PlayerEquityResult`)
#[derive(Debug, Serialize)]
pub struct PlayerEquityOutput {
    pub index: usize,
    pub hand_description: String,
    pub equity: f64,
    pub win_rate: f64,
    pub tie_rate: f64,
    pub combos: usize,
}

// ============================================================================
// Draw Analysis Types
// ============================================================================

/// Flush draw info (matches TypeScript `FlushDrawInfo`)
#[derive(Debug, Serialize)]
pub struct FlushDrawOutput {
    pub suit: String,
    pub suit_symbol: String,
    pub cards_held: usize,
    pub outs: Vec<String>,
    pub out_count: usize,
    pub is_nut: bool,
    pub draw_type: String,
}

/// Straight draw info (matches TypeScript `StraightDrawInfo`)
#[derive(Debug, Serialize)]
pub struct StraightDrawOutput {
    pub draw_type: String,
    pub needed_ranks: Vec<u8>,
    pub outs: Vec<String>,
    pub out_count: usize,
    pub high_card: u8,
    pub is_nut: bool,
}

/// Draw analysis result (matches TypeScript `DrawsResponse`)
#[derive(Debug, Serialize)]
pub struct DrawAnalysisOutput {
    pub has_flush: bool,
    pub has_straight: bool,
    pub flush_draws: Vec<FlushDrawOutput>,
    pub straight_draws: Vec<StraightDrawOutput>,
    pub total_outs: usize,
    pub all_outs: Vec<String>,
    pub is_combo_draw: bool,
}

// ============================================================================
// Canonical Hands Types
// ============================================================================

/// Canonical hand info (matches TypeScript `CanonicalHandInfo`)
#[derive(Debug, Serialize)]
pub struct CanonicalHandOutput {
    pub notation: String,
    pub high_rank: String,
    pub low_rank: String,
    pub suited: bool,
    pub is_pair: bool,
    pub num_combos: usize,
    pub matrix_row: usize,
    pub matrix_col: usize,
}

impl From<&CanonicalHand> for CanonicalHandOutput {
    fn from(hand: &CanonicalHand) -> Self {
        Self {
            notation: hand.notation(),
            high_rank: hand.high_rank.to_char().to_string(),
            low_rank: hand.low_rank.to_char().to_string(),
            suited: hand.suited,
            is_pair: hand.is_pair(),
            num_combos: hand.num_combos(),
            matrix_row: hand.matrix_row(),
            matrix_col: hand.matrix_col(),
        }
    }
}

/// Response for canonical hands (matches TypeScript `CanonicalHandsResponse`)
#[derive(Debug, Serialize)]
pub struct CanonicalHandsOutput {
    pub hands: Vec<CanonicalHandOutput>,
    pub total: usize,
}

// ============================================================================
// Parse Cards Types
// ============================================================================

/// Card info (matches TypeScript `CardInfo`)
#[derive(Debug, Serialize)]
pub struct CardInfoOutput {
    pub notation: String,
    pub rank: String,
    pub suit: String,
    pub suit_symbol: String,
}

impl From<Card> for CardInfoOutput {
    fn from(card: Card) -> Self {
        Self {
            notation: card.to_string(),
            rank: card.rank.to_char().to_string(),
            suit: card.suit.to_char().to_string(),
            suit_symbol: suit_symbol(card.suit),
        }
    }
}

/// Parse cards result (matches TypeScript `ParseCardsResponse`)
#[derive(Debug, Serialize)]
pub struct ParseCardsOutput {
    pub cards: Vec<CardInfoOutput>,
    pub formatted: String,
    pub valid: bool,
    pub error: Option<String>,
}

// ============================================================================
// Hand Evaluation Types
// ============================================================================

/// Hand evaluation result (matches TypeScript `EvaluateResponse`)
#[derive(Debug, Serialize)]
pub struct EvaluateOutput {
    pub hand_type: String,
    pub description: String,
    pub primary_ranks: Vec<u8>,
    pub kickers: Vec<u8>,
}

// ============================================================================
// Health Check
// ============================================================================

/// Health check response (matches TypeScript `HealthResponse`)
#[derive(Debug, Serialize)]
pub struct HealthOutput {
    pub status: String,
    pub version: String,
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Get Unicode suit symbol
pub fn suit_symbol(suit: Suit) -> String {
    match suit {
        Suit::Hearts => "\u{2665}".to_string(),   // ♥
        Suit::Diamonds => "\u{2666}".to_string(), // ♦
        Suit::Clubs => "\u{2663}".to_string(),    // ♣
        Suit::Spades => "\u{2660}".to_string(),   // ♠
    }
}

/// Convert draw type to string
pub fn draw_type_string(dt: DrawType) -> String {
    match dt {
        DrawType::FlushDraw => "flush_draw".to_string(),
        DrawType::BackdoorFlush => "backdoor_flush".to_string(),
        DrawType::OpenEnded => "open_ended".to_string(),
        DrawType::Gutshot => "gutshot".to_string(),
        DrawType::DoubleGutshot => "double_gutshot".to_string(),
        DrawType::BackdoorStraight => "backdoor_straight".to_string(),
    }
}

/// Parse card strings to Card objects
pub fn parse_card_strings(strings: &[String]) -> Result<Vec<Card>, String> {
    strings
        .iter()
        .map(|s| Card::parse(s).map_err(|e| e.to_string()))
        .collect()
}
