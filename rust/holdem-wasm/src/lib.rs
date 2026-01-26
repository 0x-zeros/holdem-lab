//! WebAssembly bindings for holdem-core.
//!
//! This crate provides WASM-compatible functions that can be called from JavaScript.
//! All functions use JSON serialization via `serde-wasm-bindgen` for data exchange.

use wasm_bindgen::prelude::*;

use holdem_core::{
    canonize,
    card,
    draws,
    equity::{self, PlayerHand},
    Card,
};

mod types;
mod utils;

use types::*;

// ============================================================================
// Module Initialization
// ============================================================================

/// Initialize the WASM module. Called automatically on load.
#[wasm_bindgen(start)]
pub fn init() {
    utils::set_panic_hook();
}

// ============================================================================
// Health Check
// ============================================================================

/// Health check endpoint for API consistency.
#[wasm_bindgen]
pub fn wasm_health() -> JsValue {
    let output = HealthOutput {
        status: "healthy".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
    };
    serde_wasm_bindgen::to_value(&output).unwrap_or(JsValue::NULL)
}

// ============================================================================
// Equity Calculation
// ============================================================================

/// Calculate equity for multiple players.
///
/// # Arguments
/// * `request` - JsValue containing `EquityRequest` (players, board, dead_cards, num_simulations)
///
/// # Returns
/// JsValue containing `EquityResponse` (players with equity, win_rate, tie_rate, etc.)
#[wasm_bindgen]
pub fn wasm_calculate_equity(request: JsValue) -> Result<JsValue, JsValue> {
    let req: EquityRequestInput = serde_wasm_bindgen::from_value(request)
        .map_err(|e| JsValue::from_str(&format!("Failed to parse request: {e}")))?;

    let result = calculate_equity_impl(req)
        .map_err(|e| JsValue::from_str(&e))?;

    serde_wasm_bindgen::to_value(&result)
        .map_err(|e| JsValue::from_str(&format!("Failed to serialize result: {e}")))
}

fn calculate_equity_impl(request: EquityRequestInput) -> Result<EquityResultOutput, String> {
    // Parse board
    let board = parse_card_strings(&request.board)?;

    // Parse dead cards
    let dead_cards = parse_card_strings(&request.dead_cards)?;

    // First pass: collect all specific cards from players
    let mut specific_cards: Vec<Card> = Vec::new();
    for player_input in &request.players {
        if let Some(cards) = &player_input.cards {
            if !cards.is_empty() {
                if let Ok(parsed) = parse_card_strings(cards) {
                    specific_cards.extend(parsed);
                }
            }
        }
    }

    // Parse players
    let mut players: Vec<PlayerHand> = Vec::new();
    let mut hand_descriptions: Vec<String> = Vec::new();
    let mut combo_counts: Vec<usize> = Vec::new();

    for (i, player_input) in request.players.iter().enumerate() {
        if let Some(cards) = &player_input.cards {
            if !cards.is_empty() {
                let parsed = parse_card_strings(cards)?;
                if parsed.len() != 2 {
                    return Err(format!(
                        "Player {} must have exactly 2 cards, got {}",
                        i + 1,
                        parsed.len()
                    ));
                }
                hand_descriptions.push(format!("{}{}", parsed[0], parsed[1]));
                combo_counts.push(1);
                players.push(PlayerHand::new(parsed));
            }
        } else if let Some(range) = &player_input.range {
            if range.is_empty() {
                return Err(format!("Player {} has empty range", i + 1));
            }

            let canonical = canonize::CanonicalHand::parse(&range[0])
                .map_err(|e| format!("Invalid range '{}': {}", range[0], e))?;

            // Combine dead cards, board cards, and specific cards from other players
            let mut excluded: Vec<Card> = dead_cards.clone();
            excluded.extend(board.iter().cloned());
            excluded.extend(specific_cards.iter().cloned());

            let combos = canonize::get_combos_excluding(&canonical, &excluded);
            if combos.is_empty() {
                return Err(format!(
                    "No valid combos for player {} range '{}'",
                    i + 1,
                    range[0]
                ));
            }

            hand_descriptions.push(range.join(", "));
            combo_counts.push(combos.len());

            // Use first available combo
            let (c1, c2) = combos[0];
            players.push(PlayerHand::new(vec![c1, c2]));
        } else if player_input.random {
            hand_descriptions.push("Random".to_string());
            combo_counts.push(1326); // C(52,2) total possible hands
            players.push(PlayerHand::random());
        } else {
            return Err(format!(
                "Player {} has no cards, range, or random specified",
                i + 1
            ));
        }
    }

    if players.len() < 2 {
        return Err("Need at least 2 players".to_string());
    }

    // Build equity request
    let eq_request = equity::EquityRequest::new(players, board)
        .with_simulations(request.num_simulations)
        .with_dead_cards(dead_cards);

    // Use js_sys::Date for timing in WASM (std::time::Instant not available)
    let start = js_sys::Date::now();
    let result = equity::calculate_equity(&eq_request)
        .map_err(|e| e.to_string())?;
    let elapsed_ms = js_sys::Date::now() - start;

    // Convert to output format
    Ok(EquityResultOutput {
        players: result
            .players
            .iter()
            .enumerate()
            .map(|(i, p)| PlayerEquityOutput {
                index: p.index,
                hand_description: hand_descriptions.get(i).cloned().unwrap_or_default(),
                equity: p.equity,
                win_rate: p.win_rate,
                tie_rate: p.tie_rate,
                combos: combo_counts.get(i).copied().unwrap_or(1),
            })
            .collect(),
        total_simulations: result.total_simulations,
        elapsed_ms,
    })
}

// ============================================================================
// Draw Analysis
// ============================================================================

/// Analyze draws for given hole cards and board.
///
/// # Arguments
/// * `hole_cards` - JsValue array of card strings (e.g., ["Ah", "Kh"])
/// * `board` - JsValue array of board cards (e.g., ["Qh", "Jh", "2c"])
/// * `dead_cards` - JsValue array of dead cards (optional)
///
/// # Returns
/// JsValue containing `DrawsResponse`
#[wasm_bindgen]
pub fn wasm_analyze_draws(
    hole_cards: JsValue,
    board: JsValue,
    dead_cards: JsValue,
) -> Result<JsValue, JsValue> {
    let hole: Vec<String> = serde_wasm_bindgen::from_value(hole_cards)
        .map_err(|e| JsValue::from_str(&format!("Failed to parse hole cards: {e}")))?;

    let board_cards: Vec<String> = serde_wasm_bindgen::from_value(board)
        .map_err(|e| JsValue::from_str(&format!("Failed to parse board: {e}")))?;

    let dead: Vec<String> = serde_wasm_bindgen::from_value(dead_cards).unwrap_or_default();

    let result = analyze_draws_impl(hole, board_cards, dead)
        .map_err(|e| JsValue::from_str(&e))?;

    serde_wasm_bindgen::to_value(&result)
        .map_err(|e| JsValue::from_str(&format!("Failed to serialize result: {e}")))
}

fn analyze_draws_impl(
    hole_cards: Vec<String>,
    board: Vec<String>,
    dead_cards: Vec<String>,
) -> Result<DrawAnalysisOutput, String> {
    let hole = parse_card_strings(&hole_cards)?;
    if hole.len() != 2 {
        return Err(format!("Need exactly 2 hole cards, got {}", hole.len()));
    }

    let board = parse_card_strings(&board)?;
    if board.len() > 5 {
        return Err(format!("Board cannot exceed 5 cards, got {}", board.len()));
    }

    let dead = parse_card_strings(&dead_cards)?;

    let analysis = draws::analyze_draws(&hole, &board, &dead)
        .map_err(|e| e.to_string())?;

    Ok(DrawAnalysisOutput {
        has_flush: analysis.has_flush,
        has_straight: analysis.has_straight,
        flush_draws: analysis
            .flush_draws
            .iter()
            .map(|d| FlushDrawOutput {
                suit: d.suit.to_char().to_string(),
                suit_symbol: suit_symbol(d.suit),
                cards_held: d.cards_held,
                outs: d.outs.iter().map(ToString::to_string).collect(),
                out_count: d.out_count(),
                is_nut: d.is_nut,
                draw_type: draw_type_string(d.draw_type()),
            })
            .collect(),
        straight_draws: analysis
            .straight_draws
            .iter()
            .map(|d| StraightDrawOutput {
                draw_type: draw_type_string(d.draw_type),
                needed_ranks: d.needed_ranks.clone(),
                outs: d.outs.iter().map(ToString::to_string).collect(),
                out_count: d.out_count(),
                high_card: d.high_card,
                is_nut: d.is_nut,
            })
            .collect(),
        total_outs: analysis.total_outs,
        all_outs: analysis.all_outs.iter().map(ToString::to_string).collect(),
        is_combo_draw: analysis.is_combo_draw(),
    })
}

// ============================================================================
// Canonical Hands
// ============================================================================

/// Get all 169 canonical starting hands.
///
/// # Returns
/// JsValue containing `CanonicalHandsResponse` with array of hands and total count
#[wasm_bindgen]
pub fn wasm_get_canonical_hands() -> Result<JsValue, JsValue> {
    let hands: Vec<CanonicalHandOutput> = canonize::get_all_canonical_hands()
        .iter()
        .map(CanonicalHandOutput::from)
        .collect();

    let output = CanonicalHandsOutput {
        total: hands.len(),
        hands,
    };

    serde_wasm_bindgen::to_value(&output)
        .map_err(|e| JsValue::from_str(&format!("Failed to serialize result: {e}")))
}

// ============================================================================
// Card Parsing
// ============================================================================

/// Parse cards from string notation.
///
/// # Arguments
/// * `input` - Card string (e.g., "AhKh" or "Ah Kh" or "Ah,Kh")
///
/// # Returns
/// JsValue containing `ParseCardsResponse`
#[wasm_bindgen]
pub fn wasm_parse_cards(input: &str) -> JsValue {
    let output = match card::parse_cards(input) {
        Ok(cards) => {
            let formatted = cards
                .iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>()
                .join(" ");
            ParseCardsOutput {
                cards: cards.into_iter().map(CardInfoOutput::from).collect(),
                formatted,
                valid: true,
                error: None,
            }
        }
        Err(e) => ParseCardsOutput {
            cards: Vec::new(),
            formatted: String::new(),
            valid: false,
            error: Some(e.to_string()),
        },
    };

    serde_wasm_bindgen::to_value(&output).unwrap_or(JsValue::NULL)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_health() {
        // Can't test JsValue in regular tests, just ensure it compiles
    }
}
