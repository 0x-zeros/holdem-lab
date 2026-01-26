//! Tauri IPC commands for the holdem app.

use holdem_core::{
    canonize::{self, CanonicalHand},
    card::{self, Card, Suit},
    draws::{self, DrawType},
    equity::{self, PlayerHand},
};
use serde::{Deserialize, Serialize};

/// Player input for equity calculation
#[derive(Debug, Deserialize)]
pub struct PlayerInput {
    /// Specific cards (e.g., ["Ah", "Kh"])
    pub cards: Option<Vec<String>>,
    /// Range notation (e.g., ["AA", "AKs"])
    pub range: Option<Vec<String>>,
}

/// Request for equity calculation from frontend
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

/// Equity result for frontend
#[derive(Debug, Serialize)]
pub struct EquityResultOutput {
    pub players: Vec<PlayerEquityOutput>,
    pub total_simulations: u64,
    pub elapsed_ms: f64,
}

#[derive(Debug, Serialize)]
pub struct PlayerEquityOutput {
    pub index: usize,
    pub hand_description: String,
    pub equity: f64,
    pub win_rate: f64,
    pub tie_rate: f64,
    pub combos: usize,
}

/// Parse card strings to Card objects
fn parse_card_strings(strings: &[String]) -> Result<Vec<Card>, String> {
    strings
        .iter()
        .map(|s| Card::parse(s).map_err(|e| e.to_string()))
        .collect()
}

/// Calculate equity for multiple players
#[tauri::command]
pub fn calculate_equity(request: EquityRequestInput) -> Result<EquityResultOutput, String> {
    // Parse board
    let board = parse_card_strings(&request.board)?;

    // Parse dead cards
    let dead_cards = parse_card_strings(&request.dead_cards)?;

    // Parse players
    let mut players: Vec<PlayerHand> = Vec::new();

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
                players.push(PlayerHand::new(parsed));
            }
        } else if let Some(range) = &player_input.range {
            // For range-based players, we need to expand the range
            // For now, just take the first combo of the first hand in range
            // TODO: Implement proper range vs range calculation
            if range.is_empty() {
                return Err(format!("Player {} has empty range", i + 1));
            }

            let canonical = canonize::CanonicalHand::parse(&range[0])
                .map_err(|e| format!("Invalid range '{}': {}", range[0], e))?;

            let combos = canonize::get_combos_excluding(&canonical, &dead_cards);
            if combos.is_empty() {
                return Err(format!(
                    "No valid combos for player {} range '{}'",
                    i + 1,
                    range[0]
                ));
            }

            // Use first available combo
            let (c1, c2) = combos[0];
            players.push(PlayerHand::new(vec![c1, c2]));
        } else {
            return Err(format!("Player {} has no cards or range specified", i + 1));
        }
    }

    // Track original player count before adding virtual opponent
    let original_player_count = players.len();

    // Handle single player: add a virtual opponent with full range (like PokerStove)
    if players.len() == 1 {
        // Get all 169 canonical hands and use first available combo as opponent
        let all_hands = canonize::get_all_canonical_hands();
        // Exclude player cards, board cards, and dead cards
        let mut excluded_cards: Vec<Card> = players.iter().flat_map(|p| p.cards.clone()).collect();
        excluded_cards.extend(board.iter().cloned());
        excluded_cards.extend(dead_cards.iter().cloned());
        let mut found_opponent = false;

        for hand in &all_hands {
            let combos = canonize::get_combos_excluding(hand, &excluded_cards);
            if let Some((c1, c2)) = combos.first() {
                players.push(PlayerHand::new(vec![*c1, *c2]));
                found_opponent = true;
                break;
            }
        }

        if !found_opponent {
            return Err("Could not find valid opponent hand".to_string());
        }
    }

    if players.is_empty() {
        return Err("Need at least 1 player".to_string());
    }

    // Build equity request
    let eq_request = equity::EquityRequest::new(players, board)
        .with_simulations(request.num_simulations)
        .with_dead_cards(dead_cards);

    let result = equity::calculate_equity(&eq_request);

    // Convert to output format (only return original players, exclude virtual opponent)
    Ok(EquityResultOutput {
        players: result
            .players
            .iter()
            .take(original_player_count)
            .map(|p| PlayerEquityOutput {
                index: p.index,
                hand_description: p.hand_description.clone(),
                equity: p.equity,
                win_rate: p.win_rate,
                tie_rate: p.tie_rate,
                combos: p.combos,
            })
            .collect(),
        total_simulations: result.total_simulations,
        elapsed_ms: result.elapsed_ms,
    })
}

/// Flush draw info for frontend
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

/// Straight draw info for frontend
#[derive(Debug, Serialize)]
pub struct StraightDrawOutput {
    pub draw_type: String,
    pub needed_ranks: Vec<u8>,
    pub outs: Vec<String>,
    pub out_count: usize,
    pub high_card: u8,
    pub is_nut: bool,
}

/// Draw analysis result for frontend
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

/// Get suit symbol
fn suit_symbol(suit: Suit) -> String {
    match suit {
        Suit::Hearts => "\u{2665}".to_string(),   // ♥
        Suit::Diamonds => "\u{2666}".to_string(), // ♦
        Suit::Clubs => "\u{2663}".to_string(),    // ♣
        Suit::Spades => "\u{2660}".to_string(),   // ♠
    }
}

/// Get draw type string
fn draw_type_string(dt: DrawType) -> String {
    match dt {
        DrawType::FlushDraw => "flush_draw".to_string(),
        DrawType::BackdoorFlush => "backdoor_flush".to_string(),
        DrawType::OpenEnded => "open_ended".to_string(),
        DrawType::Gutshot => "gutshot".to_string(),
        DrawType::DoubleGutshot => "double_gutshot".to_string(),
        DrawType::BackdoorStraight => "backdoor_straight".to_string(),
    }
}

/// Analyze draws for given hole cards and board
#[tauri::command]
pub fn analyze_draws(
    hole_cards: Vec<String>,
    board: Vec<String>,
    dead_cards: Option<Vec<String>>,
) -> Result<DrawAnalysisOutput, String> {
    let hole = parse_card_strings(&hole_cards)?;
    if hole.len() != 2 {
        return Err(format!("Need exactly 2 hole cards, got {}", hole.len()));
    }

    let board = parse_card_strings(&board)?;
    if board.len() > 5 {
        return Err(format!("Board cannot exceed 5 cards, got {}", board.len()));
    }

    let dead = dead_cards
        .map(|d| parse_card_strings(&d))
        .transpose()?
        .unwrap_or_default();

    let analysis = draws::analyze_draws(&hole, &board, &dead);

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

/// Canonical hand info for frontend
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

/// Get all 169 canonical starting hands
#[tauri::command]
pub fn get_canonical_hands() -> Vec<CanonicalHandOutput> {
    canonize::get_all_canonical_hands()
        .iter()
        .map(CanonicalHandOutput::from)
        .collect()
}

/// Parse cards result for frontend
#[derive(Debug, Serialize)]
pub struct ParseCardsOutput {
    pub cards: Vec<CardInfoOutput>,
    pub formatted: String,
    pub valid: bool,
    pub error: Option<String>,
}

/// Card info for frontend
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

/// Parse cards from string notation
#[tauri::command]
pub fn parse_cards(input: String) -> ParseCardsOutput {
    match card::parse_cards(&input) {
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
    }
}
