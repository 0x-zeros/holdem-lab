//! Precompute preflop equity for all 169 canonical starting hands.
//!
//! Usage:
//!   cargo run --release --bin precompute -- --simulations 1000000
//!   cargo run --release --bin precompute -- --players 2 --simulations 100000

use holdem_core::canonize::{get_all_canonical_hands, get_all_combos, CanonicalHand};
use holdem_core::equity::{calculate_equity, EquityRequest, PlayerHand};
use serde_json::json;
use std::collections::BTreeMap;
use std::env;
use std::fs;
use std::time::Instant;

fn main() {
    let args: Vec<String> = env::args().collect();

    // Parse arguments
    let mut simulations: u32 = 1_000_000;
    let mut players: Option<usize> = None;
    let mut output_path: Option<String> = None;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--simulations" | "-s" => {
                if i + 1 < args.len() {
                    simulations = args[i + 1].parse().unwrap_or(1_000_000);
                    i += 1;
                }
            }
            "--players" | "-p" => {
                if i + 1 < args.len() {
                    players = args[i + 1].parse().ok();
                    i += 1;
                }
            }
            "--output" | "-o" => {
                if i + 1 < args.len() {
                    output_path = Some(args[i + 1].clone());
                    i += 1;
                }
            }
            "--help" | "-h" => {
                print_help();
                return;
            }
            _ => {}
        }
        i += 1;
    }

    // Get all canonical hands
    let hands = get_all_canonical_hands();

    // Determine which player counts to compute
    let player_counts: Vec<usize> = match players {
        Some(n) => vec![n],
        None => (2..=10).collect(),
    };

    // Print header
    println!("========================================");
    println!("Preflop Equity Precompute");
    println!("Simulations per hand: {}", format_number(simulations as u64));
    if players.is_some() {
        println!("Players: {}", players.unwrap());
    } else {
        println!("Players: 2-10 (all)");
    }
    println!("========================================");
    println!();

    let total_start = Instant::now();
    let mut results: BTreeMap<String, BTreeMap<String, f64>> = BTreeMap::new();

    for &num_players in &player_counts {
        println!("[{} players]", num_players);

        let subtotal_start = Instant::now();
        let mut player_results: BTreeMap<String, f64> = BTreeMap::new();

        for (idx, hand) in hands.iter().enumerate() {
            let hand_start = Instant::now();

            let equity = compute_hand_equity(hand, num_players, simulations);
            let equity_pct = (equity * 1000.0).round() / 10.0; // Round to 1 decimal

            let elapsed = hand_start.elapsed().as_secs_f64();

            println!(
                "[{:>3}/169] {:<4} ... {:>5.1}%  ({:.1}s)",
                idx + 1,
                hand.notation(),
                equity_pct,
                elapsed
            );

            player_results.insert(hand.notation(), equity_pct);
        }

        let subtotal_elapsed = subtotal_start.elapsed();
        println!("Subtotal: {}", format_duration(subtotal_elapsed.as_secs()));
        println!();

        results.insert(num_players.to_string(), player_results);
    }

    let total_elapsed = total_start.elapsed();

    // Output JSON
    let json = serde_json::to_string_pretty(&json!(results)).unwrap();

    if let Some(path) = &output_path {
        fs::write(path, &json).expect("Failed to write output file");
        println!("========================================");
        println!("Done! Total time: {}", format_duration(total_elapsed.as_secs()));
        println!("Output: {}", path);
        println!("========================================");
    } else {
        println!("========================================");
        println!("Done! Total time: {}", format_duration(total_elapsed.as_secs()));
        println!("========================================");
        println!();
        println!("{}", json);
    }
}

fn compute_hand_equity(hand: &CanonicalHand, num_players: usize, simulations: u32) -> f64 {
    // Get all actual card combinations for this hand
    let combos = get_all_combos(hand);

    if combos.is_empty() {
        return 0.0;
    }

    // Calculate equity for each combo and average
    let mut total_equity = 0.0;
    let sims_per_combo = simulations / combos.len() as u32;

    for (card1, card2) in &combos {
        // Build player hands: hero + (num_players - 1) random opponents
        let mut players = vec![PlayerHand::new(vec![*card1, *card2])];
        for _ in 1..num_players {
            players.push(PlayerHand::random());
        }

        let request = EquityRequest::new(players, vec![])
            .with_simulations(sims_per_combo);

        match calculate_equity(&request) {
            Ok(result) => {
                if !result.players.is_empty() {
                    total_equity += result.players[0].equity;
                }
            }
            Err(e) => {
                eprintln!("Error computing {}: {}", hand.notation(), e);
            }
        }
    }

    total_equity / combos.len() as f64
}

fn format_number(n: u64) -> String {
    let s = n.to_string();
    let mut result = String::new();
    for (i, c) in s.chars().rev().enumerate() {
        if i > 0 && i % 3 == 0 {
            result.insert(0, ',');
        }
        result.insert(0, c);
    }
    result
}

fn format_duration(secs: u64) -> String {
    if secs < 60 {
        format!("{}s", secs)
    } else if secs < 3600 {
        format!("{}m {}s", secs / 60, secs % 60)
    } else {
        format!("{}h {}m {}s", secs / 3600, (secs % 3600) / 60, secs % 60)
    }
}

fn print_help() {
    println!("Preflop Equity Precompute");
    println!();
    println!("Usage:");
    println!("  cargo run --release --bin precompute [OPTIONS]");
    println!();
    println!("Options:");
    println!("  -s, --simulations N    Simulations per hand (default: 1,000,000)");
    println!("  -p, --players N        Only compute for N players (default: 2-10 all)");
    println!("  -o, --output PATH      Output file path (default: stdout)");
    println!("  -h, --help             Show this help");
    println!();
    println!("Examples:");
    println!("  # Quick test (100k sims, 2 players only)");
    println!("  cargo run --release --bin precompute -- -s 100000 -p 2");
    println!();
    println!("  # Full precompute with output file");
    println!("  cargo run --release --bin precompute -- -o preflop-equity.json");
}
