#!/usr/bin/env python3
"""Generate preflop equity data for all 169 starting hands.

This script calculates the equity of each canonical starting hand
against random opponents for different player counts (2-10).

Usage:
    cd holdem-core
    uv run python scripts/generate_preflop_equity.py
"""

import json
import sys
from pathlib import Path

from holdem_lab import (
    EquityRequest,
    PlayerHand,
    calculate_equity,
    get_all_canonical_hands,
    get_all_combos,
)


def calculate_preflop_equity(
    num_players: int, num_simulations: int = 50000
) -> dict[str, float]:
    """Calculate equity for all 169 hands with specified number of players.

    Args:
        num_players: Total number of players (including hero)
        num_simulations: Monte Carlo simulations per hand

    Returns:
        Dictionary mapping hand notation to equity percentage
    """
    hands = get_all_canonical_hands()
    result = {}

    for i, hand in enumerate(hands):
        # Get all combos for this hand and use the first one as representative
        combos = get_all_combos(hand)
        c1, c2 = combos[0]

        # Build player list: hero with specific hand, others random
        players = [PlayerHand((c1, c2))]
        for _ in range(num_players - 1):
            players.append(PlayerHand(is_random=True))

        # Calculate equity
        req = EquityRequest(players=players, num_simulations=num_simulations)
        res = calculate_equity(req)

        # Store as percentage with 1 decimal place
        equity = round(res.players[0].equity * 100, 1)
        notation = str(hand)
        result[notation] = equity

        # Progress output
        print(f"[{num_players}p] {i + 1:3d}/169: {notation:4s} = {equity:5.1f}%")

    return result


def main():
    """Generate preflop equity data for all player counts."""
    output_path = Path(__file__).parent.parent.parent / "web/frontend/src/data/preflop-equity.json"

    print("=" * 50)
    print("Preflop Equity Generator")
    print("=" * 50)
    print(f"Output: {output_path}")
    print()

    all_data = {}

    for num_players in range(2, 11):
        print(f"\n{'=' * 50}")
        print(f"Calculating {num_players} players...")
        print("=" * 50)

        all_data[str(num_players)] = calculate_preflop_equity(
            num_players, num_simulations=50000
        )

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Write JSON file
    with open(output_path, "w") as f:
        json.dump(all_data, f, indent=2)

    print(f"\n{'=' * 50}")
    print(f"Data saved to {output_path}")
    print("=" * 50)


if __name__ == "__main__":
    main()
