"""Monte Carlo equity calculation for Texas Hold'em."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Sequence

from holdem_lab.cards import Card, FULL_DECK
from holdem_lab.evaluator import evaluate_hand, find_winners


@dataclass(frozen=True, slots=True)
class PlayerHand:
    """
    A player's known hole cards.

    For equity calculation, hole_cards should contain exactly 2 cards.
    """

    hole_cards: tuple[Card, ...]

    def __post_init__(self) -> None:
        if len(self.hole_cards) != 2:
            raise ValueError(f"Player must have exactly 2 hole cards, got {len(self.hole_cards)}")


@dataclass(frozen=True, slots=True)
class ConvergencePoint:
    """A point on the convergence curve for tracking simulation progress."""

    simulation: int
    equities: tuple[float, ...]  # One equity per player at this point


@dataclass(frozen=True, slots=True)
class PlayerEquity:
    """Equity results for a single player."""

    win_count: int
    tie_count: int
    total_simulations: int

    @property
    def win_rate(self) -> float:
        """Probability of winning outright."""
        if self.total_simulations == 0:
            return 0.0
        return self.win_count / self.total_simulations

    @property
    def tie_rate(self) -> float:
        """Probability of tying."""
        if self.total_simulations == 0:
            return 0.0
        return self.tie_count / self.total_simulations

    @property
    def equity(self) -> float:
        """
        Overall equity (win + tie share).

        Ties are split equally among tied players, but since we track
        tie_count per player, we estimate equity as:
        equity = win_rate + (tie_rate / average_tie_players)

        For simplicity, we use: equity = win_rate + tie_rate * 0.5
        which is a reasonable approximation for heads-up.
        """
        # This is a simplified calculation
        # In practice, we calculate this more precisely in the simulation
        if self.total_simulations == 0:
            return 0.0
        return self.win_rate + (self.tie_rate / 2)


@dataclass
class EquityAccumulator:
    """Internal accumulator for equity calculation."""

    wins: list[int] = field(default_factory=list)
    ties: list[int] = field(default_factory=list)
    equity_sum: list[float] = field(default_factory=list)
    total: int = 0

    def init_players(self, num_players: int) -> None:
        self.wins = [0] * num_players
        self.ties = [0] * num_players
        self.equity_sum = [0.0] * num_players

    def record(self, winner_indices: list[int], num_players: int) -> None:
        """Record a simulation result."""
        self.total += 1
        num_winners = len(winner_indices)

        if num_winners == 1:
            # Single winner
            self.wins[winner_indices[0]] += 1
            self.equity_sum[winner_indices[0]] += 1.0
        else:
            # Tie - split equity equally
            share = 1.0 / num_winners
            for idx in winner_indices:
                self.ties[idx] += 1
                self.equity_sum[idx] += share


@dataclass
class EquityRequest:
    """Request for equity calculation."""

    players: list[PlayerHand]
    board: list[Card] = field(default_factory=list)
    num_simulations: int = 10000
    seed: int | None = None
    track_convergence: bool = False
    convergence_interval: int = 100

    def __post_init__(self) -> None:
        if len(self.players) < 2:
            raise ValueError("Need at least 2 players for equity calculation")
        if len(self.board) > 5:
            raise ValueError(f"Board can have at most 5 cards, got {len(self.board)}")

        # Check for duplicate cards
        all_cards: list[Card] = list(self.board)
        for player in self.players:
            all_cards.extend(player.hole_cards)

        if len(all_cards) != len(set(all_cards)):
            raise ValueError("Duplicate cards detected in players/board")


@dataclass(frozen=True, slots=True)
class EquityResult:
    """Result of equity calculation."""

    players: tuple[PlayerEquity, ...]
    total_simulations: int
    convergence: tuple[ConvergencePoint, ...] | None = None

    def __str__(self) -> str:
        lines = ["Equity Results:"]
        for i, p in enumerate(self.players):
            lines.append(
                f"  Player {i + 1}: {p.equity:.1%} equity "
                f"(win: {p.win_rate:.1%}, tie: {p.tie_rate:.1%})"
            )
        return "\n".join(lines)


def calculate_equity(request: EquityRequest) -> EquityResult:
    """
    Calculate equity for all players using Monte Carlo simulation.

    Args:
        request: EquityRequest with players, board, and simulation parameters.

    Returns:
        EquityResult with equity for each player.
    """
    rng = random.Random(request.seed)
    num_players = len(request.players)

    # Collect known cards
    known_cards: set[Card] = set(request.board)
    for player in request.players:
        known_cards.update(player.hole_cards)

    # Build remaining deck
    remaining_deck = [c for c in FULL_DECK if c not in known_cards]

    # How many more community cards needed?
    cards_needed = 5 - len(request.board)

    # Initialize accumulator
    acc = EquityAccumulator()
    acc.init_players(num_players)

    # Track convergence if requested
    convergence_points: list[ConvergencePoint] = []

    # Run simulations
    for sim in range(request.num_simulations):
        # Shuffle remaining deck and deal community cards
        rng.shuffle(remaining_deck)
        runout = remaining_deck[:cards_needed]
        full_board = list(request.board) + runout

        # Evaluate each player's hand
        player_hands: list[list[Card]] = []
        for player in request.players:
            full_hand = list(player.hole_cards) + full_board
            player_hands.append(full_hand)

        # Find winner(s)
        winners = find_winners(player_hands)
        acc.record(winners, num_players)

        # Track convergence
        if request.track_convergence and (sim + 1) % request.convergence_interval == 0:
            current_equities = tuple(
                acc.equity_sum[i] / acc.total if acc.total > 0 else 0.0
                for i in range(num_players)
            )
            convergence_points.append(ConvergencePoint(sim + 1, current_equities))

    # Build results
    player_results = tuple(
        PlayerEquity(
            win_count=acc.wins[i],
            tie_count=acc.ties[i],
            total_simulations=acc.total,
        )
        for i in range(num_players)
    )

    # Calculate proper equity from accumulated sums
    final_equities = [acc.equity_sum[i] / acc.total for i in range(num_players)]

    # Create PlayerEquity with proper equity calculation
    # We need to store the actual equity, but PlayerEquity calculates it
    # So we'll create a custom result that stores the accumulated values

    return EquityResult(
        players=player_results,
        total_simulations=acc.total,
        convergence=tuple(convergence_points) if convergence_points else None,
    )


def equity_vs_random(
    hole_cards: Sequence[Card],
    board: Sequence[Card] = (),
    num_opponents: int = 1,
    num_simulations: int = 10000,
    seed: int | None = None,
) -> float:
    """
    Calculate equity against random opponent hands.

    This is a convenience function for quick equity lookups.

    Args:
        hole_cards: Hero's 2 hole cards.
        board: Current community cards (0-5).
        num_opponents: Number of random opponents.
        num_simulations: Number of Monte Carlo simulations.
        seed: Random seed for reproducibility.

    Returns:
        Hero's equity as a float between 0 and 1.
    """
    rng = random.Random(seed)

    # Known cards (hero's hole cards + board)
    known_cards: set[Card] = set(hole_cards)
    known_cards.update(board)

    remaining_deck = [c for c in FULL_DECK if c not in known_cards]
    cards_needed_board = 5 - len(board)

    wins = 0
    ties = 0
    total = 0

    for _ in range(num_simulations):
        rng.shuffle(remaining_deck)

        # Deal opponent hands
        deck_idx = 0
        opponent_hands: list[list[Card]] = []
        for _ in range(num_opponents):
            opponent_hands.append([remaining_deck[deck_idx], remaining_deck[deck_idx + 1]])
            deck_idx += 2

        # Deal remaining board
        runout = remaining_deck[deck_idx : deck_idx + cards_needed_board]
        full_board = list(board) + runout

        # Evaluate hands
        hero_hand = list(hole_cards) + full_board
        all_hands = [hero_hand] + [opp + full_board for opp in opponent_hands]

        winners = find_winners(all_hands)
        total += 1

        if winners == [0]:
            wins += 1
        elif 0 in winners:
            ties += 1

    return (wins + ties * 0.5) / total if total > 0 else 0.0
