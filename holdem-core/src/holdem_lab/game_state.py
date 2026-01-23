"""Game state machine for Texas Hold'em hands."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Sequence

from holdem_lab.cards import Card, Deck
from holdem_lab.equity import EquityRequest, EquityResult, PlayerHand, calculate_equity
from holdem_lab.evaluator import HandRank, evaluate_hand, find_winners
from holdem_lab.event_log import EventLog, EventType


class Street(Enum):
    """Game street (betting round)."""

    PRE_DEAL = auto()
    PRE_FLOP = auto()
    FLOP = auto()
    TURN = auto()
    RIVER = auto()
    SHOWDOWN = auto()
    COMPLETE = auto()


@dataclass
class PlayerState:
    """State of a single player during the hand."""

    seat: int
    hole_cards: list[Card] = field(default_factory=list)
    is_active: bool = True  # Still in the hand (not folded)

    def full_hand(self, board: list[Card]) -> list[Card]:
        """Get player's full 7-card hand (hole cards + board)."""
        return self.hole_cards + board


@dataclass
class PotResult:
    """Result of pot resolution."""

    winners: list[int]  # Winning player seat indices
    winning_hand: HandRank
    pot_share: dict[int, float]  # seat -> share of pot

    def __str__(self) -> str:
        if len(self.winners) == 1:
            return f"Player {self.winners[0]} wins with {self.winning_hand.describe()}"
        return f"Players {self.winners} split with {self.winning_hand.describe()}"


class GameState:
    """
    State machine for a Texas Hold'em hand.

    Tracks the game state through each street and provides methods
    for dealing cards and resolving the hand.

    This is a simplified "check-down" model without betting.
    """

    def __init__(
        self,
        num_players: int = 2,
        deck: Deck | None = None,
        seed: int | None = None,
        log_events: bool = True,
    ):
        """
        Initialize a new hand.

        Args:
            num_players: Number of players (2-10).
            deck: Optional pre-configured deck.
            seed: Random seed for deck shuffling.
            log_events: Whether to log events for replay.
        """
        if not 2 <= num_players <= 10:
            raise ValueError(f"Number of players must be 2-10, got {num_players}")

        self.num_players = num_players
        self.deck = deck if deck else Deck(seed=seed)
        self.seed = seed
        self._street = Street.PRE_DEAL
        self._board: list[Card] = []
        self._players = [PlayerState(seat=i) for i in range(num_players)]
        self._pot = 0.0  # Simplified, no actual betting
        self._result: PotResult | None = None

        # Event logging
        self._log_events = log_events
        self._event_log = EventLog() if log_events else None

        if self._event_log is not None:
            self._event_log.append(
                EventType.INIT_HAND,
                num_players=num_players,
                seed=seed,
            )

    @property
    def street(self) -> Street:
        """Current street."""
        return self._street

    @property
    def board(self) -> list[Card]:
        """Current community cards."""
        return list(self._board)

    @property
    def players(self) -> list[PlayerState]:
        """List of player states."""
        return list(self._players)

    @property
    def event_log(self) -> EventLog | None:
        """Event log for this hand."""
        return self._event_log

    @property
    def result(self) -> PotResult | None:
        """Hand result, if resolved."""
        return self._result

    def get_player(self, seat: int) -> PlayerState:
        """Get player state by seat index."""
        if not 0 <= seat < self.num_players:
            raise ValueError(f"Invalid seat {seat}, must be 0-{self.num_players - 1}")
        return self._players[seat]

    def get_hole_cards(self, seat: int) -> list[Card]:
        """Get a player's hole cards."""
        return list(self.get_player(seat).hole_cards)

    def deal_hole_cards(self) -> None:
        """
        Deal hole cards to all players.

        Must be called before any community cards are dealt.
        """
        if self._street != Street.PRE_DEAL:
            raise RuntimeError(f"Cannot deal hole cards in {self._street}")

        for player in self._players:
            player.hole_cards = self.deck.deal(2)
            if self._event_log is not None:
                self._event_log.append(
                    EventType.DEAL_HOLE,
                    player=player.seat,
                    cards=player.hole_cards,
                )

        self._street = Street.PRE_FLOP

    def deal_flop(self) -> list[Card]:
        """
        Deal the flop (3 community cards).

        Returns:
            The 3 flop cards.
        """
        if self._street != Street.PRE_FLOP:
            raise RuntimeError(f"Cannot deal flop in {self._street}")

        flop = self.deck.deal(3)
        self._board.extend(flop)
        self._street = Street.FLOP

        if self._event_log is not None:
            self._event_log.append(EventType.DEAL_FLOP, cards=flop)

        return flop

    def deal_turn(self) -> Card:
        """
        Deal the turn (4th community card).

        Returns:
            The turn card.
        """
        if self._street != Street.FLOP:
            raise RuntimeError(f"Cannot deal turn in {self._street}")

        turn = self.deck.deal_one()
        self._board.append(turn)
        self._street = Street.TURN

        if self._event_log is not None:
            self._event_log.append(EventType.DEAL_TURN, card=turn)

        return turn

    def deal_river(self) -> Card:
        """
        Deal the river (5th community card).

        Returns:
            The river card.
        """
        if self._street != Street.TURN:
            raise RuntimeError(f"Cannot deal river in {self._street}")

        river = self.deck.deal_one()
        self._board.append(river)
        self._street = Street.RIVER

        if self._event_log is not None:
            self._event_log.append(EventType.DEAL_RIVER, card=river)

        return river

    def resolve(self) -> PotResult:
        """
        Resolve the hand and determine winner(s).

        Must be called after all 5 community cards are dealt.

        Returns:
            PotResult with winners and hand information.
        """
        if self._street != Street.RIVER:
            raise RuntimeError(f"Cannot resolve in {self._street}, must be at RIVER")

        if len(self._board) != 5:
            raise RuntimeError(f"Board must have 5 cards, has {len(self._board)}")

        # Get all active player hands
        active_players = [p for p in self._players if p.is_active]
        if not active_players:
            raise RuntimeError("No active players to resolve")

        hands = [p.full_hand(self._board) for p in active_players]
        winner_indices = find_winners(hands)

        # Map back to original seat indices
        winners = [active_players[i].seat for i in winner_indices]
        winning_hand = evaluate_hand(hands[winner_indices[0]])

        # Calculate pot shares (equal split for ties)
        pot_share = {seat: 1.0 / len(winners) for seat in winners}

        self._result = PotResult(
            winners=winners,
            winning_hand=winning_hand,
            pot_share=pot_share,
        )
        self._street = Street.SHOWDOWN

        if self._event_log is not None:
            self._event_log.append(
                EventType.SHOWDOWN,
                winners=winners,
                hand_type=str(winning_hand.hand_type),
                hand_description=winning_hand.describe(),
            )
            self._event_log.append(EventType.PAYOUT, payouts=pot_share)

        self._street = Street.COMPLETE
        return self._result

    def run_to_showdown(self) -> PotResult:
        """
        Convenience method to deal all cards and resolve.

        Deals any remaining cards and resolves the hand.

        Returns:
            PotResult with winners and hand information.
        """
        if self._street == Street.PRE_DEAL:
            self.deal_hole_cards()

        if self._street == Street.PRE_FLOP:
            self.deal_flop()

        if self._street == Street.FLOP:
            self.deal_turn()

        if self._street == Street.TURN:
            self.deal_river()

        return self.resolve()

    def calculate_equity(
        self,
        num_simulations: int = 10000,
        track_convergence: bool = False,
    ) -> EquityResult:
        """
        Calculate current equity for all players.

        Uses Monte Carlo simulation to estimate win probabilities
        given the current board state.

        Args:
            num_simulations: Number of simulations to run.
            track_convergence: Whether to track convergence data.

        Returns:
            EquityResult with equity for each player.
        """
        if self._street == Street.PRE_DEAL:
            raise RuntimeError("Cannot calculate equity before dealing hole cards")

        # Build player hands for equity calculation
        player_hands = [
            PlayerHand(hole_cards=tuple(p.hole_cards))
            for p in self._players
            if p.is_active
        ]

        request = EquityRequest(
            players=player_hands,
            board=list(self._board),
            num_simulations=num_simulations,
            track_convergence=track_convergence,
        )

        result = calculate_equity(request)

        # Log equity snapshot
        if self._event_log is not None:
            equities = {i: result.players[i].equity for i in range(len(result.players))}
            self._event_log.append(
                EventType.EQUITY_SNAPSHOT,
                street=self._street.name,
                equities=equities,
            )

        return result

    def set_hole_cards(self, seat: int, cards: Sequence[Card]) -> None:
        """
        Set specific hole cards for a player.

        Useful for analysis with known hands.

        Args:
            seat: Player seat index.
            cards: Exactly 2 cards.
        """
        if len(cards) != 2:
            raise ValueError(f"Must provide exactly 2 cards, got {len(cards)}")

        player = self.get_player(seat)
        player.hole_cards = list(cards)

        # Remove cards from deck
        self.deck.remove(cards)

        if self._event_log is not None:
            self._event_log.append(
                EventType.DEAL_HOLE,
                player=seat,
                cards=list(cards),
            )

    def set_board(self, cards: Sequence[Card]) -> None:
        """
        Set the board cards directly.

        Useful for analysis with known boards.

        Args:
            cards: 3, 4, or 5 community cards.
        """
        if not 3 <= len(cards) <= 5:
            raise ValueError(f"Board must have 3-5 cards, got {len(cards)}")

        self._board = list(cards)
        self.deck.remove(cards)

        # Update street based on board size
        if len(cards) == 3:
            self._street = Street.FLOP
            if self._event_log is not None:
                self._event_log.append(EventType.DEAL_FLOP, cards=list(cards))
        elif len(cards) == 4:
            self._street = Street.TURN
            if self._event_log is not None:
                self._event_log.append(EventType.DEAL_FLOP, cards=list(cards[:3]))
                self._event_log.append(EventType.DEAL_TURN, card=cards[3])
        else:
            self._street = Street.RIVER
            if self._event_log is not None:
                self._event_log.append(EventType.DEAL_FLOP, cards=list(cards[:3]))
                self._event_log.append(EventType.DEAL_TURN, card=cards[3])
                self._event_log.append(EventType.DEAL_RIVER, card=cards[4])

    @classmethod
    def from_known_cards(
        cls,
        hole_cards: dict[int, Sequence[Card]],
        board: Sequence[Card] = (),
        log_events: bool = True,
    ) -> GameState:
        """
        Create a game state with known cards.

        Args:
            hole_cards: Mapping of seat index to hole cards.
            board: Known community cards.
            log_events: Whether to log events.

        Returns:
            GameState with specified cards.
        """
        num_players = max(hole_cards.keys()) + 1
        game = cls(num_players=num_players, log_events=log_events)

        # Set hole cards
        for seat, cards in hole_cards.items():
            game.set_hole_cards(seat, cards)

        # Ensure PRE_FLOP street after dealing hole cards
        if game._street == Street.PRE_DEAL:
            game._street = Street.PRE_FLOP

        # Set board if provided
        if board:
            game.set_board(board)

        return game
