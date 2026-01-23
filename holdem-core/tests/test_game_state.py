"""Tests for game_state module."""

import pytest

from holdem_lab.cards import Deck, parse_cards
from holdem_lab.evaluator import HandType
from holdem_lab.game_state import GameState, PotResult, Street


class TestStreet:
    """Tests for Street enum."""

    def test_street_values(self):
        assert Street.PRE_DEAL
        assert Street.PRE_FLOP
        assert Street.FLOP
        assert Street.TURN
        assert Street.RIVER
        assert Street.SHOWDOWN
        assert Street.COMPLETE


class TestGameState:
    """Tests for GameState class."""

    def test_creation(self):
        game = GameState(num_players=2, seed=42)
        assert game.num_players == 2
        assert game.street == Street.PRE_DEAL
        assert len(game.board) == 0

    def test_invalid_player_count(self):
        with pytest.raises(ValueError):
            GameState(num_players=1)
        with pytest.raises(ValueError):
            GameState(num_players=11)

    def test_deal_hole_cards(self):
        game = GameState(num_players=2, seed=42)
        game.deal_hole_cards()

        assert game.street == Street.PRE_FLOP
        assert len(game.get_hole_cards(0)) == 2
        assert len(game.get_hole_cards(1)) == 2

    def test_deal_flop(self):
        game = GameState(num_players=2, seed=42)
        game.deal_hole_cards()
        flop = game.deal_flop()

        assert len(flop) == 3
        assert game.street == Street.FLOP
        assert len(game.board) == 3

    def test_deal_turn(self):
        game = GameState(num_players=2, seed=42)
        game.deal_hole_cards()
        game.deal_flop()
        turn = game.deal_turn()

        assert game.street == Street.TURN
        assert len(game.board) == 4

    def test_deal_river(self):
        game = GameState(num_players=2, seed=42)
        game.deal_hole_cards()
        game.deal_flop()
        game.deal_turn()
        river = game.deal_river()

        assert game.street == Street.RIVER
        assert len(game.board) == 5

    def test_resolve(self):
        game = GameState(num_players=2, seed=42)
        game.deal_hole_cards()
        game.deal_flop()
        game.deal_turn()
        game.deal_river()

        result = game.resolve()

        assert isinstance(result, PotResult)
        assert len(result.winners) >= 1
        assert result.winning_hand is not None

    def test_run_to_showdown(self):
        game = GameState(num_players=2, seed=42)
        result = game.run_to_showdown()

        assert game.street == Street.COMPLETE
        assert isinstance(result, PotResult)
        assert len(game.board) == 5

    def test_street_order_enforcement(self):
        game = GameState(num_players=2, seed=42)

        # Can't deal flop before hole cards
        with pytest.raises(RuntimeError):
            game.deal_flop()

        game.deal_hole_cards()

        # Can't deal turn before flop
        with pytest.raises(RuntimeError):
            game.deal_turn()

        game.deal_flop()

        # Can't deal river before turn
        with pytest.raises(RuntimeError):
            game.deal_river()

    def test_reproducibility(self):
        game1 = GameState(num_players=2, seed=12345)
        game1.deal_hole_cards()
        game1.deal_flop()

        game2 = GameState(num_players=2, seed=12345)
        game2.deal_hole_cards()
        game2.deal_flop()

        assert game1.get_hole_cards(0) == game2.get_hole_cards(0)
        assert game1.board == game2.board

    def test_event_logging(self):
        game = GameState(num_players=2, seed=42, log_events=True)
        game.run_to_showdown()

        log = game.event_log
        assert log is not None
        assert len(log) > 0

        # Check for expected events
        event_types = [e.event_type.value for e in log]
        assert "init_hand" in event_types
        assert "deal_hole" in event_types
        assert "deal_flop" in event_types
        assert "showdown" in event_types

    def test_no_event_logging(self):
        game = GameState(num_players=2, seed=42, log_events=False)
        game.run_to_showdown()

        assert game.event_log is None

    def test_calculate_equity(self):
        game = GameState(num_players=2, seed=42)
        game.deal_hole_cards()

        result = game.calculate_equity(num_simulations=1000)

        assert len(result.players) == 2
        # Equities should sum to approximately 1
        total = sum(p.equity for p in result.players)
        assert 0.98 < total < 1.02

    def test_calculate_equity_before_deal(self):
        game = GameState(num_players=2, seed=42)

        with pytest.raises(RuntimeError):
            game.calculate_equity()


class TestGameStateWithKnownCards:
    """Tests for GameState with pre-set cards."""

    def test_set_hole_cards(self):
        game = GameState(num_players=2, seed=42)
        cards = parse_cards("Ah Kh")
        game.set_hole_cards(0, cards)

        assert game.get_hole_cards(0) == list(cards)

    def test_set_board(self):
        game = GameState(num_players=2, seed=42)
        game.deal_hole_cards()

        board = parse_cards("2c 3d 4s")
        game.set_board(board)

        assert game.board == list(board)
        assert game.street == Street.FLOP

    def test_set_full_board(self):
        game = GameState(num_players=2, seed=42)
        game.deal_hole_cards()

        board = parse_cards("2c 3d 4s 5h 6c")
        game.set_board(board)

        assert game.board == list(board)
        assert game.street == Street.RIVER

    def test_from_known_cards(self):
        hole_cards = {
            0: parse_cards("Ah Kh"),
            1: parse_cards("Qc Qd"),
        }
        board = parse_cards("2c 3d 4s")

        game = GameState.from_known_cards(hole_cards, board)

        assert game.get_hole_cards(0) == list(hole_cards[0])
        assert game.get_hole_cards(1) == list(hole_cards[1])
        assert game.board == list(board)
        assert game.street == Street.FLOP

    def test_known_hand_result(self):
        # AA vs KK with 4-card board, then deal river
        hole_cards = {
            0: parse_cards("Ah Ad"),
            1: parse_cards("Kh Kd"),
        }
        # Use board that doesn't allow wheel (A-2-3-4-5)
        board = parse_cards("7c 8d 9s Tc")

        game = GameState.from_known_cards(hole_cards, board)
        game.deal_river()
        result = game.resolve()

        # Result depends on river card, just verify it completes
        assert result is not None
        assert len(result.winners) >= 1

    def test_known_hand_aa_wins(self):
        """Test with fully specified board where AA wins."""
        hole_cards = {
            0: parse_cards("Ah Ad"),
            1: parse_cards("Kh Kd"),
        }
        # Use board that doesn't form straight with A (no 2-3-4-5 or T-J-Q-K)
        board = parse_cards("7c 8d 2s Tc 3c")

        game = GameState.from_known_cards(hole_cards, board)
        result = game.resolve()

        assert result.winners == [0]  # AA wins
        assert result.winning_hand.hand_type == HandType.ONE_PAIR


class TestPotResult:
    """Tests for PotResult class."""

    def test_single_winner_str(self):
        from holdem_lab.evaluator import HandRank, HandType

        result = PotResult(
            winners=[0],
            winning_hand=HandRank(HandType.ONE_PAIR, (14,), (13, 12, 11)),
            pot_share={0: 1.0},
        )

        s = str(result)
        assert "Player 0 wins" in s

    def test_split_pot_str(self):
        from holdem_lab.evaluator import HandRank, HandType

        result = PotResult(
            winners=[0, 1],
            winning_hand=HandRank(HandType.STRAIGHT, (14,), ()),
            pot_share={0: 0.5, 1: 0.5},
        )

        s = str(result)
        assert "split" in s


class TestMultiPlayer:
    """Tests for multi-player games."""

    def test_three_players(self):
        game = GameState(num_players=3, seed=42)
        result = game.run_to_showdown()

        assert len(game.players) == 3
        assert all(len(p.hole_cards) == 2 for p in game.players)

    def test_six_players(self):
        game = GameState(num_players=6, seed=42)
        result = game.run_to_showdown()

        assert len(game.players) == 6

    def test_ten_players(self):
        game = GameState(num_players=10, seed=42)
        result = game.run_to_showdown()

        assert len(game.players) == 10
        # 10 players * 2 cards + 5 board = 25 cards used
        # Should still work with 52 card deck


class TestEquityIntegration:
    """Integration tests for equity calculation in game state."""

    def test_equity_on_flop(self):
        hole_cards = {
            0: parse_cards("Ah Kh"),
            1: parse_cards("2c 7d"),
        }
        board = parse_cards("Qh Jh Th")  # AK has royal flush!

        game = GameState.from_known_cards(hole_cards, board)
        result = game.calculate_equity(num_simulations=100)

        # Should be 100% for player 0
        assert result.players[0].equity > 0.99

    def test_equity_logged(self):
        game = GameState(num_players=2, seed=42, log_events=True)
        game.deal_hole_cards()
        game.calculate_equity(num_simulations=100)

        log = game.event_log
        assert log is not None

        equity_events = log.filter_by_type(
            __import__("holdem_lab.event_log", fromlist=["EventType"]).EventType.EQUITY_SNAPSHOT
        )
        assert len(equity_events) == 1
