"""Tests for equity module."""

import pytest

from holdem_lab.cards import parse_cards
from holdem_lab.equity import (
    EquityRequest,
    PlayerHand,
    calculate_equity,
    equity_vs_random,
)


class TestPlayerHand:
    """Tests for PlayerHand class."""

    def test_valid_hand(self):
        cards = tuple(parse_cards("Ah Kh"))
        hand = PlayerHand(hole_cards=cards)
        assert len(hand.hole_cards) == 2

    def test_invalid_hand_count(self):
        with pytest.raises(ValueError):
            PlayerHand(hole_cards=tuple(parse_cards("Ah")))
        with pytest.raises(ValueError):
            PlayerHand(hole_cards=tuple(parse_cards("Ah Kh Qh")))


class TestEquityRequest:
    """Tests for EquityRequest class."""

    def test_valid_request(self):
        request = EquityRequest(
            players=[
                PlayerHand(hole_cards=tuple(parse_cards("Ah Kh"))),
                PlayerHand(hole_cards=tuple(parse_cards("Qc Qd"))),
            ],
            board=parse_cards("2c 3d 4s"),
        )
        assert len(request.players) == 2
        assert len(request.board) == 3

    def test_too_few_players(self):
        with pytest.raises(ValueError):
            EquityRequest(
                players=[PlayerHand(hole_cards=tuple(parse_cards("Ah Kh")))],
            )

    def test_duplicate_cards(self):
        with pytest.raises(ValueError):
            EquityRequest(
                players=[
                    PlayerHand(hole_cards=tuple(parse_cards("Ah Kh"))),
                    PlayerHand(hole_cards=tuple(parse_cards("Ah Qd"))),  # Ah duplicate
                ],
            )

    def test_board_too_large(self):
        with pytest.raises(ValueError):
            EquityRequest(
                players=[
                    PlayerHand(hole_cards=tuple(parse_cards("Ah Kh"))),
                    PlayerHand(hole_cards=tuple(parse_cards("Qc Qd"))),
                ],
                board=parse_cards("2c 3d 4s 5h 6c 7d"),  # 6 cards
            )


class TestCalculateEquity:
    """Tests for calculate_equity function."""

    def test_basic_calculation(self):
        request = EquityRequest(
            players=[
                PlayerHand(hole_cards=tuple(parse_cards("Ah Kh"))),
                PlayerHand(hole_cards=tuple(parse_cards("2c 7d"))),
            ],
            num_simulations=1000,
            seed=42,
        )
        result = calculate_equity(request)

        assert len(result.players) == 2
        assert result.total_simulations == 1000

        # AK should beat 72o most of the time
        assert result.players[0].equity > 0.5
        assert result.players[1].equity < 0.5

    def test_aa_vs_kk(self):
        """AA vs KK should be approximately 82% vs 18%."""
        request = EquityRequest(
            players=[
                PlayerHand(hole_cards=tuple(parse_cards("Ah Ad"))),
                PlayerHand(hole_cards=tuple(parse_cards("Kh Kd"))),
            ],
            num_simulations=10000,
            seed=42,
        )
        result = calculate_equity(request)

        # Allow for Monte Carlo variance
        assert 0.78 < result.players[0].equity < 0.86
        assert 0.14 < result.players[1].equity < 0.22

    def test_pair_vs_overcards(self):
        """Pair vs overcards is roughly a coin flip."""
        request = EquityRequest(
            players=[
                PlayerHand(hole_cards=tuple(parse_cards("Jh Jd"))),
                PlayerHand(hole_cards=tuple(parse_cards("Ah Kd"))),
            ],
            num_simulations=5000,
            seed=42,
        )
        result = calculate_equity(request)

        # Should be close to 50-50
        assert 0.45 < result.players[0].equity < 0.60
        assert 0.40 < result.players[1].equity < 0.55

    def test_with_board(self):
        """Test equity calculation with partial board."""
        request = EquityRequest(
            players=[
                PlayerHand(hole_cards=tuple(parse_cards("Ah Kh"))),
                PlayerHand(hole_cards=tuple(parse_cards("Qc Qd"))),
            ],
            board=parse_cards("Qh Jh Tc"),  # AK has nut straight draw
            num_simulations=5000,
            seed=42,
        )
        result = calculate_equity(request)

        # QQ has a set, AK has a made straight
        # AK actually has the straight already! So AK should be ahead
        assert result.players[0].equity > 0.6

    def test_convergence_tracking(self):
        """Test convergence data collection."""
        request = EquityRequest(
            players=[
                PlayerHand(hole_cards=tuple(parse_cards("Ah Kh"))),
                PlayerHand(hole_cards=tuple(parse_cards("2c 7d"))),
            ],
            num_simulations=1000,
            seed=42,
            track_convergence=True,
            convergence_interval=100,
        )
        result = calculate_equity(request)

        assert result.convergence is not None
        assert len(result.convergence) == 10  # 1000 / 100

        # Check convergence points are sequential
        for i, point in enumerate(result.convergence):
            assert point.simulation == (i + 1) * 100

    def test_reproducibility(self):
        """Same seed should produce same results."""
        request1 = EquityRequest(
            players=[
                PlayerHand(hole_cards=tuple(parse_cards("Ah Kh"))),
                PlayerHand(hole_cards=tuple(parse_cards("2c 7d"))),
            ],
            num_simulations=1000,
            seed=12345,
        )
        request2 = EquityRequest(
            players=[
                PlayerHand(hole_cards=tuple(parse_cards("Ah Kh"))),
                PlayerHand(hole_cards=tuple(parse_cards("2c 7d"))),
            ],
            num_simulations=1000,
            seed=12345,
        )

        result1 = calculate_equity(request1)
        result2 = calculate_equity(request2)

        assert result1.players[0].win_count == result2.players[0].win_count
        assert result1.players[0].tie_count == result2.players[0].tie_count

    def test_three_way(self):
        """Test 3-way pot."""
        request = EquityRequest(
            players=[
                PlayerHand(hole_cards=tuple(parse_cards("Ah Ad"))),
                PlayerHand(hole_cards=tuple(parse_cards("Kh Kd"))),
                PlayerHand(hole_cards=tuple(parse_cards("Qh Qd"))),
            ],
            num_simulations=5000,
            seed=42,
        )
        result = calculate_equity(request)

        assert len(result.players) == 3
        # AA should have highest equity
        assert result.players[0].equity > result.players[1].equity
        assert result.players[1].equity > result.players[2].equity


class TestEquityVsRandom:
    """Tests for equity_vs_random function."""

    def test_basic(self):
        equity = equity_vs_random(
            hole_cards=parse_cards("Ah Ad"),
            num_opponents=1,
            num_simulations=1000,
            seed=42,
        )
        # AA should have high equity vs random
        assert equity > 0.8

    def test_with_board(self):
        equity = equity_vs_random(
            hole_cards=parse_cards("Ah Kh"),
            board=parse_cards("Qh Jh Th"),  # Royal flush!
            num_opponents=1,
            num_simulations=100,
            seed=42,
        )
        # Royal flush should win 100%
        assert equity == 1.0

    def test_multi_opponent(self):
        equity_1 = equity_vs_random(
            hole_cards=parse_cards("Ah Ad"),
            num_opponents=1,
            num_simulations=1000,
            seed=42,
        )
        equity_5 = equity_vs_random(
            hole_cards=parse_cards("Ah Ad"),
            num_opponents=5,
            num_simulations=1000,
            seed=42,
        )
        # Equity should decrease with more opponents
        assert equity_5 < equity_1


class TestPlayerEquity:
    """Tests for PlayerEquity properties."""

    def test_equity_properties(self):
        request = EquityRequest(
            players=[
                PlayerHand(hole_cards=tuple(parse_cards("Ah Ad"))),
                PlayerHand(hole_cards=tuple(parse_cards("Kh Kd"))),
            ],
            num_simulations=1000,
            seed=42,
        )
        result = calculate_equity(request)

        player = result.players[0]
        assert player.win_rate >= 0
        assert player.tie_rate >= 0
        assert player.equity >= 0
        assert player.equity <= 1

        # Win rate + tie rate should not exceed 1
        assert player.win_rate + player.tie_rate <= 1.001  # Allow for float rounding

    def test_result_str(self):
        request = EquityRequest(
            players=[
                PlayerHand(hole_cards=tuple(parse_cards("Ah Ad"))),
                PlayerHand(hole_cards=tuple(parse_cards("Kh Kd"))),
            ],
            num_simulations=100,
            seed=42,
        )
        result = calculate_equity(request)

        result_str = str(result)
        assert "Equity Results" in result_str
        assert "Player 1" in result_str
        assert "Player 2" in result_str
