"""Tests for draws module."""

import pytest

from holdem_lab.cards import Card, Rank, Suit, parse_cards
from holdem_lab.draws import (
    DrawAnalysis,
    DrawType,
    FlushDraw,
    StraightDraw,
    analyze_draws,
    count_flush_outs,
    count_straight_outs,
    get_primary_draw,
)


class TestFlushDraws:
    """Tests for flush draw detection."""

    def test_flush_draw_9_outs(self):
        """4 cards to a flush = 9 outs."""
        hole = parse_cards("Ah Kh")
        board = parse_cards("7h 6h 2c")
        analysis = analyze_draws(hole, board)

        assert len(analysis.flush_draws) == 1
        assert analysis.flush_draws[0].out_count == 9
        assert analysis.flush_draws[0].suit == Suit.HEARTS
        assert analysis.flush_draws[0].cards_held == 4

    def test_flush_draw_is_nut(self):
        """Hero with Ace of suit has nut flush draw."""
        hole = parse_cards("Ah 2h")
        board = parse_cards("7h 6h 3c")
        analysis = analyze_draws(hole, board)

        assert analysis.flush_draws[0].is_nut is True

    def test_flush_draw_not_nut(self):
        """Hero without Ace of suit does not have nut flush draw."""
        hole = parse_cards("Kh 2h")
        board = parse_cards("7h 6h 3c")
        analysis = analyze_draws(hole, board)

        assert analysis.flush_draws[0].is_nut is False

    def test_backdoor_flush_on_flop(self):
        """3 cards to a flush on flop = backdoor flush."""
        hole = parse_cards("Ah Kh")
        board = parse_cards("7h 6c 2c")
        analysis = analyze_draws(hole, board)

        assert len(analysis.flush_draws) == 1
        assert analysis.flush_draws[0].cards_held == 3
        assert analysis.flush_draws[0].draw_type == DrawType.BACKDOOR_FLUSH

    def test_no_backdoor_on_turn(self):
        """3 cards to a flush on turn is not a draw."""
        hole = parse_cards("Ah Kh")
        board = parse_cards("7h 6c 2c 3d")
        analysis = analyze_draws(hole, board)

        # Only 2 hearts (Ah, 7h), not even backdoor
        assert len(analysis.flush_draws) == 0

    def test_already_has_flush(self):
        """No flush draw if already have flush."""
        hole = parse_cards("Ah Kh")
        board = parse_cards("Qh Jh 2h")
        analysis = analyze_draws(hole, board)

        assert analysis.has_flush is True
        assert len(analysis.flush_draws) == 0

    def test_dead_cards_reduce_outs(self):
        """Dead cards reduce flush outs."""
        hole = parse_cards("Ah Kh")
        board = parse_cards("7h 6h 2c")
        dead = parse_cards("Qh Jh")
        analysis = analyze_draws(hole, board, dead_cards=dead)

        assert analysis.flush_draws[0].out_count == 7  # 9 - 2

    def test_no_flush_draw_with_2_suited(self):
        """Only 2 cards to a flush is not a draw."""
        hole = parse_cards("Ah Kh")
        board = parse_cards("7c 6c 2c")
        analysis = analyze_draws(hole, board)

        # Check that hearts flush draw is not present (only 2 hearts)
        heart_draws = [fd for fd in analysis.flush_draws if fd.suit == Suit.HEARTS]
        assert len(heart_draws) == 0


class TestStraightDraws:
    """Tests for straight draw detection."""

    def test_open_ended_straight_draw(self):
        """4 consecutive cards = open-ended (8 outs)."""
        hole = parse_cards("9h 8c")
        board = parse_cards("7d 6s 2h")
        analysis = analyze_draws(hole, board)

        oesd = [sd for sd in analysis.straight_draws if sd.draw_type == DrawType.OPEN_ENDED]
        assert len(oesd) >= 1
        # Should have 8 outs total for OESD
        assert any(sd.out_count == 8 for sd in oesd)

    def test_gutshot(self):
        """Gap in middle = gutshot (4 outs)."""
        hole = parse_cards("9h 7c")
        board = parse_cards("6d 5s 2h")
        analysis = analyze_draws(hole, board)

        gutshots = [sd for sd in analysis.straight_draws if sd.draw_type == DrawType.GUTSHOT]
        assert len(gutshots) >= 1
        # Should need 8 for 5-6-7-8-9
        assert any(8 in sd.needed_ranks for sd in gutshots)

    def test_double_gutshot(self):
        """4 cards spanning 6 ranks with 2 internal gaps = double gutshot (8 outs)."""
        hole = parse_cards("Th 7c")
        board = parse_cards("8d 5s 2h")
        analysis = analyze_draws(hole, board)

        double_gs = [sd for sd in analysis.straight_draws if sd.draw_type == DrawType.DOUBLE_GUTSHOT]
        # 5-7-8-T needs 6 or 9
        assert len(double_gs) >= 1

    def test_wheel_draw(self):
        """A-2-3-4 = open-ended for wheel (needs 5)."""
        hole = parse_cards("Ah 4c")
        board = parse_cards("3d 2s Kh")
        analysis = analyze_draws(hole, board)

        # Should detect the wheel draw
        wheel_draws = [sd for sd in analysis.straight_draws if sd.high_card == 5]
        assert len(wheel_draws) >= 1

    def test_broadway_draw(self):
        """T-J-Q-K = needs A for Broadway."""
        hole = parse_cards("Kh Qc")
        board = parse_cards("Jd Ts 2h")
        analysis = analyze_draws(hole, board)

        broadway = [sd for sd in analysis.straight_draws if sd.high_card == 14]
        assert len(broadway) >= 1
        assert any(sd.is_nut for sd in broadway)

    def test_already_has_straight(self):
        """No straight draw if already have straight."""
        hole = parse_cards("9h 8c")
        board = parse_cards("7d 6s 5h")
        analysis = analyze_draws(hole, board)

        assert analysis.has_straight is True
        assert len(analysis.straight_draws) == 0

    def test_backdoor_straight_on_flop(self):
        """3 connected cards on flop = backdoor straight."""
        hole = parse_cards("9h 8c")
        board = parse_cards("7d 2s 2h")
        analysis = analyze_draws(hole, board)

        backdoor = [sd for sd in analysis.straight_draws if sd.draw_type == DrawType.BACKDOOR_STRAIGHT]
        # 7-8-9 is 3 connected
        assert len(backdoor) >= 1


class TestMadeHands:
    """Tests for already-made hands."""

    def test_has_flush(self):
        hole = parse_cards("Ah Kh")
        board = parse_cards("Qh Jh 2h")
        analysis = analyze_draws(hole, board)

        assert analysis.has_flush is True
        assert analysis.has_straight is False

    def test_has_straight(self):
        hole = parse_cards("9h 8c")
        board = parse_cards("7d 6s 5h")
        analysis = analyze_draws(hole, board)

        assert analysis.has_straight is True
        assert analysis.has_flush is False

    def test_has_straight_flush(self):
        hole = parse_cards("9h 8h")
        board = parse_cards("7h 6h 5h")
        analysis = analyze_draws(hole, board)

        assert analysis.has_flush is True
        assert analysis.has_straight is True


class TestOutsCounting:
    """Tests for outs calculation."""

    def test_no_double_counting(self):
        """Cards that complete both flush and straight count once."""
        hole = parse_cards("9h 8h")
        board = parse_cards("7h 6c 2h")
        analysis = analyze_draws(hole, board)

        # Flush draw outs
        flush_outs = set()
        for fd in analysis.flush_draws:
            flush_outs.update(fd.outs)

        # Straight draw outs
        straight_outs = set()
        for sd in analysis.straight_draws:
            straight_outs.update(sd.outs)

        # Total should be union, not sum
        expected_unique = len(flush_outs | straight_outs)
        assert analysis.total_outs == expected_unique

    def test_combo_draw_detection(self):
        """Combo draw has both flush and straight draws."""
        hole = parse_cards("9h 8h")
        board = parse_cards("7h 6c 2h")
        analysis = analyze_draws(hole, board)

        assert analysis.is_combo_draw is True

    def test_total_outs_includes_all(self):
        """total_outs includes all unique outs."""
        hole = parse_cards("Ah Kh")
        board = parse_cards("7h 6h 2c")
        analysis = analyze_draws(hole, board)

        # All outs should be in the all_outs tuple
        assert len(analysis.all_outs) == analysis.total_outs


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_count_flush_outs(self):
        hole = parse_cards("Ah Kh")
        board = parse_cards("7h 6h 2c")
        count = count_flush_outs(hole, board)
        assert count == 9

    def test_count_flush_outs_no_draw(self):
        hole = parse_cards("Ah Kc")
        board = parse_cards("7d 6s 2c")
        count = count_flush_outs(hole, board)
        assert count == 0

    def test_count_straight_outs_oesd(self):
        hole = parse_cards("9h 8c")
        board = parse_cards("7d 6s 2h")
        count = count_straight_outs(hole, board)
        assert count == 8  # Open-ended

    def test_count_straight_outs_gutshot(self):
        hole = parse_cards("9h 6c")
        board = parse_cards("8d 5s 2h")
        count = count_straight_outs(hole, board)
        assert count == 4  # Gutshot (needs 7)

    def test_get_primary_draw_flush(self):
        hole = parse_cards("Ah Kh")
        board = parse_cards("7h 6h 2c")
        primary = get_primary_draw(hole, board)
        assert primary == DrawType.FLUSH_DRAW

    def test_get_primary_draw_oesd(self):
        hole = parse_cards("9h 8c")
        board = parse_cards("7d 6s 2h")
        primary = get_primary_draw(hole, board)
        assert primary == DrawType.OPEN_ENDED

    def test_get_primary_draw_none(self):
        hole = parse_cards("Ah Kc")
        board = parse_cards("2d 7s 9h")
        primary = get_primary_draw(hole, board)
        assert primary is None


class TestEdgeCases:
    """Tests for edge cases."""

    def test_empty_board(self):
        """Preflop has no real draws."""
        hole = parse_cards("Ah Kh")
        analysis = analyze_draws(hole, [])

        # No board means no real draws
        assert len(analysis.flush_draws) == 0
        assert len(analysis.straight_draws) == 0

    def test_flop_only(self):
        hole = parse_cards("Ah Kh")
        board = parse_cards("7h 6h 2c")
        analysis = analyze_draws(hole, board)

        assert len(analysis.board) == 3
        assert analysis.flush_draws[0].out_count == 9

    def test_turn(self):
        hole = parse_cards("Ah Kh")
        board = parse_cards("7h 6h 2c 3d")
        analysis = analyze_draws(hole, board)

        assert len(analysis.board) == 4
        assert analysis.flush_draws[0].out_count == 9

    def test_river(self):
        """On river, draws don't really matter but should still work."""
        hole = parse_cards("Ah Kh")
        board = parse_cards("7h 6h 2c 3d 9s")
        analysis = analyze_draws(hole, board)

        assert len(analysis.board) == 5
        # Still has a flush draw technically (4 hearts)
        assert len(analysis.flush_draws) == 1

    def test_invalid_hole_cards(self):
        with pytest.raises(ValueError, match="exactly 2 hole cards"):
            analyze_draws(parse_cards("Ah"), parse_cards("7h 6h 2c"))

    def test_invalid_board_too_many(self):
        with pytest.raises(ValueError, match="at most 5 cards"):
            analyze_draws(
                parse_cards("Ah Kh"),
                parse_cards("7h 6h 2c 3d 4s 5c")
            )

    def test_duplicate_cards(self):
        with pytest.raises(ValueError, match="Duplicate cards"):
            analyze_draws(
                parse_cards("Ah Kh"),
                parse_cards("Ah 6h 2c")  # Ah duplicated
            )


class TestSpecificScenarios:
    """Tests for specific poker scenarios."""

    def test_nut_flush_draw_with_oesd(self):
        """Classic combo draw: nut flush draw + open-ended."""
        hole = parse_cards("Ah Th")
        board = parse_cards("9h 8c 2h")
        analysis = analyze_draws(hole, board)

        # Should have flush draw
        assert len(analysis.flush_draws) == 1
        assert analysis.flush_draws[0].is_nut is True

        # Should have straight draw (T-9 with 8 on board)
        assert len(analysis.straight_draws) >= 1
        assert analysis.is_combo_draw is True

    def test_four_to_flush_on_board(self):
        """4 to a flush on board - hero with one heart has flush draw."""
        hole = parse_cards("Ah 2c")
        board = parse_cards("Kh Qh Jh 3d")
        analysis = analyze_draws(hole, board)

        # Hero has 4 hearts (Ah + Kh Qh Jh) - that's a flush draw
        assert analysis.has_flush is False
        assert len(analysis.flush_draws) == 1
        assert analysis.flush_draws[0].cards_held == 4

    def test_monster_draw(self):
        """15+ out monster draw scenario."""
        hole = parse_cards("Jh Th")
        board = parse_cards("9h 8c 2h")
        analysis = analyze_draws(hole, board)

        # Flush draw (9 outs) + OESD (8 outs) - some overlap
        # Total should be around 15 unique outs
        assert analysis.total_outs >= 12
        assert analysis.is_combo_draw is True
