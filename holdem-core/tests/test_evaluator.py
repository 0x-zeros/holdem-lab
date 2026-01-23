"""Tests for evaluator module."""

import pytest

from holdem_lab.cards import parse_cards
from holdem_lab.evaluator import (
    HandRank,
    HandType,
    compare_hands,
    evaluate_five,
    evaluate_hand,
    find_winners,
)


class TestHandType:
    """Tests for HandType enum."""

    def test_ordering(self):
        assert HandType.HIGH_CARD < HandType.ONE_PAIR
        assert HandType.ONE_PAIR < HandType.TWO_PAIR
        assert HandType.STRAIGHT < HandType.FLUSH
        assert HandType.STRAIGHT_FLUSH < HandType.ROYAL_FLUSH

    def test_str(self):
        assert str(HandType.HIGH_CARD) == "High Card"
        assert str(HandType.ROYAL_FLUSH) == "Royal Flush"
        assert str(HandType.THREE_OF_A_KIND) == "Three Of A Kind"


class TestEvaluateFive:
    """Tests for evaluate_five function."""

    def test_high_card(self):
        cards = parse_cards("Ah Kd Qc Js 9h")
        rank = evaluate_five(cards)
        assert rank.hand_type == HandType.HIGH_CARD
        assert rank.primary_ranks == (14,)

    def test_one_pair(self):
        cards = parse_cards("Ah Ad Kc Qs Jh")
        rank = evaluate_five(cards)
        assert rank.hand_type == HandType.ONE_PAIR
        assert rank.primary_ranks == (14,)
        assert rank.kickers == (13, 12, 11)

    def test_two_pair(self):
        cards = parse_cards("Ah Ad Kc Ks Jh")
        rank = evaluate_five(cards)
        assert rank.hand_type == HandType.TWO_PAIR
        assert rank.primary_ranks == (14, 13)
        assert rank.kickers == (11,)

    def test_three_of_a_kind(self):
        cards = parse_cards("Ah Ad Ac Ks Jh")
        rank = evaluate_five(cards)
        assert rank.hand_type == HandType.THREE_OF_A_KIND
        assert rank.primary_ranks == (14,)

    def test_straight(self):
        cards = parse_cards("Ah Kd Qc Js Th")
        rank = evaluate_five(cards)
        assert rank.hand_type == HandType.STRAIGHT
        assert rank.primary_ranks == (14,)

    def test_straight_wheel(self):
        # A-2-3-4-5 (wheel), high card is 5
        cards = parse_cards("Ah 2d 3c 4s 5h")
        rank = evaluate_five(cards)
        assert rank.hand_type == HandType.STRAIGHT
        assert rank.primary_ranks == (5,)

    def test_flush(self):
        cards = parse_cards("Ah Kh Qh Jh 9h")
        rank = evaluate_five(cards)
        assert rank.hand_type == HandType.FLUSH
        assert rank.primary_ranks == (14, 13, 12, 11, 9)

    def test_full_house(self):
        cards = parse_cards("Ah Ad Ac Ks Kh")
        rank = evaluate_five(cards)
        assert rank.hand_type == HandType.FULL_HOUSE
        assert rank.primary_ranks == (14, 13)

    def test_four_of_a_kind(self):
        cards = parse_cards("Ah Ad Ac As Kh")
        rank = evaluate_five(cards)
        assert rank.hand_type == HandType.FOUR_OF_A_KIND
        assert rank.primary_ranks == (14,)
        assert rank.kickers == (13,)

    def test_straight_flush(self):
        cards = parse_cards("9h Th Jh Qh Kh")
        rank = evaluate_five(cards)
        assert rank.hand_type == HandType.STRAIGHT_FLUSH
        assert rank.primary_ranks == (13,)

    def test_royal_flush(self):
        cards = parse_cards("Ah Kh Qh Jh Th")
        rank = evaluate_five(cards)
        assert rank.hand_type == HandType.ROYAL_FLUSH
        assert rank.primary_ranks == (14,)

    def test_steel_wheel(self):
        # A-2-3-4-5 suited is straight flush, not royal
        cards = parse_cards("Ah 2h 3h 4h 5h")
        rank = evaluate_five(cards)
        assert rank.hand_type == HandType.STRAIGHT_FLUSH
        assert rank.primary_ranks == (5,)

    def test_wrong_card_count(self):
        with pytest.raises(ValueError):
            evaluate_five(parse_cards("Ah Kh"))


class TestHandRankComparison:
    """Tests for HandRank comparison."""

    def test_different_types(self):
        pair = evaluate_five(parse_cards("Ah Ad Kc Qs Jh"))
        two_pair = evaluate_five(parse_cards("Ah Ad Kc Ks Jh"))
        assert two_pair > pair

    def test_same_type_different_primary(self):
        pair_aces = evaluate_five(parse_cards("Ah Ad Kc Qs Jh"))
        pair_kings = evaluate_five(parse_cards("Kh Kd Ac Qs Jh"))
        assert pair_aces > pair_kings

    def test_same_type_different_kicker(self):
        pair_a_kicker_k = evaluate_five(parse_cards("Ah Ad Kc Qs Jh"))
        pair_a_kicker_q = evaluate_five(parse_cards("Ah Ad Qc Js Th"))
        assert pair_a_kicker_k > pair_a_kicker_q

    def test_tie(self):
        hand1 = evaluate_five(parse_cards("Ah Kd Qc Js 9h"))
        hand2 = evaluate_five(parse_cards("As Ks Qs Jc 9s"))
        assert hand1 == hand2


class TestEvaluateHand:
    """Tests for evaluate_hand function (7 cards)."""

    def test_seven_cards(self):
        # Should find the best 5-card hand
        cards = parse_cards("Ah Ad Ac As Kh Qd 2c")
        rank = evaluate_hand(cards)
        assert rank.hand_type == HandType.FOUR_OF_A_KIND
        assert rank.primary_ranks == (14,)
        assert rank.kickers == (13,)  # King kicker, not 2

    def test_six_cards(self):
        cards = parse_cards("Ah Ad Ac Kh Qd 2c")
        rank = evaluate_hand(cards)
        assert rank.hand_type == HandType.THREE_OF_A_KIND

    def test_five_cards(self):
        cards = parse_cards("Ah Ad Kc Qs Jh")
        rank = evaluate_hand(cards)
        assert rank.hand_type == HandType.ONE_PAIR

    def test_too_few_cards(self):
        with pytest.raises(ValueError):
            evaluate_hand(parse_cards("Ah Ad"))

    def test_too_many_cards(self):
        with pytest.raises(ValueError):
            evaluate_hand(parse_cards("Ah Ad Ac As Kh Qd 2c 3c"))


class TestFindWinners:
    """Tests for find_winners function."""

    def test_single_winner(self):
        # Use cards that don't form a straight
        hands = [
            parse_cards("Ah Ad Kc 9d 2c 7h 4s"),  # Pair of aces
            parse_cards("Kh Kd Ac 9d 2c 7h 4s"),  # Pair of kings
        ]
        winners = find_winners(hands)
        assert winners == [0]

    def test_tie(self):
        # Both have same straight
        hands = [
            parse_cards("Ah Kd Qc Jd Tc 2h 3s"),
            parse_cards("As Ks Qs Js Th 2c 4d"),
        ]
        winners = find_winners(hands)
        assert winners == [0, 1]

    def test_three_players(self):
        hands = [
            parse_cards("Ah Ad 2c 3d 4c 5h 6s"),  # Pair of aces
            parse_cards("Kh Kd Kc 3d 4c 5h 6s"),  # Three kings
            parse_cards("7h 7d 7c 7s 4c 5h 6s"),  # Four sevens
        ]
        winners = find_winners(hands)
        assert winners == [2]

    def test_empty_hands(self):
        with pytest.raises(ValueError):
            find_winners([])


class TestCompareHands:
    """Tests for compare_hands function."""

    def test_first_wins(self):
        # Use cards that don't form a straight
        hand1 = parse_cards("Ah Ad Kc 9d 2c 7h 4s")
        hand2 = parse_cards("Kh Kd Ac 9d 2c 7h 4s")
        assert compare_hands(hand1, hand2) == 1

    def test_second_wins(self):
        # Use cards that don't form a straight
        hand1 = parse_cards("Kh Kd Ac 9d 2c 7h 4s")
        hand2 = parse_cards("Ah Ad Kc 9d 2c 7h 4s")
        assert compare_hands(hand1, hand2) == -1

    def test_tie(self):
        hand1 = parse_cards("Ah Kd Qc Jd Tc 2h 3s")
        hand2 = parse_cards("As Ks Qs Js Th 2c 4d")
        assert compare_hands(hand1, hand2) == 0


class TestHandRankDescribe:
    """Tests for HandRank.describe method."""

    def test_describe_royal_flush(self):
        rank = evaluate_hand(parse_cards("Ah Kh Qh Jh Th 2c 3d"))
        assert rank.describe() == "Royal Flush"

    def test_describe_straight_flush(self):
        rank = evaluate_hand(parse_cards("9h Th Jh Qh Kh 2c 3d"))
        assert "Straight Flush" in rank.describe()

    def test_describe_full_house(self):
        rank = evaluate_hand(parse_cards("Ah Ad Ac Kh Kd 2c 3d"))
        desc = rank.describe()
        assert "Full House" in desc
        assert "Ace" in desc

    def test_describe_pair(self):
        # Use cards that don't form a straight
        rank = evaluate_hand(parse_cards("Ah Ad Kc 9d 2c 7h 4s"))
        desc = rank.describe()
        assert "Pair" in desc
        assert "Ace" in desc


class TestKnownEquities:
    """Test known hand matchups."""

    def test_aa_vs_kk(self):
        # AA should beat KK most of the time
        aa = parse_cards("Ah Ad")
        kk = parse_cards("Kh Kd")
        # Use board that doesn't allow wheel straight (no A-2-3-4-5)
        board = parse_cards("2c 7d 9s Tc Jc")

        aa_rank = evaluate_hand(list(aa) + list(board))
        kk_rank = evaluate_hand(list(kk) + list(board))

        # On this board, AA wins
        assert aa_rank > kk_rank

    def test_set_over_set(self):
        # Set of Aces beats set of Kings
        aa = parse_cards("Ah Ad")
        kk = parse_cards("Kh Kd")
        board = parse_cards("Ac Kc 2s 3h 9c")

        aa_full = list(aa) + list(board)
        kk_full = list(kk) + list(board)

        aa_rank = evaluate_hand(aa_full)
        kk_rank = evaluate_hand(kk_full)

        assert aa_rank > kk_rank
        assert aa_rank.hand_type == HandType.THREE_OF_A_KIND
        assert kk_rank.hand_type == HandType.THREE_OF_A_KIND
