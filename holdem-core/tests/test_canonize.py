"""Tests for canonize module."""

import pytest

from holdem_lab.cards import Card, Rank, Suit, parse_cards
from holdem_lab.canonize import (
    CanonicalHand,
    are_strategically_equivalent,
    canonize_hole_cards,
    get_all_canonical_hands,
    get_all_combos,
    get_combos_excluding,
    parse_canonical_hand,
)


class TestCanonicalHand:
    """Tests for CanonicalHand dataclass."""

    def test_str_suited(self):
        hand = CanonicalHand(Rank.ACE, Rank.KING, suited=True)
        assert str(hand) == "AKs"

    def test_str_offsuit(self):
        hand = CanonicalHand(Rank.ACE, Rank.KING, suited=False)
        assert str(hand) == "AKo"

    def test_str_pair(self):
        hand = CanonicalHand(Rank.ACE, Rank.ACE, suited=False)
        assert str(hand) == "AA"

    def test_str_low_cards(self):
        hand = CanonicalHand(Rank.THREE, Rank.TWO, suited=True)
        assert str(hand) == "32s"

    def test_str_ten(self):
        hand = CanonicalHand(Rank.TEN, Rank.TEN, suited=False)
        assert str(hand) == "TT"

    def test_num_combos_pair(self):
        hand = CanonicalHand(Rank.ACE, Rank.ACE, suited=False)
        assert hand.num_combos == 6

    def test_num_combos_suited(self):
        hand = CanonicalHand(Rank.ACE, Rank.KING, suited=True)
        assert hand.num_combos == 4

    def test_num_combos_offsuit(self):
        hand = CanonicalHand(Rank.ACE, Rank.KING, suited=False)
        assert hand.num_combos == 12

    def test_is_pair_true(self):
        hand = CanonicalHand(Rank.QUEEN, Rank.QUEEN, suited=False)
        assert hand.is_pair is True

    def test_is_pair_false(self):
        hand = CanonicalHand(Rank.QUEEN, Rank.JACK, suited=True)
        assert hand.is_pair is False

    def test_gap_pair(self):
        hand = CanonicalHand(Rank.ACE, Rank.ACE, suited=False)
        assert hand.gap == 0

    def test_gap_connector(self):
        hand = CanonicalHand(Rank.ACE, Rank.KING, suited=True)
        assert hand.gap == 1

    def test_gap_two_gapper(self):
        hand = CanonicalHand(Rank.ACE, Rank.JACK, suited=False)
        assert hand.gap == 3

    def test_hashable(self):
        hand1 = CanonicalHand(Rank.ACE, Rank.KING, suited=True)
        hand2 = CanonicalHand(Rank.ACE, Rank.KING, suited=True)
        assert hand1 == hand2
        assert hash(hand1) == hash(hand2)
        assert len({hand1, hand2}) == 1

    def test_different_hands_not_equal(self):
        hand1 = CanonicalHand(Rank.ACE, Rank.KING, suited=True)
        hand2 = CanonicalHand(Rank.ACE, Rank.KING, suited=False)
        assert hand1 != hand2

    def test_invalid_high_low_order(self):
        with pytest.raises(ValueError, match="high_rank must be >= low_rank"):
            CanonicalHand(Rank.KING, Rank.ACE, suited=True)

    def test_invalid_pair_suited(self):
        with pytest.raises(ValueError, match="Pairs cannot be suited"):
            CanonicalHand(Rank.ACE, Rank.ACE, suited=True)


class TestCanonizeHoleCards:
    """Tests for canonize_hole_cards function."""

    def test_suited_hearts(self):
        cards = parse_cards("Ah Kh")
        result = canonize_hole_cards(cards)
        assert result == CanonicalHand(Rank.ACE, Rank.KING, suited=True)

    def test_suited_spades(self):
        cards = parse_cards("As Ks")
        result = canonize_hole_cards(cards)
        assert result == CanonicalHand(Rank.ACE, Rank.KING, suited=True)

    def test_suited_clubs(self):
        cards = parse_cards("Ac Kc")
        result = canonize_hole_cards(cards)
        assert result == CanonicalHand(Rank.ACE, Rank.KING, suited=True)

    def test_suited_diamonds(self):
        cards = parse_cards("Ad Kd")
        result = canonize_hole_cards(cards)
        assert result == CanonicalHand(Rank.ACE, Rank.KING, suited=True)

    def test_offsuit(self):
        cards = parse_cards("Ah Ks")
        result = canonize_hole_cards(cards)
        assert result == CanonicalHand(Rank.ACE, Rank.KING, suited=False)

    def test_pair(self):
        cards = parse_cards("Ah Ad")
        result = canonize_hole_cards(cards)
        assert result == CanonicalHand(Rank.ACE, Rank.ACE, suited=False)

    def test_order_independent(self):
        cards1 = parse_cards("Kh Ah")
        cards2 = parse_cards("Ah Kh")
        assert canonize_hole_cards(cards1) == canonize_hole_cards(cards2)

    def test_low_cards_suited(self):
        cards = parse_cards("2h 3h")
        result = canonize_hole_cards(cards)
        assert result == CanonicalHand(Rank.THREE, Rank.TWO, suited=True)

    def test_low_cards_offsuit(self):
        cards = parse_cards("2h 3d")
        result = canonize_hole_cards(cards)
        assert result == CanonicalHand(Rank.THREE, Rank.TWO, suited=False)

    def test_seven_two_offsuit(self):
        cards = parse_cards("7c 2d")
        result = canonize_hole_cards(cards)
        assert str(result) == "72o"

    def test_wrong_card_count_one(self):
        with pytest.raises(ValueError, match="exactly 2 cards"):
            canonize_hole_cards(parse_cards("Ah"))

    def test_wrong_card_count_three(self):
        with pytest.raises(ValueError, match="exactly 2 cards"):
            canonize_hole_cards(parse_cards("Ah Kh Qh"))

    def test_wrong_card_count_zero(self):
        with pytest.raises(ValueError, match="exactly 2 cards"):
            canonize_hole_cards([])


class TestParseCanonicalHand:
    """Tests for parse_canonical_hand function."""

    def test_parse_suited(self):
        result = parse_canonical_hand("AKs")
        assert result == CanonicalHand(Rank.ACE, Rank.KING, suited=True)

    def test_parse_offsuit(self):
        result = parse_canonical_hand("AKo")
        assert result == CanonicalHand(Rank.ACE, Rank.KING, suited=False)

    def test_parse_pair(self):
        result = parse_canonical_hand("AA")
        assert result == CanonicalHand(Rank.ACE, Rank.ACE, suited=False)

    def test_parse_case_insensitive_lower(self):
        assert parse_canonical_hand("aks") == parse_canonical_hand("AKs")

    def test_parse_case_insensitive_mixed(self):
        assert parse_canonical_hand("aKS") == parse_canonical_hand("AKs")

    def test_parse_ten(self):
        result = parse_canonical_hand("TT")
        assert result == CanonicalHand(Rank.TEN, Rank.TEN, suited=False)

    def test_parse_ten_suited(self):
        result = parse_canonical_hand("TJs")
        assert result == CanonicalHand(Rank.JACK, Rank.TEN, suited=True)

    def test_parse_low_cards(self):
        result = parse_canonical_hand("32s")
        assert result == CanonicalHand(Rank.THREE, Rank.TWO, suited=True)

    def test_parse_invalid_pair_suited(self):
        with pytest.raises(ValueError, match="Pairs cannot be suited"):
            parse_canonical_hand("AAs")

    def test_parse_either_order(self):
        """Either order is accepted and normalized."""
        assert parse_canonical_hand("KAs") == parse_canonical_hand("AKs")

    def test_parse_missing_suffix(self):
        with pytest.raises(ValueError, match="must have 's' or 'o' suffix"):
            parse_canonical_hand("AK")

    def test_parse_invalid_suffix(self):
        with pytest.raises(ValueError, match="Invalid suffix"):
            parse_canonical_hand("AKx")

    def test_parse_too_short(self):
        with pytest.raises(ValueError, match="Invalid canonical hand notation"):
            parse_canonical_hand("A")

    def test_parse_too_long(self):
        with pytest.raises(ValueError, match="Invalid canonical hand notation"):
            parse_canonical_hand("AKso")


class TestGetAllCombos:
    """Tests for get_all_combos function."""

    def test_pair_combos_count(self):
        combos = get_all_combos(CanonicalHand(Rank.ACE, Rank.ACE, suited=False))
        assert len(combos) == 6

    def test_pair_combos_all_aces(self):
        combos = get_all_combos(CanonicalHand(Rank.ACE, Rank.ACE, suited=False))
        for c1, c2 in combos:
            assert c1.rank == Rank.ACE
            assert c2.rank == Rank.ACE
            assert c1.suit != c2.suit

    def test_pair_combos_unique(self):
        combos = get_all_combos(CanonicalHand(Rank.ACE, Rank.ACE, suited=False))
        # Convert to sets of suits for comparison
        suit_pairs = {frozenset([c1.suit, c2.suit]) for c1, c2 in combos}
        assert len(suit_pairs) == 6  # All combinations unique

    def test_suited_combos_count(self):
        combos = get_all_combos(CanonicalHand(Rank.ACE, Rank.KING, suited=True))
        assert len(combos) == 4

    def test_suited_combos_same_suit(self):
        combos = get_all_combos(CanonicalHand(Rank.ACE, Rank.KING, suited=True))
        for c1, c2 in combos:
            assert c1.suit == c2.suit
            assert c1.rank == Rank.ACE
            assert c2.rank == Rank.KING

    def test_suited_combos_all_suits(self):
        combos = get_all_combos(CanonicalHand(Rank.ACE, Rank.KING, suited=True))
        suits = {c1.suit for c1, _ in combos}
        assert suits == set(Suit)

    def test_offsuit_combos_count(self):
        combos = get_all_combos(CanonicalHand(Rank.ACE, Rank.KING, suited=False))
        assert len(combos) == 12

    def test_offsuit_combos_different_suits(self):
        combos = get_all_combos(CanonicalHand(Rank.ACE, Rank.KING, suited=False))
        for c1, c2 in combos:
            assert c1.suit != c2.suit
            assert c1.rank == Rank.ACE
            assert c2.rank == Rank.KING


class TestGetAllCanonicalHands:
    """Tests for get_all_canonical_hands function."""

    def test_count(self):
        hands = get_all_canonical_hands()
        assert len(hands) == 169

    def test_unique(self):
        hands = get_all_canonical_hands()
        assert len(set(hands)) == 169

    def test_total_combos(self):
        hands = get_all_canonical_hands()
        total = sum(h.num_combos for h in hands)
        assert total == 1326  # C(52,2)

    def test_first_is_aces(self):
        hands = get_all_canonical_hands()
        assert str(hands[0]) == "AA"

    def test_pairs_count(self):
        hands = get_all_canonical_hands()
        pairs = [h for h in hands if h.is_pair]
        assert len(pairs) == 13

    def test_suited_count(self):
        hands = get_all_canonical_hands()
        suited = [h for h in hands if h.suited]
        assert len(suited) == 78

    def test_offsuit_non_pair_count(self):
        hands = get_all_canonical_hands()
        offsuit = [h for h in hands if not h.suited and not h.is_pair]
        assert len(offsuit) == 78


class TestGetCombosExcluding:
    """Tests for get_combos_excluding function."""

    def test_pair_with_one_dead(self):
        hand = CanonicalHand(Rank.ACE, Rank.ACE, suited=False)
        dead = parse_cards("Ah")
        combos = get_combos_excluding(hand, dead)
        # 6 combos - 3 that use Ah = 3 remaining
        assert len(combos) == 3
        for c1, c2 in combos:
            assert c1 != Card(Rank.ACE, Suit.HEARTS)
            assert c2 != Card(Rank.ACE, Suit.HEARTS)

    def test_pair_with_two_dead(self):
        hand = CanonicalHand(Rank.ACE, Rank.ACE, suited=False)
        dead = parse_cards("Ah Ad")
        combos = get_combos_excluding(hand, dead)
        # Only combos with Ac and As remain: 1 combo
        assert len(combos) == 1
        assert combos[0] == (Card(Rank.ACE, Suit.CLUBS), Card(Rank.ACE, Suit.SPADES))

    def test_suited_with_dead(self):
        hand = CanonicalHand(Rank.ACE, Rank.KING, suited=True)
        dead = parse_cards("Ah")
        combos = get_combos_excluding(hand, dead)
        # 4 combos - 1 (AhKh) = 3 remaining
        assert len(combos) == 3

    def test_offsuit_with_dead(self):
        hand = CanonicalHand(Rank.ACE, Rank.KING, suited=False)
        dead = parse_cards("Ah Kh")
        combos = get_combos_excluding(hand, dead)
        # 12 combos - combos using Ah (3) - combos using Kh (3) + double counted (0) = 6
        assert len(combos) == 6

    def test_no_dead_cards(self):
        hand = CanonicalHand(Rank.ACE, Rank.KING, suited=True)
        combos = get_combos_excluding(hand, [])
        assert len(combos) == 4


class TestAreStrategicallyEquivalent:
    """Tests for are_strategically_equivalent function."""

    def test_suited_equivalent(self):
        h1 = parse_cards("Ah Kh")
        h2 = parse_cards("As Ks")
        assert are_strategically_equivalent(h1, h2) is True

    def test_offsuit_equivalent(self):
        h1 = parse_cards("Ah Ks")
        h2 = parse_cards("Ac Kd")
        assert are_strategically_equivalent(h1, h2) is True

    def test_pair_equivalent(self):
        h1 = parse_cards("Ah Ad")
        h2 = parse_cards("As Ac")
        assert are_strategically_equivalent(h1, h2) is True

    def test_suited_vs_offsuit_not_equivalent(self):
        h1 = parse_cards("Ah Kh")
        h2 = parse_cards("Ah Ks")
        assert are_strategically_equivalent(h1, h2) is False

    def test_different_ranks_not_equivalent(self):
        h1 = parse_cards("Ah Kh")
        h2 = parse_cards("Ah Qh")
        assert are_strategically_equivalent(h1, h2) is False
