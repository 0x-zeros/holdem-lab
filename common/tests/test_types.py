from holdem_common import Card, GameState, PlayerState, Pot, Rank, Street, Suit


def test_card_from_code_round_trips() -> None:
    card = Card.from_code("Ah")

    assert card == Card(rank=Rank.ACE, suit=Suit.HEARTS)
    assert card.code == "Ah"


def test_game_state_exposes_pot_total_and_players() -> None:
    player = PlayerState(seat=0, stack=100, committed=10)
    state = GameState(
        hand_id="hand-1",
        street=Street.PREFLOP,
        players=(player,),
        board=(),
        pots=(Pot(amount=10, eligible_seats=frozenset({0})),),
        current_seat=0,
        button_seat=0,
        small_blind=5,
        big_blind=10,
        min_raise=10,
        to_call=0,
    )

    assert state.pot_total == 10
    assert state.active_players == (player,)
    assert state.player(0) == player
