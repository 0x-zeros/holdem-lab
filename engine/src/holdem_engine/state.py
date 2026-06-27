"""Conversion from PokerKit state to holdem-lab public state."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import replace

from holdem_common import (
    Action,
    ActionType,
    Card,
    GameState,
    PlayerState,
    Pot,
    Street,
)
from pokerkit import Card as PokerKitCard
from pokerkit import State as PokerKitState

from holdem_engine.config import HoldemConfig

_STREETS = (Street.PREFLOP, Street.FLOP, Street.TURN, Street.RIVER)


def card_from_pokerkit(card: PokerKitCard) -> Card:
    return Card.from_code(repr(card))


def cards_from_pokerkit(cards: Iterable[PokerKitCard]) -> tuple[Card, ...]:
    return tuple(card_from_pokerkit(card) for card in cards)


def legal_actions_from_pokerkit(state: PokerKitState) -> tuple[Action, ...]:
    actions: list[Action] = []

    call_amount = state.checking_or_calling_amount
    if call_amount and call_amount > 0 and state.can_fold():
        actions.append(Action(ActionType.FOLD))

    if call_amount is not None:
        action_type = ActionType.CHECK if call_amount == 0 else ActionType.CALL
        actions.append(Action(action_type, amount=call_amount))

    min_bet_to = state.min_completion_betting_or_raising_to_amount
    max_bet_to = state.max_completion_betting_or_raising_to_amount
    if min_bet_to is not None and max_bet_to is not None:
        action_type = ActionType.BET if max(state.bets) == 0 else ActionType.RAISE
        actions.append(
            Action(
                action_type,
                amount=min_bet_to,
                min_amount=min_bet_to,
                max_amount=max_bet_to,
            ),
        )
        if max_bet_to > min_bet_to:
            actions.append(
                Action(
                    ActionType.ALL_IN,
                    amount=max_bet_to,
                    min_amount=max_bet_to,
                    max_amount=max_bet_to,
                ),
            )

    return tuple(actions)


def street_from_pokerkit(state: PokerKitState) -> Street:
    if state.street_index is None or not state.status:
        return Street.SHOWDOWN
    return _STREETS[state.street_index]


def pots_from_pokerkit(
    state: PokerKitState,
    *,
    engine_to_public: tuple[int, ...],
) -> tuple[Pot, ...]:
    pots = [
        Pot(
            amount=pot.amount,
            eligible_seats=frozenset(engine_to_public[seat] for seat in pot.player_indices),
        )
        for pot in state.pots
    ]

    outstanding_bets = sum(state.bets)
    if outstanding_bets:
        eligible_seats = frozenset(
            engine_to_public[seat] for seat, active in enumerate(state.statuses) if active
        )
        if eligible_seats:
            pots.append(Pot(amount=outstanding_bets, eligible_seats=eligible_seats))

    return tuple(pots)


def game_state_from_pokerkit(
    state: PokerKitState,
    config: HoldemConfig,
    *,
    viewer_seat: int | None = None,
    engine_to_public: tuple[int, ...] | None = None,
) -> GameState:
    engine_to_public = engine_to_public or tuple(state.player_indices)
    legal_actions = legal_actions_from_pokerkit(state)
    if legal_actions:
        assert state.turn_index is not None
        current_seat = engine_to_public[state.turn_index]
    else:
        current_seat = None
    to_call = state.checking_or_calling_amount or 0
    min_raise = state.min_completion_betting_or_raising_to_amount or config.big_blind

    players = tuple(
        PlayerState(
            seat=public_seat,
            stack=state.stacks[engine_seat],
            committed=max(0, -state.payoffs[engine_seat]),
            hole_cards=cards_from_pokerkit(state.hole_cards[engine_seat]),
            active=state.statuses[engine_seat],
            all_in=state.statuses[engine_seat] and state.stacks[engine_seat] == 0,
            dealer=public_seat == config.button_seat,
            small_blind=public_seat == config.small_blind_seat,
            big_blind=public_seat == config.big_blind_seat,
        )
        for engine_seat, public_seat in enumerate(engine_to_public)
    )
    players = tuple(sorted(players, key=lambda player: player.seat))
    if viewer_seat is not None:
        if viewer_seat not in engine_to_public:
            raise KeyError(f"unknown seat: {viewer_seat}")
        players = tuple(
            player
            if player.seat == viewer_seat or not state.status
            else replace(player, hole_cards=())
            for player in players
        )

    return GameState(
        hand_id=config.hand_id,
        street=street_from_pokerkit(state),
        players=players,
        board=cards_from_pokerkit(state.get_board_cards(0)),
        pots=pots_from_pokerkit(state, engine_to_public=engine_to_public),
        current_seat=current_seat,
        button_seat=config.button_seat,
        small_blind=config.small_blind,
        big_blind=config.big_blind,
        min_raise=min_raise,
        to_call=to_call,
        legal_actions=legal_actions,
        deck_remaining=len(state.deck_cards),
        metadata={
            "payoffs": tuple(
                payoff
                for _public_seat, payoff in sorted(
                    (
                        (public_seat, state.payoffs[engine_seat])
                        for engine_seat, public_seat in enumerate(engine_to_public)
                    ),
                    key=lambda item: item[0],
                )
            ),
            "pokerkit_status": state.status,
            "viewer_seat": viewer_seat,
        },
    )
