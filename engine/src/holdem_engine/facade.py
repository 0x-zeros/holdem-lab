"""Thin facade around PokerKit's no-limit Texas Hold'em state."""

from __future__ import annotations

from collections import deque

from holdem_common import Action, ActionType, GameState
from pokerkit import Automation, Mode, NoLimitTexasHoldem
from pokerkit import Card as PokerKitCard
from pokerkit import State as PokerKitState

from holdem_engine.config import HoldemConfig
from holdem_engine.state import game_state_from_pokerkit, legal_actions_from_pokerkit

_AUTOMATIONS = (
    Automation.ANTE_POSTING,
    Automation.BET_COLLECTION,
    Automation.BLIND_OR_STRADDLE_POSTING,
    Automation.CARD_BURNING,
    Automation.HOLE_DEALING,
    Automation.BOARD_DEALING,
    Automation.HOLE_CARDS_SHOWING_OR_MUCKING,
    Automation.HAND_KILLING,
    Automation.CHIPS_PUSHING,
    Automation.CHIPS_PULLING,
)


class PokerKitFacade:
    def __init__(self, config: HoldemConfig | None = None) -> None:
        self.config = config or HoldemConfig()
        self._state: PokerKitState | None = None

    def reset(self) -> GameState:
        self._state = NoLimitTexasHoldem.create_state(
            self._automations(),
            True,
            self.config.ante,
            (self.config.small_blind, self.config.big_blind),
            self.config.big_blind,
            self.config.starting_stacks,
            len(self.config.starting_stacks),
            mode=Mode.TOURNAMENT,
        )
        if self.config.deck is not None:
            self._install_fixed_deck()
            self._deal_initial_hole_cards()
        return self.observe()

    def observe(self, seat: int | None = None) -> GameState:
        return game_state_from_pokerkit(self._require_state(), self.config, viewer_seat=seat)

    def legal_actions(self) -> tuple[Action, ...]:
        return legal_actions_from_pokerkit(self._require_state())

    def step(self, action: Action) -> GameState:
        state = self._require_state()

        match action.action_type:
            case ActionType.FOLD:
                state.fold()
            case ActionType.CHECK | ActionType.CALL:
                self._check_or_call(action)
            case ActionType.BET | ActionType.RAISE:
                state.complete_bet_or_raise_to(self._bet_to_amount(action))
            case ActionType.ALL_IN:
                max_amount = state.max_completion_betting_or_raising_to_amount
                if max_amount is None:
                    raise ValueError("all-in is not legal in the current state")
                state.complete_bet_or_raise_to(max_amount)

        return self.observe()

    @property
    def payoffs(self) -> dict[int, int]:
        state = self._require_state()
        return dict(enumerate(state.payoffs))

    def _bet_to_amount(self, action: Action) -> int:
        """Return the PokerKit bet/raise-to amount.

        Public no-limit actions use the total street commitment ("raise to X"),
        not the incremental raise size. An amount of 0 means "use the current
        legal minimum" for simple agents.
        """
        state = self._require_state()
        amount = action.amount or state.min_completion_betting_or_raising_to_amount
        if amount is None:
            raise ValueError("betting or raising is not legal in the current state")
        return amount

    def _check_or_call(self, action: Action) -> None:
        state = self._require_state()
        call_amount = state.checking_or_calling_amount
        if call_amount is None:
            raise ValueError("checking or calling is not legal in the current state")
        if call_amount == 0 and action.action_type is not ActionType.CHECK:
            raise ValueError("use CHECK when no chips are required to continue")
        if call_amount > 0 and action.action_type is not ActionType.CALL:
            raise ValueError("use CALL when chips are required to continue")
        if action.amount not in (0, call_amount):
            raise ValueError(f"call amount must be 0 or {call_amount}, got {action.amount}")
        state.check_or_call()

    def _automations(self) -> tuple[Automation, ...]:
        if self.config.deck is None:
            return _AUTOMATIONS
        return tuple(
            automation for automation in _AUTOMATIONS if automation is not Automation.HOLE_DEALING
        )

    def _install_fixed_deck(self) -> None:
        state = self._require_state()
        assert self.config.deck is not None
        state.deck_cards = deque(
            PokerKitCard.clean("".join(card.code for card in self.config.deck))
        )

    def _deal_initial_hole_cards(self) -> None:
        state = self._require_state()
        while state.can_deal_hole():
            state.deal_hole()

    def _require_state(self) -> PokerKitState:
        if self._state is None:
            raise RuntimeError("engine has not been reset")
        return self._state
