"""Temporal safety gate for Poker Legends recognition results."""

from __future__ import annotations

from dataclasses import replace
from typing import Final

from holdem_common import GameState, Street

from holdem_bot.recognize import (
    AssemblyIssue,
    AssemblyStatus,
    ContractLevel,
    Freshness,
    GameStateAssemblyResult,
    RecognitionResult,
    ValidityScope,
    VisualObservation,
)
from holdem_bot.screen_state import ScreenKind

_TEMPORAL_RULE: Final = "poker_legends_temporal_tracker"


class PokerLegendsTemporalTracker:
    """Require repeated current-frame agreement before authorizing Poker Legends state.

    The tracker consumes already-assembled single-frame results and returns a new
    ``RecognitionResult``. It never reuses a previous valid state to authorize the
    current frame. Until the current state signature has remained stable for the
    configured window, the returned result has ``state=None`` and a blocking temporal
    issue.
    """

    def __init__(self, *, required_stable_frames: int = 2) -> None:
        if required_stable_frames < 1:
            raise ValueError("required_stable_frames must be at least 1")
        self.required_stable_frames = required_stable_frames
        self._last_signature: tuple[object, ...] | None = None
        self._stable_count = 0
        self._stable_start_ms: int | None = None
        self._tracker_generation = 0
        self._action_row_generation = 0
        self._last_action_signature: tuple[object, ...] | None = None
        self._hand_generation = 0
        self._tracker_hand_id: str | None = None
        self._locked_hero_cards: tuple[str, ...] = ()
        self._locked_board_cards: tuple[str, ...] = ()
        self._overlay_requires_restabilization = False
        self._hand_boundary_pending = False

    def update(self, result: RecognitionResult) -> RecognitionResult:
        assembly = result.assembly_result
        if assembly is None:
            self._reset_window()
            return self._with_tracker_metadata(result, reasons=("missing_assembly_result",))

        if result.screen.kind is ScreenKind.BLOCKED_OVERLAY:
            self._overlay_requires_restabilization = True
            self._reset_window()
            return self._with_tracker_metadata(result, reasons=("blocked_overlay",))

        if (
            result.screen.kind is not ScreenKind.ACTIONABLE_TABLE
            or result.state is None
            or assembly.status is not AssemblyStatus.SINGLE_FRAME_VALID
        ):
            if result.screen.kind is not ScreenKind.ACTIONABLE_TABLE:
                self._reset_window()
            return self._with_tracker_metadata(result, reasons=("not_single_frame_valid",))

        state = result.state
        current_ms = _monotonic_ms(result)
        action_signature = _action_signature(result)
        if action_signature != self._last_action_signature:
            self._action_row_generation += 1
            self._last_action_signature = action_signature

        transition_issue = self._transition_issue(state)
        signature = _state_signature(result)
        if signature != self._last_signature:
            self._last_signature = signature
            self._stable_count = 1
            self._stable_start_ms = current_ms
            self._tracker_generation += 1
        else:
            self._stable_count += 1

        if transition_issue == "unsafe_transition":
            self._reset_window(keep_generation=True)
            return self._blocked_result(
                result,
                status=AssemblyStatus.UNSAFE_TRANSITION,
                reason_code="UNSAFE_TRANSITION",
                message="current frame violates locked hand transition constraints",
                current_ms=current_ms,
            )

        reasons: list[str] = []
        if transition_issue == "hand_boundary_pending":
            self._hand_boundary_pending = True
            reasons.append("HAND_BOUNDARY_PENDING")
        if (
            self._overlay_requires_restabilization
            and self._stable_count < self.required_stable_frames
        ):
            reasons.append("OVERLAY_RESTABILIZATION_REQUIRED")
        if self._stable_count < self.required_stable_frames:
            reasons.append("TEMPORALLY_UNSTABLE")

        if reasons:
            return self._blocked_result(
                result,
                status=AssemblyStatus.TEMPORALLY_UNSTABLE,
                reason_code=reasons[0],
                message="current frame has not satisfied the temporal stability window",
                current_ms=current_ms,
            )

        stable_state = self._stable_state(state)
        visual = _locked_visual_observation(result.visual_observation)
        assembly = replace(
            assembly,
            status=AssemblyStatus.TEMPORALLY_STABLE_VALID,
            validity_scope=ValidityScope.TEMPORAL_WINDOW,
            state=stable_state,
            freshness=self._freshness(result, current_ms, critical_fields_fresh=True),
            rule_consistency="temporally_consistent",
        )
        self._locked_hero_cards = _hero_cards(stable_state)
        self._locked_board_cards = _board_cards(stable_state)
        self._overlay_requires_restabilization = False
        self._hand_boundary_pending = False
        return self._replace_result(
            result,
            state=stable_state,
            assembly=assembly,
            visual_observation=visual,
            tracker_status="temporally_stable_valid",
            tracker_reasons=(),
        )

    def _transition_issue(self, state: GameState) -> str | None:
        hero_cards = _hero_cards(state)
        board_cards = _board_cards(state)
        if self._locked_hero_cards and hero_cards and hero_cards != self._locked_hero_cards:
            self._clear_hand_locks()
            return "hand_boundary_pending"
        if self._locked_board_cards:
            if len(board_cards) < len(self._locked_board_cards):
                self._clear_hand_locks()
                return "hand_boundary_pending"
            if not board_cards[: len(self._locked_board_cards)] == self._locked_board_cards:
                if state.street is Street.PREFLOP or not board_cards:
                    self._clear_hand_locks()
                    return "hand_boundary_pending"
                return "unsafe_transition"
        return None

    def _blocked_result(
        self,
        result: RecognitionResult,
        *,
        status: AssemblyStatus,
        reason_code: str,
        message: str,
        current_ms: int | None,
    ) -> RecognitionResult:
        assert result.assembly_result is not None
        issue = AssemblyIssue(
            issue_type="stale" if reason_code != "HAND_BOUNDARY_PENDING" else "ambiguous",
            reason_code=reason_code,
            field_path="temporal",
            rule_name=_TEMPORAL_RULE,
            severity="hard",
            blocking=True,
            required_by_contract=(ContractLevel.POLICY_DECISION,),
            message=message,
        )
        assembly = replace(
            result.assembly_result,
            status=status,
            validity_scope=ValidityScope.SINGLE_FRAME,
            state=None,
            contract_level=ContractLevel.OBSERVE_ONLY,
            contract_status="blocked",
            valid_for=(ContractLevel.OBSERVE_ONLY,),
            issues=(*result.assembly_result.issues, issue),
            freshness=self._freshness(result, current_ms, critical_fields_fresh=False),
            critical_min_confidence=None,
            rule_consistency="temporally_blocked",
        )
        return self._replace_result(
            result,
            state=None,
            assembly=assembly,
            visual_observation=result.visual_observation,
            tracker_status=status.value,
            tracker_reasons=(reason_code,),
        )

    def _stable_state(self, state: GameState) -> GameState:
        if self._tracker_hand_id is None or self._hand_boundary_pending:
            self._hand_generation += 1
            self._tracker_hand_id = f"plh{self._hand_generation}"
        metadata = dict(state.metadata)
        metadata["temporal_tracker_hand_id"] = self._tracker_hand_id
        metadata["temporal_tracker_generation"] = self._tracker_generation
        metadata["action_row_generation"] = self._action_row_generation
        return replace(state, hand_id=self._tracker_hand_id, metadata=metadata)

    def _freshness(
        self,
        result: RecognitionResult,
        current_ms: int | None,
        *,
        critical_fields_fresh: bool,
    ) -> Freshness:
        start_ms = self._stable_start_ms
        duration_ms = (
            max(0, current_ms - start_ms)
            if current_ms is not None and start_ms is not None
            else 0
        )
        frame_id = (
            result.frame_evidence.frame_id
            if result.frame_evidence is not None
            else result.assembly_result.observation_id
            if result.assembly_result is not None
            else ""
        )
        return Freshness(
            source_frame_id=frame_id,
            current_frame_revalidated=True,
            critical_fields_fresh=critical_fields_fresh,
            action_row_fresh=critical_fields_fresh,
            state_age_ms=0 if critical_fields_fresh else None,
            stable_frame_count=self._stable_count,
            stable_duration_ms=duration_ms,
            tracker_hand_id=self._tracker_hand_id,
            tracker_generation=self._tracker_generation,
        )

    def _replace_result(
        self,
        result: RecognitionResult,
        *,
        state: GameState | None,
        assembly: GameStateAssemblyResult,
        visual_observation: VisualObservation | None,
        tracker_status: str,
        tracker_reasons: tuple[str, ...],
    ) -> RecognitionResult:
        metadata = dict(result.metadata)
        metadata["assembly_result"] = assembly.to_dict()
        metadata["temporal_tracker"] = {
            "status": tracker_status,
            "reasons": list(tracker_reasons),
            "required_stable_frames": self.required_stable_frames,
            "stable_frame_count": self._stable_count,
            "tracker_generation": self._tracker_generation,
            "tracker_hand_id": self._tracker_hand_id,
            "action_row_generation": self._action_row_generation,
            "overlay_requires_restabilization": self._overlay_requires_restabilization,
            "hand_boundary_pending": self._hand_boundary_pending,
        }
        if visual_observation is not None:
            metadata["visual_observation"] = visual_observation.to_dict()
        if assembly.issues:
            for issue in assembly.issues:
                if issue.blocking:
                    metadata["state_block_reason"] = issue.reason_code.lower()
                    break
        else:
            metadata.pop("state_block_reason", None)
        return replace(
            result,
            state=state,
            metadata=metadata,
            visual_observation=visual_observation,
            assembly_result=assembly,
            safety_contract=assembly.contract_level,
        )

    def _with_tracker_metadata(
        self,
        result: RecognitionResult,
        *,
        reasons: tuple[str, ...],
    ) -> RecognitionResult:
        metadata = dict(result.metadata)
        metadata["temporal_tracker"] = {
            "status": "not_tracking",
            "reasons": list(reasons),
            "required_stable_frames": self.required_stable_frames,
            "stable_frame_count": self._stable_count,
            "tracker_generation": self._tracker_generation,
            "tracker_hand_id": self._tracker_hand_id,
            "action_row_generation": self._action_row_generation,
            "overlay_requires_restabilization": self._overlay_requires_restabilization,
            "hand_boundary_pending": self._hand_boundary_pending,
        }
        return replace(result, metadata=metadata)

    def _reset_window(self, *, keep_generation: bool = False) -> None:
        self._last_signature = None
        self._stable_count = 0
        self._stable_start_ms = None
        if not keep_generation:
            self._tracker_generation += 1

    def _clear_hand_locks(self) -> None:
        self._locked_hero_cards = ()
        self._locked_board_cards = ()
        self._tracker_hand_id = None


def _state_signature(result: RecognitionResult) -> tuple[object, ...]:
    assert result.state is not None
    state = result.state
    return (
        result.screen.kind.value,
        result.screen.hero_turn,
        state.street.value,
        _hero_cards(state),
        _board_cards(state),
        tuple(
            (player.seat, player.stack, player.committed, player.active, player.all_in)
            for player in state.players
        ),
        tuple((pot.amount, tuple(sorted(pot.eligible_seats))) for pot in state.pots),
        state.current_seat,
        state.button_seat,
        state.to_call,
        _legal_action_signature(state),
        _action_signature(result),
        _numeric_signature(result),
    )


def _hero_cards(state: GameState) -> tuple[str, ...]:
    if state.current_seat is None:
        return ()
    try:
        return tuple(card.code for card in state.player(state.current_seat).hole_cards)
    except KeyError:
        return ()


def _board_cards(state: GameState) -> tuple[str, ...]:
    return tuple(card.code for card in state.board)


def _legal_action_signature(state: GameState) -> tuple[tuple[object, ...], ...]:
    return tuple(
        sorted(
            (
                action.action_type.value,
                action.amount,
                action.min_amount,
                action.max_amount,
            )
            for action in state.legal_actions
        )
    )


def _action_signature(result: RecognitionResult) -> tuple[object, ...]:
    observation = result.visual_observation
    if observation is None:
        if result.state is None:
            return ()
        return _legal_action_signature(result.state)
    current_panels = tuple(
        panel
        for panel in observation.action_panels
        if panel.panel_kind == "current_action_row" and panel.visible
    )
    return tuple(
        (
            panel.panel_kind,
            panel.enabled,
            panel.hero_turn_indicator,
            tuple(
                (
                    button.slot,
                    button.visible,
                    button.enabled,
                    button.accepted_action,
                    tuple(candidate.value for candidate in button.amount_candidates),
                    button.evidence.crop_hash if button.evidence is not None else None,
                )
                for button in panel.buttons
            ),
        )
        for panel in current_panels
    )


def _numeric_signature(result: RecognitionResult) -> tuple[tuple[object, ...], ...]:
    observation = result.visual_observation
    if observation is None:
        if result.state is None:
            return ()
        return (("pot", result.state.pot_total), ("to_call", result.state.to_call))
    return tuple(
        (
            number.role,
            number.group,
            number.name,
            number.accepted_value,
            number.evidence.crop_hash if number.evidence is not None else None,
        )
        for number in observation.numbers
    )


def _monotonic_ms(result: RecognitionResult) -> int | None:
    if result.frame_evidence is None:
        return None
    return result.frame_evidence.monotonic_timestamp_ms


def _locked_visual_observation(
    observation: VisualObservation | None,
) -> VisualObservation | None:
    if observation is None:
        return None
    cards = tuple(
        replace(
            card,
            locked_card=card.accepted_card
            if card.group in {"hero_hole_cards", "board"} and card.accepted_card is not None
            else card.locked_card,
            locked_by_tracker=card.group in {"hero_hole_cards", "board"}
            and card.accepted_card is not None,
        )
        for card in observation.cards
    )
    return replace(observation, cards=cards)
