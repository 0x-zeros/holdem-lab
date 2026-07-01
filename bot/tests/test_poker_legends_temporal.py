from dataclasses import replace

from holdem_bot.recognize import (
    ActionPanelObservation,
    AssemblyStatus,
    ButtonObservation,
    ContractLevel,
    FrameEvidence,
    Freshness,
    GameStateAssemblyResult,
    LayoutObservation,
    RecognitionMode,
    RecognitionResult,
    RoiEvidence,
    ValidityScope,
    VisualObservation,
)
from holdem_bot.screen_state import ScreenState
from holdem_bot.vision import PokerLegendsTemporalTracker
from holdem_common import Action, ActionType, Card, GameState, PlayerState, Pot, Street


def test_temporal_tracker_blocks_first_single_frame_then_stabilizes() -> None:
    tracker = PokerLegendsTemporalTracker(required_stable_frames=2)

    first = tracker.update(_result("frame_001"))
    second = tracker.update(_result("frame_002"))

    assert first.state is None
    assert first.assembly_result is not None
    assert first.assembly_result.status is AssemblyStatus.TEMPORALLY_UNSTABLE
    assert first.safety_contract is ContractLevel.OBSERVE_ONLY
    assert first.metadata["state_block_reason"] == "temporally_unstable"
    assert second.state is not None
    assert second.state.hand_id == "plh1"
    assert second.assembly_result is not None
    assert second.assembly_result.status is AssemblyStatus.TEMPORALLY_STABLE_VALID
    assert second.assembly_result.validity_scope is ValidityScope.TEMPORAL_WINDOW
    assert second.assembly_result.freshness.current_frame_revalidated is True
    assert second.assembly_result.freshness.stable_frame_count == 2
    assert second.assembly_result.freshness.tracker_hand_id == "plh1"
    assert second.visual_observation is not None
    assert {card.locked_card for card in second.visual_observation.cards} == {
        "AS",
        "KH",
        "2C",
        "7D",
        "TS",
    }


def test_temporal_tracker_requires_restabilization_after_overlay_clear() -> None:
    tracker = PokerLegendsTemporalTracker(required_stable_frames=2)
    assert tracker.update(_result("frame_001")).state is None
    assert tracker.update(_result("frame_002")).state is not None

    blocked = tracker.update(_blocked_result("frame_003"))
    assert blocked.state is None

    after_clear = tracker.update(_result("frame_004"))
    restabilized = tracker.update(_result("frame_005"))

    assert after_clear.state is None
    assert after_clear.assembly_result is not None
    assert after_clear.assembly_result.status is AssemblyStatus.TEMPORALLY_UNSTABLE
    assert after_clear.metadata["state_block_reason"] == "overlay_restabilization_required"
    assert restabilized.state is not None
    assert restabilized.assembly_result is not None
    assert restabilized.assembly_result.status is AssemblyStatus.TEMPORALLY_STABLE_VALID


def test_temporal_tracker_blocks_new_hand_until_pending_boundary_stabilizes() -> None:
    tracker = PokerLegendsTemporalTracker(required_stable_frames=2)
    assert tracker.update(_result("frame_001")).state is None
    first_hand = tracker.update(_result("frame_002"))
    assert first_hand.state is not None
    assert first_hand.state.hand_id == "plh1"

    boundary = tracker.update(
        _result("frame_003", hero_cards=("QH", "6S"), board_cards=(), street=Street.PREFLOP)
    )
    second_hand = tracker.update(
        _result("frame_004", hero_cards=("QH", "6S"), board_cards=(), street=Street.PREFLOP)
    )

    assert boundary.state is None
    assert boundary.assembly_result is not None
    assert boundary.assembly_result.status is AssemblyStatus.TEMPORALLY_UNSTABLE
    assert boundary.metadata["state_block_reason"] == "hand_boundary_pending"
    assert second_hand.state is not None
    assert second_hand.state.hand_id == "plh2"
    assert second_hand.assembly_result is not None
    assert second_hand.assembly_result.freshness.tracker_hand_id == "plh2"


def _result(
    frame_id: str,
    *,
    hero_cards: tuple[str, str] = ("AS", "KH"),
    board_cards: tuple[str, ...] = ("2C", "7D", "TS"),
    street: Street = Street.FLOP,
) -> RecognitionResult:
    screen = ScreenState.actionable_table(confidence=0.99, hero_turn=True)
    state = _state(frame_id, hero_cards=hero_cards, board_cards=board_cards, street=street)
    frame = FrameEvidence(session_id=None, frame_id=frame_id)
    visual = VisualObservation(
        frame=frame,
        recognition_mode=RecognitionMode.IMAGE_ONLY_REPLAY,
        screen=screen,
        layout=LayoutObservation(
            profile_id="test",
            layout_version="v1",
            transform_type="static_roi",
            transform_residual_px=None,
        ),
        cards=tuple(
            [
                *(
                    _card_observation("hero_hole_cards", f"hero_{index}", card)
                    for index, card in enumerate(hero_cards)
                ),
                *(
                    _card_observation("board", f"board_{index}", card)
                    for index, card in enumerate(board_cards)
                ),
            ]
        ),
        action_panels=(
            ActionPanelObservation(
                panel_kind="current_action_row",
                visible=True,
                enabled=True,
                hero_turn_indicator=True,
                row_bbox=None,
                buttons=(
                    _button_observation("primary_left", "check"),
                    _button_observation("primary_middle", "raise"),
                    _button_observation("primary_right", "fold"),
                ),
            ),
        ),
    )
    assembly = GameStateAssemblyResult(
        status=AssemblyStatus.SINGLE_FRAME_VALID,
        validity_scope=ValidityScope.SINGLE_FRAME,
        state=state,
        contract_level=ContractLevel.POLICY_DECISION,
        contract_status="satisfied",
        valid_for=(
            ContractLevel.OBSERVE_ONLY,
            ContractLevel.GAME_STATE,
            ContractLevel.POLICY_DECISION,
        ),
        issues=(),
        freshness=Freshness(
            source_frame_id=frame_id,
            current_frame_revalidated=True,
            critical_fields_fresh=True,
            action_row_fresh=True,
            stable_frame_count=1,
        ),
        layout_confidence=1.0,
        screen_confidence=screen.confidence,
        rule_consistency="consistent",
        observation_id=frame_id,
    )
    return RecognitionResult(
        state=state,
        confidence=0.95,
        metadata={"assembly_result": assembly.to_dict(), "visual_observation": visual.to_dict()},
        screen=screen,
        visual_observation=visual,
        assembly_result=assembly,
        recognition_mode=RecognitionMode.IMAGE_ONLY_REPLAY,
        safety_contract=ContractLevel.POLICY_DECISION,
        frame_evidence=frame,
    )


def _blocked_result(frame_id: str) -> RecognitionResult:
    base = _result(frame_id)
    screen = ScreenState.blocked_overlay(blocking_reason="buy_in_modal", confidence=0.99)
    assert base.assembly_result is not None
    assert base.visual_observation is not None
    visual = replace(base.visual_observation, screen=screen)
    assembly = replace(
        base.assembly_result,
        status=AssemblyStatus.BLOCKED_SCREEN,
        validity_scope=ValidityScope.NONE,
        state=None,
        contract_level=ContractLevel.OBSERVE_ONLY,
        contract_status="blocked",
        valid_for=(ContractLevel.OBSERVE_ONLY,),
        freshness=replace(
            base.assembly_result.freshness,
            critical_fields_fresh=False,
            action_row_fresh=False,
            stable_frame_count=0,
        ),
    )
    return replace(
        base,
        state=None,
        screen=screen,
        visual_observation=visual,
        assembly_result=assembly,
        safety_contract=ContractLevel.OBSERVE_ONLY,
        metadata={"assembly_result": assembly.to_dict(), "visual_observation": visual.to_dict()},
    )


def _state(
    hand_id: str,
    *,
    hero_cards: tuple[str, str],
    board_cards: tuple[str, ...],
    street: Street,
) -> GameState:
    return GameState(
        hand_id=hand_id,
        street=street,
        players=(
            PlayerState(
                seat=0,
                stack=900,
                committed=0,
                hole_cards=tuple(Card.from_code(card) for card in hero_cards),
            ),
            PlayerState(seat=1, stack=800, committed=0),
        ),
        board=tuple(Card.from_code(card) for card in board_cards),
        pots=(Pot(amount=150, eligible_seats=frozenset({0, 1})),),
        current_seat=0,
        button_seat=1,
        small_blind=5,
        big_blind=10,
        min_raise=10,
        to_call=0,
        legal_actions=(
            Action(ActionType.CHECK),
            Action(ActionType.RAISE, amount=10, min_amount=10, max_amount=900),
            Action(ActionType.FOLD),
        ),
    )


def _card_observation(group: str, slot: str, card: str):
    from holdem_bot.recognize import Candidate, CardSlotObservation

    evidence = RoiEvidence(roi_id=f"card:{group}:{slot}")
    return CardSlotObservation(
        group=group,
        slot=slot,
        occupancy="face_up",
        card_candidates=(
            Candidate(value=card, confidence=0.95, source="image", evidence=(evidence,)),
        ),
        accepted_card=card,
        accepted_by_single_frame=True,
        confidence=0.95,
        evidence=evidence,
    )


def _button_observation(slot: str, action: str) -> ButtonObservation:
    return ButtonObservation(
        slot=slot,
        visible=True,
        enabled=True,
        accepted_action=action,
        confidence=0.95,
        evidence=RoiEvidence(roi_id=f"button:{slot}"),
    )
