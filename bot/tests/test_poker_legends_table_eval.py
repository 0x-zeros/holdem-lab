import json
from pathlib import Path

from holdem_bot.adapters import poker_legends_table_eval
from holdem_bot.capture import CapturedFrame
from holdem_bot.recognize import (
    ActionPanelObservation,
    AssemblyIssue,
    AssemblyStatus,
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
from holdem_common import Action, ActionType, Card, GameState, PlayerState, Pot, Street


def test_table_eval_scans_actionable_frames_and_writes_report(
    tmp_path: Path,
    monkeypatch,
) -> None:
    frames = tmp_path / "frames"
    annotations = tmp_path / "annotations"
    truth = tmp_path / "truth"
    out = tmp_path / "out"
    frames.mkdir()
    annotations.mkdir()
    truth.mkdir()
    for frame_id, screen_kind in (
        ("frame_001", "actionable_table"),
        ("frame_002", "table_observe"),
        ("frame_003", "actionable_table"),
    ):
        image = frames / f"{frame_id}.png"
        image.write_bytes(b"fake")
        (annotations / f"{frame_id}.json").write_text(
            json.dumps({"image": str(image), "regions": {}}),
            encoding="utf-8",
        )
        (truth / f"{frame_id}.json").write_text(
            json.dumps({"frame_id": frame_id, "screen": {"kind": screen_kind}}),
            encoding="utf-8",
        )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "annotation_dir": str(annotations),
                "frames": [
                    {
                        "frame_id": frame_id,
                        "image": str(frames / f"{frame_id}.png"),
                        "truth_path": str(truth / f"{frame_id}.json"),
                    }
                    for frame_id in ("frame_001", "frame_002", "frame_003")
                ],
            }
        ),
        encoding="utf-8",
    )

    class FakeRecognizerFactory:
        @classmethod
        def from_manifests(cls, **_kwargs):
            return FakeRecognizer()

    monkeypatch.setattr(
        poker_legends_table_eval,
        "PokerLegendsTableRecognizer",
        FakeRecognizerFactory,
    )

    summary = poker_legends_table_eval.evaluate_poker_legends_table_recognizer(
        dataset_manifest_path=manifest,
        card_part_manifest="unused-card-parts.json",
        card_classifier_manifest="unused-card-classifier.json",
        button_manifest="unused-buttons.json",
        output_dir=out,
    )

    assert summary["frames"] == 2
    assert summary["screen_kind_counts"] == {"actionable_table": 2}
    assert summary["authorization_events"] == 1
    assert summary["non_actionable_frames"] == 0
    assert summary["false_actionable_count"] == 0
    assert summary["false_actionable_examples"] == []
    assert summary["result_counts"] == {"missing_pot": 1, "state": 1}
    assert summary["issue_counts"] == {"POT_REQUIRED_BY_POLICY": 1}
    assert summary["action_panel_flag_counts"] == {"missing_current_action_row": 1}
    assert summary["blocking_action_panel_flag_counts"] == {"missing_current_action_row": 1}
    assert summary["examples"] == {"missing_pot": ["frame_003"], "state": ["frame_001"]}
    rows = summary["rows"]
    assert isinstance(rows, list)
    assert rows[0]["frame_id"] == "frame_001"
    assert rows[0]["state"]["street"] == "flop"
    assert rows[1]["frame_id"] == "frame_003"
    assert rows[1]["issue_codes"] == ["POT_REQUIRED_BY_POLICY"]
    assert (out / "table_recognizer_summary.json").exists()
    report = (out / "table_recognizer_report.md").read_text(encoding="utf-8")
    assert "# Poker Legends Table Recognizer Report" in report
    assert "- state: 1" in report
    assert "- Authorization events: 1" in report
    assert "- False actionable count: 0" in report
    assert "- POT_REQUIRED_BY_POLICY: 1" in report
    assert "- missing_current_action_row: 1" in report
    assert "## Blocking Action Panel Flag Counts" in report
    assert "| `frame_003` | `missing_pot` | `POT_REQUIRED_BY_POLICY` |" in report

    all_out = tmp_path / "all-out"
    all_summary = poker_legends_table_eval.evaluate_poker_legends_table_recognizer(
        dataset_manifest_path=manifest,
        card_part_manifest="unused-card-parts.json",
        card_classifier_manifest="unused-card-classifier.json",
        button_manifest="unused-buttons.json",
        output_dir=all_out,
        actionable_only=False,
    )

    assert all_summary["frames"] == 3
    assert all_summary["screen_kind_counts"] == {
        "actionable_table": 2,
        "table_observe": 1,
    }
    assert all_summary["authorization_events"] == 2
    assert all_summary["non_actionable_frames"] == 1
    assert all_summary["false_actionable_count"] == 1
    assert all_summary["false_actionable_examples"] == ["frame_002"]


class FakeRecognizer:
    def recognize(self, frame: CapturedFrame) -> RecognitionResult:
        frame_id = Path(str(frame.payload)).stem
        if frame_id == "frame_003":
            return _blocked_result(frame_id)
        state = _state(frame_id)
        evidence = FrameEvidence(session_id=None, frame_id=frame_id)
        screen = ScreenState.actionable_table(hero_turn=True)
        assembly = GameStateAssemblyResult(
            status=AssemblyStatus.SINGLE_FRAME_VALID,
            validity_scope=ValidityScope.SINGLE_FRAME,
            state=state,
            contract_level=ContractLevel.POLICY_DECISION,
            contract_status="satisfied",
            valid_for=(ContractLevel.POLICY_DECISION,),
            issues=(),
            freshness=Freshness(
                source_frame_id=frame_id,
                current_frame_revalidated=True,
                critical_fields_fresh=True,
                action_row_fresh=True,
                stable_frame_count=1,
            ),
            screen_confidence=1.0,
            observation_id=frame_id,
        )
        return RecognitionResult(
            state=state,
            confidence=1.0,
            metadata={
                "recognized_table": {
                    "street": "flop",
                    "buttons": [{"action_type": "check"}],
                },
                "assembly_result": assembly.to_dict(),
            },
            screen=screen,
            recognition_mode=RecognitionMode.TRUTH_ASSISTED_REPLAY,
            safety_contract=ContractLevel.POLICY_DECISION,
            frame_evidence=evidence,
            assembly_result=assembly,
        )


def _blocked_result(frame_id: str) -> RecognitionResult:
    evidence = FrameEvidence(session_id=None, frame_id=frame_id)
    screen = ScreenState.actionable_table(hero_turn=True)
    visual = VisualObservation(
        frame=evidence,
        recognition_mode=RecognitionMode.TRUTH_ASSISTED_REPLAY,
        screen=screen,
        layout=LayoutObservation(
            profile_id=None,
            layout_version=None,
            transform_type=None,
            transform_residual_px=None,
        ),
        action_panels=(
            ActionPanelObservation(
                panel_kind="current_action_row",
                visible=False,
                enabled=None,
                hero_turn_indicator=True,
                row_bbox=None,
                ambiguity_flags=("missing_current_action_row",),
                evidence=RoiEvidence(roi_id="action_panel:current_action_row"),
            ),
        ),
    )
    issue = AssemblyIssue(
        issue_type="missing",
        reason_code="POT_REQUIRED_BY_POLICY",
        field_path="numbers.pot",
        rule_name="single_frame_contract",
        severity="hard",
        blocking=True,
        required_by_contract=(ContractLevel.POLICY_DECISION,),
        message="missing_pot",
    )
    assembly = GameStateAssemblyResult(
        status=AssemblyStatus.NO_STATE,
        validity_scope=ValidityScope.NONE,
        state=None,
        contract_level=ContractLevel.OBSERVE_ONLY,
        contract_status="blocked",
        valid_for=(ContractLevel.OBSERVE_ONLY,),
        issues=(issue,),
        freshness=Freshness(
            source_frame_id=frame_id,
            current_frame_revalidated=True,
            critical_fields_fresh=False,
            action_row_fresh=False,
        ),
        screen_confidence=1.0,
        observation_id=frame_id,
    )
    return RecognitionResult(
        state=None,
        confidence=1.0,
        metadata={
            "state_block_reason": "missing_pot",
            "recognized_table": {
                "street": "turn",
                "pot": None,
                "buttons": [{"command": "primary_left", "action_type": "check"}],
                "seats": [{"seat": 0, "stack": 1000, "committed": 0, "current": True}],
            },
            "assembly_result": assembly.to_dict(),
        },
        screen=screen,
        recognition_mode=RecognitionMode.TRUTH_ASSISTED_REPLAY,
        safety_contract=ContractLevel.OBSERVE_ONLY,
        frame_evidence=evidence,
        visual_observation=visual,
        assembly_result=assembly,
    )


def _state(hand_id: str) -> GameState:
    return GameState(
        hand_id=hand_id,
        street=Street.FLOP,
        players=(
            PlayerState(
                seat=0,
                stack=1000,
                hole_cards=(Card.from_code("As"), Card.from_code("Kh")),
            ),
            PlayerState(seat=1, stack=1000),
        ),
        board=(Card.from_code("2c"), Card.from_code("7d"), Card.from_code("Ts")),
        pots=(Pot(amount=150, eligible_seats=frozenset({0, 1})),),
        current_seat=0,
        button_seat=1,
        small_blind=5,
        big_blind=10,
        min_raise=10,
        to_call=0,
        legal_actions=(Action(ActionType.CHECK),),
    )
