import json
from pathlib import Path

from holdem_bot.adapters import poker_legends_table_eval
from holdem_bot.capture import CapturedFrame
from holdem_bot.recognize import (
    AcceptedCriticalField,
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
        truth_payload: dict[str, object] = {
            "frame_id": frame_id,
            "screen": {"kind": screen_kind},
        }
        if frame_id == "frame_001":
            truth_payload.update(
                {
                    "buttons": [
                        {
                            "name": "primary_left",
                            "visible": True,
                            "action_type": "raise",
                            "label": "Check",
                        }
                    ],
                    "texts": [
                        {
                            "name": "pot",
                            "visible": True,
                            "value": "$150",
                            "normalized_number": 150,
                        },
                        {
                            "name": "right_top_stack",
                            "visible": True,
                            "value": "$995",
                            "normalized_number": 995,
                        }
                    ],
                }
            )
        if frame_id == "frame_003":
            truth_payload.update(
                {
                    "buttons": [
                        {
                            "name": "primary_left",
                            "visible": True,
                            "action_type": "check",
                            "label": "Check",
                        }
                    ],
                    "seats": [
                        {
                            "name": "hero",
                            "visible": True,
                            "stack": 1000,
                            "committed": 0,
                            "current": True,
                        }
                    ],
                    "texts": [
                        {
                            "name": "pot",
                            "visible": True,
                            "value": "$150",
                            "normalized_number": 150,
                        }
                    ],
                }
            )
        (truth / f"{frame_id}.json").write_text(
            json.dumps(truth_payload),
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
    assert summary["screen_confusion_counts"] == {
        "actionable_table": {"actionable_table": 2}
    }
    assert summary["screen_false_actionable_count"] == 0
    assert summary["screen_false_actionable_examples"] == []
    assert summary["screen_missed_actionable_count"] == 0
    assert summary["screen_missed_actionable_examples"] == []
    assert summary["recognition_mode_counts"] == {"truth_assisted_replay": 2}
    assert summary["contract_counts"] == {"observe_only": 1, "policy_decision": 1}
    assert summary["assembly_status_counts"] == {"no_state": 1, "single_frame_valid": 1}
    assert summary["table_readiness_flag_counts"] == {}
    assert summary["number_truth_comparison_counts"] == {}
    assert summary["number_truth_mismatch_examples"] == []
    assert summary["number_prediction_slot_counts"] == {}
    assert summary["accepted_number_prediction_slot_counts"] == {}
    assert summary["number_prediction_confidence_counts"] == {}
    assert summary["accepted_number_prediction_confidence_counts"] == {}
    assert summary["authorization_events"] == 1
    assert summary["unsafe_authorization_events"] == 0
    assert summary["stale_authorization_events"] == 0
    assert summary["truth_assisted_authorization_events"] == 1
    assert summary["non_actionable_frames"] == 0
    assert summary["false_actionable_count"] == 0
    assert summary["false_actionable_examples"] == []
    assert summary["source_policy_violation_count"] == 0
    assert summary["accepted_critical_wrong_count"] == 0
    assert summary["accepted_critical_wrong_examples"] == []
    assert summary["review_queue_frames"] == 1
    assert summary["review_queue_tag_counts"] == {"missing_pot": 1}
    assert summary["review_queue_by_tag"] == {"missing_pot": ["frame_003"]}
    assert summary["result_counts"] == {"missing_pot": 1, "state": 1}
    assert summary["issue_counts"] == {"POT_REQUIRED_BY_POLICY": 1}
    assert summary["action_panel_flag_counts"] == {"missing_current_action_row": 1}
    assert summary["blocking_action_panel_flag_counts"] == {"missing_current_action_row": 1}
    assert summary["review_tag_counts"] == {"missing_pot": 1}
    assert summary["examples"] == {"missing_pot": ["frame_003"], "state": ["frame_001"]}
    rows = summary["rows"]
    assert isinstance(rows, list)
    assert rows[0]["frame_id"] == "frame_001"
    assert rows[0]["state"]["street"] == "flop"
    assert rows[0]["recognition_mode"] == "truth_assisted_replay"
    assert rows[0]["safety_contract"] == "policy_decision"
    assert rows[0]["table_readiness_flags"] == []
    assert rows[0]["accepted_critical_fields"] == [
        {
            "evidence_refs": [],
            "field_path": "numbers.pot",
            "source": "reviewed_truth",
            "value": 150,
        },
        {
            "evidence_refs": [],
            "field_path": "actions.legal_labels",
            "source": "reviewed_truth",
            "value": ("check",),
        }
    ]
    assert rows[1]["frame_id"] == "frame_003"
    assert rows[1]["truth_path"] == str(truth / "frame_003.json")
    assert rows[1]["layout_annotation_path"] == str(annotations / "frame_003.json")
    assert rows[1]["issue_codes"] == ["POT_REQUIRED_BY_POLICY"]
    assert rows[1]["review_tags"] == ["missing_pot"]
    assert rows[1]["truth"]["buttons"] == [
        {
            "action_type": "check",
            "label": "Check",
            "name": "primary_left",
            "visible": True,
        }
    ]
    assert (out / "table_recognizer_summary.json").exists()
    review_queue = json.loads(
        (out / "table_recognizer_review_queue.json").read_text(encoding="utf-8")
    )
    assert [row["frame_id"] for row in review_queue] == ["frame_003"]
    review_queue_by_tag = json.loads(
        (out / "table_recognizer_review_queue_by_tag.json").read_text(encoding="utf-8")
    )
    assert review_queue_by_tag == {"missing_pot": ["frame_003"]}
    report = (out / "table_recognizer_report.md").read_text(encoding="utf-8")
    assert "# Poker Legends Table Recognizer Report" in report
    assert "- state: 1" in report
    assert "- Authorization events: 1" in report
    assert "- Image-only replay: False" in report
    assert "- Unsafe authorization events: 0" in report
    assert "- Truth-assisted authorization events: 1" in report
    assert "- Accepted critical wrong count: 0" in report
    assert "- False actionable count: 0" in report
    assert "- Screen false actionable count: 0" in report
    assert "- Screen missed actionable count: 0" in report
    assert "- Review queue frames: 1" in report
    assert "- POT_REQUIRED_BY_POLICY: 1" in report
    assert "## Table Readiness Flag Counts" in report
    assert "## Number Readiness Flag Counts" in report
    assert "## Number Readiness By Flag" in report
    assert "## Number Truth Comparison Counts" in report
    assert "## Number Truth Mismatch Examples" in report
    assert "## Number Prediction Slot Counts" in report
    assert "## Accepted Number Prediction Slot Counts" in report
    assert "## Number Prediction Confidence Counts" in report
    assert "## Accepted Number Prediction Confidence Counts" in report
    assert "## Number Readiness Details" in report
    assert "## Review Tag Counts" in report
    assert "## Screen Truth Confusion" in report
    assert "- actionable_table: actionable_table=2" in report
    assert "## Recognition Mode Counts" in report
    assert "- truth_assisted_replay: 2" in report
    assert "- missing_pot: 1" in report
    assert "## Review Queue By Tag" in report
    assert "- missing_pot: `frame_003`" in report
    assert "- missing_current_action_row: 1" in report
    assert "## Blocking Action Panel Flag Counts" in report
    assert "Truth Buttons" in report
    assert "`primary_left:visible=True:check:Check`" in report
    assert (
        "| `frame_003` | `missing_pot` | `missing_pot` | `POT_REQUIRED_BY_POLICY` |"
        in report
    )

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
    assert all_summary["screen_confusion_counts"] == {
        "actionable_table": {"actionable_table": 2},
        "table_observe": {"actionable_table": 1},
    }
    assert all_summary["screen_false_actionable_count"] == 1
    assert all_summary["screen_false_actionable_examples"] == ["frame_002"]
    assert all_summary["screen_missed_actionable_count"] == 0
    assert all_summary["recognition_mode_counts"] == {"truth_assisted_replay": 3}
    assert all_summary["assembly_status_counts"] == {
        "no_state": 1,
        "single_frame_valid": 2,
    }
    assert all_summary["authorization_events"] == 2
    assert all_summary["unsafe_authorization_events"] == 1
    assert all_summary["stale_authorization_events"] == 0
    assert all_summary["truth_assisted_authorization_events"] == 2
    assert all_summary["non_actionable_frames"] == 1
    assert all_summary["false_actionable_count"] == 1
    assert all_summary["false_actionable_examples"] == ["frame_002"]
    assert all_summary["source_policy_violation_count"] == 0
    assert all_summary["accepted_critical_wrong_count"] == 0
    assert all_summary["review_queue_frames"] == 1
    assert all_summary["review_queue_tag_counts"] == {"missing_pot": 1}
    assert all_summary["review_queue_by_tag"] == {"missing_pot": ["frame_003"]}

    image_only_out = tmp_path / "image-only-out"
    image_only_summary = poker_legends_table_eval.evaluate_poker_legends_table_recognizer(
        dataset_manifest_path=manifest,
        card_part_manifest="unused-card-parts.json",
        card_classifier_manifest="unused-card-classifier.json",
        button_manifest="unused-buttons.json",
        output_dir=image_only_out,
        image_only_replay=True,
    )

    assert image_only_summary["frames"] == 2
    assert image_only_summary["image_only_replay"] is True
    assert image_only_summary["screen_confusion_counts"] == {
        "actionable_table": {"actionable_table": 1, "table_observe": 1}
    }
    assert image_only_summary["screen_false_actionable_count"] == 0
    assert image_only_summary["screen_missed_actionable_count"] == 1
    assert image_only_summary["screen_missed_actionable_examples"] == ["frame_003"]
    assert image_only_summary["recognition_mode_counts"] == {"image_only_replay": 2}
    assert image_only_summary["authorization_events"] == 0
    assert image_only_summary["truth_assisted_authorization_events"] == 0
    assert image_only_summary["unsafe_authorization_events"] == 0
    assert image_only_summary["result_counts"] == {
        "missing_table_metadata": 1,
        "screen_not_actionable": 1,
    }
    assert image_only_summary["table_readiness_flag_counts"] == {
        "readiness_not_enough_players": 1
    }
    assert image_only_summary["number_readiness_flag_counts"] == {
        "readiness_low_confidence_opponent_stack": 1
    }
    assert image_only_summary["number_truth_comparison_counts"] == {
        "raw:numbers.right_top_stack:mismatch": 1
    }
    assert image_only_summary["number_truth_mismatch_examples"] == [
        {
            "confidence": 0.65,
            "expected": 995,
            "field_path": "numbers.right_top_stack",
            "frame_id": "frame_001",
            "predicted": 1000,
            "prediction_set": "raw",
            "raw": None,
            "status": "mismatch",
        }
    ]
    assert image_only_summary["number_prediction_slot_counts"] == {
        "texts:right_top_stack": 1
    }
    assert image_only_summary["accepted_number_prediction_slot_counts"] == {}
    assert image_only_summary["number_prediction_confidence_counts"] == {
        "texts:right_top_stack:conf=0.65": 1
    }
    assert image_only_summary["accepted_number_prediction_confidence_counts"] == {}
    assert image_only_summary["number_readiness_rows_count"] == 1
    assert image_only_summary["number_readiness_by_flag"] == {
        "readiness_low_confidence_opponent_stack": ["frame_001"]
    }
    assert image_only_summary["review_queue_by_tag"] == {
        "missing_table_metadata": ["frame_001"],
        "screen_missed_actionable": ["frame_003"],
    }
    number_readiness_by_flag = json.loads(
        (image_only_out / "table_recognizer_number_readiness_by_flag.json").read_text(
            encoding="utf-8"
        )
    )
    assert number_readiness_by_flag == {
        "readiness_low_confidence_opponent_stack": ["frame_001"]
    }
    number_readiness_rows = json.loads(
        (image_only_out / "table_recognizer_number_readiness_rows.json").read_text(
            encoding="utf-8"
        )
    )
    assert number_readiness_rows == [
        {
            "accepted_number_predictions": [],
            "frame_id": "frame_001",
            "layout_annotation_path": str(annotations / "frame_001.json"),
            "number_predictions": [
                {
                    "confidence": 0.65,
                    "group": "texts",
                    "name": "right_top_stack",
                    "normalized_number": 1000,
                }
            ],
            "number_readiness_flags": ["readiness_low_confidence_opponent_stack"],
            "result": "missing_table_metadata",
            "screen_kind": "actionable_table",
            "table_readiness_flags": ["readiness_not_enough_players"],
            "truth_path": str(truth / "frame_001.json"),
            "truth_screen_kind": "actionable_table",
        }
    ]
    image_only_report = (image_only_out / "table_recognizer_report.md").read_text(
        encoding="utf-8"
    )
    assert "| `frame_001` | `readiness_low_confidence_opponent_stack` |" in image_only_report


class FakeRecognizer:
    def recognize(self, frame: CapturedFrame) -> RecognitionResult:
        mode = RecognitionMode(str(frame.metadata.get("recognition_mode", "truth_assisted_replay")))
        frame_id = Path(str(frame.payload)).stem
        if mode is RecognitionMode.IMAGE_ONLY_REPLAY:
            assert "poker_legends_annotation_path" not in frame.metadata
            if frame_id == "frame_003":
                return _blocked_result(
                    frame_id,
                    mode=mode,
                    block_reason="screen_not_actionable",
                    reason_code="SCREEN_NOT_ACTIONABLE",
                    screen=ScreenState.table_observe(),
                )
            return _blocked_result(
                frame_id,
                mode=mode,
                block_reason="missing_table_metadata",
                reason_code="MISSING_TABLE_METADATA",
                recognized_table={
                    "street": "flop",
                    "pot": 150,
                    "board": (
                        {"card": "2C", "visible": True},
                        {"card": "7D", "visible": True},
                        {"card": "TS", "visible": True},
                    ),
                    "buttons": ({"command": "primary_left", "action_type": "check"},),
                    "seats": (
                        {
                            "seat": 0,
                            "stack": 1000,
                            "active": True,
                            "hole_cards": (
                                {"card": "AS", "visible": True},
                                {"card": "KH", "visible": True},
                            ),
                        },
                    ),
                },
                number_predictions=(
                    {
                        "confidence": 0.65,
                        "group": "texts",
                        "name": "right_top_stack",
                        "normalized_number": 1000,
                    },
                ),
                accepted_number_predictions=(),
            )
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
            accepted_critical_fields=(
                AcceptedCriticalField("numbers.pot", "reviewed_truth", 150),
                AcceptedCriticalField(
                    "actions.legal_labels",
                    "reviewed_truth",
                    ("check",),
                ),
            ),
        )


def _blocked_result(
    frame_id: str,
    *,
    mode: RecognitionMode = RecognitionMode.TRUTH_ASSISTED_REPLAY,
    block_reason: str = "missing_pot",
    reason_code: str = "POT_REQUIRED_BY_POLICY",
    screen: ScreenState | None = None,
    recognized_table: dict[str, object] | None = None,
    number_predictions: tuple[dict[str, object], ...] = (),
    accepted_number_predictions: tuple[dict[str, object], ...] = (),
) -> RecognitionResult:
    evidence = FrameEvidence(session_id=None, frame_id=frame_id)
    screen = screen or ScreenState.actionable_table(hero_turn=True)
    visual = VisualObservation(
        frame=evidence,
        recognition_mode=mode,
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
        reason_code=reason_code,
        field_path="numbers.pot",
        rule_name="single_frame_contract",
        severity="hard",
        blocking=True,
        required_by_contract=(ContractLevel.POLICY_DECISION,),
        message=block_reason,
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
            "state_block_reason": block_reason,
            "recognized_table": recognized_table or {
                "street": "turn",
                "pot": None,
                "buttons": [{"command": "primary_left", "action_type": "check"}],
                "seats": [{"seat": 0, "stack": 1000, "committed": 0, "current": True}],
            },
            "assembly_result": assembly.to_dict(),
            "number_predictions": list(number_predictions),
            "accepted_number_predictions": list(accepted_number_predictions),
        },
        screen=screen,
        recognition_mode=mode,
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
