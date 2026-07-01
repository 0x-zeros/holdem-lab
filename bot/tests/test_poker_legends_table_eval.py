import json
from pathlib import Path

from holdem_bot.adapters import poker_legends_table_eval
from holdem_bot.capture import CapturedFrame
from holdem_bot.recognize import (
    AssemblyStatus,
    ContractLevel,
    FrameEvidence,
    Freshness,
    GameStateAssemblyResult,
    RecognitionMode,
    RecognitionResult,
    ValidityScope,
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
                    for frame_id in ("frame_001", "frame_002")
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

    assert summary["frames"] == 1
    assert summary["result_counts"] == {"state": 1}
    assert summary["examples"] == {"state": ["frame_001"]}
    rows = summary["rows"]
    assert isinstance(rows, list)
    assert rows[0]["frame_id"] == "frame_001"
    assert rows[0]["state"]["street"] == "flop"
    assert (out / "table_recognizer_summary.json").exists()
    report = (out / "table_recognizer_report.md").read_text(encoding="utf-8")
    assert "# Poker Legends Table Recognizer Report" in report
    assert "- state: 1" in report


class FakeRecognizer:
    def recognize(self, frame: CapturedFrame) -> RecognitionResult:
        frame_id = Path(str(frame.payload)).stem
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
