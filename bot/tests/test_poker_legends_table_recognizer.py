from pathlib import Path
from typing import cast

from holdem_bot import CapturedFrame, ScreenKind
from holdem_bot.adapters import PokerLegendsTableRecognizer, poker_legends
from holdem_bot.recognize import (
    TRUTH_ASSISTED_SOURCE_BLOCK_REASON,
    AssemblyStatus,
    ContractLevel,
    RecognitionMode,
    RecognitionResult,
    ValidityScope,
    evaluate_accepted_critical_fields,
    summarize_recognition_safety,
)
from holdem_bot.vision import PokerLegendsCardConsensusPrediction, PokerLegendsNumberPrediction
from holdem_bot.vision.poker_legends_buttons import PokerLegendsButtonPrediction
from holdem_common import Action, ActionType, Street


def test_poker_legends_table_recognizer_builds_prototype_state(tmp_path: Path) -> None:
    image = tmp_path / "frame.png"
    image.write_bytes(b"not-read-by-fakes")
    annotation = actionable_truth()
    layout = {"image": str(image), "regions": {"cards": [], "board": [], "buttons": []}}
    recognizer = PokerLegendsTableRecognizer(
        card_recognizer=FakeCardRecognizer(
            (
                card_prediction("hero_hole_cards", "hero_hole_0", "AS", 0.95),
                card_prediction("hero_hole_cards", "hero_hole_1", "KH", 0.94),
                card_prediction("board", "board_0", "2C", 0.93),
                card_prediction("board", "board_1", "7D", 0.92),
                card_prediction("board", "board_2", "TS", 0.91),
            )
        ),
        button_recognizer=FakeButtonRecognizer(
            (
                button_prediction("primary_left", "check", 0.90),
                button_prediction("primary_middle", "raise", 0.90),
                button_prediction("primary_right", "fold", 0.90),
            )
        ),
        controlled_seat=0,
    )

    result = recognizer.recognize(
        CapturedFrame(
            payload=image,
            source="poker_legends_fixture",
            metadata={
                "poker_legends_annotation": annotation,
                "poker_legends_layout_annotation": layout,
            },
        )
    )

    assert result.screen.kind is ScreenKind.ACTIONABLE_TABLE
    assert result.recognition_mode is RecognitionMode.TRUTH_ASSISTED_REPLAY
    assert result.metadata["recognition_mode"] == "truth_assisted_replay"
    assert result.frame_evidence is not None
    assert result.frame_evidence.frame_id == "frame_001"
    assert result.frame_evidence.image_hash is not None
    assert result.metadata["frame_evidence"] == result.frame_evidence.to_dict()
    assert result.visual_observation is not None
    assert result.visual_observation.frame == result.frame_evidence
    assert len(result.visual_observation.cards) == 5
    assert result.visual_observation.action_panels[0].panel_kind == "current_action_row"
    assert result.assembly_result is not None
    assert result.assembly_result.status is AssemblyStatus.SINGLE_FRAME_VALID
    assert result.assembly_result.validity_scope is ValidityScope.SINGLE_FRAME
    assert result.assembly_result.contract_level is ContractLevel.POLICY_DECISION
    assert result.safety_contract is ContractLevel.POLICY_DECISION
    assert result.metadata["visual_observation"] == result.visual_observation.to_dict()
    assert result.metadata["assembly_result"] == result.assembly_result.to_dict()
    assert result.state is not None
    assert result.state.metadata["source"] == "poker_legends_prototype"
    assert result.state.street is Street.FLOP
    assert result.state.current_seat == 0
    assert [card.code for card in result.state.player(0).hole_cards] == ["As", "Kh"]
    assert [card.code for card in result.state.board] == ["2c", "7d", "Ts"]
    assert {action.action_type for action in result.state.legal_actions} == {
        ActionType.CHECK,
        ActionType.RAISE,
        ActionType.FOLD,
    }
    table = result.metadata["recognized_table"]
    assert isinstance(table, dict)
    assert table["pot"] == 150
    assert result.confidence == 0.75
    accepted_sources = {
        field.field_path: field.source for field in result.accepted_critical_fields
    }
    assert accepted_sources["cards.hero"] == "image_card_consensus"
    assert accepted_sources["numbers.pot"] == "reviewed_truth"
    assert result.metadata["accepted_critical_fields"] == [
        field.to_dict() for field in result.accepted_critical_fields
    ]


def test_poker_legends_table_recognizer_blocks_truth_assist_in_image_only_mode(
    tmp_path: Path,
) -> None:
    image = tmp_path / "frame.png"
    image.write_bytes(b"not-read-by-fakes")
    card_recognizer = FakeCardRecognizer(())
    button_recognizer = FakeButtonRecognizer(())
    recognizer = PokerLegendsTableRecognizer(
        card_recognizer=card_recognizer,
        button_recognizer=button_recognizer,
        controlled_seat=0,
    )

    result = recognizer.recognize(
        CapturedFrame(
            payload=image,
            source="poker_legends_fixture",
            metadata={
                "recognition_mode": RecognitionMode.IMAGE_ONLY_REPLAY.value,
                "poker_legends_annotation": actionable_truth(),
                "poker_legends_layout_annotation": {
                    "image": str(image),
                    "regions": {"cards": [], "board": [], "buttons": []},
                },
            },
        )
    )

    assert result.recognition_mode is RecognitionMode.IMAGE_ONLY_REPLAY
    assert result.state is None
    assert result.metadata["state_block_reason"] == TRUTH_ASSISTED_SOURCE_BLOCK_REASON
    assert result.assembly_result is not None
    assert result.assembly_result.status is AssemblyStatus.INVALID
    assert result.assembly_result.validity_scope is ValidityScope.NONE
    assert result.assembly_result.issues[0].reason_code == (
        "TRUTH_ASSISTED_FIELD_IN_IMAGE_ONLY_MODE"
    )
    assert result.safety_contract is ContractLevel.OBSERVE_ONLY
    assert result.visual_observation is not None
    assert result.source_policy_violations
    assert result.source_policy_violations[0].field_path == "poker_legends_annotation"
    assert result.metadata["source_policy_violations"] == [
        violation.to_dict() for violation in result.source_policy_violations
    ]
    assert card_recognizer.calls == 0
    assert button_recognizer.calls == 0


def test_minimal_critical_field_evaluator_reports_wrong_accepted_fields(
    tmp_path: Path,
) -> None:
    result = _recognize_actionable(tmp_path, actionable_truth())

    evaluation = evaluate_accepted_critical_fields(
        result,
        expected_values={"numbers.pot": 999, "cards.hero": ("As", "Kh")},
    )

    assert evaluation.authorization_events == 1
    assert evaluation.unsafe_authorization_events == 1
    assert [case.field_path for case in evaluation.accepted_critical_wrong_cases] == [
        "numbers.pot"
    ]


def test_recognition_safety_summary_groups_modes_and_authorizations(
    tmp_path: Path,
) -> None:
    valid = _recognize_actionable(tmp_path, actionable_truth())
    blocked = _recognize_actionable(
        tmp_path,
        actionable_truth(),
        recognition_mode=RecognitionMode.IMAGE_ONLY_REPLAY,
    )

    summary = summarize_recognition_safety(
        (valid, blocked),
        expected_screen_kind_by_frame={"frame_001": "table_observe"},
        expected_values_by_frame={"frame_001": {"numbers.pot": 999}},
    )

    assert summary.total_frames == 2
    assert summary.mode_counts == {
        "truth_assisted_replay": 1,
        "image_only_replay": 1,
    }
    assert summary.assembly_status_counts == {
        "single_frame_valid": 1,
        "invalid": 1,
    }
    assert summary.authorization_events == 1
    assert summary.truth_assisted_authorization_events == 1
    assert summary.expected_non_actionable_frames == 2
    assert summary.false_actionable_count == 1
    assert summary.source_policy_violation_count == 1
    assert summary.accepted_critical_wrong_count == 1
    assert summary.unsafe_authorization_events == 1
    assert summary.blocking_issue_counts == {
        "TRUTH_ASSISTED_FIELD_IN_IMAGE_ONLY_MODE": 1,
    }


def _recognize_actionable(
    tmp_path: Path,
    annotation: dict[str, object],
    *,
    recognition_mode: RecognitionMode | None = None,
) -> RecognitionResult:
    image = tmp_path / "frame.png"
    image.write_bytes(b"not-read-by-fakes")
    recognizer = PokerLegendsTableRecognizer(
        card_recognizer=FakeCardRecognizer(
            (
                card_prediction("hero_hole_cards", "hero_hole_0", "AS", 0.95),
                card_prediction("hero_hole_cards", "hero_hole_1", "KH", 0.94),
                card_prediction("board", "board_0", "2C", 0.93),
                card_prediction("board", "board_1", "7D", 0.92),
                card_prediction("board", "board_2", "TS", 0.91),
            )
        ),
        button_recognizer=FakeButtonRecognizer(
            (
                button_prediction("primary_left", "check", 0.90),
                button_prediction("primary_middle", "raise", 0.90),
                button_prediction("primary_right", "fold", 0.90),
            )
        ),
        controlled_seat=0,
    )
    return recognizer.recognize(
        CapturedFrame(
            payload=image,
            source="poker_legends_fixture",
            metadata={
                **(
                    {"recognition_mode": recognition_mode.value}
                    if recognition_mode is not None
                    else {}
                ),
                "poker_legends_annotation": annotation,
                "poker_legends_layout_annotation": {
                    "image": str(image),
                    "regions": {"cards": [], "board": [], "buttons": []},
                },
            },
        )
    )


def test_button_seat_recognized_from_annotation_position(tmp_path: Path) -> None:
    annotation = actionable_truth()
    seats = cast(list[dict[str, object]], annotation["seats"])
    seats[0]["position"] = "button"  # hero is on the button this hand
    result = _recognize_actionable(tmp_path, annotation)
    assert result.state is not None
    assert result.state.button_seat == 0
    assert result.state.current_seat == result.state.button_seat
    assert result.state.metadata["button_seat_source"] == "recognized"


def test_button_seat_defaults_to_out_of_position_when_unknown(tmp_path: Path) -> None:
    # The base fixture has no position info: the bot must NOT assume hero is the
    # button (the old hardcoded bug). It treats hero as out of position instead, so
    # the heuristic uses its tighter, safer ranges.
    result = _recognize_actionable(tmp_path, actionable_truth())
    assert result.state is not None
    assert result.state.button_seat != result.state.current_seat
    assert result.state.metadata["button_seat_source"] == "default_oop"


def test_blinds_are_threaded_from_annotation(tmp_path: Path) -> None:
    annotation = actionable_truth()
    annotation["table_state"] = {"street": "flop", "small_blind": 25, "big_blind": 50}
    result = _recognize_actionable(tmp_path, annotation)
    assert result.state is not None
    assert result.state.small_blind == 25
    assert result.state.big_blind == 50
    assert result.state.min_raise == 50


def test_raise_floor_accounts_for_the_amount_to_call(tmp_path: Path) -> None:
    image = tmp_path / "frame.png"
    image.write_bytes(b"not-read-by-fakes")
    annotation = actionable_truth()
    annotation["buttons"] = [
        {"name": "primary_left", "visible": True, "action_type": "call", "label": "Call 20"},
        {"name": "primary_middle", "visible": True, "action_type": "raise", "label": "Raise"},
        {"name": "primary_right", "visible": True, "action_type": "fold", "label": "Fold"},
    ]
    recognizer = PokerLegendsTableRecognizer(
        card_recognizer=FakeCardRecognizer(
            (
                card_prediction("hero_hole_cards", "hero_hole_0", "AS", 0.95),
                card_prediction("hero_hole_cards", "hero_hole_1", "KH", 0.94),
                card_prediction("board", "board_0", "2C", 0.93),
                card_prediction("board", "board_1", "7D", 0.92),
                card_prediction("board", "board_2", "TS", 0.91),
            )
        ),
        button_recognizer=FakeButtonRecognizer(
            (
                button_prediction("primary_left", "call", 0.90),
                button_prediction("primary_middle", "raise", 0.90),
                button_prediction("primary_right", "fold", 0.90),
            )
        ),
        controlled_seat=0,  # fixture hero: stack 900, committed 0; config big blind 10
    )
    result = recognizer.recognize(
        CapturedFrame(
            payload=image,
            source="poker_legends_fixture",
            metadata={
                "poker_legends_annotation": annotation,
                "poker_legends_layout_annotation": {
                    "image": str(image),
                    "regions": {"cards": [], "board": [], "buttons": []},
                },
            },
        )
    )
    assert result.state is not None
    assert result.state.to_call == 20
    raise_action = next(
        action for action in result.state.legal_actions if action.action_type is ActionType.RAISE
    )
    # Legal min raise-to = committed(0) + to_call(20) + big_blind(10) = 30, not the
    # old buggy committed + big_blind = 10, which is below the call amount.
    assert raise_action.min_amount == 30
    assert raise_action.max_amount == 900
    assert raise_action.amount == 30


def test_seat_name_is_not_treated_as_a_dealer_position(tmp_path: Path) -> None:
    # A seat literally NAMED "dealer" but with no position field must NOT be
    # promoted to a recognized button — that would silently override the OOP safety
    # default the position threading exists to provide.
    annotation = actionable_truth()
    seats = cast(list[dict[str, object]], annotation["seats"])
    seats[1]["name"] = "dealer"
    result = _recognize_actionable(tmp_path, annotation)
    assert result.state is not None
    assert result.state.metadata["button_seat_source"] == "default_oop"


def test_raise_button_dropped_when_hero_cannot_legally_raise(tmp_path: Path) -> None:
    image = tmp_path / "frame.png"
    image.write_bytes(b"not-read-by-fakes")
    annotation = actionable_truth()
    for seat in cast(list[dict[str, object]], annotation["seats"]):
        if seat.get("name") == "hero":
            seat["stack"] = 5  # underwater: stack below the 20 call, no legal raise
    annotation["buttons"] = [
        {"name": "primary_left", "visible": True, "action_type": "call", "label": "Call 20"},
        {"name": "primary_middle", "visible": True, "action_type": "raise", "label": "Raise"},
        {"name": "primary_right", "visible": True, "action_type": "fold", "label": "Fold"},
    ]
    recognizer = PokerLegendsTableRecognizer(
        card_recognizer=FakeCardRecognizer(
            (
                card_prediction("hero_hole_cards", "hero_hole_0", "AS", 0.95),
                card_prediction("hero_hole_cards", "hero_hole_1", "KH", 0.94),
                card_prediction("board", "board_0", "2C", 0.93),
                card_prediction("board", "board_1", "7D", 0.92),
                card_prediction("board", "board_2", "TS", 0.91),
            )
        ),
        button_recognizer=FakeButtonRecognizer(
            (
                button_prediction("primary_left", "call", 0.90),
                button_prediction("primary_middle", "raise", 0.90),
                button_prediction("primary_right", "fold", 0.90),
            )
        ),
        controlled_seat=0,
    )
    result = recognizer.recognize(
        CapturedFrame(
            payload=image,
            source="poker_legends_fixture",
            metadata={
                "poker_legends_annotation": annotation,
                "poker_legends_layout_annotation": {
                    "image": str(image),
                    "regions": {"cards": [], "board": [], "buttons": []},
                },
            },
        )
    )
    assert result.state is not None
    assert not any(action.action_type is ActionType.RAISE for action in result.state.legal_actions)


def test_poker_legends_table_recognizer_skips_non_actionable_without_image_work(
    tmp_path: Path,
) -> None:
    image = tmp_path / "frame.png"
    image.write_bytes(b"not-read-by-fakes")
    annotation = actionable_truth()
    annotation["screen"] = {"kind": "table_observe", "confidence": 0.99, "hero_turn": False}
    card_recognizer = FakeCardRecognizer(())
    button_recognizer = FakeButtonRecognizer(())

    recognizer = PokerLegendsTableRecognizer(
        card_recognizer=card_recognizer,
        button_recognizer=button_recognizer,
        controlled_seat=0,
    )

    result = recognizer.recognize(
        CapturedFrame(
            payload=image,
            source="poker_legends_fixture",
            metadata={
                "poker_legends_annotation": annotation,
                "poker_legends_layout_annotation": {"image": str(image), "regions": {}},
            },
        )
    )

    assert result.screen.kind is ScreenKind.TABLE_OBSERVE
    assert result.state is None
    assert card_recognizer.calls == 0
    assert button_recognizer.calls == 0


def test_poker_legends_table_recognizer_fails_closed_without_pot(tmp_path: Path) -> None:
    image = tmp_path / "frame.png"
    image.write_bytes(b"not-read-by-fakes")
    annotation = actionable_truth()
    annotation["texts"] = []
    recognizer = PokerLegendsTableRecognizer(
        card_recognizer=FakeCardRecognizer(
            (
                card_prediction("hero_hole_cards", "hero_hole_0", "AS", 0.95),
                card_prediction("hero_hole_cards", "hero_hole_1", "KH", 0.94),
                card_prediction("board", "board_0", "2C", 0.93),
                card_prediction("board", "board_1", "7D", 0.92),
                card_prediction("board", "board_2", "TS", 0.91),
            )
        ),
        button_recognizer=FakeButtonRecognizer((button_prediction("primary_left", "check", 0.90),)),
        controlled_seat=0,
    )

    result = recognizer.recognize(
        CapturedFrame(
            payload=image,
            source="poker_legends_fixture",
            metadata={
                "poker_legends_annotation": annotation,
                "poker_legends_layout_annotation": {"image": str(image), "regions": {}},
            },
        )
    )

    assert result.screen.kind is ScreenKind.ACTIONABLE_TABLE
    assert result.state is None
    assert result.metadata["state_block_reason"] == "missing_pot"


def test_poker_legends_table_recognizer_derives_pot_from_explicit_committed(
    tmp_path: Path,
) -> None:
    annotation = actionable_truth()
    annotation["texts"] = []
    for seat in cast(list[dict[str, object]], annotation["seats"]):
        seat["committed"] = 100

    result = _recognize_actionable(tmp_path, annotation)

    assert result.state is not None
    assert result.state.pots[0].amount == 200
    pot_field = next(
        field for field in result.accepted_critical_fields if field.field_path == "numbers.pot"
    )
    assert pot_field.source == "rule_inferred_committed"


def test_poker_legends_table_recognizer_rejects_low_confidence_number_ocr(
    tmp_path: Path,
) -> None:
    image = tmp_path / "frame.png"
    image.write_bytes(b"not-read-by-fakes")
    annotation = actionable_truth()
    annotation["texts"] = []
    recognizer = PokerLegendsTableRecognizer(
        card_recognizer=FakeCardRecognizer(
            (
                card_prediction("hero_hole_cards", "hero_hole_0", "AS", 0.95),
                card_prediction("hero_hole_cards", "hero_hole_1", "KH", 0.94),
                card_prediction("board", "board_0", "2C", 0.93),
                card_prediction("board", "board_1", "7D", 0.92),
                card_prediction("board", "board_2", "TS", 0.91),
            )
        ),
        button_recognizer=FakeButtonRecognizer((button_prediction("primary_left", "check", 0.90),)),
        number_recognizer=FakeNumberRecognizer((number_prediction("texts", "pot", 150, 0.55),)),
        controlled_seat=0,
    )

    result = recognizer.recognize(
        CapturedFrame(
            payload=image,
            source="poker_legends_fixture",
            metadata={
                "poker_legends_annotation": annotation,
                "poker_legends_layout_annotation": {"image": str(image), "regions": {}},
            },
        )
    )

    assert result.state is None
    assert result.metadata["state_block_reason"] == "missing_pot"


def test_poker_legends_table_recognizer_uses_number_ocr_fallbacks(
    tmp_path: Path,
) -> None:
    image = tmp_path / "frame.png"
    image.write_bytes(b"not-read-by-fakes")
    annotation = actionable_truth()
    annotation["texts"] = []
    for seat in cast(list[dict[str, object]], annotation["seats"]):
        if seat.get("name") == "hero":
            seat.pop("stack")
    annotation["buttons"] = [
        {
            "name": "primary_left",
            "visible": True,
            "action_type": "call",
            "label": "Call",
        },
        {
            "name": "primary_right",
            "visible": True,
            "action_type": "fold",
            "label": "Fold",
        },
    ]
    recognizer = PokerLegendsTableRecognizer(
        card_recognizer=FakeCardRecognizer(
            (
                card_prediction("hero_hole_cards", "hero_hole_0", "AS", 0.95),
                card_prediction("hero_hole_cards", "hero_hole_1", "KH", 0.94),
                card_prediction("board", "board_0", "2C", 0.93),
                card_prediction("board", "board_1", "7D", 0.92),
                card_prediction("board", "board_2", "TS", 0.91),
            )
        ),
        button_recognizer=FakeButtonRecognizer(
            (
                button_prediction("primary_left", "call", 0.90),
                button_prediction("primary_right", "fold", 0.90),
            )
        ),
        number_recognizer=FakeNumberRecognizer(
            (
                number_prediction("texts", "pot", 150),
                number_prediction("texts", "hero_stack", 900),
                number_prediction("buttons", "primary_left", 25),
            ),
            expected_text_names=("pot", "hero_stack"),
            expected_button_names=("primary_left",),
        ),
        controlled_seat=0,
    )

    result = recognizer.recognize(
        CapturedFrame(
            payload=image,
            source="poker_legends_fixture",
            metadata={
                "poker_legends_annotation": annotation,
                "poker_legends_layout_annotation": {"image": str(image), "regions": {}},
            },
        )
    )

    assert result.state is not None
    assert result.state.pots[0].amount == 150
    assert result.state.to_call == 25
    assert result.state.player(0).stack == 900
    assert Action(ActionType.CALL, amount=25) in result.state.legal_actions


def test_poker_legends_table_recognizer_accepts_validated_hero_stack_overlay_ocr(
    tmp_path: Path,
) -> None:
    image = tmp_path / "frame.png"
    image.write_bytes(b"not-read-by-fakes")
    annotation = actionable_truth()
    for seat in cast(list[dict[str, object]], annotation["seats"]):
        if seat.get("name") == "hero":
            seat.pop("stack")
            seat["committed"] = 10
    recognizer = PokerLegendsTableRecognizer(
        card_recognizer=FakeCardRecognizer(
            (
                card_prediction("hero_hole_cards", "hero_hole_0", "AS", 0.95),
                card_prediction("hero_hole_cards", "hero_hole_1", "KH", 0.94),
                card_prediction("board", "board_0", "2C", 0.93),
                card_prediction("board", "board_1", "7D", 0.92),
                card_prediction("board", "board_2", "TS", 0.91),
            )
        ),
        button_recognizer=FakeButtonRecognizer((button_prediction("primary_left", "check", 0.90),)),
        number_recognizer=FakeNumberRecognizer(
            (
                PokerLegendsNumberPrediction(
                    name="hero_stack",
                    group="texts",
                    visible=True,
                    raw="$990+10",
                    numbers=(990, 10),
                    first_number=990,
                    sum_number=1000,
                    normalized_number=990,
                    confidence=0.90,
                    base_number=990,
                    overlay_number=10,
                    total_number=1000,
                ),
            ),
            expected_text_names=("hero_stack",),
            expected_button_names=(),
        ),
        controlled_seat=0,
    )

    result = recognizer.recognize(
        CapturedFrame(
            payload=image,
            source="poker_legends_fixture",
            metadata={
                "poker_legends_annotation": annotation,
                "poker_legends_layout_annotation": {"image": str(image), "regions": {}},
            },
        )
    )

    assert result.state is not None
    assert result.state.player(0).stack == 990
    assert result.state.player(0).committed == 10
    assert result.metadata["accepted_number_predictions"][0]["name"] == "hero_stack"
    assert result.metadata["number_prediction_rejections"] == []


def test_poker_legends_table_recognizer_validates_hero_stack_overlay_with_current_bet_ocr() -> None:
    hero_stack = PokerLegendsNumberPrediction(
        name="hero_stack",
        group="texts",
        visible=True,
        raw="$990+10",
        numbers=(990, 10),
        first_number=990,
        sum_number=1000,
        normalized_number=990,
        confidence=0.90,
        base_number=990,
        overlay_number=10,
        total_number=1000,
    )
    hero_current_bet = number_prediction("texts", "hero_current_bet", 10, 0.90)

    assert (
        poker_legends._number_prediction_rejection_reason(
            hero_stack,
            min_confidence=0.70,
            annotation=None,
            number_predictions=(hero_stack, hero_current_bet),
        )
        is None
    )
    assert (
        poker_legends._number_prediction_rejection_reason(
            hero_stack,
            min_confidence=0.70,
            annotation=None,
            number_predictions=(hero_stack,),
        )
        == "unverified_stack_overlay"
    )


def test_poker_legends_table_recognizer_synthesizes_missing_hero_seat(
    tmp_path: Path,
) -> None:
    image = tmp_path / "frame.png"
    image.write_bytes(b"not-read-by-fakes")
    annotation = actionable_truth()
    annotation["seats"] = [
        {
            "name": "villain",
            "visible": True,
            "stack": 1200,
            "committed": 0,
            "active": True,
            "current": False,
            "confidence": 1.0,
        }
    ]
    annotation["texts"] = [
        {"name": "pot_size", "visible": True, "normalized_number": 150, "confidence": 1.0},
        {"name": "hero_stack", "visible": True, "normalized_number": 900, "confidence": 1.0},
    ]
    recognizer = PokerLegendsTableRecognizer(
        card_recognizer=FakeCardRecognizer(
            (
                card_prediction("hero_hole_cards", "hero_hole_0", "AS", 0.95),
                card_prediction("hero_hole_cards", "hero_hole_1", "KH", 0.94),
                card_prediction("board", "board_0", "2C", 0.93),
                card_prediction("board", "board_1", "7D", 0.92),
                card_prediction("board", "board_2", "TS", 0.91),
            )
        ),
        button_recognizer=FakeButtonRecognizer((button_prediction("primary_left", "check", 0.90),)),
        controlled_seat=0,
    )

    result = recognizer.recognize(
        CapturedFrame(
            payload=image,
            source="poker_legends_fixture",
            metadata={
                "poker_legends_annotation": annotation,
                "poker_legends_layout_annotation": {"image": str(image), "regions": {}},
            },
        )
    )

    assert result.state is not None
    assert result.state.player(0).stack == 900
    assert result.state.player(1).stack == 1200
    assert result.state.pots[0].amount == 150


def test_poker_legends_table_recognizer_adds_single_truth_opponent_from_stack_text(
    tmp_path: Path,
) -> None:
    image = tmp_path / "frame.png"
    image.write_bytes(b"not-read-by-fakes")
    annotation = actionable_truth()
    annotation["seats"] = [
        {
            "name": "hero",
            "visible": True,
            "stack": 900,
            "committed": 0,
            "active": True,
            "current": True,
            "confidence": 1.0,
        }
    ]
    annotation["texts"] = [
        {"name": "pot", "visible": True, "normalized_number": 150, "confidence": 1.0},
        {
            "name": "right_top_stack",
            "visible": True,
            "normalized_number": 1200,
            "confidence": 1.0,
        },
    ]
    recognizer = PokerLegendsTableRecognizer(
        card_recognizer=FakeCardRecognizer(
            (
                card_prediction("hero_hole_cards", "hero_hole_0", "AS", 0.95),
                card_prediction("hero_hole_cards", "hero_hole_1", "KH", 0.94),
                card_prediction("board", "board_0", "2C", 0.93),
                card_prediction("board", "board_1", "7D", 0.92),
                card_prediction("board", "board_2", "TS", 0.91),
            )
        ),
        button_recognizer=FakeButtonRecognizer((button_prediction("primary_left", "check", 0.90),)),
        controlled_seat=0,
    )

    result = recognizer.recognize(
        CapturedFrame(
            payload=image,
            source="poker_legends_fixture",
            metadata={
                "poker_legends_annotation": annotation,
                "poker_legends_layout_annotation": {"image": str(image), "regions": {}},
            },
        )
    )

    assert result.state is not None
    assert len(result.state.players) == 2
    assert result.state.player(1).stack == 1200
    seats = result.metadata["recognized_table"]["seats"]
    assert seats[1]["seat"] == 1
    assert seats[1]["ui_slot"] == "right_top"


def test_poker_legends_table_recognizer_adds_single_ocr_opponent_from_stack_text(
    tmp_path: Path,
) -> None:
    image = tmp_path / "frame.png"
    image.write_bytes(b"not-read-by-fakes")
    annotation = actionable_truth()
    annotation["seats"] = [
        {
            "name": "hero",
            "visible": True,
            "stack": 900,
            "committed": 0,
            "active": True,
            "current": True,
            "confidence": 1.0,
        }
    ]
    recognizer = PokerLegendsTableRecognizer(
        card_recognizer=FakeCardRecognizer(
            (
                card_prediction("hero_hole_cards", "hero_hole_0", "AS", 0.95),
                card_prediction("hero_hole_cards", "hero_hole_1", "KH", 0.94),
                card_prediction("board", "board_0", "2C", 0.93),
                card_prediction("board", "board_1", "7D", 0.92),
                card_prediction("board", "board_2", "TS", 0.91),
            )
        ),
        button_recognizer=FakeButtonRecognizer((button_prediction("primary_left", "check", 0.90),)),
        number_recognizer=FakeNumberRecognizer(
            (number_prediction("texts", "right_top_stack", 1200),),
            expected_text_names=("right_top_stack",),
            expected_button_names=(),
        ),
        controlled_seat=0,
    )

    result = recognizer.recognize(
        CapturedFrame(
            payload=image,
            source="poker_legends_fixture",
            metadata={
                "poker_legends_annotation": annotation,
                "poker_legends_layout_annotation": {"image": str(image), "regions": {}},
            },
        )
    )

    assert result.state is not None
    assert len(result.state.players) == 2
    assert result.state.player(1).stack == 1200
    seats = result.metadata["recognized_table"]["seats"]
    assert seats[1]["seat"] == 1
    assert seats[1]["ui_slot"] == "right_top"


def test_poker_legends_table_recognizer_rejects_unverified_overlay_stack_ocr(
    tmp_path: Path,
) -> None:
    image = tmp_path / "frame.png"
    image.write_bytes(b"not-read-by-fakes")
    annotation = actionable_truth()
    annotation["seats"] = [
        {
            "name": "hero",
            "visible": True,
            "stack": 900,
            "committed": 0,
            "active": True,
            "current": True,
            "confidence": 1.0,
        }
    ]
    recognizer = PokerLegendsTableRecognizer(
        card_recognizer=FakeCardRecognizer(
            (
                card_prediction("hero_hole_cards", "hero_hole_0", "AS", 0.95),
                card_prediction("hero_hole_cards", "hero_hole_1", "KH", 0.94),
                card_prediction("board", "board_0", "2C", 0.93),
                card_prediction("board", "board_1", "7D", 0.92),
                card_prediction("board", "board_2", "TS", 0.91),
            )
        ),
        button_recognizer=FakeButtonRecognizer((button_prediction("primary_left", "check", 0.90),)),
        number_recognizer=FakeNumberRecognizer(
            (
                PokerLegendsNumberPrediction(
                    name="right_top_stack",
                    group="texts",
                    visible=True,
                    raw="$1200+10",
                    numbers=(1200, 10),
                    first_number=1200,
                    sum_number=1210,
                    normalized_number=1200,
                    confidence=0.90,
                    base_number=1200,
                    overlay_number=10,
                    total_number=1210,
                ),
            ),
            expected_text_names=("right_top_stack",),
            expected_button_names=(),
        ),
        controlled_seat=0,
    )

    result = recognizer.recognize(
        CapturedFrame(
            payload=image,
            source="poker_legends_fixture",
            metadata={
                "poker_legends_annotation": annotation,
                "poker_legends_layout_annotation": {"image": str(image), "regions": {}},
            },
        )
    )

    assert result.state is None
    assert result.metadata["state_block_reason"] == "not_enough_players"
    assert result.metadata["accepted_number_predictions"] == []
    assert result.metadata["number_prediction_rejections"][0]["name"] == "right_top_stack"
    assert (
        result.metadata["number_prediction_rejections"][0]["rejection_reason"]
        == "unverified_stack_overlay"
    )


def test_poker_legends_table_recognizer_uses_direct_truth_buttons_when_image_buttons_missing(
    tmp_path: Path,
) -> None:
    image = tmp_path / "frame.png"
    image.write_bytes(b"not-read-by-fakes")
    annotation = actionable_truth()
    annotation["buttons"] = [
        {"name": "check_fold", "visible": True, "action_type": "check", "label": "Check/Fold"},
        {"name": "call_any", "visible": True, "action_type": "call", "label": "Call Any"},
        {"name": "check", "visible": True, "action_type": "check", "label": "Check"},
        {"name": "raise", "visible": True, "action_type": "raise", "label": "Raise"},
        {"name": "fold", "visible": True, "action_type": "fold", "label": "Fold"},
    ]
    recognizer = PokerLegendsTableRecognizer(
        card_recognizer=FakeCardRecognizer(
            (
                card_prediction("hero_hole_cards", "hero_hole_0", "AS", 0.95),
                card_prediction("hero_hole_cards", "hero_hole_1", "KH", 0.94),
                card_prediction("board", "board_0", "2C", 0.93),
                card_prediction("board", "board_1", "7D", 0.92),
                card_prediction("board", "board_2", "TS", 0.91),
            )
        ),
        button_recognizer=FakeButtonRecognizer(()),
        controlled_seat=0,
    )

    result = recognizer.recognize(
        CapturedFrame(
            payload=image,
            source="poker_legends_fixture",
            metadata={
                "poker_legends_annotation": annotation,
                "poker_legends_layout_annotation": {"image": str(image), "regions": {}},
            },
        )
    )

    assert result.state is not None
    assert {action.action_type for action in result.state.legal_actions} == {
        ActionType.CHECK,
        ActionType.RAISE,
        ActionType.FOLD,
    }
    assert result.visual_observation is not None
    panel = result.visual_observation.action_panels[0]
    assert panel.visible is True
    assert "missing_current_action_row" not in panel.ambiguity_flags
    assert [button.slot for button in panel.buttons] == ["check", "raise", "fold"]


def test_poker_legends_table_recognizer_accepts_explicit_truth_button_suffix(
    tmp_path: Path,
) -> None:
    image = tmp_path / "frame.png"
    image.write_bytes(b"not-read-by-fakes")
    annotation = actionable_truth()
    annotation["buttons"] = [
        {
            "name": "check_button",
            "visible": True,
            "action_type": "check",
            "label": "CHECK",
        }
    ]
    recognizer = PokerLegendsTableRecognizer(
        card_recognizer=FakeCardRecognizer(
            (
                card_prediction("hero_hole_cards", "hero_hole_0", "AS", 0.95),
                card_prediction("hero_hole_cards", "hero_hole_1", "KH", 0.94),
                card_prediction("board", "board_0", "2C", 0.93),
                card_prediction("board", "board_1", "7D", 0.92),
                card_prediction("board", "board_2", "TS", 0.91),
            )
        ),
        button_recognizer=FakeButtonRecognizer(()),
        controlled_seat=0,
    )

    result = recognizer.recognize(
        CapturedFrame(
            payload=image,
            source="poker_legends_fixture",
            metadata={
                "poker_legends_annotation": annotation,
                "poker_legends_layout_annotation": {"image": str(image), "regions": {}},
            },
        )
    )

    assert result.state is not None
    assert result.state.legal_actions == (Action(ActionType.CHECK),)


def test_poker_legends_table_recognizer_uses_shifted_truth_call_amount(
    tmp_path: Path,
) -> None:
    image = tmp_path / "frame.png"
    image.write_bytes(b"not-read-by-fakes")
    annotation = actionable_truth()
    annotation["buttons"] = [
        {"name": "primary_left", "visible": False, "action_type": None, "label": None},
        {"name": "primary_middle", "visible": True, "action_type": "raise", "label": "Call $100"},
        {"name": "primary_right", "visible": True, "action_type": "fold", "label": "Raise"},
    ]
    recognizer = PokerLegendsTableRecognizer(
        card_recognizer=FakeCardRecognizer(
            (
                card_prediction("hero_hole_cards", "hero_hole_0", "AS", 0.95),
                card_prediction("hero_hole_cards", "hero_hole_1", "KH", 0.94),
                card_prediction("board", "board_0", "2C", 0.93),
                card_prediction("board", "board_1", "7D", 0.92),
                card_prediction("board", "board_2", "TS", 0.91),
            )
        ),
        button_recognizer=FakeButtonRecognizer((button_prediction("primary_left", "call", 0.90),)),
        controlled_seat=0,
    )

    result = recognizer.recognize(
        CapturedFrame(
            payload=image,
            source="poker_legends_fixture",
            metadata={
                "poker_legends_annotation": annotation,
                "poker_legends_layout_annotation": {"image": str(image), "regions": {}},
            },
        )
    )

    assert result.state is not None
    assert result.state.to_call == 100
    assert Action(ActionType.CALL, amount=100) in result.state.legal_actions


def test_poker_legends_table_recognizer_blocks_preselect_call_any(
    tmp_path: Path,
) -> None:
    image = tmp_path / "frame.png"
    image.write_bytes(b"not-read-by-fakes")
    annotation = actionable_truth()
    annotation["buttons"] = [
        {
            "name": "primary_left",
            "visible": True,
            "action_type": "call",
            "label": "Call Any",
        },
        {
            "name": "primary_middle",
            "visible": True,
            "action_type": "raise",
            "label": "Raise",
        },
        {
            "name": "primary_right",
            "visible": True,
            "action_type": "fold",
            "label": "Fold",
        },
    ]
    recognizer = PokerLegendsTableRecognizer(
        card_recognizer=FakeCardRecognizer(
            (
                card_prediction("hero_hole_cards", "hero_hole_0", "AS", 0.95),
                card_prediction("hero_hole_cards", "hero_hole_1", "KH", 0.94),
                card_prediction("board", "board_0", "2C", 0.93),
                card_prediction("board", "board_1", "7D", 0.92),
                card_prediction("board", "board_2", "TS", 0.91),
            )
        ),
        button_recognizer=FakeButtonRecognizer(
            (
                button_prediction("primary_left", "call", 0.90),
                button_prediction("primary_middle", "raise", 0.90),
                button_prediction("primary_right", "fold", 0.90),
            )
        ),
        controlled_seat=0,
    )

    result = recognizer.recognize(
        CapturedFrame(
            payload=image,
            source="poker_legends_fixture",
            metadata={
                "poker_legends_annotation": annotation,
                "poker_legends_layout_annotation": {"image": str(image), "regions": {}},
            },
        )
    )

    assert result.state is None
    assert result.metadata["state_block_reason"] == "preselect_ambiguous"
    assert result.assembly_result is not None
    assert result.assembly_result.issues[0].reason_code == "PRESELECT_AMBIGUOUS"
    assert result.visual_observation is not None
    panels = {panel.panel_kind: panel for panel in result.visual_observation.action_panels}
    assert "preselect_shortcut_label" in panels["current_action_row"].ambiguity_flags
    assert panels["preselect_strip"].visible is True
    assert panels["preselect_strip"].buttons[0].slot == "primary_left"


def test_poker_legends_table_recognizer_marks_missing_current_action_row(
    tmp_path: Path,
) -> None:
    image = tmp_path / "frame.png"
    image.write_bytes(b"not-read-by-fakes")
    annotation = actionable_truth()
    annotation["buttons"] = []
    recognizer = PokerLegendsTableRecognizer(
        card_recognizer=FakeCardRecognizer(
            (
                card_prediction("hero_hole_cards", "hero_hole_0", "AS", 0.95),
                card_prediction("hero_hole_cards", "hero_hole_1", "KH", 0.94),
                card_prediction("board", "board_0", "2C", 0.93),
                card_prediction("board", "board_1", "7D", 0.92),
                card_prediction("board", "board_2", "TS", 0.91),
            )
        ),
        button_recognizer=FakeButtonRecognizer(()),
        controlled_seat=0,
    )

    result = recognizer.recognize(
        CapturedFrame(
            payload=image,
            source="poker_legends_fixture",
            metadata={
                "poker_legends_annotation": annotation,
                "poker_legends_layout_annotation": {"image": str(image), "regions": {}},
            },
        )
    )

    assert result.state is None
    assert result.metadata["state_block_reason"] == "missing_current_action_row"
    assert result.visual_observation is not None
    panel = result.visual_observation.action_panels[0]
    assert panel.panel_kind == "current_action_row"
    assert panel.visible is False
    assert "missing_current_action_row" in panel.ambiguity_flags


def test_poker_legends_table_recognizer_requires_passive_action_when_checking_is_free(
    tmp_path: Path,
) -> None:
    image = tmp_path / "frame.png"
    image.write_bytes(b"not-read-by-fakes")
    annotation = actionable_truth()
    annotation["buttons"] = [
        {"name": "primary_middle", "visible": True, "action_type": "raise", "label": "Raise"},
        {"name": "primary_right", "visible": True, "action_type": "fold", "label": "Fold"},
    ]
    recognizer = PokerLegendsTableRecognizer(
        card_recognizer=FakeCardRecognizer(
            (
                card_prediction("hero_hole_cards", "hero_hole_0", "AS", 0.95),
                card_prediction("hero_hole_cards", "hero_hole_1", "KH", 0.94),
                card_prediction("board", "board_0", "2C", 0.93),
                card_prediction("board", "board_1", "7D", 0.92),
                card_prediction("board", "board_2", "TS", 0.91),
            )
        ),
        button_recognizer=FakeButtonRecognizer(
            (
                button_prediction("primary_middle", "raise", 0.90),
                button_prediction("primary_right", "fold", 0.90),
            )
        ),
        controlled_seat=0,
    )

    result = recognizer.recognize(
        CapturedFrame(
            payload=image,
            source="poker_legends_fixture",
            metadata={
                "poker_legends_annotation": annotation,
                "poker_legends_layout_annotation": {"image": str(image), "regions": {}},
            },
        )
    )

    assert result.state is None
    assert result.metadata["state_block_reason"] == "missing_passive_action"
    assert result.visual_observation is not None
    panel = result.visual_observation.action_panels[0]
    assert "missing_passive_action" in panel.ambiguity_flags


def test_poker_legends_table_recognizer_filters_truth_hidden_primary_buttons(
    tmp_path: Path,
) -> None:
    image = tmp_path / "frame.png"
    image.write_bytes(b"not-read-by-fakes")
    annotation = actionable_truth()
    annotation["buttons"] = [
        {
            "name": "primary_middle",
            "visible": False,
            "action_type": None,
            "label": None,
        },
        {
            "name": "primary_right",
            "visible": True,
            "action_type": "fold",
            "label": "Fold",
        },
    ]
    recognizer = PokerLegendsTableRecognizer(
        card_recognizer=FakeCardRecognizer(
            (
                card_prediction("hero_hole_cards", "hero_hole_0", "AS", 0.95),
                card_prediction("hero_hole_cards", "hero_hole_1", "KH", 0.94),
                card_prediction("board", "board_0", "2C", 0.93),
                card_prediction("board", "board_1", "7D", 0.92),
                card_prediction("board", "board_2", "TS", 0.91),
            )
        ),
        button_recognizer=FakeButtonRecognizer(
            (
                button_prediction("primary_middle", "raise", 0.90),
                button_prediction("primary_right", "fold", 0.90),
            )
        ),
        controlled_seat=0,
    )

    result = recognizer.recognize(
        CapturedFrame(
            payload=image,
            source="poker_legends_fixture",
            metadata={
                "poker_legends_annotation": annotation,
                "poker_legends_layout_annotation": {"image": str(image), "regions": {}},
            },
        )
    )

    assert result.state is None
    assert result.metadata["state_block_reason"] == "missing_passive_action"
    table = result.metadata["recognized_table"]
    assert isinstance(table, dict)
    assert [button["action_type"] for button in table["buttons"]] == ["fold"]


def test_poker_legends_table_recognizer_derives_button_action_from_reviewed_label(
    tmp_path: Path,
) -> None:
    image = tmp_path / "frame.png"
    image.write_bytes(b"not-read-by-fakes")
    annotation = actionable_truth()
    annotation["buttons"] = [
        {
            "name": "primary_middle",
            "visible": True,
            "action_type": "call",
            "label": "Call $100",
        }
    ]
    recognizer = PokerLegendsTableRecognizer(
        card_recognizer=FakeCardRecognizer(
            (
                card_prediction("hero_hole_cards", "hero_hole_0", "AS", 0.95),
                card_prediction("hero_hole_cards", "hero_hole_1", "KH", 0.94),
                card_prediction("board", "board_0", "2C", 0.93),
                card_prediction("board", "board_1", "7D", 0.92),
                card_prediction("board", "board_2", "TS", 0.91),
            )
        ),
        button_recognizer=FakeButtonRecognizer(
            (button_prediction("primary_middle", "raise", 0.90),)
        ),
        controlled_seat=0,
    )

    result = recognizer.recognize(
        CapturedFrame(
            payload=image,
            source="poker_legends_fixture",
            metadata={
                "poker_legends_annotation": annotation,
                "poker_legends_layout_annotation": {"image": str(image), "regions": {}},
            },
        )
    )

    assert result.state is not None
    assert result.state.to_call == 100
    assert {action.action_type for action in result.state.legal_actions} == {ActionType.CALL}
    table = result.metadata["recognized_table"]
    assert isinstance(table, dict)
    assert table["buttons"][0]["action_type"] == "call"
    assert result.visual_observation is not None
    panel = result.visual_observation.action_panels[0]
    assert "button_label_action_mismatch" not in panel.ambiguity_flags


def test_poker_legends_table_recognizer_derives_call_amount_from_committed_gap(
    tmp_path: Path,
) -> None:
    image = tmp_path / "frame.png"
    image.write_bytes(b"not-read-by-fakes")
    annotation = actionable_truth()
    annotation["buttons"] = [
        {
            "name": "primary_middle",
            "visible": True,
            "action_type": "call",
            "label": "Call",
        }
    ]
    seats = cast(list[dict[str, object]], annotation["seats"])
    seats[0]["committed"] = 0
    seats[1]["committed"] = 100
    recognizer = PokerLegendsTableRecognizer(
        card_recognizer=FakeCardRecognizer(
            (
                card_prediction("hero_hole_cards", "hero_hole_0", "AS", 0.95),
                card_prediction("hero_hole_cards", "hero_hole_1", "KH", 0.94),
                card_prediction("board", "board_0", "2C", 0.93),
                card_prediction("board", "board_1", "7D", 0.92),
                card_prediction("board", "board_2", "TS", 0.91),
            )
        ),
        button_recognizer=FakeButtonRecognizer(
            (button_prediction("primary_middle", "raise", 0.90),)
        ),
        controlled_seat=0,
    )

    result = recognizer.recognize(
        CapturedFrame(
            payload=image,
            source="poker_legends_fixture",
            metadata={
                "poker_legends_annotation": annotation,
                "poker_legends_layout_annotation": {"image": str(image), "regions": {}},
            },
        )
    )

    assert result.state is not None
    assert result.state.to_call == 100
    assert {
        field.field_path: field.source for field in result.accepted_critical_fields
    }["numbers.call_amount"] == "rule_inferred_committed"


def test_poker_legends_table_recognizer_infers_street_when_truth_street_lags_board(
    tmp_path: Path,
) -> None:
    image = tmp_path / "frame.png"
    image.write_bytes(b"not-read-by-fakes")
    annotation = actionable_truth()
    annotation["table_state"] = {"street": "flop"}
    recognizer = PokerLegendsTableRecognizer(
        card_recognizer=FakeCardRecognizer(
            (
                card_prediction("hero_hole_cards", "hero_hole_0", "AS", 0.95),
                card_prediction("hero_hole_cards", "hero_hole_1", "KH", 0.94),
                card_prediction("board", "board_0", "2C", 0.93),
                card_prediction("board", "board_1", "7D", 0.92),
                card_prediction("board", "board_2", "TS", 0.91),
                card_prediction("board", "board_3", "4H", 0.91),
            )
        ),
        button_recognizer=FakeButtonRecognizer((button_prediction("primary_left", "check", 0.90),)),
        controlled_seat=0,
    )

    result = recognizer.recognize(
        CapturedFrame(
            payload=image,
            source="poker_legends_fixture",
            metadata={
                "poker_legends_annotation": annotation,
                "poker_legends_layout_annotation": {"image": str(image), "regions": {}},
            },
        )
    )

    assert result.state is not None
    assert result.state.street is Street.TURN
    assert len(result.state.board) == 4


class FakeCardRecognizer:
    def __init__(self, predictions: tuple[PokerLegendsCardConsensusPrediction, ...]) -> None:
        self.predictions = predictions
        self.calls = 0

    def recognize(
        self,
        _image_path: str | Path,
        _annotation: object,
        *,
        frame_id: str,
        exclude_frame_id: str | None = None,
        exclude_card: str | None = None,
    ) -> tuple[PokerLegendsCardConsensusPrediction, ...]:
        self.calls += 1
        assert frame_id == "frame_001"
        assert exclude_frame_id is None
        assert exclude_card is None
        return self.predictions


class FakeButtonRecognizer:
    def __init__(self, predictions: tuple[PokerLegendsButtonPrediction, ...]) -> None:
        self.predictions = predictions
        self.calls = 0

    def recognize(
        self,
        _image_path: str | Path,
        _annotation: object,
        *,
        frame_id: str,
        exclude_frame_id: str | None = None,
    ) -> tuple[PokerLegendsButtonPrediction, ...]:
        self.calls += 1
        assert frame_id == "frame_001"
        assert exclude_frame_id is None
        return self.predictions


class FakeNumberRecognizer:
    def __init__(
        self,
        predictions: tuple[PokerLegendsNumberPrediction, ...],
        *,
        expected_text_names: tuple[str, ...] | None = None,
        expected_button_names: tuple[str, ...] | None = None,
    ) -> None:
        self.predictions = predictions
        self.expected_text_names = expected_text_names
        self.expected_button_names = expected_button_names

    def recognize(
        self,
        _image_path: str | Path,
        _annotation: object,
        *,
        text_names: tuple[str, ...] = ("pot", "hero_stack", "right_top_stack"),
        button_names: tuple[str, ...] = ("primary_left",),
    ) -> tuple[PokerLegendsNumberPrediction, ...]:
        if self.expected_text_names is not None:
            assert text_names == self.expected_text_names
        if self.expected_button_names is not None:
            assert button_names == self.expected_button_names
        return self.predictions


def actionable_truth() -> dict[str, object]:
    return {
        "frame_id": "frame_001",
        "screen": {
            "kind": "actionable_table",
            "confidence": 0.99,
            "hero_turn": True,
            "reason": "hero can act",
        },
        "table_state": {"street": "flop"},
        "buttons": [
            {
                "name": "primary_left",
                "visible": True,
                "action_type": "check",
                "label": "Check",
            },
            {
                "name": "primary_middle",
                "visible": True,
                "action_type": "raise",
                "label": "Raise",
            },
            {
                "name": "primary_right",
                "visible": True,
                "action_type": "fold",
                "label": "Fold",
            },
        ],
        "texts": [
            {
                "name": "pot",
                "visible": True,
                "normalized_number": 150,
                "confidence": 1.0,
            }
        ],
        "seats": [
            {
                "name": "hero",
                "visible": True,
                "stack": 900,
                "committed": 0,
                "active": True,
                "current": True,
                "confidence": 1.0,
            },
            {
                "name": "villain",
                "visible": True,
                "stack": 1200,
                "committed": 0,
                "active": True,
                "current": False,
                "confidence": 1.0,
            },
        ],
    }


def card_prediction(
    group: str,
    slot: str,
    card: str,
    confidence: float,
) -> PokerLegendsCardConsensusPrediction:
    return PokerLegendsCardConsensusPrediction(
        frame_id="frame_001",
        group=group,
        slot=slot,
        visible=True,
        card=card,
        confidence=confidence,
        method="test",
        full_card=card,
        part_card=card,
        classifier_card=card,
        full_confidence=confidence,
        part_confidence=confidence,
        classifier_confidence=confidence,
    )


def button_prediction(
    slot: str,
    action_type: str,
    confidence: float,
) -> PokerLegendsButtonPrediction:
    return PokerLegendsButtonPrediction(
        frame_id="frame_001",
        slot=slot,
        visible=True,
        action_type=action_type,
        confidence=confidence,
    )


def number_prediction(
    group: str,
    name: str,
    number: int,
    confidence: float = 0.88,
) -> PokerLegendsNumberPrediction:
    return PokerLegendsNumberPrediction(
        name=name,
        group=group,
        visible=True,
        raw=str(number),
        numbers=(number,),
        first_number=number,
        sum_number=None,
        normalized_number=number,
        confidence=confidence,
    )
