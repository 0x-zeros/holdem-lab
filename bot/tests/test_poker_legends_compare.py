from holdem_bot.vision import (
    PokerLegendsRoiButton,
    PokerLegendsRoiCard,
    PokerLegendsRoiResult,
    PokerLegendsRoiText,
    compare_llm_candidate_to_roi,
)


def test_compare_llm_candidate_to_roi_counts_matches() -> None:
    candidate = {
        "frame_id": "keyframe_000001",
        "board": [
            {"slot": "board_0", "visible": True, "card": "AS"},
            {"slot": "board_1", "visible": True, "card": "TD"},
        ],
        "hero_hole_cards": [{"slot": "hero_hole_0", "visible": True, "card": "7H"}],
        "buttons": [
            {"name": "primary_left", "visible": True, "action_type": "check"},
        ],
        "texts": [{"name": "pot", "visible": True, "normalized_number": 160}],
    }
    roi = PokerLegendsRoiResult(
        frame_id="keyframe_000001",
        image="frames/keyframe_000001.png",
        board=(
            PokerLegendsRoiCard("board_0", True, "AS", 0.65),
            PokerLegendsRoiCard("board_1", True, "TD", 0.65),
        ),
        hero_hole_cards=(PokerLegendsRoiCard("hero_hole_0", True, "7H", 0.65),),
        buttons=(PokerLegendsRoiButton("primary_left", True, "Check", "check", 0.6),),
        texts=(PokerLegendsRoiText("pot", True, "$160", (160,), 160, None, 0.7),),
    )

    comparisons = compare_llm_candidate_to_roi(candidate, roi)

    assert {comparison.status for comparison in comparisons} == {"match"}
    assert len(comparisons) == 5


def test_compare_llm_candidate_to_roi_reports_mismatch_missing_and_false_positive() -> None:
    candidate = {
        "frame_id": "keyframe_000002",
        "board": [
            {"slot": "board_0", "visible": True, "card": "AS"},
            {"slot": "board_1", "visible": True, "card": "TD"},
            {"slot": "board_2", "visible": False, "card": None},
        ],
        "hero_hole_cards": [],
        "buttons": [
            {"name": "primary_left", "visible": True, "action_type": "call"},
        ],
        "texts": [{"name": "pot", "visible": True, "normalized_number": 160}],
    }
    roi = PokerLegendsRoiResult(
        frame_id="keyframe_000002",
        image="frames/keyframe_000002.png",
        board=(
            PokerLegendsRoiCard("board_0", True, "KS", 0.65),
            PokerLegendsRoiCard("board_1", False, None, 0.90),
            PokerLegendsRoiCard("board_2", True, "3C", 0.65),
        ),
        hero_hole_cards=(),
        buttons=(PokerLegendsRoiButton("primary_left", True, "Raise", "raise", 0.6),),
        texts=(PokerLegendsRoiText("pot", True, "$80", (80,), 80, None, 0.7),),
    )

    comparisons = compare_llm_candidate_to_roi(candidate, roi)
    statuses = {
        (comparison.group, comparison.field): comparison.status for comparison in comparisons
    }

    assert statuses[("board", "board_0")] == "mismatch"
    assert statuses[("board", "board_1")] == "missing"
    assert statuses[("board", "board_2")] == "false_positive"
    assert statuses[("buttons", "primary_left")] == "mismatch"
    assert statuses[("text_numbers", "pot")] == "mismatch"


def test_compare_llm_candidate_to_roi_accepts_text_sum_number() -> None:
    candidate = {
        "frame_id": "keyframe_000003",
        "board": [],
        "hero_hole_cards": [],
        "buttons": [],
        "texts": [{"name": "hero_stack", "visible": True, "normalized_number": 1226}],
    }
    roi = PokerLegendsRoiResult(
        frame_id="keyframe_000003",
        image="frames/keyframe_000003.png",
        board=(),
        hero_hole_cards=(),
        buttons=(),
        texts=(PokerLegendsRoiText("hero_stack", True, "$1146+80", (1146, 80), 1146, 1226, 0.7),),
    )

    comparisons = compare_llm_candidate_to_roi(candidate, roi)

    assert len(comparisons) == 1
    assert comparisons[0].status == "match"
    assert comparisons[0].observed == 1226
