"""Tests for the CV action-button detector and the vision click planner (pure, offline)."""

from __future__ import annotations

from typing import Any, cast

import cv2
import numpy as np
from holdem_bot.adapters.poker_legends_host import (
    PokerLegendsVisionClickPlanner,
    _plan_click_targets,
    _resolve_click_point,
)
from holdem_bot.recognize import RecognitionResult
from holdem_bot.vision.poker_legends_action_buttons import (
    ActionButtonDetection,
    detect_action_buttons,
)
from holdem_common import Action, ActionType

# BGR fills whose HSV hue lands in each action band: blue->call, gold->raise, red->fold.
_CALL_BGR = (255, 120, 0)
_RAISE_BGR = (0, 215, 255)
_FOLD_BGR = (0, 0, 255)


def _table_with_buttons() -> np.ndarray:
    """A dark frame with a clean Call/Raise/Fold circle row at the bottom-right + distractors."""
    image = np.full((400, 600, 3), 30, dtype=np.uint8)  # dark felt
    row_y = 340
    cv2.circle(image, (380, row_y), 30, _CALL_BGR, -1)
    cv2.circle(image, (460, row_y), 30, _RAISE_BGR, -1)
    cv2.circle(image, (540, row_y), 30, _FOLD_BGR, -1)
    # distractor 1: a gold raise-shortcut chip ABOVE the row (must be dropped by row anchoring)
    cv2.circle(image, (460, 250), 16, _RAISE_BGR, -1)
    # distractor 2: an elongated bright panel (must be dropped by the aspect filter)
    cv2.rectangle(image, (250, 355), (340, 375), (255, 150, 0), -1)
    return image


def test_detects_three_buttons_left_to_right() -> None:
    detections = detect_action_buttons(_table_with_buttons())
    assert [d.color_class for d in detections] == ["call", "raise", "fold"]
    assert [d.slot for d in detections] == ["primary_left", "primary_middle", "primary_right"]
    # centres land on the drawn circles
    centers = {d.color_class: (d.x, d.y) for d in detections}
    assert abs(centers["call"][0] - 380) <= 4 and abs(centers["call"][1] - 340) <= 4
    assert abs(centers["raise"][0] - 460) <= 4 and abs(centers["raise"][1] - 340) <= 4
    assert abs(centers["fold"][0] - 540) <= 4 and abs(centers["fold"][1] - 340) <= 4


def test_drops_shortcut_and_panel_distractors() -> None:
    detections = detect_action_buttons(_table_with_buttons())
    assert len(detections) == 3  # the above-row gold chip and the wide panel are excluded
    raises = [d for d in detections if d.color_class == "raise"]
    assert len(raises) == 1 and raises[0].y == 340  # the in-row raise, not the shortcut at y=250


def test_empty_frame_detects_nothing() -> None:
    assert detect_action_buttons(np.full((400, 600, 3), 30, dtype=np.uint8)) == ()


def test_planner_maps_action_to_button_center_in_screen_space() -> None:
    detections = detect_action_buttons(_table_with_buttons())
    planner = PokerLegendsVisionClickPlanner(detections)
    fold_center = next(d for d in detections if d.color_class == "fold")
    plan = planner.plan(
        Action(ActionType.FOLD), origin=(1736, 232), coordinate_space="screen"
    )
    assert plan is not None
    assert plan.command == "primary_right"
    assert plan.coordinate_space == "screen"
    assert plan.x == 1736 + fold_center.x
    assert plan.y == 232 + fold_center.y


def test_planner_maps_raise_and_call_to_their_slots() -> None:
    planner = PokerLegendsVisionClickPlanner(detect_action_buttons(_table_with_buttons()))

    def command_for(action: Action) -> str | None:
        plan = planner.plan(action)
        return plan.command if plan is not None else None

    assert command_for(Action(ActionType.RAISE, amount=50)) == "primary_middle"
    assert command_for(Action(ActionType.CALL, amount=10)) == "primary_left"
    assert command_for(Action(ActionType.CHECK)) == "primary_left"


def test_planner_fails_closed_when_button_absent() -> None:
    # only call + raise on screen: a fold decision has no button to click -> None (never guesses)
    only_two = (
        ActionButtonDetection("primary_left", "call", 380, 340, 30, 0.9),
        ActionButtonDetection("primary_middle", "raise", 460, 340, 30, 0.6),
    )
    planner = PokerLegendsVisionClickPlanner(only_two)
    assert planner.plan(Action(ActionType.FOLD)) is None
    assert planner.plan(Action(ActionType.CALL, amount=10)) is not None


def test_from_image_classmethod() -> None:
    planner = PokerLegendsVisionClickPlanner.from_image(_table_with_buttons())
    assert planner.plan(Action(ActionType.FOLD)) is not None


def _fold_det() -> ActionButtonDetection:
    return ActionButtonDetection("primary_right", "fold", 540, 340, 30, 0.9)


def test_resolve_agree_uses_cv_center() -> None:
    point, source = _resolve_click_point((545, 344), _fold_det(), frame_height=400)
    assert source == "llm+cv" and point == (540, 340)  # close -> pixel-exact CV centre


def test_resolve_conflict_when_far_apart() -> None:
    point, source = _resolve_click_point((100, 100), _fold_det(), frame_height=400)
    assert source == "conflict" and point == (540, 340)


def test_resolve_single_source() -> None:
    assert _resolve_click_point(None, _fold_det(), frame_height=400) == ((540, 340), "cv")
    assert _resolve_click_point((200, 210), None, frame_height=400) == ((200, 210), "llm")


def test_resolve_none_when_neither() -> None:
    assert _resolve_click_point(None, None, frame_height=400) == (None, "none")


class _StubDecision:
    """Minimal PolicyDecision stand-in: _plan_click_targets only reads .action."""

    def __init__(self, action: Action) -> None:
        self.action = action


def test_plan_click_targets_hybrid_screen_coords() -> None:
    frame = _table_with_buttons()
    # LLM box centre near the CV fold button (540,340) -> both agree
    recognition = RecognitionResult(
        state=None, metadata={"llm_button_boxes": {"primary_right": [541, 342]}}
    )
    decision = cast(Any, _StubDecision(Action(ActionType.FOLD)))
    _detections, click = _plan_click_targets(frame, recognition, decision, origin=(1000, 500))
    assert click is not None
    assert click.source == "llm+cv"
    assert click.plan.command == "primary_right"
    assert click.plan.coordinate_space == "screen"
    assert abs(click.plan.x - (1000 + 540)) <= 5  # screen = origin + CV frame centre
    assert abs(click.plan.y - (500 + 340)) <= 5


def test_plan_click_targets_none_decision() -> None:
    detections, click = _plan_click_targets(
        _table_with_buttons(), RecognitionResult(state=None), None, origin=None
    )
    assert detections == () and click is None
