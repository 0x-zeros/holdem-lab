"""Host-side Poker Legends capture and dry-run automation helpers."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol, cast

from holdem_common import Action, ActionType, GameState

from holdem_bot.adapters.poker_legends import PokerLegendsTableRecognizer
from holdem_bot.capture import Capture, CapturedFrame
from holdem_bot.orchestrator import BotOrchestrator, BotStepResult
from holdem_bot.vision.annotations import ScreenRect

_Runner = Callable[[Sequence[str]], None]


class _Clock(Protocol):
    def __call__(self) -> float: ...


@dataclass(frozen=True, slots=True)
class PokerLegendsClickPlan:
    action_type: str
    amount: int
    command: str
    x: int
    y: int
    coordinate_space: str = "image"

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class MacOSScreenCapture:
    """Capture the macOS screen or one window with Apple's built-in screencapture."""

    def __init__(
        self,
        *,
        output_dir: str | Path,
        source: str = "macos_screencapture",
        window_id: int | None = None,
        runner: _Runner | None = None,
        now: _Clock = time.time,
        system_name: Callable[[], str] = platform.system,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.source = source
        self.window_id = window_id
        self.runner = runner or _run_command
        self.now = now
        self.system_name = system_name

    def capture(self) -> CapturedFrame:
        if self.system_name() != "Darwin":
            raise RuntimeError("MacOSScreenCapture requires macOS host execution")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        image_path = self.output_dir / f"frame_{int(self.now() * 1000)}.png"
        command = ["screencapture", "-x"]
        if self.window_id is not None:
            command.extend(["-l", str(self.window_id)])
        command.append(str(image_path))
        self.runner(command)
        return CapturedFrame(
            payload=image_path,
            source=self.source,
            metadata={
                "poker_legends_image_path": str(image_path),
                "capture_command": command,
                "coordinate_space": "image",
            },
        )


class PokerLegendsImageCapture:
    """Capture adapter for one saved Poker Legends image."""

    def __init__(
        self,
        *,
        image_path: str | Path,
        layout_annotation_path: str | Path,
        annotation_path: str | Path | None = None,
        source: str = "poker_legends_image",
    ) -> None:
        self.image_path = Path(image_path)
        self.layout_annotation_path = Path(layout_annotation_path)
        self.annotation_path = Path(annotation_path) if annotation_path is not None else None
        self.source = source

    def capture(self) -> CapturedFrame:
        metadata: dict[str, object] = {
            "poker_legends_image_path": str(self.image_path),
            "poker_legends_layout_annotation_path": str(self.layout_annotation_path),
            "coordinate_space": "image",
        }
        if self.annotation_path is not None:
            metadata["poker_legends_annotation_path"] = str(self.annotation_path)
        return CapturedFrame(
            payload=self.image_path,
            source=self.source,
            metadata=metadata,
        )


class PokerLegendsCaptureMetadata:
    """Attach Poker Legends recognizer metadata to another capture source."""

    def __init__(
        self,
        *,
        capture: Capture,
        layout_annotation_path: str | Path,
        annotation_path: str | Path | None = None,
    ) -> None:
        self.capture_source = capture
        self.layout_annotation_path = Path(layout_annotation_path)
        self.annotation_path = Path(annotation_path) if annotation_path is not None else None

    def capture(self) -> CapturedFrame:
        frame = self.capture_source.capture()
        metadata = dict(frame.metadata)
        metadata["poker_legends_layout_annotation_path"] = str(self.layout_annotation_path)
        metadata["coordinate_space"] = "image"
        if self.annotation_path is not None:
            metadata["poker_legends_annotation_path"] = str(self.annotation_path)
        return CapturedFrame(
            payload=frame.payload,
            source=frame.source,
            metadata=metadata,
        )


class PokerLegendsLayoutClickPlanner:
    """Map canonical poker actions to primary Poker Legends button centers."""

    def __init__(self, layout_annotation: Mapping[str, object]) -> None:
        self.layout_annotation = layout_annotation

    @classmethod
    def from_annotation_path(cls, path: str | Path) -> PokerLegendsLayoutClickPlanner:
        return cls(_read_json_object(path))

    def plan(self, action: Action) -> PokerLegendsClickPlan:
        command = _command_for_action(action)
        rect = _button_rect(self.layout_annotation, command)
        return PokerLegendsClickPlan(
            action_type=action.action_type.value,
            amount=action.amount,
            command=command,
            x=rect.x + rect.width // 2,
            y=rect.y + rect.height // 2,
        )


class PokerLegendsDryRunAutomator:
    """Automator that records intended Poker Legends clicks without clicking."""

    def __init__(
        self,
        *,
        planner: PokerLegendsLayoutClickPlanner,
        log_path: str | Path,
        now: _Clock = time.time,
    ) -> None:
        self.planner = planner
        self.log_path = Path(log_path)
        self.now = now
        self.records: list[dict[str, object]] = []

    def perform(self, action: Action, state: GameState) -> None:
        plan = self.planner.plan(action)
        record: dict[str, object] = {
            "timestamp": self.now(),
            "mode": "dry_run",
            "hand_id": state.hand_id,
            "street": state.street.value,
            "current_seat": state.current_seat,
            "action": {
                "type": action.action_type.value,
                "amount": action.amount,
                "min_amount": action.min_amount,
                "max_amount": action.max_amount,
            },
            "click_plan": plan.to_dict(),
            "executed": False,
        }
        self.records.append(record)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a", encoding="utf-8") as output:
            output.write(json.dumps(record, sort_keys=True) + "\n")


def capture_main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Capture a macOS screen/window image for Poker Legends dry-run analysis."
    )
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--window-id", type=int)
    args = parser.parse_args(argv)
    frame = MacOSScreenCapture(
        output_dir=args.out_dir,
        window_id=args.window_id,
    ).capture()
    print(
        json.dumps(
            {
                "image": str(frame.payload),
                "source": frame.source,
                "metadata": dict(frame.metadata),
            },
            indent=2,
            sort_keys=True,
        )
    )


def plan_click_main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Create a dry-run Poker Legends click plan from a layout annotation."
    )
    parser.add_argument("--layout-annotation", required=True)
    parser.add_argument(
        "--action",
        required=True,
        choices=tuple(action.value for action in ActionType),
    )
    parser.add_argument("--amount", type=int, default=0)
    parser.add_argument("--out-jsonl")
    args = parser.parse_args(argv)

    action = Action(ActionType(args.action), amount=args.amount)
    planner = PokerLegendsLayoutClickPlanner.from_annotation_path(args.layout_annotation)
    plan = planner.plan(action)
    record = {
        "mode": "dry_run_click_plan",
        "action": {
            "type": action.action_type.value,
            "amount": action.amount,
            "min_amount": action.min_amount,
            "max_amount": action.max_amount,
        },
        "click_plan": plan.to_dict(),
        "executed": False,
    }
    if args.out_jsonl is not None:
        output = Path(args.out_jsonl)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    print(json.dumps(record, indent=2, sort_keys=True))


def dry_run_once_main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run one safe Poker Legends recognize -> AI -> dry-run step."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--image")
    source.add_argument("--capture-out-dir")
    parser.add_argument("--window-id", type=int)
    parser.add_argument("--annotation")
    parser.add_argument("--layout-annotation", required=True)
    parser.add_argument("--card-part-manifest", required=True)
    parser.add_argument("--card-classifier-manifest", required=True)
    parser.add_argument("--button-manifest", required=True)
    parser.add_argument("--card-template-manifest")
    parser.add_argument("--seat", type=int, default=0)
    parser.add_argument("--min-confidence", type=float, default=0.80)
    parser.add_argument("--log-jsonl", required=True)
    args = parser.parse_args(argv)

    capture: Capture
    if args.image is not None:
        capture = PokerLegendsImageCapture(
            image_path=args.image,
            layout_annotation_path=args.layout_annotation,
            annotation_path=args.annotation,
        )
    else:
        capture = PokerLegendsCaptureMetadata(
            capture=MacOSScreenCapture(
                output_dir=args.capture_out_dir,
                window_id=args.window_id,
            ),
            layout_annotation_path=args.layout_annotation,
            annotation_path=args.annotation,
        )

    recognizer = PokerLegendsTableRecognizer.from_manifests(
        card_part_manifest=args.card_part_manifest,
        card_classifier_manifest=args.card_classifier_manifest,
        button_manifest=args.button_manifest,
        card_template_manifest=args.card_template_manifest,
        controlled_seat=args.seat,
    )
    planner = PokerLegendsLayoutClickPlanner.from_annotation_path(args.layout_annotation)
    automator = PokerLegendsDryRunAutomator(
        planner=planner,
        log_path=args.log_jsonl,
    )
    orchestrator = BotOrchestrator(
        capture=capture,
        recognizer=recognizer,
        automator=automator,
        seat=args.seat,
        min_confidence=args.min_confidence,
    )
    result = orchestrator.run_once()
    record = _bot_step_result_to_dict(
        result,
        dry_run_record=automator.records[-1] if automator.records else None,
    )
    print(json.dumps(record, indent=2, sort_keys=True))


def _command_for_action(action: Action) -> str:
    if action.action_type in {ActionType.CHECK, ActionType.CALL}:
        return "primary_left"
    if action.action_type in {ActionType.BET, ActionType.RAISE, ActionType.ALL_IN}:
        return "primary_middle"
    if action.action_type is ActionType.FOLD:
        return "primary_right"
    raise ValueError(f"unsupported Poker Legends action: {action.action_type.value}")


def _bot_step_result_to_dict(
    result: BotStepResult,
    *,
    dry_run_record: Mapping[str, object] | None,
) -> dict[str, object]:
    return {
        "acted": result.acted,
        "reason": result.reason,
        "confidence": result.confidence,
        "screen": _screen_to_dict(result),
        "state": _state_summary(result.state),
        "action": _action_to_dict(result.action),
        "policy_decision": _policy_decision_to_dict(result),
        "dry_run_record": dict(dry_run_record) if dry_run_record is not None else None,
    }


def _screen_to_dict(result: BotStepResult) -> dict[str, object] | None:
    if result.screen is None:
        return None
    return {
        "kind": result.screen.kind.value,
        "confidence": result.screen.confidence,
        "reason": result.screen.reason,
        "blocking_reason": result.screen.blocking_reason,
        "hero_turn": result.screen.hero_turn,
    }


def _state_summary(state: GameState | None) -> dict[str, object] | None:
    if state is None:
        return None
    return {
        "hand_id": state.hand_id,
        "street": state.street.value,
        "current_seat": state.current_seat,
        "pot_total": state.pot_total,
        "to_call": state.to_call,
        "legal_actions": [_action_to_dict(action) for action in state.legal_actions],
    }


def _policy_decision_to_dict(result: BotStepResult) -> dict[str, object] | None:
    if result.policy_decision is None:
        return None
    return {
        "reason": result.policy_decision.reason,
        "strength": result.policy_decision.strength,
        "required_equity": result.policy_decision.required_equity,
        "metadata": dict(result.policy_decision.metadata),
        "action": _action_to_dict(result.policy_decision.action),
    }


def _action_to_dict(action: Action | None) -> dict[str, object] | None:
    if action is None:
        return None
    return {
        "type": action.action_type.value,
        "amount": action.amount,
        "min_amount": action.min_amount,
        "max_amount": action.max_amount,
    }


def _button_rect(annotation: Mapping[str, object], name: str) -> ScreenRect:
    regions = annotation.get("regions")
    if not isinstance(regions, Mapping):
        raise ValueError("layout annotation has no regions")
    buttons = regions.get("buttons")
    if not isinstance(buttons, list):
        raise ValueError("layout annotation has no button regions")
    for region in buttons:
        if isinstance(region, Mapping) and str(region.get("name") or "") == name:
            raw_rect = region.get("rect")
            if not isinstance(raw_rect, Mapping):
                raise ValueError(f"button region has no rect: {name}")
            return ScreenRect(
                x=_to_int(raw_rect["x"]),
                y=_to_int(raw_rect["y"]),
                width=_to_int(raw_rect["width"]),
                height=_to_int(raw_rect["height"]),
            )
    raise KeyError(f"unknown Poker Legends button region: {name}")


def _run_command(command: Sequence[str]) -> None:
    subprocess.run(command, check=True)


def _read_json_object(path: str | Path) -> dict[str, object]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object: {path}")
    return cast(dict[str, object], data)


def _to_int(value: object) -> int:
    if type(value) is int:
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str) and value.strip():
        return int(value)
    raise TypeError(f"expected int-like value: {value!r}")
