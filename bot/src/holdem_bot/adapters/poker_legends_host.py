"""Host-side Poker Legends capture and dry-run automation helpers."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Protocol, cast

import cv2
import numpy as np
from holdem_ai import PolicyDecision
from holdem_ai.field import FieldExploitPolicy
from holdem_ai.opponents import OpponentModel
from holdem_common import Action, ActionType, GameState
from numpy.typing import NDArray

from holdem_bot.adapters.poker_legends import PokerLegendsTableRecognizer
from holdem_bot.adapters.poker_legends_llm import (
    DEFAULT_GEMINI_MODEL,
    DEFAULT_MAX_EDGE,
    PokerLegendsLlmRecognizer,
)
from holdem_bot.capture import Capture, CapturedFrame
from holdem_bot.hand_tracker import HandTracker
from holdem_bot.orchestrator import BotOrchestrator, BotStepResult
from holdem_bot.recognize import RecognitionResult, Recognizer, summarize_recognition_safety
from holdem_bot.screen_state import SafetyDecision, evaluate_safety
from holdem_bot.vision.annotations import ScreenRect
from holdem_bot.vision.perception_overlay import render_overlay
from holdem_bot.vision.poker_legends_action_buttons import (
    ActionButtonDetection,
    detect_action_buttons,
)
from holdem_bot.vision.poker_legends_temporal import PokerLegendsTemporalTracker

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


class PokerLegendsVisionClickPlanner:
    """Plan a click from CV-detected action buttons (no static layout needed).

    The buttons are found live in the captured frame by colour+shape, so this works at any
    window size/framing where the static-ROI planner cannot. ``plan`` maps the AI's action to
    its button slot and returns the button centre; it returns ``None`` when that button is not
    on screen, so the caller fails closed (never clicks a button it cannot see).
    """

    def __init__(self, detections: Sequence[ActionButtonDetection]) -> None:
        self._by_slot = {detection.slot: detection for detection in detections}

    @classmethod
    def from_image(cls, image: NDArray[np.uint8]) -> PokerLegendsVisionClickPlanner:
        return cls(detect_action_buttons(image))

    def plan(
        self,
        action: Action,
        *,
        origin: tuple[int, int] = (0, 0),
        coordinate_space: str = "image",
    ) -> PokerLegendsClickPlan | None:
        command = _command_for_action(action)
        detection = self._by_slot.get(command)
        if detection is None:
            return None
        left, top = origin
        return PokerLegendsClickPlan(
            action_type=action.action_type.value,
            amount=action.amount,
            command=command,
            x=left + detection.x,
            y=top + detection.y,
            coordinate_space=coordinate_space,
        )


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
    # An opponent-aware policy carrying a persistent per-seat read. On a single
    # frame the read is still UNKNOWN (it needs many hands), so the decision is
    # identical to the bare heuristic; the model is what a continuous session
    # (e.g. the replay runner) populates to start exploiting the field.
    policy = FieldExploitPolicy()
    orchestrator = BotOrchestrator(
        capture=capture,
        recognizer=recognizer,
        automator=automator,
        seat=args.seat,
        min_confidence=args.min_confidence,
        policy_explainer=policy.explain,
    )
    result = orchestrator.run_once()
    record = _bot_step_result_to_dict(
        result,
        dry_run_record=automator.records[-1] if automator.records else None,
        opponent_reads=_opponent_reads(policy.model, result.state, controlled_seat=args.seat),
    )
    print(json.dumps(record, indent=2, sort_keys=True))


def replay_dry_run_main(argv: Sequence[str] | None = None) -> None:
    """Replay a directory of saved frames through ONE persistent opponent-aware policy.

    A single dry-run frame can never build a read (each invocation is a fresh
    process). This feeds a whole sequence of saved frames through one
    ``FieldExploitPolicy`` so the per-seat ``OpponentModel`` accumulates exactly as
    it would over a live session, and reports the read it ends up with — the
    offline proxy for "does real-frame recognition support an opponent model". Each
    frame's own annotation JSON supplies its ROIs (and, with ``--use-truth``, the
    ground-truth state, isolating the read logic from recognition quality). Never
    clicks.
    """
    parser = argparse.ArgumentParser(description="Replay saved Poker Legends frames; build a read.")
    parser.add_argument("--frames-dir", required=True)
    parser.add_argument("--annotations-dir", required=True)
    parser.add_argument("--card-part-manifest", required=True)
    parser.add_argument("--card-classifier-manifest", required=True)
    parser.add_argument("--button-manifest", required=True)
    parser.add_argument("--card-template-manifest")
    parser.add_argument("--seat", type=int, default=0)
    parser.add_argument("--min-confidence", type=float, default=0.80)
    parser.add_argument("--log-jsonl", required=True)
    parser.add_argument("--safety-report", help="Optional Markdown safety summary output path.")
    parser.add_argument(
        "--temporal-window",
        type=int,
        default=2,
        help="Stable-frame window required before replay dry-run authorization.",
    )
    parser.add_argument(
        "--use-truth",
        action="store_true",
        help="Bypass CV with each frame's truth annotation (isolates the read from recognition).",
    )
    parser.add_argument("--limit", type=int)
    args = parser.parse_args(argv)

    recognizer = PokerLegendsTableRecognizer.from_manifests(
        card_part_manifest=args.card_part_manifest,
        card_classifier_manifest=args.card_classifier_manifest,
        button_manifest=args.button_manifest,
        card_template_manifest=args.card_template_manifest,
        controlled_seat=args.seat,
    )
    policy = FieldExploitPolicy()
    temporal_tracker = PokerLegendsTemporalTracker(
        required_stable_frames=args.temporal_window,
    )

    frame_paths = sorted(Path(args.frames_dir).glob("*.png"))
    if args.limit is not None:
        frame_paths = frame_paths[: args.limit]
    annotations_dir = Path(args.annotations_dir)

    steps: list[dict[str, object]] = []
    recognitions: list[RecognitionResult] = []
    expected_screen_kind_by_frame: dict[str, str] = {}
    opponent_seats: set[int] = set()
    for frame_path in frame_paths:
        annotation_path = annotations_dir / f"{frame_path.stem}.json"
        if not annotation_path.exists():
            continue
        expected_screen_kind = _expected_screen_kind_from_annotation(annotation_path)
        if expected_screen_kind is not None:
            expected_screen_kind_by_frame[frame_path.stem] = expected_screen_kind
        capture = PokerLegendsImageCapture(
            image_path=str(frame_path),
            layout_annotation_path=str(annotation_path),
            annotation_path=str(annotation_path) if args.use_truth else None,
        )
        automator = PokerLegendsDryRunAutomator(
            planner=PokerLegendsLayoutClickPlanner.from_annotation_path(str(annotation_path)),
            log_path=args.log_jsonl,
        )
        frame = capture.capture()
        recognition, decision, policy_decision = _recognize_and_decide(
            recognizer=recognizer,
            frame=frame,
            policy=policy,
            seat=args.seat,
            min_confidence=args.min_confidence,
            temporal_tracker=temporal_tracker,
        )
        recognitions.append(recognition)
        acted = False
        if decision.allowed and decision.state is not None and policy_decision is not None:
            automator.perform(policy_decision.action, decision.state)
            acted = True
        if decision.state is not None:
            opponent_seats.update(
                player.seat for player in decision.state.players if player.seat != args.seat
            )
        steps.append(
            {
                "frame": frame_path.stem,
                "acted": acted,
                "reason": "acted" if acted else decision.reason,
                "recognition_mode": recognition.recognition_mode.value,
                "expected_screen_kind": expected_screen_kind,
                "observed_screen_kind": recognition.screen.kind.value,
                "assembly_status": recognition.assembly_result.status.value
                if recognition.assembly_result is not None
                else None,
                "validity_scope": recognition.assembly_result.validity_scope.value
                if recognition.assembly_result is not None
                else None,
                "stable_frame_count": recognition.assembly_result.freshness.stable_frame_count
                if recognition.assembly_result is not None
                else None,
                "safety_contract": recognition.safety_contract.value,
                "state_block_reason": recognition.metadata.get("state_block_reason"),
                "temporal_tracker": recognition.metadata.get("temporal_tracker"),
                "exploit": policy_decision.metadata.get("exploit")
                if policy_decision is not None
                else None,
            }
        )

    safety_summary = summarize_recognition_safety(
        recognitions,
        expected_screen_kind_by_frame=expected_screen_kind_by_frame,
    )
    summary = {
        "frames": len(steps),
        "actionable": sum(1 for step in steps if step["acted"]),
        "opponent_reads": [_seat_read_dict(policy.model, seat) for seat in sorted(opponent_seats)],
        "safety_summary": safety_summary.to_dict(),
        "steps": steps,
    }
    if args.safety_report is not None:
        _write_replay_safety_report(Path(args.safety_report), summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


def _expected_screen_kind_from_annotation(path: str | Path) -> str | None:
    annotation = _read_json_object(path)
    screen = annotation.get("screen")
    if not isinstance(screen, Mapping):
        return None
    kind = screen.get("kind")
    return kind if isinstance(kind, str) and kind else None


def _write_replay_safety_report(path: Path, summary: Mapping[str, object]) -> None:
    safety = summary.get("safety_summary")
    safety_summary = safety if isinstance(safety, Mapping) else {}
    steps = summary.get("steps")
    step_rows = (
        [step for step in steps if isinstance(step, Mapping)] if isinstance(steps, list) else []
    )
    false_actionable_rows = [
        step
        for step in step_rows
        if step.get("expected_screen_kind") not in {None, "actionable_table"}
        and step.get("observed_screen_kind") == "actionable_table"
    ]

    lines = [
        "# Poker Legends Replay Safety Report",
        "",
        "## Summary",
        "",
        f"- Frames: {summary.get('frames', 0)}",
        f"- Dry-run actions: {summary.get('actionable', 0)}",
        f"- Authorization events: {safety_summary.get('authorization_events', 0)}",
        f"- Unsafe authorization events: {safety_summary.get('unsafe_authorization_events', 0)}",
        f"- Stale authorization events: {safety_summary.get('stale_authorization_events', 0)}",
        "- Truth-assisted authorization events: "
        f"{safety_summary.get('truth_assisted_authorization_events', 0)}",
        "- Expected non-actionable frames: "
        f"{safety_summary.get('expected_non_actionable_frames', 0)}",
        f"- False actionable count: {safety_summary.get('false_actionable_count', 0)}",
        f"- Source-policy violations: {safety_summary.get('source_policy_violation_count', 0)}",
        "",
        "## Distributions",
        "",
        "### Recognition Modes",
        "",
        *_markdown_count_lines(safety_summary.get("mode_counts")),
        "",
        "### Assembly Statuses",
        "",
        *_markdown_count_lines(safety_summary.get("assembly_status_counts")),
        "",
        "### Blocking Issues",
        "",
        *_markdown_count_lines(safety_summary.get("blocking_issue_counts")),
        "",
        "## False Actionable Frames",
        "",
    ]
    if false_actionable_rows:
        lines.append("| Frame | Expected | Observed | Assembly | Block |")
        lines.append("| --- | --- | --- | --- | --- |")
        for step in false_actionable_rows:
            lines.append(
                "| "
                f"`{step.get('frame')}` | "
                f"`{step.get('expected_screen_kind')}` | "
                f"`{step.get('observed_screen_kind')}` | "
                f"`{step.get('assembly_status')}` | "
                f"`{step.get('state_block_reason')}` |"
            )
    else:
        lines.append("No false actionable frames.")
    lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _markdown_count_lines(value: object) -> list[str]:
    if not isinstance(value, Mapping) or not value:
        return ["- none"]
    return [f"- `{key}`: {count}" for key, count in sorted(value.items())]


def watch_main(argv: Sequence[str] | None = None) -> None:
    """Live perception HUD: capture -> recognise -> overlay reads + planned click, never click.

    With ``--image`` it renders one saved frame's overlay to disk headlessly (no GUI,
    no ``mss``) -- the offline path used by tests and for sending annotated evidence.
    Otherwise it opens a live ``mss`` capture loop and draws the overlay each frame in
    an OpenCV window: ``s`` dumps frame+overlay+json, ``q`` quits. When the hero can act
    it also locates the action buttons (CV) and draws the click target the bot *would*
    press, but it only ever reads the screen -- it never performs a click.
    """
    parser = argparse.ArgumentParser(
        description="Live Poker Legends perception HUD (overlay only; never clicks)."
    )
    parser.add_argument("--image", help="render ONE saved frame headlessly instead of live capture")
    parser.add_argument("--monitor", type=int, default=1, help="mss monitor index for live capture")
    parser.add_argument(
        "--region", help="live capture region 'left,top,width,height' (overrides --monitor)"
    )
    parser.add_argument("--layout-annotation")
    parser.add_argument("--card-part-manifest")
    parser.add_argument("--card-classifier-manifest")
    parser.add_argument("--button-manifest")
    parser.add_argument("--card-template-manifest")
    parser.add_argument("--annotation", help="truth annotation for --image (bypasses CV)")
    parser.add_argument("--seat", type=int, default=0)
    parser.add_argument("--min-confidence", type=float, default=0.80)
    parser.add_argument(
        "--overlay-out", help="output PNG for --image (default: <image>.overlay.png)"
    )
    parser.add_argument("--dump-dir", help="live mode: directory for 's' frame dumps")
    parser.add_argument("--fps", type=float, default=4.0)
    parser.add_argument(
        "--temporal-window",
        type=int,
        default=2,
        help="Stable-frame window required before live HUD policy decisions.",
    )
    parser.add_argument(
        "--llm",
        action="store_true",
        help="read the table with a vision LLM (Gemini) instead of template CV",
    )
    parser.add_argument("--llm-model", default=DEFAULT_GEMINI_MODEL)
    parser.add_argument(
        "--llm-min-interval",
        type=float,
        default=1.5,
        help="live --llm: min seconds between LLM calls (re-read only on frame change)",
    )
    parser.add_argument(
        "--llm-max-edge",
        type=int,
        default=DEFAULT_MAX_EDGE,
        help="downscale frames to this longest edge before the LLM (0 = full res)",
    )
    parser.add_argument(
        "--llm-no-crop",
        action="store_true",
        help="--llm: send the full frame instead of cropping to the located game window",
    )
    parser.add_argument(
        "--window",
        action="store_true",
        help="live: capture only the game's OS window (exact; no desktop, never clips UI)",
    )
    parser.add_argument(
        "--window-title", default="Poker Legends", help="window-title substring for --window"
    )
    args = parser.parse_args(argv)

    recognizer: Recognizer
    if args.llm:
        recognizer = PokerLegendsLlmRecognizer.gemini(
            controlled_seat=args.seat,
            model=args.llm_model,
            max_edge=args.llm_max_edge,
            crop=not args.llm_no_crop and not args.window,
        )
    else:
        missing = [
            f"--{name.replace('_', '-')}"
            for name in (
                "card_part_manifest",
                "card_classifier_manifest",
                "button_manifest",
                "layout_annotation",
            )
            if getattr(args, name) is None
        ]
        if missing:
            parser.error("without --llm these are required: " + ", ".join(missing))
        recognizer = PokerLegendsTableRecognizer.from_manifests(
            card_part_manifest=args.card_part_manifest,
            card_classifier_manifest=args.card_classifier_manifest,
            button_manifest=args.button_manifest,
            card_template_manifest=args.card_template_manifest,
            controlled_seat=args.seat,
        )
    policy = FieldExploitPolicy()
    layout = _read_json_object(args.layout_annotation) if args.layout_annotation else None

    if args.image is not None:
        _run_watch_once(
            image=args.image,
            layout_path=args.layout_annotation,
            layout=layout,
            annotation=args.annotation,
            recognizer=recognizer,
            policy=policy,
            seat=args.seat,
            min_confidence=args.min_confidence,
            overlay_out=args.overlay_out,
        )
        return

    region = _parse_region(args.region) if args.region else None
    if args.window:
        from holdem_bot.adapters.window_region import find_game_window

        window = find_game_window(args.window_title)
        if window is None:
            parser.error(
                f"no window matching '{args.window_title}' found (is Poker Legends open/visible?)"
            )
        region = window.region()
        print(f"capturing window '{window.title}' region {region}")
    _run_watch_live(
        monitor=args.monitor,
        region=region,
        layout_path=args.layout_annotation,
        layout=layout,
        recognizer=recognizer,
        policy=policy,
        seat=args.seat,
        min_confidence=args.min_confidence,
        fps=args.fps,
        dump_dir=args.dump_dir,
        temporal_window=args.temporal_window,
        recognize_min_interval=args.llm_min_interval if args.llm else 0.0,
    )


def _parse_region(text: str) -> dict[str, int]:
    parts = text.split(",")
    if len(parts) != 4:
        raise ValueError("region must be 'left,top,width,height'")
    left, top, width, height = (int(part) for part in parts)
    return {"left": left, "top": top, "width": width, "height": height}


def _watch_frame(image_path: Path, layout_path: str | None) -> CapturedFrame:
    """Build a CapturedFrame pointing at a saved/temp PNG (and its layout, if any)."""
    metadata: dict[str, object] = {
        "poker_legends_image_path": str(image_path),
        "coordinate_space": "image",
    }
    if layout_path is not None:
        metadata["poker_legends_layout_annotation_path"] = str(layout_path)
    return CapturedFrame(payload=image_path, source="mss_live", metadata=metadata)


def _overlay_layout(
    layout: Mapping[str, object] | None, frame: NDArray[np.uint8]
) -> Mapping[str, object]:
    """The given layout, or a region-free layout sized to the frame (LLM mode draws no ROIs)."""
    if layout is not None:
        return layout
    height, width = frame.shape[:2]
    return {"width": width, "height": height, "regions": {}}


def _overlay_base(frame: NDArray[np.uint8], recognition: RecognitionResult) -> NDArray[np.uint8]:
    """The exact image submitted to the LLM (HUD shows what the model saw), else the raw frame."""
    submitted = recognition.metadata.get("submitted_image")
    if isinstance(submitted, str):
        image = cv2.imread(submitted)
        if image is not None:
            return cast(NDArray[np.uint8], image)
    return frame


def _frame_changed(
    current: NDArray[np.uint8], previous: NDArray[np.uint8], *, threshold: float = 2.5
) -> bool:
    """True if the frame changed enough to warrant a fresh (costly) recognition pass."""
    small_current = cv2.resize(current, (64, 64)).astype(np.int16)
    small_previous = cv2.resize(previous, (64, 64)).astype(np.int16)
    return float(np.mean(np.abs(small_current - small_previous))) > threshold


@dataclass(frozen=True, slots=True)
class _ClickTarget:
    """The resolved click target: the screen/image-space plan plus its frame-space point.

    ``source`` records how the point was obtained -- ``llm+cv`` (both agree; pixel-exact CV centre),
    ``llm`` / ``cv`` (only one source), or ``conflict`` (both present but far apart). A future real
    click should fire only on ``llm+cv``; the HUD surfaces the rest so disagreement is visible.
    """

    plan: PokerLegendsClickPlan
    frame_x: int
    frame_y: int
    source: str


def _llm_box_center(metadata: Mapping[str, object], slot: str) -> tuple[int, int] | None:
    """The LLM's frame-space click centre for a button slot, if it returned a box_2d for it."""
    boxes = metadata.get("llm_button_boxes")
    if isinstance(boxes, Mapping):
        center = boxes.get(slot)
        if isinstance(center, list | tuple) and len(center) == 2:
            return int(center[0]), int(center[1])
    return None


def _resolve_click_point(
    llm_center: tuple[int, int] | None,
    cv_detection: ActionButtonDetection | None,
    frame_height: int,
) -> tuple[tuple[int, int] | None, str]:
    """Combine the LLM box centre and the CV button into one click point + an agreement source."""
    if cv_detection is not None and llm_center is not None:
        distance = float(np.hypot(llm_center[0] - cv_detection.x, llm_center[1] - cv_detection.y))
        tolerance = max(float(cv_detection.radius), frame_height * 0.03) * 1.5
        point = (cv_detection.x, cv_detection.y)  # CV centre is pixel-exact when present
        return point, ("llm+cv" if distance <= tolerance else "conflict")
    if cv_detection is not None:
        return (cv_detection.x, cv_detection.y), "cv"
    if llm_center is not None:
        return llm_center, "llm"
    return None, "none"


def _hero_turn_from_buttons(detections: Sequence[ActionButtonDetection]) -> bool:
    """Action buttons visible => it is the hero's turn -- the cheap CV trigger for an LLM read.

    Requires the Fold circle (the canonical "you can act" control) or a 2+ button row, so a stray
    coloured circle elsewhere does not look like a turn.
    """
    if any(detection.color_class == "fold" for detection in detections):
        return True
    return len(detections) >= 2


def _plan_click_targets(
    frame: NDArray[np.uint8],
    recognition: RecognitionResult,
    policy_decision: PolicyDecision | None,
    *,
    origin: tuple[int, int] | None,
    detections: Sequence[ActionButtonDetection] | None = None,
) -> tuple[tuple[ActionButtonDetection, ...], _ClickTarget | None]:
    """Plan the chosen action's click target from the LLM box + CV buttons (read-only).

    Primary coordinate = the LLM's ``box_2d`` centre (robust, free with recognition); CV verifies
    and, when both agree, supplies the pixel-exact centre. ``detections`` may be passed in to reuse
    a detection already done for the turn gate. Returns the CV detections (for the HUD) and the
    resolved target, or ``None`` when no decision is pending or neither source found the button
    (fail closed -- never click blind).
    """
    buttons = tuple(detections) if detections is not None else detect_action_buttons(frame)
    if policy_decision is None:
        return buttons, None
    action = policy_decision.action
    command = _command_for_action(action)
    cv_detection = next((det for det in buttons if det.slot == command), None)
    llm_center = _llm_box_center(recognition.metadata, command)
    point, source = _resolve_click_point(llm_center, cv_detection, frame.shape[0])
    if point is None:
        return buttons, None
    left, top = origin or (0, 0)
    plan = PokerLegendsClickPlan(
        action_type=action.action_type.value,
        amount=action.amount,
        command=command,
        x=left + point[0],
        y=top + point[1],
        coordinate_space="screen" if origin is not None else "image",
    )
    return buttons, _ClickTarget(plan=plan, frame_x=point[0], frame_y=point[1], source=source)


def _draw_click_targets(
    overlay: NDArray[np.uint8],
    frame: NDArray[np.uint8],
    detections: Sequence[ActionButtonDetection],
    click: _ClickTarget | None,
) -> None:
    """Draw CV buttons (faint) + the resolved click target (bright, labelled), scaled to overlay.

    Only valid when the overlay is a uniform downscale of the full frame (window / no-crop); the
    caller skips this for a cropped overlay. The plan's coordinates are always correct regardless.
    """
    frame_h, frame_w = frame.shape[:2]
    overlay_h, overlay_w = overlay.shape[:2]
    if frame_w == 0 or frame_h == 0:
        return
    scale_x, scale_y = overlay_w / frame_w, overlay_h / frame_h
    radius_scale = (scale_x + scale_y) / 2.0
    for detection in detections:
        cx = int(round(detection.x * scale_x))
        cy = int(round(detection.y * scale_y))
        radius = max(4, int(round(detection.radius * radius_scale)))
        cv2.circle(overlay, (cx, cy), radius, (170, 170, 170), 1)
    if click is None:
        return
    px, py = int(round(click.frame_x * scale_x)), int(round(click.frame_y * scale_y))
    color = (0, 255, 255) if click.source == "llm+cv" else (0, 165, 255)
    if click.source == "conflict":
        color = (0, 0, 255)
    cv2.circle(overlay, (px, py), max(8, int(round(28 * radius_scale))), color, 3)
    cv2.drawMarker(overlay, (px, py), color, cv2.MARKER_CROSS, 22, 2)
    cv2.putText(
        overlay,
        f"CLICK [{click.source}]",
        (px - 48, py + max(20, int(round(40 * radius_scale)))),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        color,
        2,
        cv2.LINE_AA,
    )


def _recognize_and_decide(
    recognizer: Recognizer,
    frame: CapturedFrame,
    policy: FieldExploitPolicy,
    *,
    seat: int,
    min_confidence: float,
    hand_tracker: HandTracker | None = None,
    temporal_tracker: PokerLegendsTemporalTracker | None = None,
) -> tuple[RecognitionResult, SafetyDecision, PolicyDecision | None]:
    """Faithful per-frame pipeline: recognise -> the same safety gate -> decide if allowed.

    Mirrors ``BotOrchestrator.run_once`` (so the HUD shows exactly what the bot would
    do) but keeps the full ``RecognitionResult`` -- notably ``metadata`` -- so the
    overlay can surface *why* state assembly failed. When a ``hand_tracker`` is given it
    restamps ``hand_id`` with a stable per-hand id (the live recogniser's is the constant
    temp frame name), so the opponent model can actually accumulate per-hand reads.
    """
    recognition = recognizer.recognize(frame)
    if temporal_tracker is not None:
        recognition = temporal_tracker.update(recognition)
    if hand_tracker is not None and recognition.state is not None:
        stamped = replace(recognition.state, hand_id=hand_tracker.hand_id(recognition.state))
        recognition = replace(recognition, state=stamped)
    decision = evaluate_safety(
        screen=recognition.screen,
        state=recognition.state,
        recognition_confidence=recognition.confidence,
        controlled_seat=seat,
        min_confidence=min_confidence,
    )
    policy_decision: PolicyDecision | None = None
    if decision.allowed and decision.state is not None:
        policy_decision = policy.explain(decision.state)
    return recognition, decision, policy_decision


def _run_watch_once(
    *,
    image: str,
    layout_path: str | None,
    layout: Mapping[str, object] | None,
    annotation: str | None,
    recognizer: Recognizer,
    policy: FieldExploitPolicy,
    seat: int,
    min_confidence: float,
    overlay_out: str | None,
) -> None:
    if layout_path is not None:
        frame_capture = PokerLegendsImageCapture(
            image_path=image, layout_annotation_path=layout_path, annotation_path=annotation
        ).capture()
    else:
        frame_capture = _watch_frame(Path(image), None)
    recognition, decision, policy_decision = _recognize_and_decide(
        recognizer, frame_capture, policy, seat=seat, min_confidence=min_confidence
    )
    raw = cv2.imread(image)
    if raw is None:
        raise FileNotFoundError(f"could not read image: {image}")
    frame = cast(NDArray[np.uint8], raw)
    # offline single frame: no screen origin, so the click plan is in image space
    detections, click = _plan_click_targets(frame, recognition, policy_decision, origin=None)
    lines = _watch_summary_lines(
        recognition,
        decision,
        policy_decision,
        policy.model,
        controlled_seat=seat,
        click=click,
        click_searched=policy_decision is not None,
    )
    base = _overlay_base(frame, recognition)
    overlay = render_overlay(base, _overlay_layout(layout, base), lines)
    if recognition.metadata.get("game_region_fraction") is None:
        _draw_click_targets(overlay, frame, detections, click)
    out_path = Path(overlay_out) if overlay_out else Path(image).with_suffix(".overlay.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), overlay)
    record = _watch_record(
        recognition,
        decision,
        policy_decision,
        policy.model,
        controlled_seat=seat,
        click=click,
    )
    record["overlay_out"] = str(out_path)
    print(json.dumps(record, indent=2, sort_keys=True))


def _run_watch_live(
    *,
    monitor: int,
    region: dict[str, int] | None,
    layout_path: str | None,
    layout: Mapping[str, object] | None,
    recognizer: Recognizer,
    policy: FieldExploitPolicy,
    seat: int,
    min_confidence: float,
    fps: float,
    dump_dir: str | None,
    temporal_window: int,
    recognize_min_interval: float = 0.0,
) -> None:
    try:
        import mss
    except ImportError as exc:  # pragma: no cover - host-only dependency
        raise SystemExit("live capture needs the 'mss' package: pip install mss") from exc

    interval = 1.0 / max(fps, 1.0)
    tmp_frame = Path(tempfile.gettempdir()) / "holdem_watch_frame.png"
    window = "Poker Legends HUD  [s] dump  [q] quit"
    dump_path = Path(dump_dir) if dump_dir else None
    if dump_path is not None:
        dump_path.mkdir(parents=True, exist_ok=True)
    dump_index = 0
    llm_reads = 0
    hand_tracker = HandTracker()
    temporal_tracker = PokerLegendsTemporalTracker(required_stable_frames=temporal_window)
    cached: tuple[RecognitionResult, SafetyDecision, PolicyDecision | None] | None = None
    last_recognized: NDArray[np.uint8] | None = None
    last_recognize_at = 0.0

    with mss.mss() as sct:
        target = region if region is not None else sct.monitors[monitor]
        print(f"watching {target} at ~{fps:g} fps; focus the HUD window and press q to quit")
        while True:  # pragma: no cover - interactive host-only loop
            start = time.time()
            shot = sct.grab(target)
            frame = cast(NDArray[np.uint8], np.ascontiguousarray(np.asarray(shot)[:, :, :3]))
            # CV turn gate: detect the action buttons every frame (cheap, ~ms). In LLM mode spend a
            # (slow, costly) read only when the buttons are up -- i.e. it is the hero's turn -- so
            # LLM latency tracks our own turn, not the game's animation / opponent-action speed.
            cv_buttons = detect_action_buttons(frame)
            hero_turn = _hero_turn_from_buttons(cv_buttons)
            changed = last_recognized is None or _frame_changed(frame, last_recognized)
            due = (start - last_recognize_at) >= recognize_min_interval
            if recognize_min_interval <= 0.0:
                should_read = True  # ungated (non-LLM): the CV read is cheap, refresh every frame
            else:
                should_read = cached is None or (hero_turn and changed and due)
            if should_read:
                cv2.imwrite(str(tmp_frame), frame)
                cached = _recognize_and_decide(
                    recognizer,
                    _watch_frame(tmp_frame, layout_path),
                    policy,
                    seat=seat,
                    min_confidence=min_confidence,
                    hand_tracker=hand_tracker,
                    temporal_tracker=temporal_tracker,
                )
                last_recognized = frame
                last_recognize_at = start
                llm_reads += 1
            assert cached is not None  # should_read includes `cached is None`, so it is set by now
            recognition, decision, policy_decision = cached
            # Plan/draw a click only when CV says it is our turn, so a stale decision from the last
            # turn is not shown while the opponents act.
            origin = (int(target["left"]), int(target["top"]))
            turn_decision = policy_decision if hero_turn else None
            detections, click = _plan_click_targets(
                frame, recognition, turn_decision, origin=origin, detections=cv_buttons
            )
            lines = [
                f"cv {len(cv_buttons)}btn  hero_turn~{hero_turn}  "
                f"llm_reads {llm_reads}  {'READ' if should_read else 'cached'}",
                *_watch_summary_lines(
                    recognition,
                    decision,
                    policy_decision,
                    policy.model,
                    controlled_seat=seat,
                    click=click,
                    click_searched=hero_turn,
                ),
            ]
            base = _overlay_base(frame, recognition)
            overlay = render_overlay(base, _overlay_layout(layout, base), lines)
            if recognition.metadata.get("game_region_fraction") is None:
                _draw_click_targets(overlay, frame, detections, click)
            cv2.imshow(window, overlay)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("s") and dump_path is not None:
                dump_index += 1
                _dump_watch_frame(
                    dump_path,
                    dump_index,
                    frame,
                    overlay,
                    recognition,
                    decision,
                    policy_decision,
                    policy.model,
                    controlled_seat=seat,
                    click=click,
                )
                print(f"dumped frame {dump_index} -> {dump_path}")
            elapsed = time.time() - start
            if elapsed < interval:
                time.sleep(interval - elapsed)
    cv2.destroyAllWindows()


def _dump_watch_frame(
    dump_dir: Path,
    index: int,
    frame: NDArray[np.uint8],
    overlay: NDArray[np.uint8],
    recognition: RecognitionResult,
    decision: SafetyDecision,
    policy_decision: PolicyDecision | None,
    model: OpponentModel,
    *,
    controlled_seat: int,
    click: _ClickTarget | None = None,
) -> None:
    stem = f"watch_{index:04d}"
    cv2.imwrite(str(dump_dir / f"{stem}.png"), frame)
    cv2.imwrite(str(dump_dir / f"{stem}.overlay.png"), overlay)
    record = _watch_record(
        recognition,
        decision,
        policy_decision,
        model,
        controlled_seat=controlled_seat,
        click=click,
    )
    (dump_dir / f"{stem}.json").write_text(
        json.dumps(record, indent=2, sort_keys=True), encoding="utf-8"
    )


def _watch_summary_lines(
    recognition: RecognitionResult,
    decision: SafetyDecision,
    policy_decision: PolicyDecision | None,
    model: OpponentModel,
    *,
    controlled_seat: int,
    click: _ClickTarget | None = None,
    click_searched: bool = False,
) -> list[str]:
    """Compact overlay text: gate, screen, why-blocked, state, decision, click target, reads."""
    screen = recognition.screen
    lines = [
        f"seat {controlled_seat}  conf {recognition.confidence:.2f}  gate {decision.reason}",
        f"screen {screen.kind.value}  hero_turn {screen.hero_turn}",
    ]
    if screen.blocking_reason:
        lines.append(f"block {screen.blocking_reason}")
    block_reason = recognition.metadata.get("state_block_reason")
    if block_reason is not None:
        lines.append(f"state_block {block_reason}")
    state = recognition.state
    if state is None:
        lines.append("state <none>")
    else:
        legal = "/".join(action.action_type.value for action in state.legal_actions) or "-"
        lines.append(
            f"hand {state.hand_id}  pot {state.pot_total}  to_call {state.to_call}  legal {legal}"
        )
    if policy_decision is not None:
        action = policy_decision.action
        lines.append(
            f"policy {action.action_type.value}:{action.amount} ({policy_decision.reason})"
        )
        exploit = policy_decision.metadata.get("exploit")
        if exploit is not None:
            lines.append(f"exploit {exploit}")
    if click is not None:
        lines.append(
            f"click {click.plan.command} @{click.plan.coordinate_space} "
            f"({click.plan.x},{click.plan.y})  [{click.source}, read-only]"
        )
    elif click_searched:
        lines.append("click <button not located>")
    if state is not None:
        for player in state.players:
            if player.seat == controlled_seat:
                continue
            read = model.read(player.seat)
            if read.hands:
                lines.append(
                    f"  s{read.seat} {read.profile.value}"
                    f"  v={_fmt_ratio(read.vpip)} p={_fmt_ratio(read.pfr)} n={read.hands}"
                )
    return lines


def _fmt_ratio(value: float | None) -> str:
    return f"{value:.2f}" if value is not None else "-"


def _watch_record(
    recognition: RecognitionResult,
    decision: SafetyDecision,
    policy_decision: PolicyDecision | None,
    model: OpponentModel,
    *,
    controlled_seat: int,
    click: _ClickTarget | None = None,
) -> dict[str, object]:
    """Structured per-frame diagnostic record (mirrors the dry-run record shape)."""
    screen = recognition.screen
    return {
        "gate": {"allowed": decision.allowed, "reason": decision.reason},
        "confidence": recognition.confidence,
        "screen": {
            "kind": screen.kind.value,
            "confidence": screen.confidence,
            "reason": screen.reason,
            "blocking_reason": screen.blocking_reason,
            "hero_turn": screen.hero_turn,
        },
        "state": _state_summary(recognition.state),
        "state_block_reason": recognition.metadata.get("state_block_reason"),
        "policy_decision": _watch_policy_dict(policy_decision),
        "click_plan": (
            None
            if click is None
            else {**click.plan.to_dict(), "source": click.source, "executed": False}
        ),
        "opponent_reads": _opponent_reads(
            model, recognition.state, controlled_seat=controlled_seat
        ),
    }


def _watch_policy_dict(policy_decision: PolicyDecision | None) -> dict[str, object] | None:
    if policy_decision is None:
        return None
    return {
        "reason": policy_decision.reason,
        "strength": policy_decision.strength,
        "required_equity": policy_decision.required_equity,
        "metadata": dict(policy_decision.metadata),
        "action": _action_to_dict(policy_decision.action),
    }


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
    opponent_reads: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    return {
        "acted": result.acted,
        "reason": result.reason,
        "confidence": result.confidence,
        "screen": _screen_to_dict(result),
        "state": _state_summary(result.state),
        "action": _action_to_dict(result.action),
        "policy_decision": _policy_decision_to_dict(result),
        "opponent_reads": opponent_reads,
        "dry_run_record": dict(dry_run_record) if dry_run_record is not None else None,
    }


def _opponent_reads(
    model: OpponentModel,
    state: GameState | None,
    *,
    controlled_seat: int,
) -> list[dict[str, object]] | None:
    """Per-seat VPIP/PFR/profile read for the opponents in the recognised state.

    On a single frame these are mostly ``unknown`` (the read needs many hands); a
    continuous session accumulates them, and this is the channel the operator
    watches to see whether real-frame recognition supports an opponent model.
    """
    if state is None:
        return None
    return [
        _seat_read_dict(model, player.seat)
        for player in state.players
        if player.seat != controlled_seat
    ]


def _seat_read_dict(model: OpponentModel, seat: int) -> dict[str, object]:
    read = model.read(seat)
    return {
        "seat": read.seat,
        "profile": read.profile.value,
        "hands": read.hands,
        "vpip": round(read.vpip, 3) if read.vpip is not None else None,
        "pfr": round(read.pfr, 3) if read.pfr is not None else None,
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
