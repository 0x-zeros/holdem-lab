"""Host-side Poker Legends capture and dry-run automation helpers."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
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
from holdem_bot.capture import Capture, CapturedFrame
from holdem_bot.orchestrator import BotOrchestrator, BotStepResult
from holdem_bot.recognize import RecognitionResult
from holdem_bot.screen_state import SafetyDecision, evaluate_safety
from holdem_bot.vision.annotations import ScreenRect
from holdem_bot.vision.perception_overlay import render_overlay

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

    frame_paths = sorted(Path(args.frames_dir).glob("*.png"))
    if args.limit is not None:
        frame_paths = frame_paths[: args.limit]
    annotations_dir = Path(args.annotations_dir)

    steps: list[dict[str, object]] = []
    opponent_seats: set[int] = set()
    for frame_path in frame_paths:
        annotation_path = annotations_dir / f"{frame_path.stem}.json"
        if not annotation_path.exists():
            continue
        capture = PokerLegendsImageCapture(
            image_path=str(frame_path),
            layout_annotation_path=str(annotation_path),
            annotation_path=str(annotation_path) if args.use_truth else None,
        )
        automator = PokerLegendsDryRunAutomator(
            planner=PokerLegendsLayoutClickPlanner.from_annotation_path(str(annotation_path)),
            log_path=args.log_jsonl,
        )
        orchestrator = BotOrchestrator(
            capture=capture,
            recognizer=recognizer,
            automator=automator,
            seat=args.seat,
            min_confidence=args.min_confidence,
            policy_explainer=policy.explain,
        )
        result = orchestrator.run_once()
        if result.state is not None:
            opponent_seats.update(
                player.seat for player in result.state.players if player.seat != args.seat
            )
        steps.append(
            {
                "frame": frame_path.stem,
                "acted": result.acted,
                "reason": result.reason,
                "exploit": result.policy_decision.metadata.get("exploit")
                if result.policy_decision is not None
                else None,
            }
        )

    summary = {
        "frames": len(steps),
        "actionable": sum(1 for step in steps if step["acted"]),
        "opponent_reads": [_seat_read_dict(policy.model, seat) for seat in sorted(opponent_seats)],
        "steps": steps,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


def watch_main(argv: Sequence[str] | None = None) -> None:
    """Live perception HUD: capture -> recognise -> overlay ROIs + reads, never click.

    With ``--image`` it renders one saved frame's overlay to disk headlessly (no GUI,
    no ``mss``) -- the offline path used by tests and for sending annotated evidence.
    Otherwise it opens a live ``mss`` capture loop and draws the overlay each frame in
    an OpenCV window: ``s`` dumps frame+overlay+json, ``q`` quits. It only ever reads
    the screen; it never plans or performs a click.
    """
    parser = argparse.ArgumentParser(
        description="Live Poker Legends perception HUD (overlay only; never clicks)."
    )
    parser.add_argument("--image", help="render ONE saved frame headlessly instead of live capture")
    parser.add_argument("--monitor", type=int, default=1, help="mss monitor index for live capture")
    parser.add_argument(
        "--region", help="live capture region 'left,top,width,height' (overrides --monitor)"
    )
    parser.add_argument("--layout-annotation", required=True)
    parser.add_argument("--card-part-manifest", required=True)
    parser.add_argument("--card-classifier-manifest", required=True)
    parser.add_argument("--button-manifest", required=True)
    parser.add_argument("--card-template-manifest")
    parser.add_argument("--annotation", help="truth annotation for --image (bypasses CV)")
    parser.add_argument("--seat", type=int, default=0)
    parser.add_argument("--min-confidence", type=float, default=0.80)
    parser.add_argument(
        "--overlay-out", help="output PNG for --image (default: <image>.overlay.png)"
    )
    parser.add_argument("--dump-dir", help="live mode: directory for 's' frame dumps")
    parser.add_argument("--fps", type=float, default=4.0)
    args = parser.parse_args(argv)

    recognizer = PokerLegendsTableRecognizer.from_manifests(
        card_part_manifest=args.card_part_manifest,
        card_classifier_manifest=args.card_classifier_manifest,
        button_manifest=args.button_manifest,
        card_template_manifest=args.card_template_manifest,
        controlled_seat=args.seat,
    )
    policy = FieldExploitPolicy()
    layout = _read_json_object(args.layout_annotation)

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
    )


def _parse_region(text: str) -> dict[str, int]:
    parts = text.split(",")
    if len(parts) != 4:
        raise ValueError("region must be 'left,top,width,height'")
    left, top, width, height = (int(part) for part in parts)
    return {"left": left, "top": top, "width": width, "height": height}


def _recognize_and_decide(
    recognizer: PokerLegendsTableRecognizer,
    frame: CapturedFrame,
    policy: FieldExploitPolicy,
    *,
    seat: int,
    min_confidence: float,
) -> tuple[RecognitionResult, SafetyDecision, PolicyDecision | None]:
    """Faithful per-frame pipeline: recognise -> the same safety gate -> decide if allowed.

    Mirrors ``BotOrchestrator.run_once`` (so the HUD shows exactly what the bot would
    do) but keeps the full ``RecognitionResult`` -- notably ``metadata`` -- so the
    overlay can surface *why* state assembly failed.
    """
    recognition = recognizer.recognize(frame)
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
    layout_path: str,
    layout: Mapping[str, object],
    annotation: str | None,
    recognizer: PokerLegendsTableRecognizer,
    policy: FieldExploitPolicy,
    seat: int,
    min_confidence: float,
    overlay_out: str | None,
) -> None:
    capture = PokerLegendsImageCapture(
        image_path=image, layout_annotation_path=layout_path, annotation_path=annotation
    )
    recognition, decision, policy_decision = _recognize_and_decide(
        recognizer, capture.capture(), policy, seat=seat, min_confidence=min_confidence
    )
    raw = cv2.imread(image)
    if raw is None:
        raise FileNotFoundError(f"could not read image: {image}")
    frame = cast(NDArray[np.uint8], raw)
    lines = _watch_summary_lines(
        recognition, decision, policy_decision, policy.model, controlled_seat=seat
    )
    overlay = render_overlay(frame, layout, lines)
    out_path = Path(overlay_out) if overlay_out else Path(image).with_suffix(".overlay.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), overlay)
    record = _watch_record(
        recognition, decision, policy_decision, policy.model, controlled_seat=seat
    )
    record["overlay_out"] = str(out_path)
    print(json.dumps(record, indent=2, sort_keys=True))


def _run_watch_live(
    *,
    monitor: int,
    region: dict[str, int] | None,
    layout_path: str,
    layout: Mapping[str, object],
    recognizer: PokerLegendsTableRecognizer,
    policy: FieldExploitPolicy,
    seat: int,
    min_confidence: float,
    fps: float,
    dump_dir: str | None,
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

    with mss.mss() as sct:
        target = region if region is not None else sct.monitors[monitor]
        print(f"watching {target} at ~{fps:g} fps; focus the HUD window and press q to quit")
        while True:  # pragma: no cover - interactive host-only loop
            start = time.time()
            shot = sct.grab(target)
            frame = cast(NDArray[np.uint8], np.ascontiguousarray(np.asarray(shot)[:, :, :3]))
            cv2.imwrite(str(tmp_frame), frame)
            capture = PokerLegendsImageCapture(
                image_path=tmp_frame, layout_annotation_path=layout_path
            )
            recognition, decision, policy_decision = _recognize_and_decide(
                recognizer, capture.capture(), policy, seat=seat, min_confidence=min_confidence
            )
            lines = _watch_summary_lines(
                recognition, decision, policy_decision, policy.model, controlled_seat=seat
            )
            overlay = render_overlay(frame, layout, lines)
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
) -> None:
    stem = f"watch_{index:04d}"
    cv2.imwrite(str(dump_dir / f"{stem}.png"), frame)
    cv2.imwrite(str(dump_dir / f"{stem}.overlay.png"), overlay)
    record = _watch_record(
        recognition, decision, policy_decision, model, controlled_seat=controlled_seat
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
) -> list[str]:
    """Compact overlay text: gate, screen, why-blocked, state, decision, reads."""
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
        lines.append(f"pot {state.pot_total}  to_call {state.to_call}  legal {legal}")
    if policy_decision is not None:
        action = policy_decision.action
        lines.append(
            f"policy {action.action_type.value}:{action.amount} ({policy_decision.reason})"
        )
        exploit = policy_decision.metadata.get("exploit")
        if exploit is not None:
            lines.append(f"exploit {exploit}")
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
