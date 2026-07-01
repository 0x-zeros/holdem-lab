"""Compare local VLMs (via LM Studio / Ollama) against Gemini and CV on Poker Legends frames.

Runs the SAME runtime prompt through each backend and scores three things:
  * box_2d button centres vs the CV detector (objective pixel ground truth on hero-turn frames),
  * structured fields (hero cards / board / pot / seats / actionable) vs the Gemini baseline (the
    current production reader, used here as the reference),
  * wall-clock latency per frame.

Two backend quirks learned empirically and baked in here:
  * box_2d axis order differs per model -- Gemini emits [ymin,xmin,ymax,xmax]; Qwen-VL ignores the
    prompt and emits its native [x1,y1,x2,y2]. See ``_button_centers_px``.
  * local models echo the prompt's section headers as JSON keys (``Buttons``/``GAME_REGION``), so
    top-level keys are lower-cased before use. See ``_canon``.

Usage::

    scripts/dev/py -m holdem_bot.eval.local_vlm \
        --frames-dir artifacts/poker-legends-videos/card_review_selection_v2/frames --limit 10 \
        --backends gemini,qwen/qwen3-vl-30b,cv --out docs/local-vlm-eval.md

    scripts/dev/py -m holdem_bot.eval.local_vlm \\
        --frames-dir <auto_review_selection_v2/frames> \\
        --reference-dir <truth_overlay_v2/truth_overlays> \\
        --backends qwen/qwen3-vl-8b,qwen/qwen3-vl-30b,cv --timeout 60
"""

from __future__ import annotations

import argparse
import base64
import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import cv2

from holdem_bot.adapters.poker_legends_llm import (
    DEFAULT_GEMINI_MODEL,
    RUNTIME_PROMPT,
    _downscale_image,
    read_image_with_gemini,
)
from holdem_bot.vision.poker_legends_action_buttons import detect_action_buttons

DEFAULT_LMSTUDIO_URL = "http://host.docker.internal:1234/v1"
DEFAULT_MAX_EDGE = 1280
#: action button name -> colour class (matches the CV detector's ``color_class``)
NAME_TO_COLOR = {
    "call": "call",
    "check": "call",
    "raise": "raise",
    "bet": "raise",
    "all_in": "raise",
    "fold": "fold",
}


@dataclass
class Read:
    """One backend's attempt to read one frame."""

    backend: str
    model: str
    ok: bool
    latency: float
    annotation: dict[str, Any] | None = None
    note: str = ""
    mode: str = ""
    finish: str = ""


# --------------------------------------------------------------------------- parsing helpers


def _data_url(image: Any) -> str:
    ok, buf = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError("png encode failed")
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode()


def _extract_json(text: str) -> dict[str, Any]:
    """First complete JSON object in text, tolerating ``` fences and trailing prose."""
    text = text.strip()
    if "```" in text:
        text = max(text.split("```"), key=len)
        if text[:4].lower() == "json":
            text = text[4:]
    i = text.find("{")
    if i < 0:
        raise ValueError("no JSON object found")
    obj, _ = json.JSONDecoder().raw_decode(text[i:])
    if not isinstance(obj, dict):
        raise ValueError("top-level JSON is not an object")
    return obj


def _canon(ann: dict[str, Any]) -> dict[str, Any]:
    """Lower-case top-level keys (Buttons -> buttons, GAME_REGION -> game_region)."""
    return {str(k).lower(): v for k, v in ann.items()}


# --------------------------------------------------------------------------- backends


def read_gemini(image: Any, model: str = DEFAULT_GEMINI_MODEL) -> Read:
    t0 = time.time()
    try:
        ann = dict(read_image_with_gemini(image, model=model))
        return Read(
            "gemini", model, True, time.time() - t0, _canon(ann), mode="json_schema", finish="stop"
        )
    except Exception as exc:  # noqa: BLE001
        return Read("gemini", model, False, time.time() - t0, note=repr(exc)[:160])


def read_openai_compat(
    image: Any, model: str, base_url: str = DEFAULT_LMSTUDIO_URL, timeout: float = 300.0
) -> Read:
    from openai import OpenAI

    client = OpenAI(base_url=base_url, api_key="local", timeout=timeout)
    is_qwen = "qwen" in model.lower()
    prompt = RUNTIME_PROMPT + ("\n\n/no_think" if is_qwen else "")
    extra = {"chat_template_kwargs": {"enable_thinking": False}} if is_qwen else {}
    data_url = _data_url(image)
    t0 = time.time()
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                }
            ],
            temperature=0,
            max_tokens=6000,
            response_format={"type": "text"},  # LM Studio rejects json_object; json_schema stalls
            extra_body=extra,
        )
    except Exception as exc:  # noqa: BLE001
        return Read("local", model, False, time.time() - t0, note=repr(exc)[:160], mode="text")
    dt = time.time() - t0
    choice = resp.choices[0]
    content = choice.message.content or ""
    reasoning = getattr(choice.message, "reasoning_content", None) or ""
    finish = choice.finish_reason or "?"
    src = content if content.strip() else reasoning  # salvage if mis-routed to reasoning
    try:
        ann = _canon(_extract_json(src))
        return Read("local", model, True, dt, ann, mode="text", finish=finish)
    except Exception as exc:  # noqa: BLE001
        return Read(
            "local", model, False, dt, note=f"parse:{exc!r}"[:140], mode="text", finish=finish
        )


def read_cv(image: Any) -> Read:
    t0 = time.time()
    dets = detect_action_buttons(image)
    dt = time.time() - t0
    buttons = [{"name": d.color_class, "label": "", "_px": [d.x, d.y]} for d in dets]
    return Read("cv", "cv", True, dt, {"buttons": buttons}, mode="cv", finish="stop")


# --------------------------------------------------------------------------- scoring


def _button_centers_px(read: Read, w: int, h: int) -> dict[str, tuple[float, float]]:
    """colour class -> (x, y) full-frame px, decoding box_2d per the backend's axis order."""
    ann = read.annotation or {}
    btns = ann.get("buttons") or ann.get("Buttons") or []
    xyxy = "qwen" in read.model.lower()  # Qwen-VL emits [x1,y1,x2,y2]; others [ymin,xmin,ymax,xmax]
    out: dict[str, tuple[float, float]] = {}
    for b in btns:
        if not isinstance(b, dict):
            continue
        color = NAME_TO_COLOR.get(str(b.get("name", "")).lower())
        if color is None:
            continue
        if "_px" in b:  # CV detector reports pixels directly
            out[color] = (float(b["_px"][0]), float(b["_px"][1]))
            continue
        box = b.get("box_2d")
        if not isinstance(box, (list, tuple)) or len(box) != 4:
            continue
        if xyxy:
            x1, y1, x2, y2 = box
        else:
            y1, x1, y2, x2 = box
        out[color] = (
            (float(x1) + float(x2)) / 2 / 1000 * w,
            (float(y1) + float(y2)) / 2 / 1000 * h,
        )
    return out


_SUIT_GLYPH = {"♠": "S", "♥": "H", "♦": "D", "♣": "C"}


def _norm_card(code: str) -> str:
    for glyph, letter in _SUIT_GLYPH.items():  # local models emit ♠♥♦♣ instead of S/H/D/C
        code = code.replace(glyph, letter)
    code = code.upper().replace(" ", "")
    if code.startswith("10"):  # local models write 10C; canonical is TC
        code = "T" + code[2:]
    return code


def _card_str(c: Any) -> str:
    """A card code like ``4S`` from any shape: canonical {card}, {rank,suit}, or a bare string."""
    if isinstance(c, dict):
        if c.get("visible") is False:
            return ""
        if c.get("card"):  # canonical schema: {slot, visible, card, confidence}
            return _norm_card(str(c["card"]))
        if c.get("rank") or c.get("suit"):
            return _norm_card(f"{c.get('rank') or ''}{c.get('suit') or ''}")
        return ""
    return _norm_card(str(c))


def _cards(v: Any) -> frozenset[str]:
    return frozenset(s for s in (_card_str(c) for c in (v or [])) if s)


def _pot(ann: dict[str, Any]) -> Any:
    """Pot total, from a top-level pot (local models) or a 'pot' text entry (canonical schema)."""
    pot = ann.get("pot")
    if isinstance(pot, dict) and pot.get("normalized_number") is not None:
        return pot.get("normalized_number")
    if isinstance(pot, (int, float)):
        return int(pot)
    for t in ann.get("texts") or []:
        if isinstance(t, dict) and "pot" in str(t.get("name", "")).lower():
            if t.get("normalized_number") is not None:
                return t.get("normalized_number")
    return None


def _seats(ann: dict[str, Any]) -> list[dict[str, Any]]:
    """Active players only -- drop empty/sitting-out seats (stack 0 or missing)."""
    return [s for s in (ann.get("seats") or [])
            if isinstance(s, dict) and s.get("stack") not in (None, 0)]


def _stacks(ann: dict[str, Any]) -> tuple[int, ...]:
    return tuple(
        sorted(int(s["stack"]) for s in _seats(ann) if isinstance(s.get("stack"), (int, float)))
    )


def _street(ann: dict[str, Any]) -> Any:
    s = (ann.get("table_state") or {}).get("street")
    return str(s).lower() if s else None


FIELD_FUNCS: dict[str, Callable[[dict[str, Any]], Any]] = {
    "hero_cards": lambda a: _cards(a.get("hero_hole_cards")),
    "board": lambda a: _cards(a.get("board")),
    "pot": _pot,
    "seat_count": lambda a: len(_seats(a)),
    "stacks": _stacks,
    "street": _street,
    "actionable": lambda a: (a.get("table_state") or {}).get("is_actionable"),
}


def compare_fields(pred: dict[str, Any], ref: dict[str, Any]) -> dict[str, tuple[bool, Any, Any]]:
    out: dict[str, tuple[bool, Any, Any]] = {}
    for key, fn in FIELD_FUNCS.items():
        try:
            pv = fn(pred)
        except Exception:  # noqa: BLE001
            pv = None
        try:
            rv = fn(ref)
        except Exception:  # noqa: BLE001
            rv = None
        out[key] = (pv == rv, pv, rv)
    return out


# --------------------------------------------------------------------------- runner


@dataclass
class FrameResult:
    frame: str
    hero_turn: bool = False  # gemini is_actionable -> a genuine hero decision frame
    reads: dict[str, Read] = field(default_factory=dict)
    box_err: dict[str, dict[str, float | None]] = field(default_factory=dict)
    fields: dict[str, dict[str, tuple[bool, Any, Any]]] = field(default_factory=dict)


def _load_reference(frame_path: str, reference_dir: str) -> dict[str, Any] | None:
    path = Path(reference_dir) / f"{Path(frame_path).stem}.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"reference is not a JSON object: {path}")
    return _canon(data)


def run(
    frames: list[str],
    backends: list[str],
    base_url: str,
    max_edge: int,
    *,
    reference_dir: str = "",
    timeout: float = 300.0,
) -> list[FrameResult]:
    results: list[FrameResult] = []
    for frame_path in frames:
        image = cv2.imread(frame_path)
        if image is None:
            print(f"  SKIP unreadable {frame_path}")
            continue
        h, w = image.shape[:2]
        sent = _downscale_image(cast(Any, image), max_edge)
        cv_truth = _button_centers_px(read_cv(image), w, h)  # objective box ground truth
        fr = FrameResult(frame=frame_path)
        reference_ann = _load_reference(frame_path, reference_dir) if reference_dir else None
        print(f"\n# {Path(frame_path).name}  {w}x{h}  CV buttons={sorted(cv_truth)}", flush=True)
        for be in backends:
            if be == "cv":
                read = read_cv(image)
            elif be == "gemini" or be.startswith("gemini:"):
                read = read_gemini(sent, be.split(":", 1)[1] if ":" in be else DEFAULT_GEMINI_MODEL)
            else:
                read = read_openai_compat(sent, be, base_url=base_url, timeout=timeout)
            fr.reads[be] = read
            if reference_ann is None and (
                be in ("gemini", DEFAULT_GEMINI_MODEL) or be.startswith("gemini:")
            ):
                reference_ann = read.annotation
            centers = _button_centers_px(read, w, h)
            fr.box_err[be] = {
                c: (
                    None
                    if c not in centers
                    else ((centers[c][0] - t[0]) ** 2 + (centers[c][1] - t[1]) ** 2) ** 0.5
                )
                for c, t in cv_truth.items()
            }
            errs = [e for e in fr.box_err[be].values() if e is not None]
            mean_err = sum(errs) / len(errs) if errs else None
            box_str = f"{mean_err:.0f}px" if mean_err is not None else "n/a"
            missed_n = sum(1 for e in fr.box_err[be].values() if e is None)
            print(
                f"  {be:24} ok={read.ok!s:5} {read.latency:5.1f}s finish={read.finish:7} "
                f"box_err_mean={box_str:>7} missed={missed_n} {read.note}",
                flush=True,
            )
        # field agreement vs reviewed truth if provided, else Gemini reference
        if reference_ann is not None:
            fr.hero_turn = bool(FIELD_FUNCS["actionable"](reference_ann))
            source = "reference" if reference_dir else "gemini"
            print(f"  -> hero_turn ({source} is_actionable) = {fr.hero_turn}", flush=True)
            for be, read in fr.reads.items():
                if read.annotation is not None and be != "cv" and (
                    reference_dir or not be.startswith("gemini")
                ):
                    fr.fields[be] = compare_fields(read.annotation, reference_ann)
                    miss = {
                        k: f"{pv!r}!={rv!r}"
                        for k, (good, pv, rv) in fr.fields[be].items()
                        if not good
                    }
                    if miss:
                        print(f"    {be} vs {source} mismatch: {miss}", flush=True)
        results.append(fr)
    return results


def _md_report(
    results: list[FrameResult],
    backends: list[str],
    stamp: str,
    *,
    reference_label: str,
) -> str:
    ht = [r for r in results if r.hero_turn]  # genuine hero decision frames
    action_source = f"{reference_label} `is_actionable=true`"
    lines = [
        "# 本地多模态 vs Gemini vs CV — Poker Legends 识别评测",
        "",
        f"_生成于 {stamp}；共 {len(results)} 帧，其中 {len(ht)} 帧为真·hero 决策帧"
        f"({action_source})。box 误差与字段一致率**仅在这些决策帧上**统计"
        f"(box 真值=CV 检测中心；字段基线={reference_label})。延迟/解析率统计全部帧。_",
        "",
    ]
    lines += [
        "## 汇总(仅 hero 决策帧)",
        "",
        "| 后端 | 解析成功 | box 平均误差 | 漏检按钮 | "
        f"字段一致率(vs {reference_label}) | 平均延迟 |",
        "|---|---|---|---|---|---|",
    ]
    for be in backends:
        reads = [r.reads[be] for r in results if be in r.reads]
        oks = [r for r in reads if r.ok]
        parse = f"{len(oks)}/{len(reads)}"
        box_vals = [e for r in ht for e in (r.box_err.get(be) or {}).values() if e is not None]
        missed = sum(1 for r in ht for e in (r.box_err.get(be) or {}).values() if e is None)
        box_mean = f"{sum(box_vals) / len(box_vals):.0f}px" if box_vals else "—"
        fagree = [v[0] for r in ht for v in (r.fields.get(be) or {}).values()]
        fa = f"{100 * sum(fagree) / len(fagree):.0f}% (n={len(fagree)})" if fagree else "—"
        lat = f"{sum(r.latency for r in oks) / len(oks):.1f}s" if oks else "—"
        lines.append(f"| `{be}` | {parse} | {box_mean} | {missed} | {fa} | {lat} |")
    lines += [
        "",
        f"## 逐字段一致率(vs {reference_label}，仅 hero 决策帧)",
        "",
        "| 后端 | " + " | ".join(FIELD_FUNCS) + " |",
        "|---|" + "---|" * len(FIELD_FUNCS),
    ]
    for be in backends:
        if be == "cv" or (reference_label == "Gemini" and be.startswith("gemini")):
            continue
        cells = []
        for key in FIELD_FUNCS:
            vals = [r.fields[be][key][0] for r in ht if be in r.fields and key in r.fields[be]]
            cells.append(f"{100 * sum(vals) / len(vals):.0f}%" if vals else "—")
        lines.append(f"| `{be}` | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", default="", help="comma-separated frame paths")
    parser.add_argument("--frames-dir", default="", help="directory of *.png frames")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--backends", default="gemini,qwen/qwen3-vl-30b,cv")
    parser.add_argument("--base-url", default=DEFAULT_LMSTUDIO_URL)
    parser.add_argument("--max-edge", type=int, default=DEFAULT_MAX_EDGE)
    parser.add_argument("--reference-dir", default="", help="directory of reviewed truth JSON refs")
    parser.add_argument(
        "--timeout", type=float, default=300.0, help="local backend request timeout"
    )
    parser.add_argument("--out", default="")
    parser.add_argument("--stamp", default="(unstamped)", help="report timestamp string")
    args = parser.parse_args()

    from holdem_bot.vision.llm_annotation import _load_dotenv

    _load_dotenv(Path(".env"))  # GEMINI_API_KEY for the gemini backend

    frames: list[str] = [p for p in args.frames.split(",") if p]
    if args.frames_dir:
        frames += [str(p) for p in sorted(Path(args.frames_dir).glob("*.png"))][: args.limit]
    if not frames:
        parser.error("provide --frames or --frames-dir")
    backends = [b for b in args.backends.split(",") if b]

    results = run(
        frames,
        backends,
        args.base_url,
        args.max_edge,
        reference_dir=args.reference_dir,
        timeout=args.timeout,
    )
    reference_label = "reviewed truth" if args.reference_dir else "Gemini"
    report = _md_report(results, backends, args.stamp, reference_label=reference_label)
    print("\n" + report)
    if args.out:
        Path(args.out).write_text(report, encoding="utf-8")
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
