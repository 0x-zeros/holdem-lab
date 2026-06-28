"""LLM-based Poker Legends recognizer.

Instead of brittle template CV, a vision LLM reads the full screenshot into the
``annotation_output_schema`` JSON, which the validated ``recognize_from_llm_annotation``
assembly turns into a fail-closed ``GameState``. The LLM call is injectable so the
assembly is testable offline without any API key.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, cast

from holdem_bot.adapters.poker_legends import PokerLegendsTableRecognizer
from holdem_bot.capture import CapturedFrame
from holdem_bot.recognize import RecognitionResult
from holdem_bot.vision.llm_annotation import annotation_output_schema

DEFAULT_GEMINI_MODEL = "gemini-3.1-flash-lite"

RUNTIME_PROMPT = (
    "You are the perception module of a Texas Hold'em bot playing Poker Legends. "
    "Look at the full screenshot and return ONLY JSON matching the schema. Read the hero's "
    "two hole cards (face-up, bottom-center), any community/board cards, the pot, each seat's "
    "stack and committed chips, the small/big blind amounts, and the action buttons. Card codes "
    "are rank+suit using S,H,D,C (e.g. AS, TD, 7H); use null and add an 'uncertain' entry when a "
    "card or number is unclear. Normalize chip numbers (drop $ and commas; ignore '+N' deltas). "
    "Name the human player's own seat exactly 'hero' and set its 'current'=true only when it is "
    "hero's turn to act (Call/Raise/Fold buttons are shown). Give each other seat a short name and "
    "a 'position' when a marker is visible (sb, bb, button). For buttons, name is one of "
    "{call,raise,fold,check,bet,all_in} with the visible label (e.g. 'Call $10'). "
    "table_state.is_actionable is true only when hero can choose an in-hand poker action now."
)

#: A frame reader maps an image path to the LLM annotation dict.
FrameReader = Callable[[Path], Mapping[str, object]]


def runtime_annotation_schema() -> dict[str, object]:
    """The annotation schema plus blinds (table_state) and seat position for live play."""
    schema = copy.deepcopy(annotation_output_schema())
    props = cast(dict[str, Any], schema["properties"])
    table_state = cast(dict[str, Any], props["table_state"])
    table_state["properties"]["small_blind"] = {"type": ["integer", "null"]}
    table_state["properties"]["big_blind"] = {"type": ["integer", "null"]}
    table_state["required"] = [*table_state["required"], "small_blind", "big_blind"]
    seat_items = cast(dict[str, Any], cast(dict[str, Any], props["seats"])["items"])
    seat_items["properties"]["position"] = {"type": ["string", "null"]}
    seat_items["required"] = [*seat_items["required"], "position"]
    return schema


def read_frame_with_gemini(
    image_path: Path,
    *,
    model: str = DEFAULT_GEMINI_MODEL,
    api_key: str | None = None,
) -> dict[str, object]:
    """Send one full frame to Gemini and return the parsed annotation (host-side; needs a key)."""
    import os

    from google import genai
    from google.genai import types as genai_types

    key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise RuntimeError("GEMINI_API_KEY or GOOGLE_API_KEY is required for Gemini recognition")
    client = genai.Client(api_key=key)
    response = cast(Any, client.models).generate_content(
        model=model,
        contents=[
            RUNTIME_PROMPT,
            genai_types.Part.from_bytes(data=image_path.read_bytes(), mime_type="image/png"),
        ],
        config=genai_types.GenerateContentConfig(
            response_mime_type="application/json",
            response_json_schema=runtime_annotation_schema(),
            temperature=0,
            max_output_tokens=4000,
        ),
    )
    return _parse_annotation(str(getattr(response, "text", "")), image_path.stem)


def _parse_annotation(raw_text: str, frame_id: str) -> dict[str, object]:
    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        return {"frame_id": frame_id, "parse_error": str(exc), "table_state": {}}
    if isinstance(data, dict):
        return cast(dict[str, object], data)
    return {
        "frame_id": frame_id,
        "parse_error": "response was not a JSON object",
        "table_state": {},
    }


class PokerLegendsLlmRecognizer:
    """Read the table state with a vision LLM, then assemble a fail-closed GameState."""

    def __init__(self, *, reader: FrameReader, recognizer: PokerLegendsTableRecognizer) -> None:
        self._reader = reader
        self._recognizer = recognizer

    @classmethod
    def gemini(
        cls,
        *,
        controlled_seat: int = 0,
        small_blind: int = 5,
        big_blind: int = 10,
        model: str = DEFAULT_GEMINI_MODEL,
        api_key: str | None = None,
    ) -> PokerLegendsLlmRecognizer:
        def reader(path: Path) -> Mapping[str, object]:
            return read_frame_with_gemini(path, model=model, api_key=api_key)

        return cls(
            reader=reader,
            recognizer=PokerLegendsTableRecognizer.for_llm(
                controlled_seat=controlled_seat, small_blind=small_blind, big_blind=big_blind
            ),
        )

    def recognize(self, frame: CapturedFrame) -> RecognitionResult:
        image_path = _image_path(frame)
        annotation = self._reader(image_path)
        return self._recognizer.recognize_from_llm_annotation(
            annotation, image=str(image_path), frame_id=image_path.stem
        )


def _image_path(frame: CapturedFrame) -> Path:
    payload = frame.payload
    if isinstance(payload, str | Path):
        path = Path(payload)
        if path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
            return path
    meta = frame.metadata.get("poker_legends_image_path")
    if isinstance(meta, str):
        return Path(meta)
    raise ValueError("CapturedFrame has no image path for LLM recognition")
