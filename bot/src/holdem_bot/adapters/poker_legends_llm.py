"""LLM-based Poker Legends recognizer.

A vision LLM reads the full screenshot into the ``annotation_output_schema`` JSON, which
the validated ``recognize_from_llm_annotation`` assembly turns into a fail-closed
``GameState``. The LLM also reports the game's bounding box (``game_region``); after the
first read the recognizer crops every later frame to that box, so only the game area --
not the whole desktop -- is sent to the model. The exact image submitted to the LLM is
exposed via ``metadata['submitted_image']`` so the HUD can draw on what the model saw.

The LLM call is injectable, so the assembly is testable offline without an API key.
"""

from __future__ import annotations

import copy
import json
import tempfile
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, cast

import cv2
import numpy as np
from numpy.typing import NDArray

from holdem_bot.adapters.poker_legends import PokerLegendsTableRecognizer
from holdem_bot.capture import CapturedFrame
from holdem_bot.recognize import RecognitionResult
from holdem_bot.vision.llm_annotation import annotation_output_schema

BgrImage = NDArray[np.uint8]

DEFAULT_GEMINI_MODEL = "gemini-3.1-flash-lite"
#: Longest-edge cap (px) for frames sent to the LLM. Downscaling cuts upload size for lower
#: latency; reads stay accurate down to ~1024px. 0 disables resizing.
DEFAULT_MAX_EDGE = 1280
#: Fraction to expand the located game box on each side before cropping (safety margin).
DEFAULT_MARGIN = 0.04
#: Re-locate the game box after this many consecutive non-table frames (window moved/closed).
DEFAULT_RELOCATE_AFTER = 3

RUNTIME_PROMPT = (
    "You are the perception module of a Texas Hold'em bot playing Poker Legends. Return ONLY JSON "
    "matching the schema for the CURRENT table.\n"
    "HERO: the human player you control sits at the BOTTOM-CENTER and is the ONLY seat showing two "
    "face-up hole cards. Put those two cards in hero_hole_cards. In 'seats', that seat has "
    "name='hero', stack = the chip number shown directly BELOW hero's two cards, and current=true "
    "only when in-hand action buttons (Call/Raise/Fold) are visible.\n"
    "SEATS: list EVERY seated player (up to 6). 'name' = the player's name-plate text "
    "(use 'hero' for the hero seat). stack = the chip number on that player's plate. "
    "committed = chips pushed in front of that player THIS round (blinds/calls; 0 if none). "
    "current=true ONLY for the player whose turn it is now. position = sb/bb/button when a "
    "marker is shown, else null.\n"
    "POT: the pot is the shared chip total in the CENTER of the table. NEVER use a player's "
    "stack as the pot. If no central pot number is visible, set the pot text "
    "normalized_number to the SUM of all seats' committed chips (never 0 during a live hand).\n"
    "GAME_REGION: the bounding box {x,y,width,height} of the WHOLE game window: from the "
    "LEFTMOST game element to the RIGHTMOST one. The action buttons and any side rail/panel "
    "on the right ARE inside the window, so the right edge MUST be at or past the rightmost "
    "button. Span the full game UI top-to-bottom; exclude ONLY the OS menu bar and the "
    "desktop wallpaper outside the window. null only if no game window is visible.\n"
    "Cards: rank+suit S,H,D,C (AS, TD, 7H); null + an 'uncertain' entry if unclear. Normalize chip "
    "numbers (drop $ and commas; ignore '+N'). Buttons: name in {call,raise,fold,check,bet,all_in} "
    "with the visible label (e.g. 'Call $10'). table_state.is_actionable=true only when hero can "
    "choose an in-hand poker action now (modals/lobbies are not actionable)."
)

#: Focused prompt for the one-time game-window locate call. The full annotation prompt has too
#: many tasks for the model to reliably box the whole window (it defaults to the felt), so the
#: locate is a separate single-purpose call.
LOCATE_PROMPT = (
    "Return game_region = the pixel bounding box {x,y,width,height} of the WHOLE game window: "
    "from the LEFTMOST game element to the RIGHTMOST one. The action buttons and any side "
    "rail/panel on the right ARE inside the window, so the right edge MUST be at or past the "
    "rightmost button. Top and bottom span the full game UI; exclude ONLY the OS menu bar and "
    "the desktop wallpaper outside the window."
)

#: A frame reader maps a prepared (cropped + downscaled) BGR image to the LLM annotation.
FrameReader = Callable[[BgrImage], Mapping[str, object]]


def runtime_annotation_schema() -> dict[str, object]:
    """The annotation schema plus blinds, seat position, and the game-region box."""
    schema = copy.deepcopy(annotation_output_schema())
    props = cast(dict[str, Any], schema["properties"])
    table_state = cast(dict[str, Any], props["table_state"])
    table_state["properties"]["small_blind"] = {"type": ["integer", "null"]}
    table_state["properties"]["big_blind"] = {"type": ["integer", "null"]}
    table_state["required"] = [*table_state["required"], "small_blind", "big_blind"]
    seat_items = cast(dict[str, Any], cast(dict[str, Any], props["seats"])["items"])
    seat_items["properties"]["position"] = {"type": ["string", "null"]}
    seat_items["required"] = [*seat_items["required"], "position"]
    props["game_region"] = {
        "type": ["object", "null"],
        "additionalProperties": False,
        "required": ["x", "y", "width", "height"],
        "properties": {
            "x": {"type": "integer"},
            "y": {"type": "integer"},
            "width": {"type": "integer"},
            "height": {"type": "integer"},
        },
    }
    schema["required"] = [*cast(list[str], schema["required"]), "game_region"]
    return schema


def _downscale_image(image: BgrImage, max_edge: int) -> BgrImage:
    """Resize so the longest edge is <= max_edge (0 = no resize)."""
    if max_edge <= 0:
        return image
    height, width = image.shape[:2]
    longest = max(height, width)
    if longest <= max_edge:
        return image
    scale = max_edge / longest
    resized = cv2.resize(
        image, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA
    )
    return cast(BgrImage, resized)


def read_image_with_gemini(
    image: BgrImage,
    *,
    model: str = DEFAULT_GEMINI_MODEL,
    api_key: str | None = None,
) -> dict[str, object]:
    """Send one prepared BGR image to Gemini and return the parsed annotation (needs a key)."""
    import os

    from google import genai
    from google.genai import types as genai_types

    key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise RuntimeError("GEMINI_API_KEY or GOOGLE_API_KEY is required for Gemini recognition")
    ok, buffer = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError("failed to PNG-encode the frame for Gemini")
    client = genai.Client(api_key=key)
    response = cast(Any, client.models).generate_content(
        model=model,
        contents=[
            RUNTIME_PROMPT,
            genai_types.Part.from_bytes(data=bytes(buffer.tobytes()), mime_type="image/png"),
        ],
        config=genai_types.GenerateContentConfig(
            response_mime_type="application/json",
            response_json_schema=runtime_annotation_schema(),
            temperature=0,
            max_output_tokens=4000,
        ),
    )
    return _parse_annotation(str(getattr(response, "text", "")), "frame")


def read_frame_with_gemini(
    image_path: Path,
    *,
    model: str = DEFAULT_GEMINI_MODEL,
    api_key: str | None = None,
    max_edge: int = DEFAULT_MAX_EDGE,
) -> dict[str, object]:
    """Load a saved frame, downscale to max_edge, and read it with Gemini (host-side helper)."""
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"could not read image: {image_path}")
    return read_image_with_gemini(
        _downscale_image(cast(BgrImage, image), max_edge), model=model, api_key=api_key
    )


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


def _table_state(annotation: Mapping[str, object]) -> Mapping[str, object]:
    table_state = annotation.get("table_state")
    return table_state if isinstance(table_state, Mapping) else {}


def _fraction_from_box(box: Mapping[str, object], height: int, width: int) -> _Region | None:
    """Convert a pixel box (in an image of the given size) to clamped (fx, fy, fw, fh) fractions."""
    if width <= 0 or height <= 0:
        return None
    try:
        x = float(cast(float, box["x"]))
        y = float(cast(float, box["y"]))
        box_w = float(cast(float, box["width"]))
        box_h = float(cast(float, box["height"]))
    except (KeyError, TypeError, ValueError):
        return None
    if box_w <= 0 or box_h <= 0:
        return None
    fx = min(max(x / width, 0.0), 1.0)
    fy = min(max(y / height, 0.0), 1.0)
    fw = min(box_w / width, 1.0 - fx)
    fh = min(box_h / height, 1.0 - fy)
    if fw <= 0.0 or fh <= 0.0:
        return None
    return (fx, fy, fw, fh)


def _crop_to_fraction(image: BgrImage, region: _Region, margin: float) -> BgrImage:
    """Crop to the fractional region, expanded by margin on each side and clamped to the image."""
    height, width = image.shape[:2]
    fx, fy, fw, fh = region
    fx = max(fx - margin, 0.0)
    fy = max(fy - margin, 0.0)
    fw = min(fw + 2.0 * margin, 1.0 - fx)
    fh = min(fh + 2.0 * margin, 1.0 - fy)
    x0 = int(round(fx * width))
    y0 = int(round(fy * height))
    x1 = min(width, int(round((fx + fw) * width)))
    y1 = min(height, int(round((fy + fh) * height)))
    if x1 - x0 < 1 or y1 - y0 < 1:
        return image
    return image[y0:y1, x0:x1]


_Region = tuple[float, float, float, float]
#: A region locator maps a full frame to the game-window box as (fx, fy, fw, fh) fractions.
RegionLocator = Callable[[BgrImage], _Region | None]


def _locate_schema() -> dict[str, object]:
    box = {
        "type": "object",
        "additionalProperties": False,
        "required": ["x", "y", "width", "height"],
        "properties": {key: {"type": "integer"} for key in ("x", "y", "width", "height")},
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["game_region"],
        "properties": {"game_region": box},
    }


def locate_region_with_gemini(
    image: BgrImage, *, model: str = DEFAULT_GEMINI_MODEL, api_key: str | None = None
) -> _Region | None:
    """One focused Gemini call returning the game-window box as (fx, fy, fw, fh) fractions."""
    import os

    from google import genai
    from google.genai import types as genai_types

    key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise RuntimeError("GEMINI_API_KEY or GOOGLE_API_KEY is required for Gemini recognition")
    ok, buffer = cv2.imencode(".png", image)
    if not ok:
        return None
    height, width = image.shape[:2]
    prompt = f"This is a {width}x{height} screenshot of the Poker Legends game. " + LOCATE_PROMPT
    client = genai.Client(api_key=key)
    response = cast(Any, client.models).generate_content(
        model=model,
        contents=[
            prompt,
            genai_types.Part.from_bytes(data=bytes(buffer.tobytes()), mime_type="image/png"),
        ],
        config=genai_types.GenerateContentConfig(
            response_mime_type="application/json",
            response_json_schema=_locate_schema(),
            temperature=0,
            max_output_tokens=500,
        ),
    )
    data = _parse_annotation(str(getattr(response, "text", "")), "locate")
    box = data.get("game_region")
    if isinstance(box, Mapping):
        return _fraction_from_box(box, height, width)
    return None


class PokerLegendsLlmRecognizer:
    """LLM table reader: crops to the located game box and exposes the submitted image."""

    def __init__(
        self,
        *,
        reader: FrameReader,
        recognizer: PokerLegendsTableRecognizer,
        max_edge: int = DEFAULT_MAX_EDGE,
        margin: float = DEFAULT_MARGIN,
        relocate_after: int = DEFAULT_RELOCATE_AFTER,
        submitted_path: str | Path | None = None,
        crop: bool = True,
        locator: RegionLocator | None = None,
    ) -> None:
        self._reader = reader
        self._locator = locator
        self._recognizer = recognizer
        self._max_edge = max_edge
        self._margin = margin
        self._relocate_after = relocate_after
        self._crop = crop
        self._submitted_path = (
            Path(submitted_path)
            if submitted_path is not None
            else Path(tempfile.gettempdir()) / "holdem_submitted.png"
        )
        self._region: _Region | None = None
        self._miss = 0

    @classmethod
    def gemini(
        cls,
        *,
        controlled_seat: int = 0,
        small_blind: int = 5,
        big_blind: int = 10,
        model: str = DEFAULT_GEMINI_MODEL,
        api_key: str | None = None,
        max_edge: int = DEFAULT_MAX_EDGE,
        margin: float = DEFAULT_MARGIN,
        crop: bool = True,
    ) -> PokerLegendsLlmRecognizer:
        def reader(image: BgrImage) -> Mapping[str, object]:
            return read_image_with_gemini(image, model=model, api_key=api_key)

        def locator(image: BgrImage) -> _Region | None:
            return locate_region_with_gemini(image, model=model, api_key=api_key)

        return cls(
            reader=reader,
            locator=locator,
            recognizer=PokerLegendsTableRecognizer.for_llm(
                controlled_seat=controlled_seat, small_blind=small_blind, big_blind=big_blind
            ),
            max_edge=max_edge,
            margin=margin,
            crop=crop,
        )

    def recognize(self, frame: CapturedFrame) -> RecognitionResult:
        image_path = _image_path(frame)
        full = cv2.imread(str(image_path))
        if full is None:
            raise FileNotFoundError(f"could not read image: {image_path}")
        full_bgr = cast(BgrImage, full)
        if self._crop and self._region is None and self._locator is not None:
            self._region = self._locator(_downscale_image(full_bgr, self._max_edge))
        region = self._region if self._crop else None
        cropped = (
            _crop_to_fraction(full_bgr, region, self._margin) if region is not None else full_bgr
        )
        submitted = _downscale_image(cropped, self._max_edge)
        annotation = self._reader(submitted)
        if self._crop and region is not None:
            self._note_outcome(annotation)
        cv2.imwrite(str(self._submitted_path), submitted)
        result = self._recognizer.recognize_from_llm_annotation(
            annotation, image=str(self._submitted_path), frame_id=image_path.stem
        )
        metadata = dict(result.metadata)
        metadata["submitted_image"] = str(self._submitted_path)
        metadata["game_region_fraction"] = list(self._region) if self._region is not None else None
        return RecognitionResult(
            state=result.state,
            confidence=result.confidence,
            metadata=metadata,
            screen=result.screen,
        )

    def _note_outcome(self, annotation: Mapping[str, object]) -> None:
        """Drop the cached box after several non-table frames in a row (window moved/closed)."""
        if _table_state(annotation).get("is_table"):
            self._miss = 0
        else:
            self._miss += 1
            if self._miss >= self._relocate_after:
                self._region = None
                self._miss = 0


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
