"""Tests for the local-VLM eval harness scoring (the empirically learned parsing quirks)."""

from __future__ import annotations

from pathlib import Path

from holdem_bot.eval.local_vlm import (
    Read,
    _button_centers_px,
    _canon,
    _cards,
    _extract_json,
    _load_reference,
    _norm_card,
    _pot,
    _seats,
    compare_fields,
)


def test_extract_json_tolerates_fence_and_trailing_prose() -> None:
    raw = '```json\n{"a": 1, "b": [2, 3]}\n```\nthat is my answer.'
    assert _extract_json(raw) == {"a": 1, "b": [2, 3]}


def test_canon_lowercases_top_level_keys() -> None:
    # local models echo the prompt's section headers (Buttons / GAME_REGION) as keys
    assert _canon({"Buttons": [1], "GAME_REGION": {}, "seats": []}) == {
        "buttons": [1],
        "game_region": {},
        "seats": [],
    }


def test_norm_card_handles_suit_glyphs_and_ten() -> None:
    assert _norm_card("5♥") == "5H"
    assert _norm_card("J♠") == "JS"
    assert _norm_card("10c") == "TC"
    assert _norm_card("AS") == "AS"


def test_cards_normalises_canonical_objects_and_bare_strings_equally() -> None:
    canonical = [{"slot": "a", "visible": True, "card": "JS", "confidence": 1.0},
                 {"slot": "b", "visible": True, "card": "5H", "confidence": 1.0}]
    glyph_strings = ["J♠", "5♥"]
    assert _cards(canonical) == _cards(glyph_strings) == frozenset({"JS", "5H"})
    # hidden / null slots drop out
    assert _cards([{"slot": "a", "visible": False, "card": None, "confidence": 0.0}]) == frozenset()


def test_button_centers_decodes_per_backend_axis_order() -> None:
    box = {"name": "fold", "box_2d": [840, 850, 940, 950]}
    qwen = Read("local", "qwen/qwen3-vl-30b", True, 0.0, {"buttons": [box]})
    gemini = Read("gemini", "gemini-3.1-flash-lite", True, 0.0, {"buttons": [box]})
    # Qwen native xyxy -> x=(840+940)/2, y=(850+950)/2
    assert _button_centers_px(qwen, 1000, 1000)["fold"] == (890.0, 900.0)
    # Gemini yxyx -> x=(850+950)/2, y=(840+940)/2
    assert _button_centers_px(gemini, 1000, 1000)["fold"] == (900.0, 890.0)


def test_button_centers_uses_cv_pixels_directly() -> None:
    cv = Read("cv", "cv", True, 0.0, {"buttons": [{"name": "call", "_px": [100, 200]}]})
    assert _button_centers_px(cv, 1396, 814)["call"] == (100.0, 200.0)


def test_pot_from_top_level_or_texts() -> None:
    assert _pot({"pot": {"normalized_number": 65}}) == 65  # local-model shape
    assert _pot({"texts": [{"name": "Pot", "normalized_number": 120}]}) == 120  # canonical shape
    assert _pot({"texts": [{"name": "blinds", "normalized_number": 10}]}) is None


def test_seats_filters_empty_seats() -> None:
    ann = {"seats": [{"name": "a", "stack": 990}, {"name": "b", "stack": 0},
                     {"name": "c", "stack": None}, {"name": "d", "stack": 1052}]}
    assert len(_seats(ann)) == 2


def test_compare_fields_glyph_cards_match_canonical() -> None:
    local = {"hero_hole_cards": ["J♠", "5♥"], "table_state": {"is_actionable": True}}
    ref = {"hero_hole_cards": [{"slot": "a", "visible": True, "card": "JS", "confidence": 1.0},
                               {"slot": "b", "visible": True, "card": "5H", "confidence": 1.0}],
           "table_state": {"is_actionable": True}}
    out = compare_fields(local, ref)
    assert out["hero_cards"][0] is True
    assert out["actionable"][0] is True


def test_load_reference_matches_frame_stem(tmp_path: Path) -> None:
    reference_dir = tmp_path / "refs"
    reference_dir.mkdir()
    (reference_dir / "frame_001.json").write_text('{"Buttons": [], "table_state": {}}')

    assert _load_reference("/frames/frame_001.png", str(reference_dir)) == {
        "buttons": [],
        "table_state": {},
    }
    assert _load_reference("/frames/missing.png", str(reference_dir)) is None
