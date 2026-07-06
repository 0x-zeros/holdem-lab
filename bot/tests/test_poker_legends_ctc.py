from pathlib import Path

import pytest
from holdem_bot.vision.poker_legends_ctc import (
    CtcAlignmentError,
    CtcAlphabet,
    ctc_required_timesteps,
    validate_ctc_log_probs_shape,
    validate_ctc_time_step_budget,
)
from holdem_bot.vision.poker_legends_ctc_sanity import run_poker_legends_ctc_sanity
from holdem_bot.vision.poker_legends_number_chars import (
    StringPrediction,
    apply_number_text_target_contract,
)


def test_ctc_alphabet_keeps_blank_out_of_targets_and_decodes_repeats() -> None:
    alphabet = CtcAlphabet.from_texts(["$1000", "+55"])

    encoded = alphabet.encode_target("$1000")

    assert alphabet.blank_index == 0
    assert encoded.target_length == 5
    assert encoded.required_timesteps == 7
    assert 0 not in encoded.indices

    label_to_index = alphabet.label_to_index()
    path = [
        label_to_index["$"],
        label_to_index["$"],
        alphabet.blank_index,
        label_to_index["1"],
        label_to_index["0"],
        alphabet.blank_index,
        label_to_index["0"],
        alphabet.blank_index,
        label_to_index["0"],
    ]
    text, confidences = alphabet.greedy_decode(path, [0.9] * len(path))

    assert text == "$1000"
    assert confidences == (0.9, 0.9, 0.9, 0.9, 0.9)


def test_ctc_required_timesteps_accounts_for_adjacent_repeats() -> None:
    assert ctc_required_timesteps("1000") == 6
    assert ctc_required_timesteps("$1000") == 7
    assert ctc_required_timesteps("+1000") == 7
    assert ctc_required_timesteps("$43,044") == 8
    assert ctc_required_timesteps("+55") == 4


def test_ctc_time_step_budget_fails_loudly_when_input_is_too_short() -> None:
    with pytest.raises(CtcAlignmentError, match="CTC time-step budget failed"):
        validate_ctc_time_step_budget("$1000", input_timesteps=13)

    budget = validate_ctc_time_step_budget("$1000", input_timesteps=14)

    assert budget.valid
    assert budget.required_timesteps == 7
    assert budget.ratio == 2.0


def test_ctc_log_probs_shape_requires_tnc_layout_and_blank_class() -> None:
    shape = validate_ctc_log_probs_shape((40, 2, 8), batch_size=2, class_count=8)

    assert shape.timesteps == 40
    assert shape.batch_size == 2
    assert shape.class_count == 8

    with pytest.raises(CtcAlignmentError, match="batch dimension"):
        validate_ctc_log_probs_shape((2, 40, 8), batch_size=2, class_count=8)

    with pytest.raises(CtcAlignmentError, match="include blank"):
        validate_ctc_log_probs_shape((40, 2, 7), batch_size=2, class_count=8)


def test_ctc_sanity_runner_writes_offline_summary(tmp_path: Path) -> None:
    summary = run_poker_legends_ctc_sanity(
        output_dir=tmp_path,
        targets=("base",),
        synthetic_count=4,
        epochs=1,
        batch_size=2,
    )

    assert (tmp_path / "ctc_sanity_summary.json").exists()
    assert summary["too_short_case"] == {
        "error": (
            "CTC time-step budget failed for '$1000': T=13, "
            "target_len=5, required=7, ratio=1.857, "
            "reason=insufficient_timesteps"
        ),
        "failed_loudly": True,
        "input_timesteps": 13,
        "text": "$1000",
    }
    datasets = summary["datasets"]
    assert isinstance(datasets, list)
    assert len(datasets) == 1
    assert datasets[0]["name"] == "synthetic_base"
    assert datasets[0]["sample_count"] == 4
    assert datasets[0]["time_step_budget"]["failed"] == 0


def test_ctc_target_contract_rejects_incomplete_stack_text() -> None:
    prediction = StringPrediction(
        method="crnn_ctc",
        text="$",
        confidence=0.95,
        accepted=True,
        reason="accepted",
        char_confidences=(0.95,),
    )

    contracted = apply_number_text_target_contract(
        prediction,
        target="base",
        is_stack=True,
    )

    assert not contracted.accepted
    assert contracted.reason == "format"
