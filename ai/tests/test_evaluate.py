import json
from typing import cast

import pytest
from holdem_ai.evaluate import (
    evaluate_heads_up,
    evaluate_match,
    evaluate_profile_matrix,
    main,
    profile_from_name,
)


def test_evaluate_heads_up_returns_balanced_profile_stats() -> None:
    result = evaluate_heads_up(
        profile_from_name("current"),
        profile_from_name("no_equity"),
        hands=4,
        seed=7,
        starting_stack=100,
    )
    report = result.to_dict()
    profiles = cast(dict[str, dict[str, object]], report["profiles"])
    current = profiles["current"]
    no_equity = profiles["no_equity"]

    assert current["hands"] == 4
    assert no_equity["hands"] == 4
    assert cast(int, current["chips"]) + cast(int, no_equity["chips"]) == 0
    assert current["actions"]
    assert no_equity["actions"]


def test_evaluate_heads_up_rejects_duplicate_profile_names() -> None:
    with pytest.raises(ValueError, match="different"):
        evaluate_heads_up(profile_from_name("current"), profile_from_name("current"), hands=1)


def test_evaluate_cli_outputs_json(capsys: pytest.CaptureFixture[str]) -> None:
    main(["--hands", "2", "--seed", "3", "--profile-a", "current", "--profile-b", "tight"])

    report = json.loads(capsys.readouterr().out)
    assert report["hands"] == 2
    assert set(report["profiles"]) == {"current", "tight"}


def test_evaluate_profile_matrix_returns_pairings_and_leaderboard() -> None:
    result = evaluate_profile_matrix(("current", "no_equity", "tight"), hands=2, seed=5)
    report = result.to_dict()
    pairings = cast(dict[str, object], report["pairings"])
    leaderboard = cast(list[dict[str, object]], report["leaderboard"])

    assert set(pairings) == {
        "current_vs_no_equity",
        "current_vs_tight",
        "no_equity_vs_tight",
    }
    assert [entry["profile"] for entry in leaderboard]
    assert {entry["profile"] for entry in leaderboard} == {
        "current",
        "no_equity",
        "tight",
    }


def test_evaluate_matrix_cli_outputs_json(capsys: pytest.CaptureFixture[str]) -> None:
    main(["--matrix", "current", "no_equity", "tight", "--hands", "2", "--seed", "9"])

    report = json.loads(capsys.readouterr().out)
    assert report["hands_per_pairing"] == 2
    assert len(report["pairings"]) == 3
    assert len(report["leaderboard"]) == 3


def test_evaluate_match_balances_positions_and_brackets_estimate() -> None:
    report = evaluate_match(
        profile_from_name("current"),
        profile_from_name("rock"),
        pairs=60,
        seed=3,
        starting_stack=20,
        bootstrap=500,
    )
    assert report.hands == 120  # 2 * pairs
    # Each pair plays the focal on the button once and the big blind once.
    assert report.by_position["button"]["hands"] == 60
    assert report.by_position["big_blind"]["hands"] == 60
    # The bootstrap interval brackets the point estimate.
    assert report.ci_low <= report.bb_per_100 <= report.ci_high
    # current crushes a rock at 10bb effective (steals relentlessly).
    assert report.bb_per_100 > 0


def test_evaluate_match_is_crn_zero_sum_under_label_swap() -> None:
    # The two mirrored hands per pair are the same physical games regardless of
    # which side we call "focal", so the focal chips just negate when swapped.
    forward = evaluate_match(
        profile_from_name("current"), profile_from_name("rock"), pairs=40, seed=11, bootstrap=1
    )
    reverse = evaluate_match(
        profile_from_name("rock"), profile_from_name("current"), pairs=40, seed=11, bootstrap=1
    )
    assert forward.focal_chips == -reverse.focal_chips


def test_crn_invariance_holds_for_equity_driven_policies() -> None:
    # tag and maniac drive postflop showdowns (estimate_showdown_equity sampling),
    # so this only negates exactly if both mirrored halves hash to the SAME policy
    # RNG. It failed when the per-hand hand_id encoded focal_seat.
    forward = evaluate_match(
        profile_from_name("tag"), profile_from_name("maniac"), pairs=30, seed=4, bootstrap=1
    )
    reverse = evaluate_match(
        profile_from_name("maniac"), profile_from_name("tag"), pairs=30, seed=4, bootstrap=1
    )
    assert forward.focal_chips == -reverse.focal_chips


def test_evaluate_match_is_reproducible() -> None:
    kwargs = dict(pairs=40, seed=5, starting_stack=20, bootstrap=300)
    first = evaluate_match(profile_from_name("current"), profile_from_name("maniac"), **kwargs)
    second = evaluate_match(profile_from_name("current"), profile_from_name("maniac"), **kwargs)
    assert first.bb_per_100 == second.bb_per_100
    assert (first.ci_low, first.ci_high) == (second.ci_low, second.ci_high)


def test_evaluate_match_rejects_duplicate_names() -> None:
    with pytest.raises(ValueError, match="different"):
        evaluate_match(profile_from_name("current"), profile_from_name("current"), pairs=1)


def test_match_cli_outputs_json(capsys: pytest.CaptureFixture[str]) -> None:
    main(["--match", "current", "rock", "--pairs", "20", "--seed", "4", "--bootstrap", "200"])

    report = json.loads(capsys.readouterr().out)
    assert report["focal"] == "current"
    assert report["opponent"] == "rock"
    assert report["hands"] == 40
    assert report["ci95_low"] <= report["bb_per_100"] <= report["ci95_high"]
