import json
from typing import cast

import pytest
from holdem_ai.evaluate import evaluate_heads_up, main, profile_from_name


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
