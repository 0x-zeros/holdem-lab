import json

import pytest
from holdem_ai.cfr import (
    CFRResult,
    build_arg_parser,
    main,
    nolimit_holdem_abstraction,
    train_cfr,
)


def test_cfr_plus_drives_kuhn_exploitability_toward_zero() -> None:
    result = train_cfr("kuhn_poker", iterations=100, variant="cfr_plus", eval_every=20)

    assert isinstance(result, CFRResult)
    assert result.checkpoints
    assert result.checkpoints[0].iteration == 20
    assert result.checkpoints[-1].iteration == 100
    # Exploitability falls across training and approaches the Nash value (~0).
    assert result.checkpoints[0].exploitability > result.final_exploitability
    assert result.final_exploitability < 0.01


def test_cfr_plus_beats_vanilla_cfr_at_equal_iterations() -> None:
    iters = 60
    plus = train_cfr("kuhn_poker", iterations=iters, variant="cfr_plus", eval_every=iters)
    vanilla = train_cfr("kuhn_poker", iterations=iters, variant="cfr", eval_every=iters)

    assert plus.final_exploitability < vanilla.final_exploitability


def test_train_cfr_validates_input() -> None:
    with pytest.raises(ValueError, match="iterations"):
        train_cfr("kuhn_poker", iterations=0)
    with pytest.raises(ValueError, match="variant"):
        train_cfr("kuhn_poker", iterations=5, variant="nope")


def test_result_to_dict_is_json_serializable() -> None:
    result = train_cfr("kuhn_poker", iterations=10, variant="cfr", eval_every=5)
    payload = json.loads(json.dumps(result.to_dict()))

    assert payload["game"] == "kuhn_poker"
    assert payload["variant"] == "cfr"
    assert payload["iterations"] == 10
    assert len(payload["checkpoints"]) == 2


def test_cli_outputs_exploitability_json(capsys: pytest.CaptureFixture[str]) -> None:
    main(
        [
            "--game",
            "kuhn_poker",
            "--iterations",
            "40",
            "--variant",
            "cfr_plus",
            "--eval-every",
            "20",
        ]
    )

    report = json.loads(capsys.readouterr().out)
    assert report["game"] == "kuhn_poker"
    assert report["final_exploitability"] < 0.1


def test_arg_parser_defaults() -> None:
    args = build_arg_parser().parse_args([])
    assert args.variant == "cfr_plus"
    assert args.iterations == 200


def test_cfr_solves_small_nolimit_holdem_abstraction() -> None:
    # A genuine no-limit hold'em abstraction (fcpa betting), reduced enough to be
    # tractable for tabular CFR. Exploitability must fall toward the Nash value.
    game = nolimit_holdem_abstraction(suits=2, ranks=3, hole_cards=1, stack=6)
    result = train_cfr(game, iterations=50, variant="cfr_plus", eval_every=25)

    assert result.checkpoints[0].exploitability > result.final_exploitability
    assert result.final_exploitability < 0.05


def test_nlhe_abstraction_string_is_nolimit_fcpa() -> None:
    game = nolimit_holdem_abstraction()
    assert "betting=nolimit" in game
    assert "bettingAbstraction=fcpa" in game
