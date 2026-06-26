from holdem_common import ActionType, Card
from holdem_engine import HoldemConfig, HoldemEnv
from holdem_engine.adapters import legal_action_ids, rlcard_action_to_action, to_rlcard_observation
from rlcard.games.nolimitholdem.round import Action as RLCardAction  # type: ignore[import-untyped]


def cards(*codes: str) -> tuple[Card, ...]:
    return tuple(Card.from_code(code) for code in codes)


def test_rlcard_observation_encodes_visible_cards_and_chips() -> None:
    env = HoldemEnv(
        HoldemConfig(
            starting_stacks=(100, 100, 100),
            deck=cards("As", "Ks", "Qs", "Ah", "Kh", "Qh"),
        ),
    )
    state = env.reset()

    observation = to_rlcard_observation(state, seat=state.current_seat)

    assert observation.obs.shape == (54,)
    assert observation.raw_obs["current_player"] == state.current_seat
    assert observation.raw_obs["pot"] == state.pot_total
    assert observation.obs[0] == 0.0
    assert observation.obs[12] == 0.0
    assert observation.obs[52] == float(state.player(state.current_seat or 0).committed)
    assert observation.obs[53] == 2.0


def test_rlcard_observation_uses_seat_private_cards() -> None:
    env = HoldemEnv(
        HoldemConfig(
            starting_stacks=(100, 100, 100),
            deck=cards("As", "Ks", "Qs", "Ah", "Kh", "Qh"),
        ),
    )
    state = env.reset()

    observation = to_rlcard_observation(state, seat=0)

    assert observation.obs[0] == 1.0
    assert observation.obs[13] == 1.0
    assert observation.raw_obs["hand"] == ["SA", "HA"]


def test_rlcard_legal_action_ids_and_decode_call_all_in() -> None:
    env = HoldemEnv(HoldemConfig(starting_stacks=(100, 100, 100)))
    state = env.reset(seed=1)

    legal_ids = legal_action_ids(state)

    assert RLCardAction.FOLD.value in legal_ids
    assert RLCardAction.CHECK_CALL.value in legal_ids
    assert RLCardAction.ALL_IN.value in legal_ids
    assert rlcard_action_to_action(RLCardAction.CHECK_CALL, state).action_type is ActionType.CALL
    assert rlcard_action_to_action(RLCardAction.CHECK_CALL, state).amount == state.to_call
    assert rlcard_action_to_action(RLCardAction.ALL_IN, state).action_type is ActionType.ALL_IN


def test_rlcard_raise_action_decodes_to_total_commitment_amount() -> None:
    env = HoldemEnv(HoldemConfig(starting_stacks=(100, 100, 100)))
    state = env.reset(seed=1)

    action = rlcard_action_to_action(RLCardAction.RAISE_POT, state)

    assert action.action_type is ActionType.RAISE
    assert action.amount == action.min_amount
    assert action.amount >= state.min_raise
