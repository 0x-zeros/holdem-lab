"""Adapters between holdem-lab states and external poker/AI libraries."""

from holdem_engine.adapters.openspiel import (
    OpenSpielAction,
    OpenSpielObservation,
    openspiel_action_to_action,
    openspiel_legal_action_ids,
    registered_poker_games,
    to_openspiel_observation,
)
from holdem_engine.adapters.rlcard import (
    RLCardObservation,
    legal_action_ids,
    rlcard_action_to_action,
    to_rlcard_observation,
)

__all__ = [
    "OpenSpielAction",
    "OpenSpielObservation",
    "RLCardObservation",
    "legal_action_ids",
    "openspiel_action_to_action",
    "openspiel_legal_action_ids",
    "registered_poker_games",
    "rlcard_action_to_action",
    "to_openspiel_observation",
    "to_rlcard_observation",
]
