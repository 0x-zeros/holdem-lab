import pytest
from holdem_ai import PolicyDecision
from holdem_ai.field import FieldExploitPolicy
from holdem_ai.opponents import OpponentModel, OpponentProfile
from holdem_bot import BotOrchestrator, CapturedFrame, RecognitionResult, ScreenState
from holdem_bot.adapters import ActionCallbackAutomator, StateCapture, StateRecognizer
from holdem_common import Action, ActionType, Card, GameState, PlayerState, Pot, Street


def cards(*codes: str) -> tuple[Card, ...]:
    return tuple(Card.from_code(code) for code in codes)


def make_state(
    *,
    current_seat: int | None = 0,
    legal_actions: tuple[Action, ...] = (Action(ActionType.CHECK),),
) -> GameState:
    return GameState(
        hand_id="bot-test",
        street=Street.PREFLOP,
        players=(
            PlayerState(seat=0, stack=100, hole_cards=cards("7c", "2d")),
            PlayerState(seat=1, stack=100, hole_cards=()),
        ),
        board=(),
        pots=(Pot(amount=3, eligible_seats=frozenset({0, 1})),),
        current_seat=current_seat,
        button_seat=0,
        small_blind=1,
        big_blind=2,
        min_raise=4,
        to_call=0,
        legal_actions=legal_actions,
    )


class LowConfidenceRecognizer:
    def __init__(self, state: GameState) -> None:
        self.state = state

    def recognize(self, _frame: CapturedFrame) -> RecognitionResult:
        return RecognitionResult(state=self.state, confidence=0.25)


class FixedRecognition:
    def __init__(self, result: RecognitionResult) -> None:
        self.result = result

    def recognize(self, _frame: CapturedFrame) -> RecognitionResult:
        return self.result


def test_orchestrator_acts_when_it_is_controlled_seat_turn() -> None:
    state = make_state()
    performed: list[Action] = []
    orchestrator = BotOrchestrator(
        capture=StateCapture(lambda: state),
        recognizer=StateRecognizer(),
        automator=ActionCallbackAutomator(performed.append),
        seat=0,
    )

    result = orchestrator.run_once()

    assert result.acted
    assert result.reason == "acted"
    assert result.action is not None
    assert result.action.action_type is ActionType.CHECK
    assert result.policy_decision is not None
    assert result.policy_decision.reason == "preflop_check"
    assert performed == [result.action]


def test_orchestrator_can_use_custom_policy_explainer() -> None:
    state = make_state()
    performed: list[Action] = []

    def explain(state: GameState) -> PolicyDecision:
        return PolicyDecision(
            action=state.legal_actions[0],
            reason="custom_profile",
            strength=0.5,
            required_equity=None,
            metadata={},
        )

    orchestrator = BotOrchestrator(
        capture=StateCapture(lambda: state),
        recognizer=StateRecognizer(),
        automator=ActionCallbackAutomator(performed.append),
        seat=0,
        policy_explainer=explain,
    )

    result = orchestrator.run_once()

    assert result.acted
    assert result.policy_decision is not None
    assert result.policy_decision.reason == "custom_profile"
    assert performed == [result.action]


def test_orchestrator_with_field_exploit_accumulates_opponent_read() -> None:
    # A persistent FieldExploitPolicy injected as the policy_explainer must build a
    # per-seat read across frames -- the core of the bot-side opponent model. Seat 1
    # voluntarily commits (a limp) every hand and never raises -> reads as a station.
    policy = FieldExploitPolicy(model=OpponentModel(min_hands=4))
    counter = {"hand": 0}

    def next_state() -> GameState:
        hand = counter["hand"]
        counter["hand"] += 1
        return GameState(
            hand_id=f"hand-{hand}",
            street=Street.PREFLOP,
            players=(
                PlayerState(seat=0, stack=100, committed=2, hole_cards=cards("7c", "2d")),
                PlayerState(seat=1, stack=100, committed=2),  # voluntary limp, never raises
            ),
            board=(),
            pots=(Pot(amount=4, eligible_seats=frozenset({0, 1})),),
            current_seat=0,
            button_seat=0,
            small_blind=1,
            big_blind=2,
            min_raise=4,
            to_call=0,
            legal_actions=(Action(ActionType.CHECK),),
        )

    orchestrator = BotOrchestrator(
        capture=StateCapture(next_state),
        recognizer=StateRecognizer(),
        automator=ActionCallbackAutomator(lambda _action: None),
        seat=0,
        policy_explainer=policy.explain,
    )

    for _ in range(10):
        result = orchestrator.run_once()
        assert result.acted

    read = policy.model.read(1)
    assert read.hands >= 4
    assert read.vpip == 1.0
    assert read.profile is OpponentProfile.STATION


def test_orchestrator_waits_when_other_seat_is_to_act() -> None:
    state = make_state(current_seat=1)
    performed: list[Action] = []
    orchestrator = BotOrchestrator(
        capture=StateCapture(lambda: state),
        recognizer=StateRecognizer(),
        automator=ActionCallbackAutomator(performed.append),
        seat=0,
    )

    result = orchestrator.run_once()

    assert not result.acted
    assert result.reason == "waiting"
    assert performed == []


def test_orchestrator_skips_low_confidence_recognition() -> None:
    state = make_state()
    performed: list[Action] = []
    orchestrator = BotOrchestrator(
        capture=StateCapture(lambda: state),
        recognizer=LowConfidenceRecognizer(state),
        automator=ActionCallbackAutomator(performed.append),
        seat=0,
        min_confidence=0.8,
    )

    result = orchestrator.run_once()

    assert not result.acted
    assert result.reason == "low_confidence"
    assert performed == []


def test_orchestrator_skips_terminal_state() -> None:
    state = make_state(current_seat=None, legal_actions=())
    performed: list[Action] = []
    orchestrator = BotOrchestrator(
        capture=StateCapture(lambda: state),
        recognizer=StateRecognizer(),
        automator=ActionCallbackAutomator(performed.append),
        seat=0,
    )

    result = orchestrator.run_once()

    assert not result.acted
    assert result.reason == "terminal"
    assert performed == []


def test_orchestrator_blocks_overlay_before_deciding() -> None:
    state = make_state()
    performed: list[Action] = []
    recognizer = FixedRecognition(
        RecognitionResult(
            state=state,
            screen=ScreenState.blocked_overlay(blocking_reason="challenge_overlay"),
        )
    )
    orchestrator = BotOrchestrator(
        capture=StateCapture(lambda: state),
        recognizer=recognizer,
        automator=ActionCallbackAutomator(performed.append),
        seat=0,
    )

    result = orchestrator.run_once()

    assert not result.acted
    assert result.reason == "blocked_overlay"
    assert result.screen is not None
    assert result.screen.blocking_reason == "challenge_overlay"
    assert performed == []


def test_orchestrator_blocks_non_table_screen_without_state() -> None:
    state = make_state()
    performed: list[Action] = []
    recognizer = FixedRecognition(
        RecognitionResult(
            state=None,
            screen=ScreenState.non_table_ui(reason="lobby"),
        )
    )
    orchestrator = BotOrchestrator(
        capture=StateCapture(lambda: state),
        recognizer=recognizer,
        automator=ActionCallbackAutomator(performed.append),
        seat=0,
    )

    result = orchestrator.run_once()

    assert not result.acted
    assert result.reason == "non_table_ui"
    assert result.state is None
    assert performed == []


def test_orchestrator_blocks_unknown_screen() -> None:
    state = make_state()
    performed: list[Action] = []
    recognizer = FixedRecognition(
        RecognitionResult(
            state=state,
            screen=ScreenState.unknown_or_transition(confidence=0.9, reason="animation"),
        )
    )
    orchestrator = BotOrchestrator(
        capture=StateCapture(lambda: state),
        recognizer=recognizer,
        automator=ActionCallbackAutomator(performed.append),
        seat=0,
    )

    result = orchestrator.run_once()

    assert not result.acted
    assert result.reason == "unknown_or_transition"
    assert performed == []


def test_orchestrator_requires_state_for_actionable_table() -> None:
    state = make_state()
    performed: list[Action] = []
    recognizer = FixedRecognition(
        RecognitionResult(
            state=None,
            screen=ScreenState.actionable_table(hero_turn=True),
        )
    )
    orchestrator = BotOrchestrator(
        capture=StateCapture(lambda: state),
        recognizer=recognizer,
        automator=ActionCallbackAutomator(performed.append),
        seat=0,
    )

    result = orchestrator.run_once()

    assert not result.acted
    assert result.reason == "no_game_state"
    assert performed == []


def test_state_recognizer_rejects_non_state_payload() -> None:
    recognizer = StateRecognizer()

    with pytest.raises(TypeError, match="GameState"):
        recognizer.recognize(CapturedFrame(payload=object(), source="test"))
