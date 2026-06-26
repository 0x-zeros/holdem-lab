import pytest
from holdem_bot import BotOrchestrator, CapturedFrame, RecognitionResult
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
    assert performed == [result.action]


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


def test_state_recognizer_rejects_non_state_payload() -> None:
    recognizer = StateRecognizer()

    with pytest.raises(TypeError, match="GameState"):
        recognizer.recognize(CapturedFrame(payload=object(), source="test"))
