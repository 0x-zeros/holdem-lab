# holdem-lab

Texas Hold'em poker equity calculator and game engine.

## Installation

```bash
cd holdem-core
uv pip install -e ".[dev]"
```

## Quick Start

```python
from holdem_lab import parse_cards, calculate_equity, EquityRequest, PlayerHand

# Parse cards
hole_cards = parse_cards("Ah Kh")
board = parse_cards("Qh Jh 2c")

# Calculate equity
request = EquityRequest(
    players=[
        PlayerHand(hole_cards=parse_cards("Ah Kh")),
        PlayerHand(hole_cards=parse_cards("Qc Qd")),
    ],
    board=board,
    num_simulations=10000,
)
result = calculate_equity(request)
print(f"Player 1 equity: {result.players[0].equity:.1%}")
```

## Running Tests

```bash
cd holdem-core
uv run pytest
```

## Modules

- `cards.py` - Card representation, parsing, and deck management
- `evaluator.py` - Hand evaluation (7-card to 5-card best hand)
- `equity.py` - Monte Carlo equity calculation
- `event_log.py` - Event logging and hand replay
- `game_state.py` - Game state machine for hand simulation
