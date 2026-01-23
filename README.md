# holdem-lab

Texas Hold'em poker probability calculator and game engine for analysis and validation.

## Features

- **Equity Calculator**: Monte Carlo simulation for hand equity
- **Hand Evaluator**: 7-card to 5-card best hand evaluation
- **Game Engine**: Complete game state machine for hand simulation
- **Event Logging**: Record and replay hands
- **Analysis Tools**: Jupyter notebooks for equity analysis

## Project Structure

```
holdem-lab/
├── README.md
├── CLAUDE.md              # Development guidance
└── holdem-core/           # Core Python library
    ├── src/holdem_lab/    # Main package
    ├── tests/             # Unit tests (148 tests)
    ├── analysis/          # Jupyter notebooks
    └── fixtures/          # Test data
```

## Installation

```bash
cd holdem-core
uv pip install -e ".[dev]"
```

For analysis notebooks:

```bash
uv pip install -e ".[analysis]"
```

## Quick Start

```python
from holdem_lab import (
    parse_cards,
    EquityRequest, PlayerHand, calculate_equity,
    GameState,
)

# Calculate AA vs KK equity
request = EquityRequest(
    players=[
        PlayerHand(hole_cards=tuple(parse_cards("Ah Ad"))),
        PlayerHand(hole_cards=tuple(parse_cards("Kh Kd"))),
    ],
    num_simulations=10000,
)
result = calculate_equity(request)
print(f"AA: {result.players[0].equity:.1%}")
print(f"KK: {result.players[1].equity:.1%}")

# Run a complete hand
game = GameState(num_players=2, seed=42)
result = game.run_to_showdown()
print(f"Winner: Player {result.winners[0]}")
```

## Running Tests

```bash
cd holdem-core
uv run pytest
```

## Modules

| Module | Description |
|--------|-------------|
| `cards.py` | Card representation, parsing, and deck management |
| `evaluator.py` | Hand evaluation (7-card to 5-card best hand) |
| `equity.py` | Monte Carlo equity calculation |
| `event_log.py` | Event logging and hand replay |
| `game_state.py` | Game state machine for hand simulation |

## License

MIT
