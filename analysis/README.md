# Analysis Notebooks

Jupyter notebooks for analyzing poker equity and replaying hands.

## Setup

```bash
cd holdem-core
uv pip install -e ".[analysis]"
```

## Notebooks

- `00_smoke_tests.ipynb` - Quick verification that the library works
- `01_equity_preflop.ipynb` - Preflop equity analysis and charts
- `02_equity_postflop.ipynb` - Postflop equity with board textures
- `03_replay_viewer.ipynb` - Hand replay with interactive controls

## Running

```bash
cd analysis
jupyter notebook
```
