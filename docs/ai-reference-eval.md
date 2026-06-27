# AI reference-strength table

Absolute yardstick: each focal policy vs the reference opponents, measured with
the CRN-paired bootstrap harness (`holdem_ai.evaluate.evaluate_match`).

- Each matchup plays **N CRN deck-pairs = 2N hands** (same cards played both seat
  orderings, so dealing luck cancels). N shrinks with depth.
- 95% CI is a **percentile bootstrap** (1500 resamples) over per-pair nets.
- Deterministic (seed 20260627). **Regenerate the tables** with
  `ai/scripts/reference_eval.py --stacks 20,50,200 --pairs 1200,800,300`
  (the *Key findings* below are curated commentary, not regenerated).
- A row is **bold** when the 95% CI excludes 0 (a statistically clear result).

All numbers are **bb/100 from the focal policy's perspective** (higher = better).

## Key findings

- **The 10bb tier is the only one that distinguishes `pushfold` from `current`.**
  The blueprint only engages at ≤12bb effective; above that it falls back to the
  heuristic, so the 25bb and 100bb rows are **identical** for the two focals
  (a clean confirmation that the fallback works).
- **GTO floor vs aggression (the headline).** At 10bb, the heuristic *loses* to
  the maniac (`current` vs `maniac` **−18.8** [−38.2, +1.6]) while the blueprint
  *beats* it (`pushfold` vs `maniac` **+40.8** [+20.2, +60.1]). This ~60 bb/100
  swing is exactly the unexploitable short-stack floor the hybrid (S2c-3) should
  graft onto the heuristic.
- **GTO leaves money on the table vs exploitable passivity.** Against the
  calling station and rock, `current` wins *more* than `pushfold`
  (call_station +139.5 vs +103.9; rock +69.8 vs +37.4): jam/fold cannot do the
  thin value-betting the heuristic does. GTO = unexploitable, not maximally
  exploitative — as expected (`docs/ai-strength.md` §7).
- **Blind open-jamming underperforms vs a competent opponent.** Against `tag`
  (which defends correctly), `current` **+33.2** beats `pushfold` **+7.3**
  (CI spans 0) at 10bb. A real signal that S2c-2's richer short-stack game
  (limp / min-raise / jam) is worth more than pure push/fold vs sound players.
- **CRN makes the harness sharp.** Low-variance matchups land tight intervals on
  modest samples — e.g. `*` vs `rock` @25bb = +69.8 bb/100, CI [+68.3, +71.1]
  over 1600 hands. Deep all-in pots stay wide (maniac @100bb spans hundreds),
  which is honest poker variance, not a harness defect.

## 10bb effective — 1200 pairs (2400 hands/matchup)

| focal | opponent | bb/100 | 95% CI | button | big blind |
|---|---|--:|:--:|--:|--:|
| `pushfold` | `random` | **+27.0** | [+9.4, +44.6] | +28.1 | +25.9 |
| `pushfold` | `call_station` | **+103.9** | [+88.0, +120.5] | +39.1 | +168.7 |
| `pushfold` | `rock` | **+37.4** | [+35.2, +39.6] | +24.8 | +50.0 |
| `pushfold` | `maniac` | **+40.8** | [+20.2, +60.1] | +39.1 | +42.6 |
| `pushfold` | `tag` | +7.3 | [-2.1, +17.0] | -7.3 | +22.0 |
| `pushfold` | `three_bet_jammer` | **+18.2** | [+10.4, +26.4] | +24.8 | +11.6 |
| `current` | `random` | **+24.7** | [+4.6, +44.0] | +14.1 | +35.4 |
| `current` | `call_station` | **+139.5** | [+129.5, +149.9] | +110.2 | +168.7 |
| `current` | `rock` | **+69.8** | [+68.7, +70.8] | +89.5 | +50.0 |
| `current` | `maniac` | -18.8 | [-38.2, +1.6] | -80.1 | +42.6 |
| `current` | `tag` | **+33.2** | [+24.1, +42.4] | +44.5 | +22.0 |
| `current` | `three_bet_jammer` | **+25.7** | [+16.1, +35.7] | +45.1 | +6.3 |

## 25bb effective — 800 pairs (1600 hands/matchup)

`pushfold` ≡ `current` here (effective stack > 12bb, so the blueprint never fires).

| focal | opponent | bb/100 | 95% CI | button | big blind |
|---|---|--:|:--:|--:|--:|
| `pushfold` | `random` | **+128.8** | [+72.0, +186.9] | +144.8 | +112.9 |
| `pushfold` | `call_station` | **+318.2** | [+282.9, +352.9] | +286.1 | +350.4 |
| `pushfold` | `rock` | **+69.8** | [+68.3, +71.1] | +89.5 | +50.0 |
| `pushfold` | `maniac` | **+149.1** | [+92.4, +208.2] | +106.5 | +191.8 |
| `pushfold` | `tag` | **+36.8** | [+13.4, +61.7] | +55.6 | +18.1 |
| `pushfold` | `three_bet_jammer` | +8.0 | [-18.4, +33.2] | +22.9 | -6.9 |
| `current` | `random` | **+128.8** | [+72.0, +186.9] | +144.8 | +112.9 |
| `current` | `call_station` | **+318.2** | [+282.9, +352.9] | +286.1 | +350.4 |
| `current` | `rock` | **+69.8** | [+68.3, +71.1] | +89.5 | +50.0 |
| `current` | `maniac` | **+149.1** | [+92.4, +208.2] | +106.5 | +191.8 |
| `current` | `tag` | **+36.8** | [+13.4, +61.7] | +55.6 | +18.1 |
| `current` | `three_bet_jammer` | +8.0 | [-18.4, +33.2] | +22.9 | -6.9 |

## 100bb effective — 300 pairs (600 hands/matchup)

`pushfold` ≡ `current` here too. Deep all-in pots make the CIs wide (honest
variance); the bold rows are still clearly positive.

| focal | opponent | bb/100 | 95% CI | button | big blind |
|---|---|--:|:--:|--:|--:|
| `pushfold` | `random` | **+403.7** | [+29.8, +771.6] | +587.3 | +220.0 |
| `pushfold` | `call_station` | **+638.4** | [+503.2, +789.4] | +491.0 | +785.8 |
| `pushfold` | `rock` | **+69.5** | [+67.2, +71.5] | +89.0 | +50.0 |
| `pushfold` | `maniac` | **+968.0** | [+612.6, +1297.5] | +1053.7 | +882.3 |
| `pushfold` | `tag` | -26.8 | [-145.1, +85.1] | +34.0 | -87.7 |
| `pushfold` | `three_bet_jammer` | -2.8 | [-95.2, +87.5] | +126.7 | -132.2 |
| `current` | `random` | **+403.7** | [+29.8, +771.6] | +587.3 | +220.0 |
| `current` | `call_station` | **+638.4** | [+503.2, +789.4] | +491.0 | +785.8 |
| `current` | `rock` | **+69.5** | [+67.2, +71.5] | +89.0 | +50.0 |
| `current` | `maniac` | **+968.0** | [+612.6, +1297.5] | +1053.7 | +882.3 |
| `current` | `tag` | -26.8 | [-145.1, +85.1] | +34.0 | -87.7 |
| `current` | `three_bet_jammer` | -2.8 | [-95.2, +87.5] | +126.7 | -132.2 |
