# Poker Legends host dry-run — operator guide

Goal: run the bot's `capture → recognize → AI decide → log the intended click` loop
**once** on a real screen, **without clicking**, to see whether perception produces a
usable `GameState` + action or fails closed (and why). This is the cheapest way to
surface the real-world perception gaps that can only be found on the live game.

**It never clicks.** `PokerLegendsDryRunAutomator` only appends the *intended* click
to a JSONL log with `"executed": false`. Safe to run with Poker Legends open.

The dry-run CLI is `holdem-bot-run-poker-legends-dry-run`
(`holdem_bot.adapters.poker_legends_host:dry_run_once_main`).

---

## Step 1 — verify your checkout (no macOS screen needed; run this first)

Runs the **full** pipeline (recognizer + the upgraded AI + dry-run logging) on a
bundled saved frame. Confirms your checkout works before touching the live game.
Run from the repo root (`/workspace` here; your repo root on the host):

```bash
A=artifacts/poker-legends-videos
uv run holdem-bot-run-poker-legends-dry-run \
  --image "$A/session_001_selection/frames/keyframe_000042.png" \
  --layout-annotation "$A/session_001_selection/annotations/keyframe_000042.json" \
  --card-part-manifest "$A/multi_source_templates_v2/card_part_threshold_r02_s02/card_part_template_manifest.json" \
  --card-classifier-manifest "$A/multi_source_templates_v2/card_classifier_v1/card_classifier_manifest.json" \
  --button-manifest "$A/multi_source_templates_v1/button_templates/button_template_manifest.json" \
  --card-template-manifest "$A/multi_source_templates_v1/card_templates/card_template_manifest.json" \
  --log-jsonl /tmp/poker-legends-dryrun.jsonl
```

Expected for this exact frame (verified): `"acted": false`, `"reason":
"blocked_overlay"`, `screen.blocking_reason: "buy_in_prompt"`, `state: null` — the
safety gate correctly refusing to act on a buy-in overlay. Swap the keyframe number
(e.g. `keyframe_000019`, `_000080`, `_000151`) to find frames that classify as an
actionable table and produce a non-null `state` + `action`.

> On the host use `uv run …`. (In this devcontainer the equivalent is
> `/workspace/scripts/dev/py -c "from holdem_bot.adapters.poker_legends_host import
> dry_run_once_main; dry_run_once_main([...])"`.)

---

## Step 2 — live capture on your real screen (macOS host)

1. **Capture** the screen (or a specific window) — Poker Legends visible, your turn:
   ```bash
   uv run holdem-bot-capture-macos-screen --out-dir ~/pl-frames
   # prints the saved PNG path; optionally pass --window-id <id> for one window
   ```
2. **Layout calibration is the likely sticking point.** The bundled layout
   (`keyframe_000042.json`) has pixel ROIs proportional to the *session-video*
   resolution; at your window size/aspect they may not land on the real buttons/
   cards. Send me your capture's pixel dimensions (and ideally the saved PNG path so
   I can give you an `apply-poker-legends-layout` command tuned to it) — we calibrate
   a layout for your resolution together, then re-run Step 1's command with
   `--image <your-capture.png> --layout-annotation <your-layout.json>`.
3. For a fully live one-shot once a layout exists, swap `--image` for
   `--capture-out-dir ~/pl-frames` (captures fresh, then recognizes).

---

## Reading the output JSON

| field | meaning |
|---|---|
| `acted` | did it reach the AI + log an intended click (`true`) or fail closed (`false`) |
| `reason` | `acted` / `blocked_overlay` / `low_confidence` / `no_game_state` / a `state_block_reason` (e.g. `missing_pot`, `not_enough_players`, `hero_not_current`, `missing_seat_stack`, `low_card_confidence`) |
| `screen.kind` | `actionable_table` / `table_observe` / `blocked_overlay` / `unknown_or_transition` |
| `state` | recognized `GameState` summary (pot, to_call, legal_actions) — `null` if fail-closed |
| `action` / `policy_decision` | what the AI **would** do (incl. `metadata.button_seat_source` = recognized/default_oop) |
| `dry_run_record.click_plan` | where it **would** click `(x, y)` — use this to check ROI alignment |

The `--log-jsonl` file accumulates every step's record (append-only).

---

## What to send back

For ~3–5 frames across situations — **your turn**, **opponent's turn**, and an
**overlay/transition** — paste the printed JSON. From that I build the offline fix
list (which `state_block_reason`s dominate, whether positions/blinds/legal-actions
look right, whether ROIs are mis-calibrated). The two highest-signal things:
1. On a real **your-turn** frame: does `state` come back non-null with sane
   `pot` / `to_call` / `legal_actions`, or what `reason` blocks it?
2. Does `dry_run_record.click_plan (x,y)` land on the correct button at your
   resolution (the layout-calibration check)?
