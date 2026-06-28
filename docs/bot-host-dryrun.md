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

## Quickstart — the helper script (copy-paste)

`scripts/host/poker-legends-dryrun.sh` wraps every step below so you never retype
the four `--*-manifest` flags. It **never clicks**. Run from your repo root on the
host (it resolves the bundled manifests via `git rev-parse`, and uses `uv run`):

```bash
# Fast loop (recommended) — the live perception HUD: SEE what the bot sees instead
# of the screenshot -> save -> run -> paste-JSON round-trip. 'watch-once' renders one
# bundled frame's annotated overlay to a PNG offline; 'watch' opens the live mss
# overlay loop (never clicks; s=dump frame+overlay+json, q=quit):
scripts/host/poker-legends-dryrun.sh watch-once 000080         # offline overlay PNG
scripts/host/poker-legends-dryrun.sh watch                     # live HUD on monitor 1

# Step 1 — sanity-check the checkout on a bundled frame (no live screen needed):
scripts/host/poker-legends-dryrun.sh sanity            # default keyframe 000042
scripts/host/poker-legends-dryrun.sh sanity 000080     # any bundled keyframe number

# Step 2 — capture + decide on YOUR screen (needs a layout for your resolution):
scripts/host/poker-legends-dryrun.sh capture ~/pl-frames           # save one PNG
scripts/host/poker-legends-dryrun.sh dryrun <your.png> <your-layout.json>
scripts/host/poker-legends-dryrun.sh live <your-layout.json>       # capture fresh + decide

# Step 3 — accumulate a per-seat opponent read across a frame sequence:
scripts/host/poker-legends-dryrun.sh replay <frames-dir> <annotations-dir> --use-truth
scripts/host/poker-legends-dryrun.sh replay-bundled --use-truth --limit 5   # offline demo
```

Env overrides: `SEAT=<n>` (your seat), `A=<artifacts dir>`, `LOG=<path>`,
`FRAMES_OUT=<dir>`. `scripts/host/poker-legends-dryrun.sh help` lists every
subcommand. The long-form command each step expands to is documented below.

---

## Fast loop — the perception HUD (`holdem-bot-watch-poker-legends`)

The single-shot dry-run is fine for one frame, but iterating that way (screenshot →
save → run → paste JSON → wait) is slow. The **HUD** collapses that loop: it
captures the screen with `mss` a few times a second, runs the *same* recogniser +
opponent-aware policy, and **draws what the bot perceives back onto the frame** — the
layout ROIs (cards/buttons/seats/pot, scaled to your resolution) plus a text panel
with the screen kind, the safety-gate verdict, **why state assembly blocked**
(`state_block_reason`), the recognised pot/to-call/legal-actions, the policy's
intended action, and the per-seat opponent reads. It **only reads** — there is no
click path in this tool at all.

```bash
# Offline (no GUI, no live screen): render one frame's overlay to a PNG. Great for
# sending back as evidence, and it surfaces the exact state_block_reason.
scripts/host/poker-legends-dryrun.sh watch-once 000080        # -> /tmp/poker-legends-watch-000080.overlay.png

# Live: open an OpenCV window updating ~4 fps. 's' dumps {frame,overlay,json}, 'q' quits.
scripts/host/poker-legends-dryrun.sh watch                    # bundled 1600w layout on monitor 1
scripts/host/poker-legends-dryrun.sh watch ~/pl-layout.json --region 0,0,1600,982 --dump-dir ~/pl-dumps
```

**Why this is the calibration tool.** The bundled layout's ROIs are in a fixed
`1600×982` space and get scaled to your captured frame; when the game window is a
different size/aspect, the drawn rectangles visibly drift off the real cards and
buttons. Watch the HUD, line the game window up (or hand me a `watch-once` overlay
PNG + your capture's pixel size) and we calibrate a layout for your resolution —
which is what unblocks the `no_game_state` / `missing_table_metadata` failures the
single-shot run reports. Reads only accumulate on frames where it would act (hero's
turn), exactly as the live bot sees them, so a long `watch` session with `--dump-dir`
is the highest-signal artifact to send back.

> Bundled-frame finding: `watch-once 000080` is a **real in-game table** (hero `5h 8s`,
> Call/Raise/Fold visible) yet still returns `no_game_state` with
> `state_block_reason: missing_table_metadata` — the selection-set annotations have
> ROIs placed but values pending (`roi_applied_values_pending`), so even a correct
> table can't assemble a `GameState` until the layout carries the table-metadata block.

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

## Step 3 — build an opponent read across a session (replay)

A single frame can never build a read: each `run-poker-legends-dry-run` is a fresh
process. The bot's decision policy (`FieldExploitPolicy`) carries a per-seat
`OpponentModel` that only becomes useful once it has seen many hands, so to
exercise it you replay a *sequence* of frames through **one** persistent policy:

```bash
uv run holdem-bot-replay-poker-legends-dry-run \
  --frames-dir <dir of *.png> --annotations-dir <dir of matching *.json> \
  --card-part-manifest … --card-classifier-manifest … --button-manifest … \
  --log-jsonl /tmp/poker-legends-replay.jsonl
```

It runs every frame through the same policy (never clicks) and prints a summary:
per-frame `reason`/`exploit`, and the final `opponent_reads` (`vpip`/`pfr`/`profile`
per seat). Add `--use-truth` to feed each frame's annotation as ground truth,
which isolates the read logic from recognition quality — if reads stay empty even
with `--use-truth`, the frames have no actionable table states (the bundled
`session_001_selection` set is mostly menus/overlays); if they populate with
`--use-truth` but not without, the gap is recognition (Step 2's calibration). This
is the offline proxy for "does real-frame recognition support an opponent model"
— the highest-signal thing to send back alongside the per-frame JSON.

## Reading the output JSON

| field | meaning |
|---|---|
| `acted` | did it reach the AI + log an intended click (`true`) or fail closed (`false`) |
| `reason` | `acted` / `blocked_overlay` / `low_confidence` / `no_game_state` / a `state_block_reason` (e.g. `missing_pot`, `not_enough_players`, `hero_not_current`, `missing_seat_stack`, `low_card_confidence`) |
| `screen.kind` | `actionable_table` / `table_observe` / `blocked_overlay` / `unknown_or_transition` |
| `state` | recognized `GameState` summary (pot, to_call, legal_actions) — `null` if fail-closed |
| `action` / `policy_decision` | what the AI **would** do (incl. `metadata.button_seat_source` = recognized/default_oop, and `metadata.exploit` = base/station/nit + `metadata.opponent_profiles`) |
| `opponent_reads` | per-seat `{profile, vpip, pfr, hands}` from the opponent model — `unknown` on a single frame (the read needs many hands; see Step 3) |
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
