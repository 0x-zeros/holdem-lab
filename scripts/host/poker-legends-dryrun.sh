#!/usr/bin/env bash
# Poker Legends host dry-run helper — recognise -> AI decide -> LOG the intended
# click. It NEVER clicks (PokerLegendsDryRunAutomator records "executed": false).
# Safe to run with Poker Legends open. Full guide: docs/bot-host-dryrun.md
#
# Centralises the four bundled template-manifest flags so each step is a one-liner.
#
# Subcommands:
#   sanity [KF]                 Step 1 — run the full pipeline on a bundled frame
#                               (default keyframe 000042; needs no live screen).
#   capture [OUT_DIR]           Step 2 — grab one macOS screen/window PNG.
#   dryrun IMAGE LAYOUT         Step 2 — decide on a saved capture + its layout JSON.
#   live LAYOUT [OUT_DIR]       Step 2 — capture fresh then decide (needs a layout).
#   replay FRAMES ANN [..]      Step 3 — replay a frame dir through ONE persistent
#                               policy to accumulate a per-seat opponent read.
#                               Extra args pass through (e.g. --use-truth --limit 5).
#   replay-bundled [..]         Step 3 — replay the bundled session_001_selection set.
#   watch-once [KF] [OUT]       HUD — render one bundled frame's perception overlay
#                               to a PNG (no GUI; default keyframe 000080). Shows the
#                               ROIs + recognised state + why state assembly blocked.
#   watch [LAYOUT] [..]         HUD — live mss overlay loop (never clicks). Default
#                               LAYOUT = bundled 1600w; pass yours once calibrated.
#                               Extra args pass through (--region L,T,W,H --dump-dir D).
#   watch-llm [..]              HUD — live LLM (Gemini) read; NO manifests/layout needed.
#                               Needs GEMINI_API_KEY (auto-sourced from repo .env).
#                               e.g. watch-llm --monitor 2 --dump-dir ~/pl-dumps
#
# Env overrides: REPO, A (artifacts dir), SEAT (controlled seat), LOG, FRAMES_OUT,
#   RUN (python runner; default "uv run"). On this devcontainer test the plumbing
#   with: RUN= PATH=/workspace/.venv-docker/bin:$PATH scripts/host/poker-legends-dryrun.sh sanity
set -euo pipefail

REPO="${REPO:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"
A="${A:-$REPO/artifacts/poker-legends-videos}"
RUN="${RUN:-uv run}"
SEAT="${SEAT:-0}"
LOG="${LOG:-/tmp/poker-legends-dryrun.jsonl}"
FRAMES_OUT="${FRAMES_OUT:-$HOME/pl-frames}"

PART="$A/multi_source_templates_v2/card_part_threshold_r02_s02/card_part_template_manifest.json"
CLS="$A/multi_source_templates_v2/card_classifier_v1/card_classifier_manifest.json"
BTN="$A/multi_source_templates_v1/button_templates/button_template_manifest.json"
CARD="$A/multi_source_templates_v1/card_templates/card_template_manifest.json"
# Always non-empty (8 elements) -> "${MANIFESTS[@]}" is safe even under bash 3.2 + set -u.
MANIFESTS=(
  --card-part-manifest "$PART"
  --card-classifier-manifest "$CLS"
  --button-manifest "$BTN"
  --card-template-manifest "$CARD"
)

die() { echo "error: $*" >&2; exit 2; }
need_file() { [ -e "$1" ] || die "missing file: $1 (is \$A=$A correct for this checkout?)"; }

cmd_sanity() {
  local kf="${1:-000042}"
  local frame="$A/session_001_selection/frames/keyframe_${kf}.png"
  local ann="$A/session_001_selection/annotations/keyframe_${kf}.json"
  need_file "$frame"; need_file "$ann"
  # RUN is intentionally unquoted: "uv run" must word-split into two argv words.
  # shellcheck disable=SC2086
  $RUN holdem-bot-run-poker-legends-dry-run \
    --image "$frame" --layout-annotation "$ann" \
    "${MANIFESTS[@]}" --seat "$SEAT" --log-jsonl "$LOG"
}

cmd_capture() {
  # shellcheck disable=SC2086
  $RUN holdem-bot-capture-macos-screen --out-dir "${1:-$FRAMES_OUT}"
}

cmd_dryrun() {
  local image="${1:?usage: dryrun IMAGE.png LAYOUT.json}"
  local layout="${2:?usage: dryrun IMAGE.png LAYOUT.json}"
  need_file "$image"; need_file "$layout"
  # shellcheck disable=SC2086
  $RUN holdem-bot-run-poker-legends-dry-run \
    --image "$image" --layout-annotation "$layout" \
    "${MANIFESTS[@]}" --seat "$SEAT" --log-jsonl "$LOG"
}

cmd_live() {
  local layout="${1:?usage: live LAYOUT.json [OUT_DIR]}"
  need_file "$layout"
  # shellcheck disable=SC2086
  $RUN holdem-bot-run-poker-legends-dry-run \
    --capture-out-dir "${2:-$FRAMES_OUT}" --layout-annotation "$layout" \
    "${MANIFESTS[@]}" --seat "$SEAT" --log-jsonl "$LOG"
}

cmd_replay() {
  local frames="${1:?usage: replay FRAMES_DIR ANNOTATIONS_DIR [--use-truth] [--limit N]}"
  local anns="${2:?usage: replay FRAMES_DIR ANNOTATIONS_DIR [--use-truth] [--limit N]}"
  shift 2
  [ -d "$frames" ] || die "missing frames dir: $frames"
  [ -d "$anns" ] || die "missing annotations dir: $anns"
  # shellcheck disable=SC2086
  $RUN holdem-bot-replay-poker-legends-dry-run \
    --frames-dir "$frames" --annotations-dir "$anns" \
    "${MANIFESTS[@]}" --seat "$SEAT" --log-jsonl "${LOG%.jsonl}-replay.jsonl" "$@"
}

cmd_replay_bundled() {
  cmd_replay "$A/session_001_selection/frames" "$A/session_001_selection/annotations" "$@"
}

cmd_watch_once() {
  local kf="${1:-000080}"
  local frame="$A/session_001_selection/frames/keyframe_${kf}.png"
  local ann="$A/session_001_selection/annotations/keyframe_${kf}.json"
  local out="${2:-/tmp/poker-legends-watch-${kf}.overlay.png}"
  need_file "$frame"; need_file "$ann"
  # shellcheck disable=SC2086
  $RUN holdem-bot-watch-poker-legends --image "$frame" --layout-annotation "$ann" \
    "${MANIFESTS[@]}" --seat "$SEAT" --overlay-out "$out"
}

cmd_watch() {
  local layout
  if [ $# -gt 0 ] && [ "${1:0:2}" != "--" ]; then
    layout="$1"; shift
  else
    layout="$A/session_001_selection/annotations/keyframe_000042.json"
  fi
  need_file "$layout"
  # shellcheck disable=SC2086
  $RUN holdem-bot-watch-poker-legends --layout-annotation "$layout" \
    "${MANIFESTS[@]}" --seat "$SEAT" "$@"
}

cmd_watch_llm() {
  # Live LLM perception HUD (Gemini). No manifests/layout. Sources GEMINI_API_KEY from the
  # repo .env if present so the host run is a single command.
  if [ -f "$REPO/.env" ]; then set -a; . "$REPO/.env"; set +a; fi
  # shellcheck disable=SC2086
  $RUN holdem-bot-watch-poker-legends --llm --seat "$SEAT" "$@"
}

usage() {
  sed -n '2,/^set -euo/p' "$0" | sed '$d'
}

case "${1:-help}" in
  sanity)          shift; cmd_sanity "$@" ;;
  capture)         shift; cmd_capture "$@" ;;
  dryrun)          shift; cmd_dryrun "$@" ;;
  live)            shift; cmd_live "$@" ;;
  replay)          shift; cmd_replay "$@" ;;
  replay-bundled)  shift; cmd_replay_bundled "$@" ;;
  watch-once)      shift; cmd_watch_once "$@" ;;
  watch)           shift; cmd_watch "$@" ;;
  watch-llm)       shift; cmd_watch_llm "$@" ;;
  help|-h|--help)  usage ;;
  *)               echo "unknown subcommand: $1" >&2; echo >&2; usage; exit 2 ;;
esac
