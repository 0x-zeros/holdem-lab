# Poker Legends perception redesign: observation-first, contract-first pipeline

> Status: revised design draft after external safety review.
>
> Scope: Poker Legends perception and dry-run state assembly. This document does not
> enable real clicking, does not add dependencies, and does not change model/download
> policy.

## 1. Executive summary

The current Poker Legends prototype can assemble a prototype `GameState` for 47/57
reviewed actionable frames, but that number is not a safety proof. It is only a
regression baseline for actionable-frame assembly. Safety has to be proven against
authorization events, hard negative screens, stale-state transitions, and image-only
runtime constraints.

The redesign is:

```text
CapturedFrame
  -> preliminary ScreenState / capture evidence
  -> layout-calibrated ROI recognizers
  -> VisualObservation
  -> ObservationValidator
  -> rule-constrained TableTracker
  -> GameStateAssembler
  -> StateValidator / DecisionContractEvaluator
  -> GameStateAssemblyResult
  -> RecognitionResult compatibility wrapper
  -> evaluate_safety
```

The core rule is:

```text
Recognizers observe pixels.
Trackers and validators constrain observations.
Contracts decide what the result is valid for.
RecognitionResult only wraps the result for compatibility.
```

VLMs stay off the critical path. Gemini/Qwen may be used for teacher labels, review
assistance, anomaly explanation, and offline sanity checks. They must not be the final
authority for hero cards, board cards, critical numbers, actionability, or click
permission.

## 2. Safety review decisions

These decisions are mandatory for the first implementation.

- Do not use `stable_valid` for single-frame results. Slice 1 may output
  `single_frame_valid`; temporal stability may only be called
  `temporally_stable_valid` after the tracker validates the current frame/window.
- `GameStateAssemblyResult`, `VisualObservation`, `recognition_mode`, and the active
  safety contract must be first-class fields on `RecognitionResult`, not only entries
  inside `metadata`.
- Runtime modes must be explicit: `image_only_live`, `image_only_replay`,
  `truth_assisted_replay`, and `synthetic_test`.
- In `image_only_live` and `image_only_replay`, accepted critical fields must not come
  from `reviewed_truth`.
- Preselect/shortcut strips must be explicitly observed as negative or ambiguous
  action-panel evidence. They must not be silently ignored.
- A tracker's previous valid state must never authorize the current frame. Current
  action contracts require fresh critical fields and a fresh action row.
- The 47/57 current actionable-frame assembly count is a regression baseline, not a
  safety metric. Any preserved, dropped, or newly blocked frame must be explained.

## 3. Current repo anchors

The implementation should preserve these existing anchors while migrating semantics:

- `ScreenState` and `evaluate_safety()` remain the outer safety gate.
- `PokerLegendsTableRecognizer` remains the Poker Legends recognizer entrypoint.
- `RecognitionResult.state` remains for compatibility, but new code should consume
  `RecognitionResult.assembly_result` and `RecognitionResult.safety_contract`.
- Existing metadata keys stay for HUD/backward compatibility, especially
  `state_block_reason`, `recognized_table`, `number_predictions`, and
  `accepted_number_predictions`.
- Existing specialized recognizers are reused:
  - card consensus: full-card template, rank/suit template, and classifier consensus;
  - primary button recognizer;
  - numeric OCR fallback for pot, hero stack, and primary-left call amount;
  - session timeline tracker as the first temporal skeleton.

The first implementation should explain every delta from the current prototype:

```text
preserved_valid
blocked_by_new_contradiction
blocked_by_missing_required_evidence
blocked_due_to_truth_assisted_source
blocked_due_to_temporal_not_available
discovered_previous_false_valid
```

Coverage drops are acceptable when they identify unsafe old assumptions.

## 4. Modes, source policy, and evidence

### 4.1 Recognition mode

Every result must carry:

```python
RecognitionMode = Literal[
    "image_only_live",
    "image_only_replay",
    "truth_assisted_replay",
    "synthetic_test",
]
```

Rules:

- `image_only_live`: live screenshot, no reviewed truth may populate accepted critical
  fields.
- `image_only_replay`: saved image replay, no reviewed truth may populate accepted
  critical fields.
- `truth_assisted_replay`: reviewed truth may populate fields, but reports must label
  the run as truth-assisted and exclude it from live safety claims.
- `synthetic_test`: fixtures/fakes may provide oracle data for unit tests only.

### 4.2 Field source policy

Critical accepted fields include:

- hero cards;
- board cards for the current street;
- current action row and button labels;
- call amount when `call` is considered;
- hero stack when call/all-in/raise/sizing is considered;
- click target source if a click plan is generated.

In `image_only_live` and `image_only_replay`, those fields must have image-derived or
rule-inferred sources. `reviewed_truth` is allowed only as expected data for scoring or
as a non-accepted diagnostic candidate.

### 4.3 Frame evidence

Every `VisualObservation` should carry frame provenance:

```python
FrameEvidence(
    session_id: str | None,
    frame_id: str,
    frame_seq: int | None,
    wall_timestamp: str | None,
    monotonic_timestamp_ms: int | None,
    image_hash: str | None,
    image_size: tuple[int, int] | None,
    capture_backend: str | None,
    window_id: str | None,
    window_title_hash: str | None,
    dpi_scale: float | None,
)
```

This is required to debug stale screenshots, capture lag, window switching, duplicate
frames, and resolution/DPI changes.

### 4.4 ROI evidence

Every card/button/number/seat observation should reference ROI evidence:

```python
RoiEvidence(
    roi_id: str,
    roi_rect_screen: tuple[int, int, int, int] | None,
    roi_rect_canonical: tuple[float, float, float, float] | None,
    crop_hash: str | None,
    crop_path: str | None,
    layout_version: str | None,
    layout_transform_version: str | None,
    crop_quality: float | None,
    occlusion_score: float | None,
    blur_score: float | None,
)
```

Reports should surface ROI evidence for high-confidence wrong predictions and blocking
contradictions.

## 5. Observation model

### 5.1 Candidate values

Primitive observations expose candidates, not only top-1 values.

```python
Candidate(
    value: object,
    confidence: float,
    source: str,
    raw: object | None = None,
    evidence: tuple[RoiEvidence, ...] = (),
)
```

The implementation may use typed candidates (`CardCandidate`, `NumericCandidate`,
`ActionCandidate`) if that keeps mypy simpler. Serialized output must still include
value, confidence, source, raw evidence, and ROI evidence refs.

### 5.2 `VisualObservation`

```python
VisualObservation(
    frame: FrameEvidence,
    recognition_mode: RecognitionMode,
    screen: ScreenState,
    layout: LayoutObservation,
    cards: tuple[CardSlotObservation, ...],
    action_panels: tuple[ActionPanelObservation, ...],
    numbers: tuple[NumericObservation, ...],
    seats: tuple[SeatObservation, ...],
    warnings: tuple[str, ...],
)
```

`VisualObservation` is evidence, not a poker state. It should be serialized into
`RecognitionResult.visual_observation` and mirrored into metadata for old HUD/report
paths.

### 5.3 `LayoutObservation`

```python
LayoutObservation(
    profile_id: str | None,
    layout_version: str | None,
    transform_type: str | None,
    transform_residual_px: float | None,
    anchor_scores: dict[str, float],
    roi_generation: str | None,
    confidence: float,
    image_size: tuple[int, int] | None,
    source: str,
    warnings: tuple[str, ...],
)
```

The current layout annotations can populate this with `source="layout_annotation"`.
Even before full calibration exists, reports should distinguish layout failures from
card/OCR failures.

### 5.4 `CardSlotObservation`

```python
CardSlotObservation(
    group: str,
    slot: str,
    occupancy: str,          # empty | card_back | face_up | face_down | occluded | animating | unknown
    rank_candidates: tuple[Candidate, ...],
    suit_candidates: tuple[Candidate, ...],
    card_candidates: tuple[Candidate, ...],
    accepted_card: str | None,
    locked_card: str | None,
    accepted_by_single_frame: bool,
    locked_by_tracker: bool,
    confidence: float,
    consensus_components: dict[str, object],
    evidence: RoiEvidence,
)
```

Rules:

- Slice 1 may set `accepted_card`, not `locked_card`.
- `locked_card` may only be set by the tracker.
- Hidden, empty, occluded, and animating slots must not become accepted face-up cards.
- Duplicate accepted visible cards are hard contradictions.

### 5.5 `ActionPanelObservation`

Current-action safety is row-level, not just button-level.

```python
ActionPanelObservation(
    panel_kind: str,         # current_action_row | preselect_strip | modal_dialog | lobby_or_menu | unknown
    visible: bool,
    enabled: bool | None,
    hero_turn_indicator: bool | None,
    row_bbox: tuple[int, int, int, int] | None,
    buttons: tuple[ButtonObservation, ...],
    confidence: float,
    ambiguity_flags: tuple[str, ...],
    evidence: RoiEvidence | None,
)
```

Rules:

- Legal actions may only come from a confirmed `current_action_row`.
- `preselect_strip` is negative evidence, not ignored evidence.
- If the panel kind is ambiguous, assembly must be `no_state` or `invalid`.
- Reports must count preselect/modal false positives separately.

### 5.6 `ButtonObservation`

```python
ButtonObservation(
    slot: str,
    visible: bool,
    enabled: bool | None,
    action_candidates: tuple[Candidate, ...],
    accepted_action: str | None,
    amount_candidates: tuple[Candidate, ...],
    confidence: float,
    evidence: RoiEvidence,
)
```

`ButtonObservation` does not decide whether a button is current-action. That belongs
to `ActionPanelObservation`.

### 5.7 `NumericObservation`

Observation describes what the number appears to be. Contracts decide whether it is
critical for the current decision.

```python
NumericObservation(
    role: str,               # pot | hero_stack | call_amount | raise_amount | min_raise | seat_stack | committed | winner_amount
    group: str,
    name: str,
    visible: bool,
    raw_text: str,
    normalized_text: str | None,
    unit: str | None,
    scale: str | None,
    candidates: tuple[Candidate, ...],
    accepted_value: int | None,
    ocr_confidence: float,
    parse_confidence: float,
    value_confidence: float,
    parser_version: str,
    format_flags: tuple[str, ...],
    evidence: RoiEvidence,
)
```

Do not encode global risk as a static observation tier. Risk is a contract property:
`pot` may be non-critical for observe-only output but critical for bet/raise sizing.

### 5.8 `SeatObservation`

```python
SeatObservation(
    seat: int,
    occupied: bool | None,
    in_hand: bool | None,
    has_hole_cards: bool | None,
    folded: bool | None,
    all_in: bool | None,
    sitting_out: bool | None,
    current_actor: bool | None,
    hero_seat: bool,
    dealer_button_nearby: bool | None,
    small_blind_marker: bool | None,
    big_blind_marker: bool | None,
    stack_candidates: tuple[Candidate, ...],
    accepted_stack: int | None,
    committed_current_street: int | None,
    committed_total_hand: int | None,
    showing_cards: tuple[CardSlotObservation, ...],
    confidence: float,
    evidence: RoiEvidence | None,
)
```

Visual observation should record dealer/blind markers. Positional labels such as button
or cutoff are inferred later from seat order and dealer button state.

## 6. Assembly, contracts, and compatibility

### 6.1 Assembly statuses

```python
AssemblyStatus = Literal[
    "blocked_screen",
    "no_state",
    "single_frame_valid",
    "temporally_unstable",
    "temporally_stable_valid",
    "invalid",
    "unsafe_transition",
]
```

Slice 1 must not output `temporally_stable_valid`. It may output
`single_frame_valid` only when the current frame satisfies the single-frame contract.

### 6.2 Validity scope and freshness

```python
ValidityScope = Literal["none", "single_frame", "temporal_window", "hand_locked"]

Freshness(
    source_frame_id: str,
    current_frame_revalidated: bool,
    critical_fields_fresh: bool,
    action_row_fresh: bool,
    state_age_ms: int | None,
    stable_frame_count: int,
    stable_duration_ms: int,
    tracker_hand_id: str | None,
    tracker_generation: int | None,
)
```

A previous valid state may be used for continuity and transition validation. It must
not authorize policy decisions or click plans unless the current frame/window is
revalidated and all contract-required fields are fresh.

### 6.3 Contract levels

```python
ContractLevel = Literal[
    "observe_only",
    "game_state",
    "policy_decision",
    "fold_check_only",
    "call_decision",
    "sizing_decision",
    "click_plan",
]
```

Examples:

- `observe_only`: enough to show HUD evidence; no AI decision.
- `game_state`: enough to construct a partial/internal state; not necessarily safe for
  policy.
- `fold_check_only`: permits only passive dry-run decisions when legal actions and
  screen/action row are trusted.
- `call_decision`: requires trusted call amount and hero stack.
- `sizing_decision`: requires pot, hero stack, min raise, and sizing-relevant stack
  context.
- `click_plan`: requires a fresh action row and click target evidence; still does not
  execute a click.

If the current AI cannot explicitly operate in a lower contract mode, the assembler
must fail closed instead of guessing.

### 6.4 Contract requirements

```python
ContractRequirement(
    field_path: str,
    required_for: tuple[ContractLevel, ...],
    risk_tier: str,          # critical | strategy | auxiliary
    min_confidence: float,
    freshness_required: bool,
    allowed_sources: tuple[str, ...],
)
```

The decision contract, not the raw observation, decides when `pot`, `hero_stack`, or
`call_amount` is critical.

### 6.5 Structured issues

Do not model missing fields and contradictions as free strings.

```python
AssemblyIssue(
    issue_type: str,         # missing | contradiction | low_confidence | stale | source_policy | ambiguous
    reason_code: str,
    field_path: str,
    rule_name: str,
    severity: str,           # info | warning | hard
    blocking: bool,
    required_by_contract: tuple[ContractLevel, ...],
    observed_value: object | None,
    candidate_values: tuple[object, ...],
    source: str | None,
    evidence_refs: tuple[str, ...],
    message: str,
)
```

Suggested `reason_code` values include:

```text
SCREEN_NOT_ACTIONABLE
HERO_TURN_NOT_CONFIRMED
LAYOUT_LOW_CONFIDENCE
MISSING_HERO_CARDS
MISSING_BOARD_CARD
DUPLICATE_CARD
ACTION_ROW_UNSTABLE
PRESELECT_AMBIGUOUS
CALL_AMOUNT_UNTRUSTED
POT_REQUIRED_BY_POLICY
TEMPORALLY_UNSTABLE
STALE_TRACKER_STATE
HAND_BOUNDARY_PENDING
TRUTH_ASSISTED_FIELD_IN_IMAGE_ONLY_MODE
```

### 6.6 `GameStateAssemblyResult`

```python
GameStateAssemblyResult(
    status: AssemblyStatus,
    validity_scope: ValidityScope,
    state: GameState | None,
    contract_level: ContractLevel,
    contract_status: str,    # satisfied | blocked | not_evaluated
    valid_for: tuple[ContractLevel, ...],
    issues: tuple[AssemblyIssue, ...],
    freshness: Freshness,
    field_confidences: dict[str, float],
    critical_min_confidence: float | None,
    layout_confidence: float,
    screen_confidence: float,
    rule_consistency: str,
    observation_id: str,
)
```

Scalar confidence may be retained for display, but it must not be the main gate.
Contracts and blocking issues drive safety.

### 6.7 `RecognitionResult` migration

`RecognitionResult` should be extended, not used as a metadata-only carrier:

```python
RecognitionResult(
    state: GameState | None,
    confidence: float,
    metadata: Mapping[str, object],
    screen: ScreenState,
    visual_observation: VisualObservation | None,
    assembly_result: GameStateAssemblyResult | None,
    recognition_mode: RecognitionMode,
    safety_contract: ContractLevel,
)
```

Compatibility:

- `metadata["state_block_reason"]` mirrors the first blocking issue reason code.
- `metadata["visual_observation"]` and `metadata["assembly_result"]` may mirror the
  structured fields for old report paths.
- New safety/HUD/dry-run code should read the typed fields first.

## 7. Rules and temporal tracker

Rules are used at three levels:

- observation validation;
- tracker transition validation;
- assembled-state and decision-contract validation.

The tracker must be a rule-constrained state machine, not "update first, validate
after." Transition rules should prevent polluted hand context.

Initial tracker state:

- `hand_id` and generation;
- accepted and locked hero cards;
- accepted and locked board cards;
- inferred street;
- current action row generation;
- rolling numeric locks tied to action row/street/hand;
- blocked/overlay pause state;
- hand-boundary pending state;
- last valid result for continuity only.

Tracker rules:

- card locks require repeated agreement;
- hero cards cannot change inside a confirmed hand;
- board is append-only: `0 -> 3 -> 4 -> 5`;
- action row requires debounce and row-level confirmation;
- overlay appearance blocks action immediately;
- overlay clear requires re-stabilization before any action contract is satisfied;
- hand boundary uses `HandBoundarySuspected -> NewHandPending -> NewHandConfirmed`;
- pending hand-boundary states cannot output actionable `GameState`;
- critical numeric locks reset when action row, button label, street, hand boundary,
  overlay state, or relevant crop hash changes;
- call amount locks are bound to action-row generation, button slot, and call-label
  crop evidence.

The existing `PokerLegendsSessionTracker` should be adapted to consume
`VisualObservation` or a derived frame observation. It should not be replaced until the
new invariants are covered by tests.

## 8. Evaluation redesign

Evaluation must report authorization safety, not only field accuracy.

### 8.1 Required report partitions

Reports must separate:

```text
image_only_live
image_only_replay
truth_assisted_replay
synthetic_test
```

Mixed-mode aggregate metrics must not be used as live safety evidence.

### 8.2 Metrics

Screen and actionability:

- false actionable count;
- actionable precision/recall;
- blocked-overlay recall;
- preselect/modal false positives;
- animation/settlement rejection.

Recognizer quality:

- raw top-1 accuracy;
- accepted prediction precision;
- accepted prediction coverage;
- high-confidence wrong count;
- accepted critical wrong count.

Assembly and contracts:

- status distribution;
- contract-level distribution;
- blocking issue distribution;
- truth-assisted critical-field rejection count;
- preserved/dropped baseline frame delta from the 47/57 prototype baseline.

Authorization events:

- authorization event count;
- unsafe authorization events;
- stale authorization events;
- truth-assisted authorization events;
- decision-contract violations.

Decision-impacting errors:

- `recognized_state + AI -> decision A`;
- `canonical_state + AI -> decision B`;
- action-class disagreement;
- sizing-bucket disagreement;
- legal-action-set disagreement;
- call-amount threshold crossing;
- call amount underread/overread;
- pot underread/overread;
- hero stack underread/overread.

Temporal:

- hand-boundary accuracy;
- illegal transition count;
- stale-state blocks;
- overlay-clear re-stabilization latency;
- action-row generation resets;
- numeric lock reset correctness.

### 8.3 Hard negative set

Slice 2 must include a negative safety set, not only 57 actionable frames:

- blocked overlay;
- modal popup;
- Steam/system overlay;
- lobby/menu;
- preselect-only strip;
- showdown;
- winner banner;
- settlement;
- dealing animation;
- table observe but not hero turn;
- hero folded;
- all-in/no ordinary action;
- stale captured frame;
- layout mismatch;
- wrong resolution/DPI;
- partial obstruction.

`false actionable = 0` is meaningful only against this negative set and authorization
events.

## 9. Implementation slices

### Slice 0: mode separation and evidence invariants

- Add `RecognitionMode`.
- Add source-policy checks for accepted critical fields.
- Add frame evidence and ROI evidence IDs/hashes.
- Add a minimal evaluator that records accepted critical wrong cases.
- Keep current state assembly behavior otherwise unchanged.

Acceptance:

- image-only modes reject `reviewed_truth` as an accepted critical source;
- reports identify source mode for every frame;
- evidence IDs are present for card/button/number observations.

### Slice 1: observation schema, contracts, and compatible assembler

- Add `VisualObservation`, `ActionPanelObservation`, typed observation dataclasses,
  `GameStateAssemblyResult`, structured issues, and contract-level objects.
- Extend `RecognitionResult` with typed fields while preserving existing constructor
  compatibility where practical.
- Convert existing card/button/number predictions into observations.
- Extract current `_state_from_table` logic into a contract-aware assembler.
- Do not output `temporally_stable_valid` in this slice.
- Mirror typed results into legacy metadata for HUD/report compatibility.

Acceptance:

- current table recognizer tests pass;
- `state_block_reason` remains visible;
- every previous 47/57 valid prototype frame is categorized as preserved, blocked by a
  named issue, truth-assisted, temporal-not-available, or previous-false-valid;
- no dependency or model changes.

### Slice 2: evaluator and negative safety set

- Add grouped observation/assembly/contract report.
- Separate image-only and truth-assisted results.
- Add hard negative safety fixtures.
- Report authorization events and accepted critical wrong counts.

Acceptance:

- false actionable and unsafe authorization counts are explicit;
- high-confidence wrong critical predictions are listed with frame, slot, source, and
  evidence refs;
- the report cannot hide truth-assisted fields inside image-only metrics.

### Slice 3: temporal tracker integration

- Adapt the existing session tracker to consume `VisualObservation` or derived frame
  observations.
- Add card locks, board append-only validation, action-row generation, overlay
  re-stabilization, hand-boundary pending states, and numeric reset rules.
- Permit `temporally_stable_valid` only after current frame/window revalidation.

Acceptance:

- previous stable state cannot authorize current action;
- overlay clear requires a fresh stabilization window;
- old board/hero/action/numeric values cannot leak into a new hand;
- tracker reasons are visible in HUD/report output.

### Slice 4: critical recognizer hardening

- Improve only critical fields first: current action row, call amount, hero stack, pot,
  hero cards, board cards.
- Continue targeted work on blocker categories:
  `missing_legal_actions`, `not_enough_players`, `missing_pot`,
  `missing_call_amount`, and `hero_not_current`.
- Add reviewed samples through the existing dense-scan workflow.

Acceptance:

- recognizer coverage improves only when accepted critical wrong count remains zero in
  reviewed sets;
- no coverage improvement may depend on truth-assisted critical fields in image-only
  mode.

### Slice 5: coverage improvement under temporal contracts

- Only after Slice 3 should assembly coverage become a primary optimization target.
- Use temporal contracts to decide whether lower-confidence single-frame fields become
  acceptable after repeated agreement.

Acceptance:

- increased `temporally_stable_valid` coverage;
- no increase in unsafe authorization or stale authorization events.

## 10. Success criteria before real clicking

Before any later discussion of real clicking:

- false actionable count is zero on reviewed hard negative sets;
- unsafe authorization event count is zero;
- stale authorization event count is zero;
- truth-assisted authorization event count is zero in image-only modes;
- accepted critical wrong count is zero for hero cards, board cards, current action row,
  current button labels, and call amount;
- no duplicate visible cards enter accepted state;
- no call action is considered without trusted call amount;
- no sizing contract is satisfied without pot/stack/min-raise requirements;
- no previous stable state authorizes the current frame;
- long dry-run evidence packages contain `screen`, `visual_observation`,
  `assembly_result`, `contract`, `state`, `policy_decision`, and
  `click_plan.executed=false`.

High recall is not required early. A conservative system that acts on fewer frames but
never authorizes unsafe actions is the intended direction.
