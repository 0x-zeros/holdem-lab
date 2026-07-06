# Poker Legends CRNN+CTC OCR Plan

This plan defines the next serious OCR experiment for Poker Legends stack numbers.
It is a design and experiment plan only. It does not authorize runtime use, click
planning, or live automation.

## Current Verdict

Proceed only to **P0-strengthened offline experiment implementation**. Do not treat
the next training run as a definitive CRNN+CTC benefit test until the falsification
harness, frozen split, time-step budget checks, validation-only thresholding, and
hard-negative gate are in place.

## Decision

For the next serious sequence OCR experiment, choose **CRNN+CTC** over Transformer
OCR / ViT+CTC / encoder-decoder OCR.

Reason:

- The current main failure is brittle character segmentation, especially for
  `overlay` and combined `display` strings.
- CRNN+CTC directly addresses unsegmented text-line recognition without requiring
  large pretrained OCR models.
- The domain is narrow: fixed UI, fixed font family, short numeric strings, and a
  tiny alphabet.
- The expected near-term benefit is higher than Transformer OCR because we can train
  a small domain-specific model using real crops plus synthetic augmentation.

Transformer OCR remains a later option if CRNN+CTC still fails after data cleaning,
synthetic pretraining, and proper convergence tracking.

## Non-Goals

- Do not connect CRNN+CTC to runtime `RecognitionResult` accepted fields.
- Do not use CRNN+CTC to authorize AI decisions or click plans.
- Do not treat `display` as the primary runtime number target.
- Do not lower safety thresholds to make coverage look better.
- Do not download datasets, pretrained weights, or models from unofficial sources.

## Target Semantics

The OCR targets are component-based:

- `base`: white available-stack text, for example `$935`.
- `overlay`: cyan plus/current-bet text, for example `+80`.
- `display`: derived review target, for example `$935+80`.

Runtime direction:

```text
base observation + overlay observation -> structured numeric components
structured components + seat/current-bet rules -> validated stack semantics
```

`display` is useful for human review, but should not be the primary runtime OCR
contract.

## Required Pre-Work

### 0. CTC Falsification Harness

Before any real CRNN+CTC result can be interpreted, the implementation must prove
that the CTC pipeline itself is sane. This is a hard P0 gate.

The harness must include:

- overfit 20 clean real crops to near-100% exact
- overfit 1000 clean synthetic crops to near-100% exact
- include repeated-character cases such as `$1000`, `+1000`, `$43,044`, and `+55`
- include a deliberately too-short input case that fails loudly
- unit-test blank index, alphabet index, target encoding, target length, log-softmax
  shape, and `(T, N, C)` layout
- report CTC input time steps `T`, target length, required CTC length, and
  `T / required`

No blank-collapse result on the real dataset is meaningful until this harness passes.

Implementation anchor:

```bash
uv run holdem-bot-run-poker-legends-ctc-sanity \
  --out artifacts/poker-legends-videos/ctc_sanity_v1 \
  --target base \
  --target overlay \
  --synthetic-count 1000 \
  --epochs 600 \
  --batch-size 4 \
  --weight-decay 0.0
```

The harness is offline-only. It writes `ctc_sanity_summary.json`, records the
too-short `$1000` failure case, checks `(T, N, C)` CTC logit layout, checks
`T >= required_ctc_length * 2`, and reports overfit exact/accepted metrics. A small
smoke run may legitimately report `passed=false`; that is useful only to verify the
command and artifact path. A real P0 pass requires the configured overfit run to
reach near-100% exact with `accepted_wrong = 0`.

For reviewed real crops:

```bash
uv run holdem-bot-run-poker-legends-ctc-sanity \
  --manifest artifacts/.../number_crop_dataset_manifest.json \
  --out artifacts/poker-legends-videos/ctc_sanity_real_v1 \
  --target base \
  --target overlay \
  --real-count 20 \
  --synthetic-count 1000 \
  --epochs 120
```

The manifest passed here must already be reviewed clean / labeled-visible. The
runner does not promote manifest truth into runtime OCR and does not authorize click
planning.

Current implementation status:

- CTC blank/index/layout checks are implemented in `poker_legends_ctc.py`.
- The CRNN trainer now uses per-sample effective `input_lengths` derived from the
  non-empty sequence width; this prevents right-padding blank regions from dominating
  short stack strings.
- Accepted CTC output is additionally gated by target format, so incomplete strings
  such as `$` are rejected even when raw confidence is high.
- Small clean synthetic sanity currently passes for both `base` and `overlay`:
  20/20 raw exact and `accepted_wrong = 0` using `--epochs 600 --batch-size 4
  --weight-decay 0.0`.
- The larger 1000-synthetic overfit gate is still pending. Earlier runs before the
  effective-input-length fix failed badly and must not be used to judge the final
  CRNN route.

### 1. Data Labels

Before serious training, the dataset must distinguish:

- `labeled_visible`: target is visible and label is trusted.
- `roi_invalid`: crop does not contain the intended number.
- `no_visible_number`: crop is valid but target is not visible.
- `overlay_present_unlabeled`: overlay appears in the image but truth has no overlay
  component.
- `ambiguous_or_animation`: visual state is not stable enough to train on.

Rows like current HTML 31/32/33, where the image has no usable number but truth expects
`$43,044`, must be excluded from positive training and reported separately.

### 2. Review Evidence

Each reviewed crop should show:

- original crop
- `base` mask
- `overlay` mask
- combined mask
- current segmentation boxes
- expected component labels
- model predictions

This keeps failures diagnosable as ROI, mask, label, or model errors.

## Dataset Split

Split by source/session/frame, not by crop row. The same screen/frame must not appear
in both train and test.

Recommended splits:

- train: 70%
- validation: 15%
- test: 15%

The split must be frozen before any glyph/background extraction. Synthetic data may
only use glyphs and backgrounds from the train split. Validation and test crops must
not leak into synthetic generation.

Thresholds, beam width, parser settings, grammar settings, and calibration parameters
may be tuned only on validation. Test is for final reporting only.

The test split should include hard negatives and edge cases:

- weak cyan overlay
- missing overlay
- `$0+...`
- comma values
- tight crops
- no-pad crops
- trim variants
- ROI-invalid rows
- no-visible-number rows

## Synthetic Pretraining

CRNN+CTC should not be trained only on the current few hundred real crops.

Build synthetic data from project-owned observations:

- extract clean real glyph masks from accepted `base` and `overlay` samples
- compose strings with the allowed alphabet
- paste onto real crop backgrounds
- apply augmentation:
  - subpixel shift
  - crop jitter
  - brightness/contrast changes
  - blur
  - antialiasing
  - weak cyan opacity
  - small edge contamination

Generate separate synthetic corpora:

- `base`: `$0`, `$5`, `$185`, `$1,005`, `$43,044`
- `overlay`: `+5`, `+80`, `+1000`
- optional `display`: `$935+80`, `$1,005+10`

Synthetic data should be marked as synthetic and reported separately from real-data
metrics.

## Model

Train separate CRNN+CTC models for:

- `base`
- `overlay`

Optionally train `display` only for review comparison.

Suggested first architecture:

```text
input grayscale/mask image: H=32, W=160 or W=192
CNN feature extractor
  Conv/ReLU/Pool blocks
  keep enough horizontal time steps
BiLSTM sequence model
linear projection to alphabet + blank
CTC loss
greedy decode for baseline
```

The model must preserve enough horizontal time steps. For every sample:

```text
T >= required_ctc_length * 2
```

`required_ctc_length` must account for repeated adjacent characters, since CTC needs
blank-separated repeats. Examples such as `1000`, `$43,044`, and `+1000` are explicit
budget tests. `zero_infinity=True` must not be used to hide impossible alignments.

Alphabet:

```text
base:    $0123456789,.KM
overlay: +0123456789,.KM
display: $+0123456789,.KM
```

Keep the alphabet target-specific where possible. This reduces confusion and makes
confidence easier to interpret.

## Training Protocol

Track these metrics every epoch:

- train CTC loss
- validation CTC loss
- validation exact match
- validation character error rate / edit distance
- validation accepted precision
- validation accepted coverage
- blank decode rate
- mean blank posterior
- nonblank occupancy
- average decoded length
- per-length exact
- per-character confusion
- sequence top1/top2 margin, if beam search is enabled
- high-confidence wrong count
- calibration / reliability table for accepted thresholds

Training should stop only after one of these is true:

- validation exact/coverage plateaus
- blank rate has stabilized low enough to be useful
- accepted wrong appears and cannot be removed by thresholding
- runtime budget is exceeded

Do not judge by fixed low epoch counts. The previous 3/12 epoch experiments only show
that the quick prototype did not converge.

## Confidence / Acceptance

For offline evaluation:

- raw prediction: greedy CTC decode
- char confidence: probability at accepted nonblank timesteps
- string confidence: min char confidence, sequence NLL, or calibrated aggregate
- accepted prediction: confidence above threshold and passes format parser

Required accepted formats:

- `base`: starts with `$`
- `overlay`: starts with `+`
- no unexpected letters unless explicitly allowed for `K/M`
- numeric parser must round-trip to canonical text

Runtime candidate threshold should be tuned to keep:

```text
accepted_wrong = 0
```

Coverage is secondary.

Greedy confidence is only a baseline. If CTC is promising, add a grammar-constrained
beam search and report top1/top2 margin. Accepted predictions must pass both text
canonicalization and numeric round-trip checks.

## Evaluation

Evaluate against the current component CNN/template baseline:

- base CNN
- overlay CNN
- template
- template+CNN consensus
- CRNN+CTC raw
- CRNN+CTC accepted
- optional CRNN+CTC + CNN/template agreement

Report per target:

- raw exact
- accepted exact
- accepted wrong
- accepted precision
- accepted coverage
- blank rate
- ROI-invalid rejection behavior
- no-visible-number rejection behavior
- hard-negative accepted count
- zero-event upper bound or confidence interval for accepted wrong

The most important result is not raw accuracy. It is:

```text
accepted critical wrong = 0
with better coverage than current component CNN/template
```

## Success Criteria

CRNN+CTC is worth continuing if it achieves all of:

- accepted wrong = 0 on reviewed test set
- base accepted coverage >= current CNN accepted coverage, or close with better
  stability on segmentation-mismatch rows
- overlay accepted coverage >= current overlay CNN/template coverage, or it correctly
  handles weak overlay cases currently missed by segmentation
- materially reduces segmentation-related missing predictions
- failure reasons are explainable through blank rate, confidence, or format rejection

CRNN+CTC should not proceed toward runtime if:

- high-confidence wrong predictions appear
- blank collapse remains high after synthetic pretraining
- performance only improves by using polluted labels
- it fails to beat current component CNN/template on accepted coverage at zero wrong

The reviewed test set must be large enough to make zero accepted wrong meaningful.
Until then, reports must show the zero-event upper bound instead of treating `0 wrong`
as proof of safety.

## Experiment Slices

### Slice 0: CTC Falsification Harness

- Implement CTC encoding/decoding unit tests.
- Add time-step budget checks.
- Add clean real overfit and synthetic overfit sanity tests.
- Add repeated-character cases and too-short-input failure tests.

Output:

- sanity report
- overfit curves
- decoded examples
- pass/fail status

### Slice 1: Dataset Audit

- Add explicit row status labels.
- Exclude `roi_invalid` and `no_visible_number` from positive training.
- Generate a review report focused on label/ROI quality.
- Freeze train/validation/test split by source/session/frame.

Output:

- cleaned manifest
- audit summary
- review HTML
- frozen split file

### Slice 2: Synthetic Generator

- Build synthetic `base` and `overlay` crops from extracted glyphs and backgrounds.
- Keep synthetic and real rows separately tagged.
- Ensure synthetic assets are generated only from train split inputs.

Output:

- synthetic manifest
- sample contact sheet
- generator parameters

### Slice 3: CRNN+CTC Training Instrumentation

- Train `base` and `overlay` models.
- Save curves and epoch-level metrics.
- Report blank rate and decoded samples.
- Run input ablations:
  - base: grayscale, mask, grayscale+mask
  - overlay: RGB/grayscale, cyan mask, RGB+mask
- Run training schedule ablations:
  - real-only
  - synthetic-only
  - synthetic pretrain + real finetune
  - mixed from scratch

Output:

- training summary JSON
- model artifacts
- validation report

### Slice 4: Baseline Comparison

- Compare CRNN+CTC against CNN/template on the same cleaned test set.
- Produce side-by-side review HTML.
- Include segmentation-mismatch rows as a special slice.
- Include hard negatives and require accepted=0 for `roi_invalid`,
  `no_visible_number`, and `ambiguous_or_animation`.

Output:

- comparison report
- review HTML
- recommendation: continue / tune / stop

## Review-Driven P0/P1/P2 Checklist

### P0

- CTC sanity harness passes before real metrics are interpreted.
- Input width/stride budget is asserted per sample.
- Splits are frozen before glyph/background extraction.
- Synthetic generator cannot use validation/test assets.
- Canonical text and numeric parser round-trip are defined.
- Accepted thresholds are validation-only; test is not used for tuning.
- Hard negatives are part of formal evaluation and require accepted=0.
- Current CNN/template baselines are rerun on the same cleaned split.

### P1

- Add input ablations for base and overlay.
- Add synthetic training schedule ablations.
- Add grammar-constrained beam search.
- Add sequence confidence calibration and margin reporting.
- Add CRNN+CTC plus CNN/template agreement mode.
- Add per-slice dashboards for weak overlay, comma values, zero values, long values,
  tight crops, segmentation-mismatch rows, and invalid ROI rows.

### P2

- Run EasyOCR / PaddleOCR / docTR / TrOCR only as offline benchmarks or sanity checks.
- Pin package/model versions and model hashes for any external model.
- Disable runtime auto-downloads.
- Keep external model outputs out of accepted critical fields.

## Open Questions

- Should `overlay` be trained from cyan mask only, RGB crop, or both?
- Should `base` and `overlay` use separate model weights or share a backbone?
- How much synthetic data is needed before real validation improves?
- Is width 160 enough for `$43,044` and long display strings, or should width be
  target-specific?
- Should CTC use greedy decode only, or add small beam search with a strict numeric
  grammar?

## Current Recommendation

Proceed with CRNN+CTC only as a serious offline experiment after dataset audit and
synthetic pretraining. Do not replace the current component CNN/template baseline yet.

If we must choose between CRNN+CTC and Transformer OCR today, choose CRNN+CTC because
the expected benefit per unit of data, compute, and engineering effort is higher for
this fixed-UI numeric OCR problem.
