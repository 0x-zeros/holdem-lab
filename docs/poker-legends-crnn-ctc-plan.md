# Poker Legends CRNN+CTC OCR Plan

This plan defines the next serious OCR experiment for Poker Legends stack numbers.
It is a design and experiment plan only. It does not authorize runtime use, click
planning, or live automation.

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
- validation exact match
- validation accepted precision
- validation accepted coverage
- blank decode rate
- average decoded length
- high-confidence wrong count

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
- string confidence: min char confidence or calibrated aggregate
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

## Experiment Slices

### Slice 1: Dataset Audit

- Add explicit row status labels.
- Exclude `roi_invalid` and `no_visible_number` from positive training.
- Generate a review report focused on label/ROI quality.

Output:

- cleaned manifest
- audit summary
- review HTML

### Slice 2: Synthetic Generator

- Build synthetic `base` and `overlay` crops from extracted glyphs and backgrounds.
- Keep synthetic and real rows separately tagged.

Output:

- synthetic manifest
- sample contact sheet
- generator parameters

### Slice 3: CRNN+CTC Training Instrumentation

- Train `base` and `overlay` models.
- Save curves and epoch-level metrics.
- Report blank rate and decoded samples.

Output:

- training summary JSON
- model artifacts
- validation report

### Slice 4: Baseline Comparison

- Compare CRNN+CTC against CNN/template on the same cleaned test set.
- Produce side-by-side review HTML.
- Include segmentation-mismatch rows as a special slice.

Output:

- comparison report
- review HTML
- recommendation: continue / tune / stop

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
