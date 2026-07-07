# Poker Legends Number OCR Experiments

This note records the offline number-OCR experiments so far. None of these
recognizers are connected to runtime click or authorization paths.

## Current Data

- Dataset: `artifacts/poker-legends-videos/number_crop_dataset_v8`
- Main field evaluated: `hero_stack`
- Crop dataset: 119 reviewed frames, 714 crops, 458 labeled crops, and 36 crop-level
  review overrides applied.
- Rows used by the latest component experiments: 309 rows total, 300 positive rows
  and 9 hard-negative rows; split is 246 train / 63 test. The current deterministic
  split placed the 9 hard-negative rows outside the test split, so v6 component test
  metrics are positive-crop metrics. Hard-negative safety still has to be judged from
  full-shadow review queues and a future frozen stratified test set.
- Current reviewed issue classes:
  - Some crops have valid base text but weak or noisy overlay text.
  - Some truth rows are polluted, for example rows 31-33 in the review HTML show no
    usable number while truth still expects `$43,044`. This is now captured in
    `docs/poker-legends-number-review-overrides.json` as a no-visible-number override
    for `session_002__keyframe_000047` hero-stack variants.
  - The 24-row hard-negative review pass found 21 visible but unlabeled stack crops
    and 3 ROI-invalid player-name crops, not additional no-visible-number rows.
  - A later promotion audit labeled three additional visible stack rows:
    `$345+10`, `$900+100`, and `$365+100`.
  - `display` should be treated as a derived value from `base + overlay`, not as the
    primary runtime OCR target.

## Results So Far

### Tesseract Crop OCR Baselines

Early crop-level Tesseract was useful only as a weak baseline. It produced too many
accepted wrong results for direct runtime use.

- It helped expose ROI/truth problems.
- It should not be used as an accepted critical-field authority.

### Character Segmentation + Template/CNN

The first character-level route split crops into character boxes and classified each
glyph.

- v1: template on full stack-style text had high precision but low coverage.
- v2: switching `hero_stack` to white `base` text improved segmentation coverage from
  153/273 to 242/273.
- v3: component split added three explicit targets:
  - `base`: white available-stack text
  - `overlay`: cyan plus/current-bet text
  - `display`: combined visible text

Latest component report:

- Report: `artifacts/poker-legends-videos/number_real_hero_stack_components_cnn_template_v6/number_char_recognizer_report.md`
- HTML: `artifacts/poker-legends-videos/number_real_hero_stack_components_cnn_template_v6/number_char_recognizer_review.html`

Key metrics:

| Target | Model | Exact | Accepted | Accepted wrong |
|---|---|---:|---:|---:|
| base | CNN | 57/63 | 57/63 | 0 |
| base | template KNN | 57/63 | 57/63 | 0 |
| base | template+CNN | 57/63 | 57/63 | 0 |
| overlay | CNN | 27/30 | 27/30 | 0 |
| overlay | template+CNN | 27/30 | 27/30 | 0 |
| display | CNN | 30/63 | 30/63 | 0 |
| display | template+CNN | 30/63 | 30/63 | 0 |
| base | Tesseract | 38/63 | 63/63 | 25 |

Interpretation:

- Component split is valuable.
- The current CNN/template path is the best working baseline.
- Review overrides now carry both cleanup labels and newly reviewed visible stack
  labels. They keep known truth pollution such as `$43,044` out of positive OCR
  training/evaluation while adding reviewed rows that were previously unlabeled.
- A later manual review added 21 visible stack labels to train data:
  `$290+710`, `$1,000`, `$995+5`, `$900+100`, `$475+200`, and `$475`; it also
  excluded 3 player-name crops as ROI-invalid.
- Template KNN voting fixed 0/9 nearest-neighbor tie failures that previously made
  template misread `$990` and `$399` as `$900` / `$390`.
- Template / CNN / MLP are now trained per target (`base`, `overlay`, `display`) so
  overlay/display symbols cannot contaminate base hard-negative evaluation.
- Overlay mask cleanup removes horizontal cyan rule-line contamination. On v6 the
  overlay target segments 165/174 overall and 27/30 on the current test split.
- Template+CNN relaxed agreement accepts only same-text predictions that pass the
  target contract and stay near the normal template/CNN thresholds; this recovered
  the reviewed `+80` cases without adding accepted mismatches in the current
  promotion view.
- Template-only is not promoted to runtime authority from this small split; the safer
  base reporting baseline remains CNN or template+CNN, both at 57/63 accepted with
  accepted wrong 0; overlay template+CNN is now 27/30 accepted with accepted wrong 0.
- `display` coverage is low because combined segmentation is fragile.
- Runtime should prefer structured `base` and `overlay` observations, then derive
  `display` through rules.
- v6 uses the v8 reviewed-overrides dataset and writes full `shadow_evaluation_rows`:
  309 shadow rows total, while `evaluation_rows` remains the 63-row test split used
  for metrics and review HTML.

### Table Recognizer Shadow Reporting

The component CNN/template output can now be loaded from
`number_char_recognizer_summary.json` as table-recognizer shadow evidence. This is
reported under `shadow_number_predictions` and
`accepted_shadow_number_predictions`; it is not fed into `RecognizedTable`,
`GameStateAssemblyResult`, accepted critical fields, policy decisions, or click
authorization.

Latest shadow reports:

- Test-split truth-assisted:
  `artifacts/poker-legends-videos/multi_source_templates_v2/table_recognizer_number_shadow_v2/`
- Test-split image-only replay:
  `artifacts/poker-legends-videos/multi_source_templates_v2/table_recognizer_number_shadow_image_only_v1/`
- Full-shadow truth-assisted:
  `artifacts/poker-legends-videos/multi_source_templates_v2/table_recognizer_number_shadow_full_v2/`
- Full-shadow image-only replay:
  `artifacts/poker-legends-videos/multi_source_templates_v2/table_recognizer_number_shadow_full_image_only_v2/`

Key safety/coverage checks on the 119-frame reviewed manifest:

| Mode | Authorization events | Unsafe authorization | Shadow raw | Shadow accepted |
|---|---:|---:|---:|---:|
| truth-assisted | 39 | 0 | 57 | 54 |
| image-only replay | 0 | 0 | 57 | 54 |

Shadow truth comparison for `hero_stack`:

- raw shadow: 54 matches / 3 mismatches against reviewed truth.
- accepted shadow: 54 matches / 0 mismatches against reviewed truth.
- accepted total-number component: 54/54 match; base-number matches only 30/54 because
  reviewed stack truth sometimes represents the displayed total (`base + overlay`).
- Shadow component consensus now allows a template base fallback only when accepted
  display text exactly agrees with the assembled `base + overlay` text. This recovered
  the reviewed `$399+5` case without adding accepted shadow mismatches.

The table recognizer also writes focused review queues for shadow-number issues:

- `table_recognizer_shadow_number_review_rows.json`
- `table_recognizer_shadow_number_review_by_flag.json`
- `table_recognizer_shadow_number_review_by_class.json`
- `table_recognizer_shadow_number_promotion_rows.json`
- `table_recognizer_shadow_number_promotion_by_class.json`

Current queue on both truth-assisted and image-only reports:

- 1 review row out of 119 frames.
- `session_002__keyframe_000047`: no accepted shadow prediction; raw predictions are
  rejected by component disagreement and mismatch polluted reviewed truth (`$43,044`).

Image-only readiness diagnostics now include `shadow_number_readiness_flags`. Current
test-split result:

- 6 image-only `hero_stack` readiness gaps are tagged `shadow_missing_hero_stack`.
- None are currently tagged `shadow_covers_hero_stack_readiness_gap`.
- The reason is coverage of the component summary artifact, not a runtime promotion
  decision.

Full-shadow result:

- The component summary now exposes all 309 cleaned rows through
  `shadow_evaluation_rows`.
- In image-only table replay, all 6 current `hero_stack` readiness gaps are now tagged
  `shadow_covers_hero_stack_readiness_gap`.
- Full-shadow accepted predictions are not safe to promote: table replay reports 266
  accepted shadow `hero_stack` predictions, but 8 accepted mismatches against current
  reviewed truth and 29 shadow review rows. Treat full shadow as a diagnostic/candidate
  source until those mismatches are reviewed or filtered.
- The full-shadow review queue is now classified so the risk is easier to triage. On
  both truth-assisted and image-only full-shadow reports, the 29 review rows include:
  3 `accepted_mismatch_non_actionable`, 8 `no_accepted_non_actionable`, 12
  `rejected_variant_noise_with_accepted_match`, 20 `roi_or_segmentation_gap`, 2
  `truth_pollution_suspected`, and 3 `needs_review` rows. Classes can overlap. The
  important safety point is that the accepted mismatches currently appear in
  non-actionable frames, but this still is not enough evidence to promote full-shadow
  predictions into accepted runtime fields.
- A separate promotion view now filters to truth-actionable `hero_stack` frames and
  reports the hypothetical risk of using shadow OCR only in current action contexts.
  Truth-assisted full-shadow has 50 `candidate_match_actionable` rows and 7
  `no_shadow_hero_stack` rows. All 7 no-shadow truth-assisted rows are excluded by
  image-only ScreenState as not currently actionable. Image-only full-shadow has 41
  `candidate_match_actionable` rows and 16 `excluded_screen_not_actionable` rows, with
  no remaining `no_shadow_hero_stack`. There are currently 0
  `candidate_mismatch_actionable` rows in this promotion view; this is encouraging but
  still remains offline evidence, not a runtime gate change.

### CRNN+CTC and Transformer+CTC Prototype

Both sequence OCR variants were added as opt-in offline baselines.

- Report: `artifacts/poker-legends-videos/number_char_sequence_v1/number_char_recognizer_report.md`
- HTML: `artifacts/poker-legends-videos/number_char_sequence_v1/number_char_recognizer_review.html`
- Command shape:
  - `--enable-ctc --disable-tesseract --crnn-epochs 12 --transformer-epochs 12`

Result:

| Target | Model | Exact | Accepted | Accepted wrong |
|---|---|---:|---:|---:|
| base | CRNN+CTC | 0/57 | 0/57 | 0 |
| base | Transformer+CTC | 3/57 | 0/57 | 0 |
| overlay | CRNN+CTC | 0/24 | 0/24 | 0 |
| overlay | Transformer+CTC | 0/24 | 0/24 | 0 |
| display | CRNN+CTC | 0/57 | 0/57 | 0 |
| display | Transformer+CTC | 3/57 | 0/57 | 0 |

Interpretation:

- This prototype did not converge.
- The result does not prove sequence OCR is unsuitable.
- It only proves that a small, lightly trained sequence model on the current noisy
  dataset is not an immediate replacement for component CNN/template.

## Literature / Implementation Notes

- CRNN+CTC is a mature OCR route for unsegmented text-line recognition. It avoids
  explicit character box segmentation by learning alignments through CTC.
- CRNN-style recognition is still used in practical OCR toolkits such as EasyOCR,
  whose recognition model is documented as CRNN with CTC.
- PaddleOCR's historical PP-OCR line also used DB+CRNN, and PaddleOCR exposes multiple
  CTC-style recognizers in its recognition stack.
- Transformer OCR systems such as TrOCR and PARSeq reach strong general scene-text
  results, but their reported strength relies on large-scale pretraining/synthetic
  data and more model capacity.
- For this project, language/context modeling provides little value because the target
  alphabet is tiny and numeric: `$0123456789+,.KM`.

## Two-Way Choice

If we choose exactly one sequence OCR path for the next serious experiment:

**Choose CRNN+CTC.**

Reasons:

1. It targets the main current failure: brittle character segmentation.
2. It has a better expected data/compute fit for fixed UI, fixed font, small alphabet,
   and short numeric strings.
3. It is easier to train from synthetic Poker Legends-like data without relying on
   large pretrained OCR weights.
4. It is easier to keep fail-closed and auditable: greedy CTC text, blank rate,
   confidence by timestep, and accepted precision are straightforward to inspect.
5. Transformer OCR may be stronger in broad OCR benchmarks, but the expected marginal
   benefit here is lower unless we bring in large pretrained models or substantial
   synthetic data.

Recommended next serious CRNN+CTC experiment:

- Train separate models for `base` and `overlay`.
- Use cleaned labels only; exclude `roi_invalid` / no-visible-number rows.
- Add synthetic pretraining from extracted glyphs and real backgrounds.
- Track train loss, validation exact, blank rate, accepted precision, and accepted
  coverage.
- Compare only against current component CNN/template, with accepted wrong required to
  stay at zero.

Transformer OCR / ViT+CTC / encoder-decoder OCR should remain a later option if CRNN
does not improve after the data and synthetic-pretraining work.

## References

- CRNN paper: https://arxiv.org/abs/1507.05717
- CTC loss documentation: https://docs.pytorch.org/docs/stable/generated/torch.nn.CTCLoss.html
- CTC alignment explainer: https://distill.pub/2017/ctc
- EasyOCR repository: https://github.com/JaidedAI/EasyOCR
- TrOCR paper: https://arxiv.org/abs/2109.10282
- TrOCR documentation: https://huggingface.co/docs/transformers/en/model_doc/trocr
- PARSeq paper: https://arxiv.org/abs/2207.06966
- PaddleOCR text recognition documentation: https://paddlepaddle.github.io/PaddleX/3.1/en/module_usage/tutorials/ocr_modules/text_recognition.html
