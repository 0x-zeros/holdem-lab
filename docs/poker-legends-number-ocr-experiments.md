# Poker Legends Number OCR Experiments

This note records the offline number-OCR experiments so far. None of these
recognizers are connected to runtime click or authorization paths.

## Current Data

- Dataset: `/tmp/poker-legends-number-crop-dataset-v7`
- Main field evaluated: `hero_stack`
- Rows used by the latest component experiments: 300 rows total, 291 positive rows
  and 9 hard-negative rows; split is 243 train / 57 test, with 54 positive test rows
  and 3 hard-negative test rows. Three reviewed ROI-invalid rows are excluded.
- Current reviewed issue classes:
  - Some crops have valid base text but weak or noisy overlay text.
  - Some truth rows are polluted, for example rows 31-33 in the review HTML show no
    usable number while truth still expects `$43,044`. This is now captured in
    `docs/poker-legends-number-review-overrides.json` as a no-visible-number override
    for `session_002__keyframe_000047` hero-stack variants.
  - The 24-row hard-negative review pass found 21 visible but unlabeled stack crops
    and 3 ROI-invalid player-name crops, not additional no-visible-number rows.
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

- Report: `artifacts/poker-legends-videos/number_real_hero_stack_components_cnn_template_v4/number_char_recognizer_report.md`
- HTML: `artifacts/poker-legends-videos/number_real_hero_stack_components_cnn_template_v4/number_char_recognizer_review.html`

Key metrics:

| Target | Model | Exact | Accepted | Accepted wrong |
|---|---|---:|---:|---:|
| base | CNN | 51/54 | 51/54 | 0 |
| base | template KNN | 54/54 | 54/54 | 0 |
| base | template+CNN | 51/54 | 51/54 | 0 |
| overlay | CNN | 24/24 | 21/24 | 0 |
| overlay | template+CNN | 24/24 | 24/24 | 0 |
| display | CNN | 27/54 | 27/54 | 0 |
| display | template+CNN | 27/54 | 27/54 | 0 |

Hard-negative gate on the latest base split:

| Target | Method | Hard-negative false accepts |
|---|---|---:|
| base | CNN | 0/3 |
| base | template KNN | 0/3 |
| base | template+CNN | 0/3 |
| overlay | template+CNN | 0/3 |
| display | template+CNN | 0/3 |
| base | Tesseract | 3/3 |

Interpretation:

- Component split is valuable.
- The current CNN/template path is the best working baseline.
- Review overrides moved three polluted `$43,044` positive rows into hard-negative
  evaluation. Positive base test rows are now 54, with 3 no-visible-number hard
  negatives retained in the test artifact.
- A later manual review added 21 visible stack labels to train data:
  `$290+710`, `$1,000`, `$995+5`, `$900+100`, `$475+200`, and `$475`; it also
  excluded 3 player-name crops as ROI-invalid.
- Template KNN voting fixed 0/9 nearest-neighbor tie failures that previously made
  template misread `$990` and `$399` as `$900` / `$390`.
- Template / CNN / MLP are now trained per target (`base`, `overlay`, `display`) so
  overlay/display symbols cannot contaminate base hard-negative evaluation.
- Overlay mask cleanup removes horizontal cyan rule-line contamination, improving
  overlay segmentation from 137/153 to 144/153 overall and from 22/24 to 24/24 on
  the current test split.
- Template+CNN relaxed agreement accepts only same-text predictions that pass the
  target contract and stay near the normal template/CNN thresholds; this recovered
  the reviewed `+80` cases without adding hard-negative false accepts.
- Template-only is not promoted to runtime authority from this small split; the safer
  base reporting baseline remains CNN or template+CNN, both at 51/54 accepted with
  accepted wrong 0; overlay template+CNN is now 24/24 accepted with accepted wrong 0.
- `display` coverage is low because combined segmentation is fragile.
- Runtime should prefer structured `base` and `overlay` observations, then derive
  `display` through rules.

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
