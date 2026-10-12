# NumCog-Bench — clean, pre-stitched release

A cognitively-grounded benchmark for **numerical cognition in vision-language models**,
covering **exact counting** and **approximate numerosity discrimination (ANS)**. This is the
cleaned, self-contained release: **one question = one ready-to-use image** (paired
discrimination stimuli are already stitched left|white-divider|right), with one tidy
`metadata.csv` per experiment folder.

Regenerate from source any time with `../harness/build_dataset.py`.

## Folders (merged by question type) & tasks

| Folder | Task | Images | Stimulus |
|---|---|---:|---|
| `exp1_counting/` | Count items in one image | 1,994 | abstract shapes + real-world (FSC-133), small-n & large-n |
| `exp2_discrimination/` | Which side has more? (abstract, random sizes) | 480 | pre-stitched pairs, 6 ratio conditions |
| `exp3_controlled/` | Which side has more? (numerosity decorrelated from area/spacing) | 2,500 | pre-stitched pairs, 3 shapes × 4 colors |
| `exp4_realworld/naming/` | Name the object category | 864 | single real-world panels |
| `exp4_realworld/discrimination/` | Which side has more? (real-world) | 600 | pre-stitched same-/diff-category pairs |
| `exp5_synthetic_matched/` | Which side has more? (abstract matched to real-world) | 1,164 | pre-stitched pairs, counts/coords matched to Exp4 |

Every paired image is `left | 10px white divider | right`, panels vertically centered on white.

## Metadata schemas (differ by task — read the header)

`image` is always relative to that folder's `images/`. `count`/`*_count`/`*_numerosity` are
ground-truth object counts; for discrimination, `answer`/`correct_answer` ∈ {`left`,`right`,`tie`}.

- **exp1_counting/metadata.csv** — `image, subset, stimulus_type{abstract,realworld}, n_range{small_n,large_n}, shape, category, count`
  (`subset` ∈ {random_smalln, random_largen, rw_smalln, rw_largen}; `shape` for abstract, `category` for real-world)
- **exp2_discrimination/metadata.csv** — `image, pair_id, shape{dots,rectangles}, anchor_n, ratio_condition{0.5..0.9,n+1}, ratio_value, left_count, right_count, correct_answer`
- **exp3_controlled/metadata.csv** — `image, pair_num, pair_shape{dots,ellipse,rectangle}, pair_color{white,blue,red,yellow}, pair_ratio, left_numerosity, right_numerosity, left_TSA, right_TSA, left_FA, right_FA, correct_answer`
  (TSA = total surface area, FA = field area / convex hull — for the numerosity-vs-magnitude decomposition)
- **exp4_realworld/naming/metadata.csv** — `image, category, count, original_image_name`
- **exp4_realworld/discrimination/metadata.csv** — `image, pair_num, condition{same_cat,diff_cat}, left_category, right_category, left_count, right_count, num_ratio, answer`
- **exp5_synthetic_matched/metadata.csv** — `image, pair_num, condition, shape_type{dots,rectangles}, left_category, right_category, left_count, right_count, num_ratio, answer`

## Replicate an evaluation

`../harness/build_manifests.py` turns this dataset into trial lists, and `../harness/run_eval.py`
runs any VLM over them, emitting per-experiment `result.json` analyzed by `all_exp_analysis.qmd`.
See `../PIPELINE.md`.

## Provenance & notes
- Derived from the original `Experiment1…Experiment4`, `Experiment1_1`, `Experiment2_1`,
  `Experiment3_1` stimulus sets.
- **Excluded** (kept out of this release): Exp1 `random_largen`-in-small-n bucket, real-world
  `FSC_sampled`/`FSC_sampled_largen`, and Exp3 `ellipses_white`/`rectangles_white` (rendered
  incorrectly — replaced by the Exp3_1 re-renders). 36 synthetic pairs with missing source
  images and 1 exact-duplicate counting row were dropped.
- **License:** real-world images derive from FSC-133 (Ranjan et al., 2021) / FSC-147; abstract
  stimuli are procedurally generated and freely usable. Respect upstream FSC licensing.
