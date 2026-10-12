"""
Rebuild ANS_Benchmark HF parquets with prompt + task columns.
Also deletes all raw image/CSV/junk files from the repo.

Steps:
  1. Load all 5 manifests → lookup dict keyed by _image path
  2. For each of the 6 configs, merge manifest fields into CSV rows
  3. Push new parquets (overwrites existing)
  4. Delete all non-parquet, non-README, non-.gitattributes files
  5. Write updated README (dataset card)
"""

import csv, json, os
from pathlib import Path
from datasets import Dataset, Features, Image, Value
from huggingface_hub import HfApi, CommitOperationDelete

REPO_ID = "87cc2s/ans-benchmark"
BASE    = Path(__file__).parent
MANIFEST_DIR = BASE.parent / "manifests"

api = HfApi()


# ─────────────────────────────────────────────────────────────── manifests ──
def load_manifests():
    """Return dict: _image_path → manifest record for all 5 tags."""
    lookup = {}
    for tag in ("ans", "ans2", "ans3", "ans4", "ans5"):
        path = MANIFEST_DIR / f"{tag}.jsonl"
        with open(path) as f:
            for line in f:
                rec = json.loads(line)
                lookup[rec["_image"]] = rec
    print(f"  Manifests loaded: {len(lookup)} unique trial records")
    return lookup


def task_label(rec):
    ct = rec.get("concept_type", "")
    if ct.startswith("ans_media") or ct == "Experiment1_1":
        return "counting"
    if ct == "object_recognition":
        return "naming"
    if ct.startswith("quantity_comparison"):
        return "discrimination_realworld"
    if ct == "Experiment2_1":
        return "discrimination_synthetic"
    if ct.startswith("paired_dots") or ct.startswith("paired_rects"):
        return "discrimination_abstract"
    if ct == "Experiment3_1":
        return "discrimination_controlled"
    return "unknown"


def clean_prompt(p):
    return p.replace("<image>", "").strip()


# ─────────────────────────────────────────────────────────────── builders ──
def read_csv(rel_path):
    with open(BASE / rel_path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def build_config(config_name, rows_with_paths, lookup, images_root_rel, correct_answer_src=None):
    """
    rows_with_paths    : list of (csv_row_dict, absolute_image_path, _image_key)
    _image_key         : the key into `lookup` (e.g. "exp1_counting/images/...")
    correct_answer_src : source column to alias as `correct_answer`; skipped if
                         None or if the column is already named `correct_answer`
    """
    records = []
    missing_prompt = 0
    for csv_row, abs_img, img_key in rows_with_paths:
        manifest = lookup.get(img_key)
        if manifest is None:
            missing_prompt += 1
            prompt, task, tag, eval_id = "", "unknown", "unknown", -1
        else:
            prompt  = clean_prompt(manifest["prompt"])
            task    = task_label(manifest)
            tag     = manifest["_tag"]
            eval_id = manifest.get("eval-id", -1)

        row = dict(csv_row)
        row["image"]   = str(abs_img)   # cast_column will embed bytes
        row["prompt"]  = prompt
        row["task"]    = task
        row["tag"]     = tag
        row["eval_id"] = eval_id
        if correct_answer_src and correct_answer_src != "correct_answer":
            row["correct_answer"] = row[correct_answer_src]
        records.append(row)

    if missing_prompt:
        print(f"  WARNING [{config_name}]: {missing_prompt} rows had no manifest entry")

    # Build column-oriented dict
    cols = {k: [] for k in records[0]}
    for r in records:
        for k, v in r.items():
            cols[k].append(v)

    ds = Dataset.from_dict(cols)
    ds = ds.cast_column("image", Image())
    return ds


# ─────────────────────────────────── per-config builders ──
def exp1_counting(lookup):
    rows = read_csv("exp1_counting/metadata.csv")
    items = []
    for r in rows:
        abs_img = BASE / "exp1_counting/images" / r["image"]
        img_key = f"exp1_counting/images/{r['image']}"
        items.append((r, abs_img, img_key))
    return build_config("exp1_counting", items, lookup, "exp1_counting/images", correct_answer_src="count")


def exp2_discrimination(lookup):
    rows = read_csv("exp2_discrimination/metadata.csv")
    items = []
    for r in rows:
        abs_img = BASE / "exp2_discrimination/images" / r["image"]
        img_key = f"exp2_discrimination/images/{r['image']}"
        items.append((r, abs_img, img_key))
    return build_config("exp2_discrimination", items, lookup, "exp2_discrimination/images", correct_answer_src="correct_answer")


def exp3_controlled(lookup):
    rows = read_csv("exp3_controlled/metadata.csv")
    items = []
    for r in rows:
        abs_img = BASE / "exp3_controlled/images" / r["image"]
        img_key = f"exp3_controlled/images/{r['image']}"
        items.append((r, abs_img, img_key))
    return build_config("exp3_controlled", items, lookup, "exp3_controlled/images", correct_answer_src="correct_answer")


def exp4_realworld_naming(lookup):
    rows = read_csv("exp4_realworld/naming/metadata.csv")
    items = []
    for r in rows:
        abs_img = BASE / "exp4_realworld/naming/images" / r["image"]
        img_key = f"exp4_realworld/naming/images/{r['image']}"
        items.append((r, abs_img, img_key))
    return build_config("exp4_realworld_naming", items, lookup, "exp4_realworld/naming/images", correct_answer_src="category")


def exp4_realworld_discrimination(lookup):
    rows = read_csv("exp4_realworld/discrimination/metadata.csv")
    items = []
    for r in rows:
        abs_img = BASE / "exp4_realworld/discrimination/images" / r["image"]
        img_key = f"exp4_realworld/discrimination/images/{r['image']}"
        items.append((r, abs_img, img_key))
    return build_config("exp4_realworld_discrimination", items, lookup, "exp4_realworld/discrimination/images", correct_answer_src="answer")


def exp5_synthetic_matched(lookup):
    rows = read_csv("exp5_synthetic_matched/metadata.csv")
    items = []
    for r in rows:
        abs_img = BASE / "exp5_synthetic_matched/images" / r["image"]
        img_key = f"exp5_synthetic_matched/images/{r['image']}"
        items.append((r, abs_img, img_key))
    return build_config("exp5_synthetic_matched", items, lookup, "exp5_synthetic_matched/images", correct_answer_src="answer")


# ─────────────────────────────────────────────────────── delete raw files ──
def delete_raw_files():
    print("\n── Deleting raw files from HF repo ──")
    keep_suffixes = {".parquet", ".md", ".gitattributes"}
    files = list(api.list_repo_files(REPO_ID, repo_type="dataset"))
    to_delete = [
        f for f in files
        if not any(f.endswith(s) for s in keep_suffixes)
        and f not in ("README.md",)
    ]
    print(f"  Files to delete: {len(to_delete)}")
    if not to_delete:
        print("  Nothing to delete.")
        return

    # Batch into chunks of 500 to avoid oversized commits
    BATCH = 500
    for i in range(0, len(to_delete), BATCH):
        batch = to_delete[i : i + BATCH]
        ops = [CommitOperationDelete(path_in_repo=f) for f in batch]
        api.create_commit(
            repo_id=REPO_ID,
            repo_type="dataset",
            operations=ops,
            commit_message=f"Remove raw image/CSV files (batch {i//BATCH + 1})",
        )
        print(f"  Deleted batch {i//BATCH + 1}: {len(batch)} files")
    print("  Raw file cleanup done.")


# ─────────────────────────────────────────────────────── README ──
README = """\
---
configs:
- config_name: exp1_counting
  data_files:
  - split: train
    path: exp1_counting/train-*.parquet
- config_name: exp2_discrimination
  data_files:
  - split: train
    path: exp2_discrimination/train-*.parquet
- config_name: exp3_controlled
  data_files:
  - split: train
    path: exp3_controlled/train-*.parquet
- config_name: exp4_realworld_naming
  data_files:
  - split: train
    path: exp4_realworld_naming/train-*.parquet
- config_name: exp4_realworld_discrimination
  data_files:
  - split: train
    path: exp4_realworld_discrimination/train-*.parquet
- config_name: exp5_synthetic_matched
  data_files:
  - split: train
    path: exp5_synthetic_matched/train-*.parquet
---

# ANS Benchmark

A cognitively-grounded benchmark for **numerical cognition in vision-language models**,
covering **exact counting** and **approximate numerosity discrimination (ANS)**.

Each row is a self-contained trial: the stimulus image is embedded inline, and `prompt`
holds the exact question text sent to the model during evaluation.

## Configs (6)

| Config | Task | Rows | Stimuli |
|---|---|---|---|
| `exp1_counting` | Count items in one image | 1,994 | Abstract shapes + real-world (FSC-133), small-n & large-n |
| `exp2_discrimination` | Which side has more? (abstract, random sizes) | 480 | Pre-stitched pairs, 6 ratio conditions |
| `exp3_controlled` | Which side has more? (numerosity decorrelated from area/spacing) | 2,500 | Pre-stitched pairs, 3 shapes × 4 colors |
| `exp4_realworld_naming` | Name the object category | 864 | Single real-world panels |
| `exp4_realworld_discrimination` | Which side has more? (real-world) | 600 | Pre-stitched same/diff-category pairs |
| `exp5_synthetic_matched` | Which side has more? (abstract matched to real-world) | 1,164 | Pre-stitched pairs, counts/coords matched to Exp4 |

Every paired discrimination image is `left | 10 px white divider | right`.

## Common columns (all configs)

| Column | Type | Description |
|---|---|---|
| `image` | `Image` | Stimulus image (embedded bytes, loads as PIL) |
| `prompt` | `string` | Exact question text used in evaluation |
| `task` | `string` | Task type — see values below |
| `tag` | `string` | Analysis tag (`ans`, `ans2`, `ans3`, `ans4`, `ans5`) |
| `eval_id` | `int` | Global sequential trial index within each tag |
| `answer` / `correct_answer` | `string` | Ground-truth label (`left`/`right`/`tie` or integer count or category name) |

**`task` values:**

| Value | Description |
|---|---|
| `counting` | Exact count of objects |
| `discrimination_abstract` | Which side has more? — abstract shapes, unconstrained |
| `discrimination_controlled` | Which side has more? — numerosity decorrelated from surface area |
| `discrimination_realworld` | Which side has more? — real-world object images |
| `discrimination_synthetic` | Which side has more? — synthetic shapes matched to real-world stimuli |
| `naming` | Identify the object category |

## Config-specific columns

**exp1_counting** — `subset`, `stimulus_type` {abstract, realworld}, `n_range` {small_n, large_n}, `shape`, `category`, `count`

**exp2_discrimination** — `pair_id`, `shape` {dots, rectangles}, `anchor_n`, `ratio_condition`, `ratio_value`, `left_count`, `right_count`, `correct_answer`

**exp3_controlled** — `pair_num`, `pair_shape` {dots, ellipse, rectangle}, `pair_color`, `pair_ratio`, `left_numerosity`, `right_numerosity`, `left_TSA`, `right_TSA`, `left_FA`, `right_FA`, `correct_answer`
(TSA = total surface area; FA = field area / convex hull)

**exp4_realworld_naming** — `category`, `count`, `original_image_name`

**exp4_realworld_discrimination** — `pair_num`, `condition` {same_cat, diff_cat}, `left_category`, `right_category`, `left_count`, `right_count`, `num_ratio`, `answer`

**exp5_synthetic_matched** — `pair_num`, `condition`, `shape_type` {dots, rectangles}, `left_category`, `right_category`, `left_count`, `right_count`, `num_ratio`, `answer`

## Prompt templates

**5 semantically-equivalent templates per task**, assigned **one per stimulus** deterministically
(md5 hash of stimulus id). Each image is tested exactly once — prompts rotate across stimuli,
not within.

- **Counting:** `"How many {item} are there in the image? Please answer with an integer directly."`  (+ 4 variants)
- **Discrimination abstract:** `"Two images of {desc} appear side by side. Which image contains more shapes? Respond with 'left' or 'right'."` (+ 4 variants)
- **Discrimination same-category:** `"Look at these two images. Which image has more {cat}? … Answer with 'left' or 'right'."` (+ 2 variants)
- **Discrimination diff-category:** `"Left image contains {lcat}, right image contains {rcat}. Which image has more items? … Answer with 'left' or 'right'."` (+ 2 variants)
- **Naming:** `"What is(are) the object(s) in the image called? Please answer with one object category name directly."` (+ 4 variants)

## Quick start

```python
from datasets import load_dataset

# Load one config
ds = load_dataset("87cc2s/ans-benchmark", "exp1_counting")
row = ds["train"][0]
row["image"]   # PIL Image
row["prompt"]  # "How many dots are there in the image? ..."
row["answer"]  # "1"
row["task"]    # "counting"

# Load discrimination
ds2 = load_dataset("87cc2s/ans-benchmark", "exp2_discrimination")
row2 = ds2["train"][0]
row2["prompt"]          # "Two images of white dots appear side by side ..."
row2["correct_answer"]  # "right"
row2["task"]            # "discrimination_abstract"
```

## Provenance & notes

- Derived from original Experiment1–Experiment4, Experiment1_1, Experiment2_1, Experiment3_1 stimulus sets.
- **Excluded:** Exp1 `random_largen`-in-small-n bucket, real-world `FSC_sampled`/`FSC_sampled_largen`,
  Exp3 `ellipses_white`/`rectangles_white` (rendered incorrectly — replaced by Exp3_1 re-renders).
  36 synthetic pairs with missing source images and 1 exact-duplicate counting row were dropped.
- **License:** Real-world images derive from FSC-133 (Ranjan et al., 2021) / FSC-147. Abstract stimuli
  are procedurally generated and freely usable. Respect upstream FSC licensing.
"""


def push_readme():
    print("\n── Pushing updated README ──")
    from huggingface_hub import CommitOperationAdd
    readme_bytes = README.encode("utf-8")
    ops = [CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=readme_bytes)]
    api.create_commit(
        repo_id=REPO_ID,
        repo_type="dataset",
        operations=ops,
        commit_message="Update dataset card: document parquet format, prompts, task column",
    )
    print("  README pushed.")


# ─────────────────────────────────────────────────────────────── main ──
CONFIGS = [
    ("exp1_counting",              exp1_counting),
    ("exp2_discrimination",        exp2_discrimination),
    ("exp3_controlled",            exp3_controlled),
    ("exp4_realworld_naming",      exp4_realworld_naming),
    ("exp4_realworld_discrimination", exp4_realworld_discrimination),
    ("exp5_synthetic_matched",     exp5_synthetic_matched),
]

def main():
    print(f"Logged in as: {api.whoami()['name']}")
    print(f"Target: {REPO_ID}\n")

    lookup = load_manifests()

    # 1. Rebuild parquets
    print("\n── Rebuilding parquets with prompt + task columns ──")
    for config_name, builder_fn in CONFIGS:
        print(f"\n  [{config_name}]")
        ds = builder_fn(lookup)
        # Spot-check: verify prompt + task are populated
        sample = ds[0]
        print(f"  rows={len(ds)}  task={sample['task']}  prompt[:60]={sample['prompt'][:60]}")
        ds.push_to_hub(REPO_ID, config_name=config_name, split="train", token=api.token)
        print(f"  pushed.")

    # 2. Delete raw files
    delete_raw_files()

    # 3. Push README
    push_readme()

    print("\n\nAll done!")
    print(f"View: https://huggingface.co/datasets/{REPO_ID}")


if __name__ == "__main__":
    main()
