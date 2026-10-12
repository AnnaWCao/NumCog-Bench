"""
Convert ANS_Benchmark CSV+images folders to HF parquet datasets with embedded images.
Pushes 6 configs to 87cc2s/ans-benchmark.
"""

import os
import csv
from pathlib import Path
from datasets import Dataset, Features, Image, Value
import huggingface_hub

REPO_ID = "87cc2s/ans-benchmark"
BASE = Path(__file__).parent

SPLITS = [
    {
        "config": "exp1_counting",
        "csv": BASE / "exp1_counting/metadata.csv",
        "images_dir": BASE / "exp1_counting/images",
    },
    {
        "config": "exp2_discrimination",
        "csv": BASE / "exp2_discrimination/metadata.csv",
        "images_dir": BASE / "exp2_discrimination/images",
    },
    {
        "config": "exp3_controlled",
        "csv": BASE / "exp3_controlled/metadata.csv",
        "images_dir": BASE / "exp3_controlled/images",
    },
    {
        "config": "exp4_realworld_naming",
        "csv": BASE / "exp4_realworld/naming/metadata.csv",
        "images_dir": BASE / "exp4_realworld/naming/images",
    },
    {
        "config": "exp4_realworld_discrimination",
        "csv": BASE / "exp4_realworld/discrimination/metadata.csv",
        "images_dir": BASE / "exp4_realworld/discrimination/images",
    },
    {
        "config": "exp5_synthetic_matched",
        "csv": BASE / "exp5_synthetic_matched/metadata.csv",
        "images_dir": BASE / "exp5_synthetic_matched/images",
    },
]


def load_split(split):
    config = split["config"]
    csv_path = split["csv"]
    images_dir = split["images_dir"]

    print(f"\n{'='*60}")
    print(f"  Config: {config}")
    print(f"  CSV:    {csv_path}")
    print(f"  Images: {images_dir}")

    rows = []
    missing = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            img_rel = row["image"]
            img_path = images_dir / img_rel
            if not img_path.exists():
                missing.append(img_rel)
                continue
            row["image"] = str(img_path)  # absolute path; datasets will read + embed
            rows.append(row)

    if missing:
        print(f"  WARNING: {len(missing)} missing images skipped: {missing[:5]}")
    print(f"  Rows loaded: {len(rows)}")

    # Build column dict
    cols = {k: [] for k in rows[0].keys()}
    for row in rows:
        for k, v in row.items():
            cols[k].append(v)

    # Cast image column to datasets.Image() (embeds bytes in parquet)
    ds = Dataset.from_dict(cols)
    ds = ds.cast_column("image", Image())
    print(f"  Features: {ds.features}")
    return ds


def main():
    api = huggingface_hub.HfApi()
    user = api.whoami()["name"]
    print(f"Logged in as: {user}")
    print(f"Target repo:  {REPO_ID}")

    for split in SPLITS:
        ds = load_split(split)
        config = split["config"]
        print(f"\n  Pushing '{config}' ({len(ds)} rows) to {REPO_ID}...")
        ds.push_to_hub(
            REPO_ID,
            config_name=config,
            split="train",
            token=api.token,
        )
        print(f"  Done: {config}")

    print("\n\nAll configs pushed successfully!")
    print(f"View at: https://huggingface.co/datasets/{REPO_ID}")


if __name__ == "__main__":
    main()
