from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT / "FakeAVCeleb_v1.2" / "FakeAVCeleb_v1.2"
META_CSV = DATA_ROOT / "meta_data.csv"
OUT_CSV = ROOT / "fakeavceleb_unique_manifest.csv"
BALANCED_OUT_CSV = ROOT / "fakeavceleb_unique_manifest_balanced.csv"


def slugify(text: str) -> str:
    text = re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_")
    return text.lower()


def build_video_uid(raw_relpath: str, source_id: str, clip_name: str) -> str:
    digest = hashlib.sha1(raw_relpath.encode("utf-8")).hexdigest()[:10]
    stem = slugify(Path(clip_name).stem)[:80]
    source_slug = slugify(source_id)
    return f"{source_slug}__{stem}__{digest}"


def load_manifest() -> pd.DataFrame:
    df = pd.read_csv(META_CSV).rename(columns={"Unnamed: 9": "dataset_dir"})
    df["dataset_dir"] = df["dataset_dir"].astype(str).str.replace(r"^FakeAVCeleb/", "", regex=True)
    df["clip_name"] = df["path"].astype(str)
    df["raw_video_relpath"] = df["dataset_dir"].str.rstrip("/") + "/" + df["clip_name"]
    df["raw_video_path"] = df["raw_video_relpath"].map(lambda rel: str(DATA_ROOT / rel))
    df["metadata_row_count"] = df.groupby("raw_video_relpath")["raw_video_relpath"].transform("size")
    df = df.drop_duplicates(subset=["raw_video_relpath"]).reset_index(drop=True)

    df["is_video_fake"] = (df["type"].astype(str).str.contains("FakeVideo")).astype(int)
    df["is_audio_fake"] = (df["type"].astype(str).str.contains("FakeAudio")).astype(int)
    df["true_label"] = ((df["is_video_fake"] == 1) | (df["is_audio_fake"] == 1)).astype(int)
    df["label_name"] = df["true_label"].map({0: "REAL", 1: "FAKE"})

    df["video_uid"] = [
        build_video_uid(raw_relpath, source_id, clip_name)
        for raw_relpath, source_id, clip_name in zip(df["raw_video_relpath"], df["source"], df["clip_name"])
    ]

    if df["video_uid"].duplicated().any():
        dupes = df.loc[df["video_uid"].duplicated(keep=False), ["video_uid", "raw_video_relpath"]]
        raise ValueError(f"Generated duplicate video_uid values:\n{dupes.head().to_string(index=False)}")

    exists = df["raw_video_path"].map(lambda p: Path(p).exists())
    if not exists.all():
        missing = df.loc[~exists, ["video_uid", "raw_video_relpath", "raw_video_path"]]
        raise FileNotFoundError(f"Missing raw videos:\n{missing.head().to_string(index=False)}")

    return df[
        [
            "video_uid",
            "source",
            "target1",
            "target2",
            "method",
            "category",
            "type",
            "race",
            "gender",
            "clip_name",
            "dataset_dir",
            "raw_video_relpath",
            "raw_video_path",
            "metadata_row_count",
            "is_video_fake",
            "is_audio_fake",
            "true_label",
            "label_name",
        ]
    ].copy()


def build_balanced_subset(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    real_df = df[df["true_label"] == 0].copy()
    fake_df = df[df["true_label"] == 1].copy()
    n_real = len(real_df)
    if n_real == 0:
        raise ValueError("No REAL samples found in manifest.")

    fake_parts = []
    fake_types = sorted(fake_df["type"].unique().tolist())
    remaining = n_real
    for idx, fake_type in enumerate(fake_types):
        pool = fake_df[fake_df["type"] == fake_type].copy()
        take = remaining // (len(fake_types) - idx)
        take = min(len(pool), take)
        sampled = pool.sample(n=take, random_state=seed + idx)
        fake_parts.append(sampled)
        remaining -= len(sampled)

    if remaining > 0:
        taken_ids = set(pd.concat(fake_parts)["video_uid"].tolist()) if fake_parts else set()
        leftovers = fake_df[~fake_df["video_uid"].isin(taken_ids)]
        fake_parts.append(leftovers.sample(n=remaining, random_state=seed + 99))

    balanced = pd.concat([real_df, *fake_parts], ignore_index=True)
    balanced = balanced.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    stratify_cols = balanced["type"].where(balanced["true_label"] == 1, other="RealVideo-RealAudio")
    train_idx, val_idx = train_test_split(
        balanced.index,
        test_size=0.15,
        random_state=seed,
        stratify=stratify_cols,
    )
    balanced["split"] = "train"
    balanced.loc[val_idx, "split"] = "val"
    return balanced


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = load_manifest()
    df.to_csv(OUT_CSV, index=False)

    balanced = build_balanced_subset(df, seed=args.seed)
    balanced.to_csv(BALANCED_OUT_CSV, index=False)

    print(f"Saved full manifest to {OUT_CSV}")
    print(f"Saved balanced manifest to {BALANCED_OUT_CSV}")
    print("\nFull manifest counts:")
    print(df["type"].value_counts().to_string())
    print("\nBalanced subset counts by type:")
    print(balanced["type"].value_counts().to_string())
    print("\nBalanced subset split counts:")
    print(balanced.groupby(["split", "type"]).size().to_string())
    print("\nExample rows:")
    print(
        balanced[
            ["video_uid", "type", "label_name", "raw_video_relpath", "split"]
        ].head(5).to_string(index=False)
    )


if __name__ == "__main__":
    main()
