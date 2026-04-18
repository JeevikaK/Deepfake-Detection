from __future__ import annotations

import argparse
import os
import pickle
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score


ROOT = Path(__file__).resolve().parent
SYNCNET_ROOT = ROOT / "external" / "syncnet_python"
SYNCNET_MODEL = SYNCNET_ROOT / "data" / "syncnet_v2.model"
MANIFEST_CSV = ROOT / "fakeavceleb_unique_manifest_balanced.csv"
OUT_CSV = ROOT / "syncnet_scores_pretrained.csv"
MODEL_OUT = ROOT / "models" / "syncnet_score_calibrator.joblib"
WORK_DIR = ROOT / "syncnet_work_pretrained"
MIN_TRACK = 25


def ensure_syncnet_imports() -> None:
    syncnet_path = str(SYNCNET_ROOT)
    if syncnet_path not in sys.path:
        sys.path.insert(0, syncnet_path)


def min_distance_from_dists(dists: np.ndarray) -> float:
    dists = np.asarray(dists, dtype=np.float32)
    if dists.ndim == 1:
        return float(dists.min())
    return float(dists.mean(axis=0).min())


def run_pipeline(video_path: str, reference: str, data_dir: Path) -> list[Path]:
    for subdir in ["pywork", "pycrop", "pyavi", "pyframes", "pytmp"]:
        ref_dir = data_dir / subdir / reference
        if ref_dir.exists():
            shutil.rmtree(ref_dir)
    cmd = [
        sys.executable,
        "run_pipeline.py",
        "--videofile",
        video_path,
        "--reference",
        reference,
        "--data_dir",
        str(data_dir),
        "--min_track",
        str(MIN_TRACK),
    ]
    subprocess.run(cmd, cwd=SYNCNET_ROOT, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    crop_dir = data_dir / "pycrop" / reference
    return sorted(crop_dir.glob("0*.avi"))


def load_track_count(data_dir: Path, reference: str) -> int:
    tracks_path = data_dir / "pywork" / reference / "tracks.pckl"
    if not tracks_path.exists():
        return 0
    with tracks_path.open("rb") as fh:
        tracks = pickle.load(fh, encoding="latin1")
    return len(tracks)


def score_video(syncnet, video_path: str, reference: str, work_dir: Path) -> dict[str, object]:
    opt = SimpleNamespace(batch_size=20, vshift=15, tmp_dir=str(work_dir / "tmp"), reference=reference)
    try:
        crop_files = run_pipeline(video_path, reference=reference, data_dir=work_dir)
    except subprocess.CalledProcessError as exc:
        return {
            "sync_status": "pipeline_failed",
            "sync_error": exc.stderr.decode("utf-8", errors="ignore")[:4000],
            "n_tracks": 0,
        }

    n_tracks = load_track_count(work_dir, reference=reference)
    if not crop_files:
        return {
            "sync_status": "no_tracks",
            "sync_error": "",
            "n_tracks": n_tracks,
        }

    best = None
    for crop_path in crop_files:
        offset, conf, dists = syncnet.evaluate(opt, videofile=str(crop_path))
        candidate = {
            "crop_path": str(crop_path),
            "sync_offset": int(offset),
            "sync_confidence": float(conf),
            "sync_min_distance": min_distance_from_dists(dists),
            "n_tracks": n_tracks,
            "sync_status": "ok",
            "sync_error": "",
        }
        if best is None or candidate["sync_confidence"] > best["sync_confidence"]:
            best = candidate
    return best


def build_scores(df: pd.DataFrame) -> pd.DataFrame:
    ensure_syncnet_imports()
    from SyncNetInstance import SyncNetInstance  # pylint: disable=import-error

    WORK_DIR.mkdir(exist_ok=True)
    (ROOT / "models").mkdir(exist_ok=True)

    scored = []
    if OUT_CSV.exists():
        existing = pd.read_csv(OUT_CSV)
        existing_scored = existing[existing["sync_status"] == "ok"].copy()
        scored = existing_scored.to_dict("records")
        done = set(existing_scored["video_uid"].tolist())
    else:
        done = set()

    syncnet = SyncNetInstance()
    syncnet.loadParameters(str(SYNCNET_MODEL))

    todo = df[~df["video_uid"].isin(done)].reset_index(drop=True)
    print(f"Videos already scored: {len(done)}")
    print(f"Videos to score now: {len(todo)}")

    for idx, row in todo.iterrows():
        result = score_video(syncnet, video_path=row.raw_video_path, reference=row.video_uid, work_dir=WORK_DIR)
        merged = {**row.to_dict(), **result}
        scored.append(merged)

        if (idx + 1) % 10 == 0 or idx + 1 == len(todo):
            pd.DataFrame(scored).to_csv(OUT_CSV, index=False)
            print(f"Saved progress: {idx + 1}/{len(todo)} new videos")

    out_df = pd.DataFrame(scored)
    out_df.to_csv(OUT_CSV, index=False)
    return out_df


def add_calibrated_scores(df: pd.DataFrame) -> pd.DataFrame:
    feat_df = df[df["sync_status"] == "ok"].copy()
    if feat_df.empty:
        df["sync_score"] = 0.5
        return df

    feat_df["abs_offset"] = feat_df["sync_offset"].abs()
    feat_df["neg_confidence"] = -feat_df["sync_confidence"]

    feature_cols = ["sync_min_distance", "neg_confidence", "abs_offset"]
    train_df = feat_df[feat_df["split"] == "train"].copy()
    val_df = feat_df[feat_df["split"] == "val"].copy()

    if train_df.empty:
        raw = feat_df["sync_min_distance"] + feat_df["neg_confidence"] + 0.25 * feat_df["abs_offset"]
        scaled = (raw - raw.min()) / (raw.max() - raw.min() + 1e-6)
        feat_df["sync_score"] = scaled.clip(0.0, 1.0)
        df = df.merge(feat_df[["video_uid", "sync_score"]], on="video_uid", how="left")
        df["sync_score"] = df["sync_score"].fillna(0.5)
        return df

    clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    clf.fit(train_df[feature_cols], train_df["is_audio_fake"])

    feat_df["sync_score"] = clf.predict_proba(feat_df[feature_cols])[:, 1]
    df = df.merge(feat_df[["video_uid", "sync_score"]], on="video_uid", how="left")
    df["sync_score"] = df["sync_score"].fillna(0.5)

    import joblib

    joblib.dump({"model": clf, "feature_cols": feature_cols}, MODEL_OUT)

    print("\nCalibration target: is_audio_fake")
    print_metrics(val_df.assign(sync_score=clf.predict_proba(val_df[feature_cols])[:, 1]), "audio-fake", "is_audio_fake")
    print_metrics(val_df.assign(sync_score=clf.predict_proba(val_df[feature_cols])[:, 1]), "overall deepfake", "true_label")
    return df


def print_metrics(df: pd.DataFrame, label: str, target_col: str) -> None:
    if df.empty or df[target_col].nunique() < 2:
        print(f"{label}: not enough scored validation examples to compute metrics.")
        return
    y_true = df[target_col].astype(int).to_numpy()
    y_prob = df["sync_score"].astype(float).to_numpy()
    y_pred = (y_prob >= 0.5).astype(int)
    auc = roc_auc_score(y_true, y_prob)
    acc = accuracy_score(y_true, y_pred)
    print(f"{label} ROC-AUC: {auc:.4f}")
    print(f"{label} accuracy: {acc:.4f}")
    print(f"{label} confusion matrix:\n{confusion_matrix(y_true, y_pred)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=str, default=str(MANIFEST_CSV))
    parser.add_argument("--split", type=str, default="all", choices=["all", "train", "val"])
    parser.add_argument("--limit", type=int, default=0, help="Optional limit after split filtering.")
    args = parser.parse_args()

    df = pd.read_csv(args.manifest)
    if args.split != "all":
        df = df[df["split"] == args.split].reset_index(drop=True)
    if args.limit > 0:
        df = df.head(args.limit).copy()

    scored_df = build_scores(df)
    merged = pd.read_csv(args.manifest).merge(
        scored_df[
            [
                "video_uid",
                "sync_offset",
                "sync_confidence",
                "sync_min_distance",
                "n_tracks",
                "sync_status",
                "sync_error",
                "crop_path",
            ]
        ],
        on="video_uid",
        how="left",
    )
    merged = add_calibrated_scores(merged)
    merged.to_csv(OUT_CSV, index=False)

    ok_count = int((merged["sync_status"] == "ok").sum())
    print(f"\nSaved scored manifest to {OUT_CSV}")
    print(f"Successfully scored videos: {ok_count}/{len(merged)}")
    print(merged["sync_status"].fillna("missing").value_counts().to_string())


if __name__ == "__main__":
    main()
