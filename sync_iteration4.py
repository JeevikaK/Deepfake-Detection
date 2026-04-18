import json
import random
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

from sync_iteration3 import (
    SEED,
    discover_videos,
    group_score,
    overlap_corr,
    preprocess_video,
    summarize,
)


ROOT = Path(".")
FEATURE_CSV = ROOT / "sync_clip_features_iter4.csv"
SCORE_CSV = ROOT / "sync_scores_pervideo_iter4.csv"
MODEL_OUT = ROOT / "models" / "sync_clip_logreg_iter4.joblib"
META_OUT = ROOT / "models" / "sync_clip_logreg_iter4_meta.json"

random.seed(SEED)
np.random.seed(SEED)


def compose_pair_features(vfeat, afeat):
    motion = np.asarray(vfeat["_motion_seq"], dtype=np.float32)
    open_seq = np.asarray(vfeat["_open_seq"][:-1], dtype=np.float32)
    env = np.asarray(afeat["_env_seq"], dtype=np.float32)

    grid_mean = np.array(
        [val for key, val in sorted(vfeat.items()) if key.startswith("v_grid_mean_")],
        dtype=np.float32,
    )
    grid_std = np.array(
        [val for key, val in sorted(vfeat.items()) if key.startswith("v_grid_std_")],
        dtype=np.float32,
    )
    band_mean = np.array(
        [val for key, val in sorted(afeat.items()) if key.startswith("a_band_mean_")],
        dtype=np.float32,
    )

    row = {}
    row.update(summarize("motion", motion))
    row.update(summarize("open", open_seq))
    row.update(summarize("env", env))
    row.update(summarize("grid_mean", grid_mean))
    row.update(summarize("grid_std", grid_std))
    row.update(summarize("band_mean", band_mean))

    motion_diff = np.diff(motion)
    env_diff = np.diff(env)
    open_diff = np.diff(open_seq)
    row.update(summarize("pair_abs_motion_env", np.abs(motion - env)))
    row.update(summarize("pair_abs_open_env", np.abs(open_seq - env)))
    row.update(summarize("pair_abs_motiondiff_envdiff", np.abs(motion_diff - env_diff)))
    row["pair_motion_env_mse"] = float(np.mean((motion - env) ** 2))
    row["pair_open_env_mse"] = float(np.mean((open_seq - env) ** 2))
    row["pair_motion_env_cosine"] = float(
        np.dot(motion, env) / ((np.linalg.norm(motion) * np.linalg.norm(env)) + 1e-6)
    )
    row["pair_open_env_cosine"] = float(
        np.dot(open_seq, env) / ((np.linalg.norm(open_seq) * np.linalg.norm(env)) + 1e-6)
    )

    for i, val in enumerate(motion):
        row[f"motion_{i:02d}"] = float(val)
    for i, val in enumerate(open_seq):
        row[f"open_{i:02d}"] = float(val)
    for i, val in enumerate(env):
        row[f"env_{i:02d}"] = float(val)
    for i, val in enumerate(motion - env):
        row[f"motion_env_diff_{i:02d}"] = float(val)
    for i, val in enumerate(open_seq - env):
        row[f"open_env_diff_{i:02d}"] = float(val)
    for lag in range(-2, 3):
        row[f"corr_motion_env_{lag:+d}"] = overlap_corr(motion, env, lag)
        row[f"corr_open_env_{lag:+d}"] = overlap_corr(open_seq, env, lag)
        row[f"corr_motiondiff_envdiff_{lag:+d}"] = overlap_corr(motion_diff, env_diff, lag)
    return row


def build_feature_table():
    videos_df = discover_videos()
    clip_bank = {}
    usable_rows = []
    for row in tqdm(videos_df.itertuples(index=False), total=len(videos_df), desc="Preprocessing clip bank"):
        clips = preprocess_video(row)
        if not clips:
            continue
        clip_bank[row.uid] = clips
        usable_rows.append(
            {
                "uid": row.uid,
                "video_id": row.video_id,
                "true_label": row.true_label,
                "split": row.split,
                "n_clips": len(clips),
            }
        )

    meta_df = pd.DataFrame(usable_rows)
    rows = []
    for row in tqdm(meta_df.itertuples(index=False), total=len(meta_df), desc="Building shift-only pairs"):
        clips = clip_bank[row.uid]
        for clip in clips:
            aligned = compose_pair_features(clip["visual"], clip["audio"])
            aligned.update(
                {
                    "source_uid": row.uid,
                    "video_id": row.video_id,
                    "clip_idx": clip["clip_idx"],
                    "split": row.split,
                    "true_label": row.true_label,
                    "sync_target": 0,
                    "neg_type": "aligned",
                }
            )
            rows.append(aligned)

            shift_candidates = [c for c in clips if abs(c["clip_idx"] - clip["clip_idx"]) >= 2]
            if not shift_candidates:
                continue
            neg_clip = random.Random(f"{SEED}:shiftonly:{row.uid}:{clip['clip_idx']}").choice(shift_candidates)
            shifted = compose_pair_features(clip["visual"], neg_clip["audio"])
            shifted.update(
                {
                    "source_uid": row.uid,
                    "video_id": row.video_id,
                    "clip_idx": clip["clip_idx"],
                    "split": row.split,
                    "true_label": row.true_label,
                    "sync_target": 1,
                    "neg_type": "shift",
                }
            )
            rows.append(shifted)
    return pd.DataFrame(rows)


def choose_aggregator(aligned_df):
    candidates = ["mean", "max", "top2mean", "top3mean", "q75", "q90", "mean_std"]
    train_df = aligned_df[aligned_df["split"] == "train"].copy()
    best_name, best_auc = None, -1.0
    for name in candidates:
        grouped = (
            train_df.groupby(["source_uid", "video_id", "true_label", "split"])["sync_prob"]
            .apply(lambda x: group_score(x.values, name))
            .reset_index(name="sync_score")
        )
        auc = roc_auc_score(grouped["true_label"], grouped["sync_score"])
        if auc > best_auc:
            best_auc = auc
            best_name = name
    return best_name, float(best_auc)


def main():
    feature_df = build_feature_table()
    feature_df.to_csv(FEATURE_CSV, index=False)
    print("Saved clip feature table to:", FEATURE_CSV)
    print("Pair counts:", feature_df[["split", "sync_target", "neg_type"]].value_counts().sort_index())

    meta_cols = ["source_uid", "video_id", "clip_idx", "split", "true_label", "sync_target", "neg_type"]
    feature_cols = [c for c in feature_df.columns if c not in meta_cols]

    train_df = feature_df[feature_df["split"] == "train"].reset_index(drop=True)
    val_df = feature_df[feature_df["split"] == "val"].reset_index(drop=True)

    clf = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "logreg",
                LogisticRegression(
                    max_iter=1500,
                    class_weight="balanced",
                    random_state=SEED,
                ),
            ),
        ]
    )
    clf.fit(train_df[feature_cols], train_df["sync_target"])
    joblib.dump(clf, MODEL_OUT)
    print("Saved sync model to:", MODEL_OUT)

    for name, df in [("train", train_df), ("val", val_df)]:
        y_true = df["sync_target"].astype(int).to_numpy()
        y_prob = clf.predict_proba(df[feature_cols])[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)
        auc = roc_auc_score(y_true, y_prob)
        print(f"\n[{name}] mismatch ROC-AUC: {auc:.4f}")
        print("Confusion matrix:\n", confusion_matrix(y_true, y_pred))
        print(classification_report(y_true, y_pred, digits=4, zero_division=0))

    aligned_df = feature_df[feature_df["neg_type"] == "aligned"].copy()
    aligned_df["sync_prob"] = clf.predict_proba(aligned_df[feature_cols])[:, 1]
    agg_name, train_auc = choose_aggregator(aligned_df)
    print(f"\nSelected aggregator: {agg_name} (train deepfake ROC-AUC={train_auc:.4f})")

    score_rows = []
    grouped = aligned_df.groupby(["source_uid", "video_id", "true_label", "split"])["sync_prob"]
    for keys, values in grouped:
        source_uid, video_id, true_label, split = keys
        score_rows.append(
            {
                "video_uid": source_uid,
                "video_id": video_id,
                "true_label": int(true_label),
                "split": split,
                "sync_score": group_score(values.values, agg_name),
            }
        )
    score_df = pd.DataFrame(score_rows).sort_values(["split", "video_uid"]).reset_index(drop=True)
    score_df.to_csv(SCORE_CSV, index=False)
    print("Saved per-video sync scores to:", SCORE_CSV)

    train_threshold = float(score_df[score_df["split"] == "train"]["sync_score"].median())
    val_scores = score_df[score_df["split"] == "val"].copy()
    y_true = val_scores["true_label"].astype(int).to_numpy()
    y_prob = val_scores["sync_score"].astype(float).to_numpy()
    y_pred = (y_prob >= train_threshold).astype(int)
    auc = roc_auc_score(y_true, y_prob)
    print("\n[val] deepfake ROC-AUC from sync_score:", round(float(auc), 4))
    print("Confusion matrix:\n", confusion_matrix(y_true, y_pred))
    print(classification_report(y_true, y_pred, digits=4, zero_division=0))
    print("\nSync score summary by label:")
    print(val_scores.groupby("true_label")["sync_score"].agg(["count", "mean", "std", "min", "max"]))

    META_OUT.write_text(
        json.dumps(
            {
                "aggregator": agg_name,
                "train_deepfake_auc": train_auc,
                "train_threshold": train_threshold,
                "feature_columns": feature_cols,
            },
            indent=2,
        )
    )
    print("Saved sync metadata to:", META_OUT)


if __name__ == "__main__":
    main()
