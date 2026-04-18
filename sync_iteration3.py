import json
import random
from pathlib import Path

import cv2
import joblib
import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torchaudio.functional as AF
import torchaudio.transforms as AT
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm


ROOT = Path(".")
MOUTH_ROOT = ROOT / "processed_perframe"
AUDIO_ROOT = ROOT / "processed_audio"

CLIP_FEATURE_CSV = ROOT / "sync_clip_features_iter3.csv"
SYNC_SCORE_CSV = ROOT / "sync_scores_pervideo_iter3.csv"
SYNC_MODEL_OUT = ROOT / "models" / "sync_clip_hgb_iter3.joblib"
SYNC_META_OUT = ROOT / "models" / "sync_clip_hgb_iter3_meta.json"

SEED = 42
IMG_SIZE = 48
CLIP_FRAMES = 7
CLIPS_PER_VIDEO = 8
MEL_BANDS = 24
MAX_LAG = 2

random.seed(SEED)
np.random.seed(SEED)

MEL_TF = AT.MelSpectrogram(
    sample_rate=16000,
    n_fft=512,
    win_length=400,
    hop_length=160,
    n_mels=MEL_BANDS,
    center=True,
    power=2.0,
)
AMP_TO_DB = AT.AmplitudeToDB(stype="power")


def resample_axis(arr, target_len, axis=-1):
    arr = np.asarray(arr, dtype=np.float32)
    src_len = arr.shape[axis]
    if src_len == target_len:
        return arr.astype(np.float32)
    if src_len <= 1:
        return np.repeat(arr, target_len, axis=axis).astype(np.float32)
    old = np.linspace(0.0, 1.0, num=src_len, dtype=np.float32)
    new = np.linspace(0.0, 1.0, num=target_len, dtype=np.float32)
    moved = np.moveaxis(arr, axis, -1)
    flat = moved.reshape(-1, src_len)
    out = np.stack([np.interp(new, old, row) for row in flat], axis=0)
    out = out.reshape(*moved.shape[:-1], target_len)
    return np.moveaxis(out, -1, axis).astype(np.float32)


def zscore(x):
    x = np.asarray(x, dtype=np.float32)
    sigma = float(x.std())
    if sigma < 1e-6:
        return np.zeros_like(x, dtype=np.float32)
    return ((x - float(x.mean())) / (sigma + 1e-6)).astype(np.float32)


def overlap_corr(a, b, lag):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    if lag > 0:
        aa, bb = a[lag:], b[:-lag]
    elif lag < 0:
        aa, bb = a[:lag], b[-lag:]
    else:
        aa, bb = a, b
    if len(aa) < 2 or len(bb) < 2:
        return 0.0
    if float(aa.std()) < 1e-6 or float(bb.std()) < 1e-6:
        return 0.0
    return float(np.corrcoef(aa, bb)[0, 1])


def summarize(prefix, x):
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    return {
        f"{prefix}_mean": float(x.mean()),
        f"{prefix}_std": float(x.std()),
        f"{prefix}_min": float(x.min()),
        f"{prefix}_max": float(x.max()),
        f"{prefix}_range": float(x.max() - x.min()),
    }


def grid_pool(img, grids=4):
    h, w = img.shape
    hs = np.array_split(np.arange(h), grids)
    ws = np.array_split(np.arange(w), grids)
    feats = []
    for hidx in hs:
        for widx in ws:
            feats.append(float(img[np.ix_(hidx, widx)].mean()))
    return np.asarray(feats, dtype=np.float32)


def load_video_frames(mouth_dir):
    fps = sorted(Path(mouth_dir).glob("frame_*.jpg"))
    frames = []
    for fp in fps:
        img = cv2.imread(str(fp), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
        frames.append(img.astype(np.float32) / 255.0)
    if len(frames) < CLIP_FRAMES + 2:
        return None
    return frames


def load_aligned_audio_mel(audio_path, target_steps):
    wav, sr = sf.read(str(audio_path), always_2d=False)
    wav = np.asarray(wav, dtype=np.float32)
    if wav.ndim == 2:
        wav = wav.mean(axis=1)
    if wav.size < 800:
        return None
    wav_t = torch.from_numpy(wav).float()
    if int(sr) != 16000:
        wav_t = AF.resample(wav_t.unsqueeze(0), int(sr), 16000).squeeze(0)
    mel = MEL_TF(wav_t.unsqueeze(0))
    mel = AMP_TO_DB(mel).squeeze(0).cpu().numpy().astype(np.float32)
    mel = resample_axis(mel, target_steps, axis=1)
    return zscore(mel)


def build_visual_clip_features(frames):
    stack = np.stack(frames, axis=0).astype(np.float32)
    center = stack[:, IMG_SIZE // 3 :, IMG_SIZE // 5 : IMG_SIZE - IMG_SIZE // 5]
    open_seq = zscore(1.0 - center.mean(axis=(1, 2)))

    diffs = np.abs(np.diff(stack, axis=0))
    motion_seq = zscore(diffs.mean(axis=(1, 2)))

    grid_feats = np.stack([grid_pool(d, grids=4) for d in diffs], axis=0)
    grid_mean = grid_feats.mean(axis=0)
    grid_std = grid_feats.std(axis=0)

    out = {}
    out.update(summarize("v_open", open_seq))
    out.update(summarize("v_motion", motion_seq))
    for i, val in enumerate(open_seq):
        out[f"v_open_{i:02d}"] = float(val)
    for i, val in enumerate(motion_seq):
        out[f"v_motion_{i:02d}"] = float(val)
    for i, val in enumerate(grid_mean):
        out[f"v_grid_mean_{i:02d}"] = float(val)
    for i, val in enumerate(grid_std):
        out[f"v_grid_std_{i:02d}"] = float(val)
    out["_open_seq"] = open_seq
    out["_motion_seq"] = motion_seq
    return out


def build_audio_clip_features(mel_clip):
    mel_clip = np.asarray(mel_clip, dtype=np.float32)
    env_seq = zscore(mel_clip.mean(axis=0))
    band_edges = np.linspace(0, mel_clip.shape[0], num=7, dtype=int)
    band_means = []
    for i in range(len(band_edges) - 1):
        lo, hi = band_edges[i], band_edges[i + 1]
        band_means.append(mel_clip[lo:hi].mean(axis=0))
    band_means = np.stack(band_means, axis=0)
    band_summary = band_means.mean(axis=1)

    out = {}
    out.update(summarize("a_env", env_seq))
    for i, val in enumerate(env_seq):
        out[f"a_env_{i:02d}"] = float(val)
    for i, val in enumerate(band_summary):
        out[f"a_band_mean_{i:02d}"] = float(val)
    out["_env_seq"] = env_seq
    return out


def compose_pair_features(vfeat, afeat):
    row = {}
    for key, val in vfeat.items():
        if not key.startswith("_"):
            row[key] = val
    for key, val in afeat.items():
        if not key.startswith("_"):
            row[key] = val

    motion = vfeat["_motion_seq"]
    open_seq = vfeat["_open_seq"][:-1]
    env = afeat["_env_seq"]

    row.update(summarize("pair_absdiff", np.abs(motion - env)))
    row["pair_motion_env_mse"] = float(np.mean((motion - env) ** 2))
    row["pair_open_env_mse"] = float(np.mean((open_seq - env) ** 2))
    row["pair_motion_env_cosine"] = float(
        np.dot(motion, env) / ((np.linalg.norm(motion) * np.linalg.norm(env)) + 1e-6)
    )
    row["pair_open_env_cosine"] = float(
        np.dot(open_seq, env) / ((np.linalg.norm(open_seq) * np.linalg.norm(env)) + 1e-6)
    )
    for lag in range(-MAX_LAG, MAX_LAG + 1):
        row[f"pair_corr_motion_env_{lag:+d}"] = overlap_corr(motion, env, lag)
        row[f"pair_corr_open_env_{lag:+d}"] = overlap_corr(open_seq, env, lag)
    return row


def choose_clip_starts(n_frames):
    max_start = n_frames - CLIP_FRAMES
    if max_start < 0:
        return []
    count = min(CLIPS_PER_VIDEO, max_start + 1)
    starts = np.linspace(0, max_start, num=count, dtype=int)
    return sorted(set(int(x) for x in starts))


def preprocess_video(row):
    frames = load_video_frames(row.mouth_dir)
    if frames is None:
        return None
    mel = load_aligned_audio_mel(row.audio_path, target_steps=len(frames) - 1)
    if mel is None or mel.shape[1] < CLIP_FRAMES - 1:
        return None

    clips = []
    for clip_idx, start in enumerate(choose_clip_starts(len(frames))):
        end = start + CLIP_FRAMES
        frame_clip = frames[start:end]
        mel_clip = mel[:, start : end - 1]
        if len(frame_clip) != CLIP_FRAMES or mel_clip.shape[1] != CLIP_FRAMES - 1:
            continue
        clips.append(
            {
                "clip_idx": clip_idx,
                "start": start,
                "visual": build_visual_clip_features(frame_clip),
                "audio": build_audio_clip_features(mel_clip),
            }
        )
    return clips if clips else None


def discover_videos():
    rows = []
    for label_name in ["REAL", "FAKE"]:
        root = MOUTH_ROOT / label_name
        if not root.exists():
            continue
        for vid_dir in sorted(root.iterdir()):
            if not vid_dir.is_dir():
                continue
            vid = vid_dir.name
            mouth_dir = vid_dir / "mouth"
            audio_path = AUDIO_ROOT / label_name / vid / "audio.wav"
            if mouth_dir.exists() and audio_path.exists():
                rows.append(
                    {
                        "uid": f"{label_name}__{vid}",
                        "video_id": vid,
                        "label_name": label_name,
                        "true_label": 1 if label_name == "FAKE" else 0,
                        "mouth_dir": str(mouth_dir),
                        "audio_path": str(audio_path),
                    }
                )
    videos_df = pd.DataFrame(rows).drop_duplicates("uid").reset_index(drop=True)
    train_uid, val_uid = train_test_split(
        videos_df["uid"].tolist(),
        test_size=0.15,
        random_state=SEED,
        stratify=videos_df["true_label"].tolist(),
    )
    split_map = {uid: "train" for uid in train_uid}
    split_map.update({uid: "val" for uid in val_uid})
    videos_df["split"] = videos_df["uid"].map(split_map)
    return videos_df


def build_feature_table(videos_df):
    clip_bank = {}
    usable_rows = []
    for row in tqdm(videos_df.itertuples(index=False), total=len(videos_df), desc="Preprocessing clip features"):
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
    by_label = {
        label: meta_df[meta_df["true_label"] == label]["uid"].tolist() for label in [0, 1]
    }

    rows = []
    for row in tqdm(meta_df.itertuples(index=False), total=len(meta_df), desc="Building pair table"):
        clips = clip_bank[row.uid]
        same_label_uids = [u for u in by_label[int(row.true_label)] if u != row.uid and u in clip_bank]
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
            if shift_candidates:
                neg_clip = random.Random(f"{SEED}:shift:{row.uid}:{clip['clip_idx']}").choice(shift_candidates)
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

            if same_label_uids:
                neg_uid = random.Random(f"{SEED}:cross:{row.uid}:{clip['clip_idx']}").choice(same_label_uids)
                neg_bank = clip_bank[neg_uid]
                neg_clip = random.Random(f"{SEED}:crossclip:{row.uid}:{clip['clip_idx']}").choice(neg_bank)
                cross = compose_pair_features(clip["visual"], neg_clip["audio"])
                cross.update(
                    {
                        "source_uid": row.uid,
                        "video_id": row.video_id,
                        "clip_idx": clip["clip_idx"],
                        "split": row.split,
                        "true_label": row.true_label,
                        "sync_target": 1,
                        "neg_type": "cross",
                    }
                )
                rows.append(cross)

    df = pd.DataFrame(rows)
    return df


def group_score(values, mode):
    arr = np.asarray(values, dtype=np.float32)
    if mode == "mean":
        return float(arr.mean())
    if mode == "max":
        return float(arr.max())
    if mode == "top2mean":
        return float(np.sort(arr)[-min(2, len(arr)) :].mean())
    if mode == "top3mean":
        return float(np.sort(arr)[-min(3, len(arr)) :].mean())
    if mode == "q75":
        return float(np.quantile(arr, 0.75))
    if mode == "q90":
        return float(np.quantile(arr, 0.90))
    if mode == "mean_std":
        return float(arr.mean() + arr.std())
    raise ValueError(mode)


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
    print("Preparing videos for sync iteration 3...")
    videos_df = discover_videos()
    print("Discovered videos:", len(videos_df))
    print(videos_df["split"].value_counts().to_dict())

    feature_df = build_feature_table(videos_df)
    feature_df.to_csv(CLIP_FEATURE_CSV, index=False)
    print("Saved clip feature table to:", CLIP_FEATURE_CSV)
    print("Pair counts:", feature_df[["split", "sync_target", "neg_type"]].value_counts().sort_index())

    meta_cols = ["source_uid", "video_id", "clip_idx", "split", "true_label", "sync_target", "neg_type"]
    feature_cols = [c for c in feature_df.columns if c not in meta_cols]

    train_df = feature_df[feature_df["split"] == "train"].reset_index(drop=True)
    val_df = feature_df[feature_df["split"] == "val"].reset_index(drop=True)

    clf = HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_depth=6,
        max_iter=350,
        min_samples_leaf=20,
        l2_regularization=0.1,
        random_state=SEED,
    )
    clf.fit(train_df[feature_cols], train_df["sync_target"])
    joblib.dump(clf, SYNC_MODEL_OUT)
    print("Saved sync model to:", SYNC_MODEL_OUT)

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
    score_df.to_csv(SYNC_SCORE_CSV, index=False)
    print("Saved per-video sync scores to:", SYNC_SCORE_CSV)

    val_scores = score_df[score_df["split"] == "val"].copy()
    y_true = val_scores["true_label"].astype(int).to_numpy()
    y_prob = val_scores["sync_score"].astype(float).to_numpy()
    y_pred = (y_prob >= np.median(score_df[score_df["split"] == "train"]["sync_score"])).astype(int)
    auc = roc_auc_score(y_true, y_prob)
    print("\n[val] deepfake ROC-AUC from sync_score:", round(float(auc), 4))
    print("Confusion matrix:\n", confusion_matrix(y_true, y_pred))
    print(classification_report(y_true, y_pred, digits=4, zero_division=0))
    print("\nSync score summary by label:")
    print(val_scores.groupby("true_label")["sync_score"].agg(["count", "mean", "std", "min", "max"]))

    meta = {
        "aggregator": agg_name,
        "train_deepfake_auc": train_auc,
        "feature_columns": feature_cols,
        "clip_frames": CLIP_FRAMES,
        "clips_per_video": CLIPS_PER_VIDEO,
    }
    SYNC_META_OUT.write_text(json.dumps(meta, indent=2))
    print("Saved sync metadata to:", SYNC_META_OUT)


if __name__ == "__main__":
    main()
