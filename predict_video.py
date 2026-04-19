from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import cv2
import joblib
import numpy as np
import pandas as pd
import soundfile as sf
import timm
import torch
import torch.nn as nn
import torchvision.transforms as T
from facenet_pytorch import MTCNN
from PIL import Image
from transformers import AutoModel, AutoProcessor


ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"
SYNCNET_ROOT = ROOT / "external" / "syncnet_python"


def choose_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


DEVICE = choose_device()


@dataclass
class BranchResult:
    score: float | None
    available: bool
    details: dict


def extract_base_id(value: str) -> str:
    match = re.search(r"id\d{5}", str(value))
    return match.group(0) if match else str(value)


def infer_ground_truth(video_path: Path) -> dict | None:
    path_str = str(video_path)
    if "RealVideo-RealAudio" in path_str:
        clip_type = "RealVideo-RealAudio"
        is_video_fake = 0
        is_audio_fake = 0
    elif "RealVideo-FakeAudio" in path_str:
        clip_type = "RealVideo-FakeAudio"
        is_video_fake = 0
        is_audio_fake = 1
    elif "FakeVideo-RealAudio" in path_str:
        clip_type = "FakeVideo-RealAudio"
        is_video_fake = 1
        is_audio_fake = 0
    elif "FakeVideo-FakeAudio" in path_str:
        clip_type = "FakeVideo-FakeAudio"
        is_video_fake = 1
        is_audio_fake = 1
    else:
        return None

    true_label = int(is_video_fake or is_audio_fake)
    return {
        "type": clip_type,
        "is_video_fake": is_video_fake,
        "is_audio_fake": is_audio_fake,
        "true_label": true_label,
        "label_name": "FAKE" if true_label else "REAL",
    }


def load_visual_model(device: torch.device):
    model = timm.create_model("tf_efficientnet_b0", pretrained=False, num_classes=2)
    state = torch.load(MODELS_DIR / "visual_effnet_b0_perframe.pth", map_location="cpu")
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


class AudioMLP(nn.Module):
    def __init__(self, input_dim: int = 768):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 2),
        )

    def forward(self, x):
        return self.net(x)


def load_audio_model(device: torch.device):
    model = AudioMLP(768)
    state = torch.load(MODELS_DIR / "audio_mlp_best.pth", map_location="cpu")
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def load_audio_embedder(device: torch.device):
    model_name = "facebook/wav2vec2-base-960h"
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    processor = AutoProcessor.from_pretrained(model_name, local_files_only=True)
    base = AutoModel.from_pretrained(model_name, local_files_only=True)
    base.to(device)
    base.eval()
    return processor, base


def sample_uniform_frames(video_path: Path, max_frames: int = 100) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        return []
    step = max(1, total_frames // max_frames)
    frames = []
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % step == 0:
            frames.append(frame[:, :, ::-1])  # BGR -> RGB
            if len(frames) >= max_frames:
                break
        frame_idx += 1
    cap.release()
    return frames


def crop_face(frame_rgb: np.ndarray, box: np.ndarray, pad_ratio: float = 0.05) -> Image.Image | None:
    h, w = frame_rgb.shape[:2]
    x1, y1, x2, y2 = box.astype(int)
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    padx = int(bw * pad_ratio)
    pady = int(bh * pad_ratio)
    x1 = max(0, x1 - padx)
    y1 = max(0, y1 - pady)
    x2 = min(w, x2 + padx)
    y2 = min(h, y2 + pady)
    if x2 <= x1 or y2 <= y1:
        return None
    crop = frame_rgb[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    return Image.fromarray(crop)


def predict_visual(video_path: Path, device: torch.device) -> BranchResult:
    frames = sample_uniform_frames(video_path)
    if not frames:
        return BranchResult(None, False, {"reason": "No frames could be read from the video."})

    # MTCNN in facenet_pytorch can hit unsupported adaptive-pooling paths on MPS,
    # so keep face detection on CPU for robustness.
    mtcnn = MTCNN(keep_all=False, device="cpu")
    model = load_visual_model(device)
    transform = T.Compose(
        [
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    probs = []
    used_frames = 0
    with torch.no_grad():
        for frame in frames:
            boxes, _ = mtcnn.detect(Image.fromarray(frame))
            if boxes is None or len(boxes) == 0:
                continue
            face = crop_face(frame, boxes[0])
            if face is None:
                continue
            xb = transform(face).unsqueeze(0).to(device)
            logits = model(xb)
            prob = torch.softmax(logits, dim=1)[0, 1].item()
            probs.append(prob)
            used_frames += 1

    if not probs:
        return BranchResult(None, False, {"reason": "No usable face crops were detected for the visual branch."})

    return BranchResult(
        float(np.mean(probs)),
        True,
        {
            "frames_sampled": len(frames),
            "frames_used": used_frames,
            "frame_probabilities": [round(float(p), 4) for p in probs[:10]],
        },
    )


def extract_audio(video_path: Path, wav_path: Path, sample_rate: int = 16000) -> None:
    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(video_path),
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-vn",
        "-f",
        "wav",
        str(wav_path),
    ]
    subprocess.run(cmd, check=True)


def build_audio_segments(audio: np.ndarray, sample_rate: int = 16000, n_segments: int = 10, seg_len_sec: float = 1.0):
    seg_samples = int(seg_len_sec * sample_rate)
    if len(audio) == 0:
        return []
    if len(audio) <= seg_samples:
        padded = np.zeros(seg_samples, dtype=np.float32)
        padded[: len(audio)] = audio[:seg_samples]
        return [padded]
    max_start = len(audio) - seg_samples
    starts = np.linspace(0, max_start, num=n_segments, dtype=int)
    return [audio[s : s + seg_samples].astype(np.float32) for s in starts]


def predict_audio(video_path: Path, device: torch.device, temp_dir: Path) -> BranchResult:
    wav_path = temp_dir / "audio.wav"
    try:
        extract_audio(video_path, wav_path)
    except subprocess.CalledProcessError as exc:
        return BranchResult(None, False, {"reason": f"Audio extraction failed: {exc}"})

    try:
        audio, sample_rate = sf.read(wav_path, always_2d=False)
    except Exception as exc:  # noqa: BLE001
        return BranchResult(None, False, {"reason": f"Audio loading failed: {exc}"})

    if isinstance(audio, np.ndarray) and audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = np.asarray(audio, dtype=np.float32)
    if sample_rate != 16000:
        return BranchResult(None, False, {"reason": f"Expected 16kHz audio after extraction, got {sample_rate}Hz."})

    segments = build_audio_segments(audio, sample_rate=sample_rate)
    if not segments:
        return BranchResult(None, False, {"reason": "Audio branch found no usable samples."})

    try:
        processor, embedder = load_audio_embedder(device)
    except Exception as exc:  # noqa: BLE001
        return BranchResult(
            None,
            False,
            {
                "reason": (
                    "Audio branch could not load the cached Wav2Vec2 model locally. "
                    f"Run once with internet access or pre-cache the model. Original error: {exc}"
                )
            },
        )
    mlp = load_audio_model(device)
    probs = []

    with torch.no_grad():
        for seg in segments:
            inputs = processor(seg, sampling_rate=16000, return_tensors="pt", padding=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            hidden = embedder(**inputs).last_hidden_state.mean(dim=1)
            logits = mlp(hidden)
            prob = torch.softmax(logits, dim=1)[0, 1].item()
            probs.append(prob)

    return BranchResult(
        float(np.mean(probs)),
        True,
        {
            "segments_used": len(segments),
            "segment_probabilities": [round(float(p), 4) for p in probs],
        },
    )


def predict_sync(video_path: Path, temp_dir: Path) -> BranchResult:
    sys.path.insert(0, str(SYNCNET_ROOT))
    from SyncNetInstance import SyncNetInstance  # pylint: disable=import-error

    syncnet = SyncNetInstance()
    syncnet.loadParameters(str(SYNCNET_ROOT / "data" / "syncnet_v2.model"))

    work_dir = temp_dir / "syncnet"
    reference = video_path.stem
    for subdir in ["pywork", "pycrop", "pyavi", "pyframes", "pytmp"]:
        (work_dir / subdir / reference).parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "run_pipeline.py",
        "--videofile",
        str(video_path),
        "--reference",
        reference,
        "--data_dir",
        str(work_dir),
        "--min_track",
        "25",
    ]
    try:
        subprocess.run(cmd, cwd=SYNCNET_ROOT, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as exc:
        return BranchResult(None, False, {"reason": exc.stderr.decode("utf-8", errors="ignore")[:1000]})

    crop_dir = work_dir / "pycrop" / reference
    crop_files = sorted(crop_dir.glob("0*.avi"))
    if not crop_files:
        return BranchResult(None, False, {"reason": "Sync branch found no valid face tracks."})

    best = None
    opt = SimpleNamespace(batch_size=20, vshift=15, tmp_dir=str(work_dir / "pytmp"), reference=reference)
    for crop_path in crop_files:
        offset, conf, dists = syncnet.evaluate(opt, videofile=str(crop_path))
        min_distance = float(np.mean(dists, axis=0).min()) if np.ndim(dists) > 1 else float(np.min(dists))
        candidate = {
            "offset": int(offset),
            "confidence": float(conf),
            "min_distance": min_distance,
            "crop_path": str(crop_path),
        }
        if best is None or candidate["confidence"] > best["confidence"]:
            best = candidate

    calibrator_bundle = joblib.load(MODELS_DIR / "syncnet_score_calibrator.joblib")
    calibrator = calibrator_bundle["model"]
    feature_cols = calibrator_bundle["feature_cols"]
    features = pd.DataFrame(
        [[best["min_distance"], -best["confidence"], abs(best["offset"])]],
        columns=feature_cols,
    )
    sync_score = float(calibrator.predict_proba(features)[:, 1][0])
    best["sync_score"] = sync_score
    return BranchResult(sync_score, True, best)


def explain_prediction(final_prob: float, visual: BranchResult, audio: BranchResult, sync: BranchResult) -> str:
    label = "deepfake" if final_prob >= 0.5 else "not deepfake"
    branch_msgs = []

    if visual.available and visual.score is not None:
        if visual.score >= 0.7:
            branch_msgs.append(f"visual artifacts look strong ({visual.score:.3f})")
        elif visual.score <= 0.3:
            branch_msgs.append(f"visual branch looks mostly authentic ({visual.score:.3f})")
    else:
        branch_msgs.append("visual branch was unavailable")

    if audio.available and audio.score is not None:
        if audio.score >= 0.7:
            branch_msgs.append(f"audio branch detected strong fake-audio cues ({audio.score:.3f})")
        elif audio.score <= 0.3:
            branch_msgs.append(f"audio branch looks mostly authentic ({audio.score:.3f})")
    else:
        branch_msgs.append("audio branch was unavailable")

    if sync.available and sync.score is not None:
        if sync.score >= 0.6:
            branch_msgs.append(f"lip-sync mismatch signal is elevated ({sync.score:.3f})")
        elif sync.score <= 0.4:
            branch_msgs.append(f"lip-sync looks reasonably consistent ({sync.score:.3f})")
    else:
        branch_msgs.append("sync branch was unavailable")

    lead = f"The final multimodal score is {final_prob:.3f}, so the system predicts this video is {label}."
    why = " Key evidence: " + "; ".join(branch_msgs) + "."
    return lead + why


def run_inference(video_path: Path) -> dict:
    fusion_model = joblib.load(MODELS_DIR / "fusion_best_model.joblib")
    temp_root = Path(tempfile.mkdtemp(prefix="deepfake_infer_"))
    ground_truth = infer_ground_truth(video_path)

    try:
        visual = predict_visual(video_path, DEVICE)
        audio = predict_audio(video_path, DEVICE, temp_root)
        sync = predict_sync(video_path, temp_root)

        row = {
            "visual_score": visual.score if visual.score is not None else np.nan,
            "audio_score": audio.score if audio.score is not None else np.nan,
            "sync_score": sync.score if sync.score is not None else 0.5,
            "visual_missing": int(not visual.available),
            "audio_missing": int(not audio.available),
        }
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)

    feature_df = pd.DataFrame([row])
    fusion_prob = float(fusion_model.predict_proba(feature_df)[0, 1])
    prediction = "YES_DEEPFAKE" if fusion_prob >= 0.5 else "NO_NOT_DEEPFAKE"

    return {
        "video_path": str(video_path),
        "device": str(DEVICE),
        "ground_truth": ground_truth,
        "prediction": prediction,
        "final_probability": round(fusion_prob, 4),
        "branch_scores": {
            "visual_score": None if visual.score is None else round(float(visual.score), 4),
            "audio_score": None if audio.score is None else round(float(audio.score), 4),
            "sync_score": None if sync.score is None else round(float(sync.score), 4),
        },
        "explanation": explain_prediction(fusion_prob, visual, audio, sync),
        "details": {
            "visual": visual.details,
            "audio": audio.details,
            "sync": sync.details,
            "fusion_features": row,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run multimodal deepfake inference on a single video.")
    parser.add_argument("video_path", type=str, help="Path to an input video file.")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON only.")
    args = parser.parse_args()

    video_path = Path(args.video_path).expanduser().resolve()
    if not video_path.exists():
        raise SystemExit(f"Video not found: {video_path}")

    result = run_inference(video_path)
    if args.json:
        print(json.dumps(result, indent=2))
        return

    print(f"Video: {result['video_path']}")
    print(f"Device: {result['device']}")
    if result["ground_truth"] is not None:
        gt = result["ground_truth"]
        print(
            "Ground truth: "
            f"{gt['label_name']} ({gt['type']}, "
            f"video_fake={gt['is_video_fake']}, audio_fake={gt['is_audio_fake']})"
        )
    print(f"Prediction: {result['prediction']}")
    print(f"Final probability: {result['final_probability']:.4f}")
    print("Branch scores:")
    for key, value in result["branch_scores"].items():
        print(f"  - {key}: {value}")
    print("\nWhy:")
    print(result["explanation"])


if __name__ == "__main__":
    main()
