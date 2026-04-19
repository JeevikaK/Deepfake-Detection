from __future__ import annotations

import re
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parent
VIS_CSV = ROOT / "models" / "visual_scores_perframe.csv"
AUDIO_CSV = ROOT / "audio_scores_pervideo.csv"
SYNC_CSV = ROOT / "syncnet_scores_pretrained.csv"

FUSION_FEATURES_CSV = ROOT / "fusion_features.csv"
FUSION_SCORES_CSV = ROOT / "fusion_scores_pervideo.csv"
FUSION_MODEL_PATH = ROOT / "models" / "fusion_best_model.joblib"


def extract_base_id(value: str) -> str:
    match = re.search(r"id\d{5}", str(value))
    return match.group(0) if match else str(value)


def build_fusion_table() -> pd.DataFrame:
    vis = pd.read_csv(VIS_CSV)
    audio = pd.read_csv(AUDIO_CSV)
    sync = pd.read_csv(SYNC_CSV)

    vis["base_id"] = vis["video_id"].map(extract_base_id)
    audio["base_id"] = audio["video_id"].map(extract_base_id)
    sync["base_id"] = sync["video_uid"].map(extract_base_id)

    vis_map = {
        (row.base_id, int(row.true_label)): float(row.prob_fake)
        for row in vis.itertuples(index=False)
    }
    audio_map = {
        (row.base_id, int(row.true_label)): float(row.prob_fake)
        for row in audio.itertuples(index=False)
    }

    rows = []
    for row in sync.itertuples(index=False):
        visual_score = vis_map.get((row.base_id, int(row.is_video_fake)))
        audio_score = audio_map.get((row.base_id, int(row.is_audio_fake)))
        rows.append(
            {
                "video_uid": row.video_uid,
                "base_id": row.base_id,
                "split": row.split,
                "type": row.type,
                "true_label": int(row.true_label),
                "is_video_fake": int(row.is_video_fake),
                "is_audio_fake": int(row.is_audio_fake),
                "visual_score": visual_score,
                "audio_score": audio_score,
                "sync_score": float(row.sync_score),
                "visual_missing": int(visual_score is None),
                "audio_missing": int(audio_score is None),
            }
        )

    fusion_df = pd.DataFrame(rows)
    fusion_df.to_csv(FUSION_FEATURES_CSV, index=False)
    return fusion_df


def evaluate_predictions(name: str, y_true, y_prob) -> dict[str, object]:
    y_pred = (y_prob >= 0.5).astype(int)
    auc = roc_auc_score(y_true, y_prob)
    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    print(f"\n{name}")
    print(f"ROC-AUC: {auc:.4f}")
    print(f"Accuracy: {acc:.4f}")
    print(cm)
    return {"model": name, "auc": auc, "acc": acc, "cm": cm}


def run_experiments(fusion_df: pd.DataFrame) -> tuple[pd.DataFrame, object, str]:
    train_df = fusion_df[fusion_df["split"] == "train"].copy()
    val_df = fusion_df[fusion_df["split"] == "val"].copy()

    numeric_features = ["visual_score", "audio_score", "sync_score"]
    feature_cols = numeric_features + ["visual_missing", "audio_missing"]

    X_train = train_df[feature_cols]
    y_train = train_df["true_label"].astype(int)
    X_val = val_df[feature_cols]
    y_val = val_df["true_label"].astype(int)

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="constant", fill_value=0.5)),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_features,
            ),
            ("flags", "passthrough", ["visual_missing", "audio_missing"]),
        ]
    )

    models = {
        "logreg_imputed": Pipeline(
            steps=[
                ("prep", preprocessor),
                ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)),
            ]
        ),
        "hgb_imputed": Pipeline(
            steps=[
                (
                    "prep",
                    ColumnTransformer(
                        transformers=[
                            ("num", SimpleImputer(strategy="constant", fill_value=0.5), numeric_features),
                            ("flags", "passthrough", ["visual_missing", "audio_missing"]),
                        ]
                    ),
                ),
                ("clf", HistGradientBoostingClassifier(random_state=42, max_depth=3, learning_rate=0.05)),
            ]
        ),
    }

    complete_train = train_df.dropna(subset=["visual_score", "audio_score", "sync_score"]).copy()
    complete_val = val_df.dropna(subset=["visual_score", "audio_score", "sync_score"]).copy()
    models["logreg_complete"] = Pipeline(
        steps=[
            (
                "prep",
                Pipeline(
                    steps=[
                        ("scaler", StandardScaler()),
                    ]
                ),
            ),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)),
        ]
    )

    results = []
    best_name = None
    best_model = None
    best_auc = -1.0

    avg_val_prob = (
        X_val[["visual_score", "audio_score", "sync_score"]]
        .fillna(0.5)
        .mean(axis=1)
        .to_numpy()
    )
    results.append(evaluate_predictions("avg_baseline", y_val.to_numpy(), avg_val_prob))
    if results[-1]["auc"] > best_auc:
        best_auc = results[-1]["auc"]
        best_name = "avg_baseline"
        best_model = None

    for name, model in models.items():
        if name == "logreg_complete":
            Xc_train = complete_train[["visual_score", "audio_score", "sync_score"]]
            yc_train = complete_train["true_label"].astype(int)
            Xc_val = complete_val[["visual_score", "audio_score", "sync_score"]]
            yc_val = complete_val["true_label"].astype(int)
            model.fit(Xc_train, yc_train)
            y_prob = model.predict_proba(Xc_val)[:, 1]
            result = evaluate_predictions(name, yc_val.to_numpy(), y_prob)
        else:
            model.fit(X_train, y_train)
            y_prob = model.predict_proba(X_val)[:, 1]
            result = evaluate_predictions(name, y_val.to_numpy(), y_prob)
        results.append(result)
        if result["auc"] > best_auc:
            best_auc = result["auc"]
            best_name = name
            best_model = model

    if best_name == "avg_baseline":
        fusion_df["fusion_score"] = (
            fusion_df[["visual_score", "audio_score", "sync_score"]].fillna(0.5).mean(axis=1)
        )
    elif best_name == "logreg_complete":
        full_complete = fusion_df.dropna(subset=["visual_score", "audio_score", "sync_score"]).copy()
        full_complete["fusion_score"] = best_model.predict_proba(
            full_complete[["visual_score", "audio_score", "sync_score"]]
        )[:, 1]
        fusion_df["fusion_score"] = 0.5
        fusion_df.loc[full_complete.index, "fusion_score"] = full_complete["fusion_score"]
    else:
        fusion_df["fusion_score"] = best_model.predict_proba(fusion_df[feature_cols])[:, 1]

    return fusion_df, best_model, best_name


def main() -> None:
    fusion_df = build_fusion_table()
    fused_scores, best_model, best_name = run_experiments(fusion_df)
    fused_scores.to_csv(FUSION_SCORES_CSV, index=False)

    if best_model is not None:
        joblib.dump(best_model, FUSION_MODEL_PATH)
    print(f"\nBest fusion model: {best_name}")
    print(f"Saved fusion features to {FUSION_FEATURES_CSV}")
    print(f"Saved fusion scores to {FUSION_SCORES_CSV}")
    if best_model is not None:
        print(f"Saved best model to {FUSION_MODEL_PATH}")


if __name__ == "__main__":
    main()
