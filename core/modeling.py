from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from .strategy import STANDARD_FEATURE_COLUMNS, StandardSystemSpec, compute_threshold


@dataclass(frozen=True)
class ModelBundle:
    model_name: str
    feature_columns: tuple[str, ...]
    threshold: float
    created_at: pd.Timestamp
    artifact_path: Path


def fit_hist_gbm(train_frame: pd.DataFrame, spec: StandardSystemSpec) -> tuple[HistGradientBoostingClassifier, float]:
    features = [feature for feature in STANDARD_FEATURE_COLUMNS if feature in train_frame.columns]
    frame = train_frame.dropna(subset=["label_stress_exec"]).copy()
    if frame.empty:
        raise ValueError("Training frame is empty.")
    model = HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_depth=4,
        max_iter=300,
        min_samples_leaf=120,
        random_state=42,
    )
    model.fit(frame[features], frame["label_stress_exec"].astype(int))
    train_scores = model.predict_proba(frame[features])[:, 1]
    threshold = compute_threshold(train_scores)
    return model, threshold


def score_candidates(model: HistGradientBoostingClassifier, candidates: pd.DataFrame) -> pd.DataFrame:
    if candidates.empty:
        return candidates.copy()
    features = [feature for feature in STANDARD_FEATURE_COLUMNS if feature in candidates.columns]
    scored = candidates.copy()
    scored["score"] = model.predict_proba(scored[features])[:, 1]
    return scored


def save_model_bundle(model: HistGradientBoostingClassifier, threshold: float, artifact_path: Path) -> ModelBundle:
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    with artifact_path.open("wb") as handle:
        pickle.dump({"model": model, "threshold": threshold, "features": STANDARD_FEATURE_COLUMNS}, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return ModelBundle(
        model_name="hist_gbm_extended_5m_start",
        feature_columns=tuple(STANDARD_FEATURE_COLUMNS),
        threshold=threshold,
        created_at=pd.Timestamp.utcnow(),
        artifact_path=artifact_path,
    )


def load_model_bundle(artifact_path: Path) -> tuple[HistGradientBoostingClassifier, float, list[str]]:
    with artifact_path.open("rb") as handle:
        payload = pickle.load(handle)
    return payload["model"], float(payload["threshold"]), list(payload["features"])
