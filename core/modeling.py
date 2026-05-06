from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from .strategy import STANDARD_FEATURE_COLUMNS, StandardSystemSpec, compute_threshold

STAGE2_RETIRED_MESSAGE = (
    "Stage-2 intraday HistGradientBoosting is retired because no production "
    "feature columns are defined. Use the breakout signal report/watchlist "
    "ranking directly, or define STANDARD_FEATURE_COLUMNS before training."
)


class Stage2RetiredError(RuntimeError):
    """Raised when legacy Stage-2 modeling code is invoked after retirement."""


@dataclass(frozen=True)
class ModelBundle:
    model_name: str
    feature_columns: tuple[str, ...]
    threshold: float
    created_at: pd.Timestamp
    artifact_path: Path


def stage2_feature_columns(frame: pd.DataFrame | None = None) -> list[str]:
    if not STANDARD_FEATURE_COLUMNS:
        raise Stage2RetiredError(STAGE2_RETIRED_MESSAGE)
    if frame is None:
        return list(STANDARD_FEATURE_COLUMNS)
    features = [feature for feature in STANDARD_FEATURE_COLUMNS if feature in frame.columns]
    if not features:
        raise Stage2RetiredError(STAGE2_RETIRED_MESSAGE)
    return features


def fit_hist_gbm(train_frame: pd.DataFrame, spec: StandardSystemSpec) -> tuple[HistGradientBoostingClassifier, float]:
    features = stage2_feature_columns(train_frame)
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
    features = stage2_feature_columns(candidates)
    scored = candidates.copy()
    scored["score"] = model.predict_proba(scored[features])[:, 1]
    return scored


def save_model_bundle(model: HistGradientBoostingClassifier, threshold: float, artifact_path: Path) -> ModelBundle:
    features = stage2_feature_columns()
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    with artifact_path.open("wb") as handle:
        pickle.dump({"model": model, "threshold": threshold, "features": features}, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return ModelBundle(
        model_name="hist_gbm_extended_5m_start",
        feature_columns=tuple(features),
        threshold=threshold,
        created_at=pd.Timestamp.utcnow(),
        artifact_path=artifact_path,
    )


def load_model_bundle(artifact_path: Path) -> tuple[HistGradientBoostingClassifier, float, list[str]]:
    with artifact_path.open("rb") as handle:
        payload = pickle.load(handle)
    return payload["model"], float(payload["threshold"]), list(payload["features"])
