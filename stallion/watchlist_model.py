from __future__ import annotations

import json
import logging
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from .features import _daily_bar_session_dates, build_stage2_labeled_panel
from .modeling import fit_hist_gbm, score_candidates
from .strategy import StandardSystemSpec, select_candidates_for_session


LOGGER = logging.getLogger(__name__)

WATCHLIST_FEATURE_COLUMNS = [
    "daily_buy_pressure_prev",
    "daily_rs_score_prev",
    "daily_rrs_prev",
    "prev_day_adr_pct",
    "industry_buy_pressure_prev",
    "sector_buy_pressure_prev",
    "industry_rs_prev",
]

WATCHLIST_FEATURE_SOURCE_COLUMNS = {
    "daily_buy_pressure_prev": "daily_buy_pressure_eod",
    "daily_rs_score_prev": "daily_rs_score_eod",
    "daily_rrs_prev": "daily_rrs_eod",
    "prev_day_adr_pct": "adr_pct_20_eod",
    "industry_buy_pressure_prev": "industry_buy_pressure_eod",
    "sector_buy_pressure_prev": "sector_buy_pressure_eod",
    "industry_rs_prev": "industry_rs_eod",
}

LABEL_COLUMN_BY_MODE = {
    "trade_and_profit": "label_watchlist_trade_and_profit",
    "trade_any": "label_watchlist_trade_any",
    "nextday_close_up": "label_watchlist_nextday_close_up",
}

LEGACY_SHORTLIST_WEIGHTS = {
    "daily_buy_pressure_prev": 0.20,
    "daily_rs_score_prev": 0.25,
    "daily_rrs_prev": 0.20,
    "prev_day_adr_pct": 0.10,
    "industry_buy_pressure_prev": 0.10,
    "sector_buy_pressure_prev": 0.05,
    "industry_rs_prev": 0.10,
}


def _normalize_date_series(values) -> pd.Series:
    series = pd.Series(values)
    series = pd.to_datetime(series, errors="coerce")
    if getattr(series.dt, "tz", None) is not None:
        series = series.dt.tz_localize(None)
    return series.dt.normalize()


@dataclass(frozen=True)
class WatchlistModelSpec:
    feature_columns: tuple[str, ...]
    label_mode: str
    shortlist_count: int
    cv_folds: int
    min_train_sessions: int
    embargo_sessions: int
    winsor_lower: float = 0.01
    winsor_upper: float = 0.99
    logistic_c: float = 1.0
    logistic_penalty: str = "l2"
    logistic_solver: str = "lbfgs"
    max_iter: int = 2000
    class_weight: str | None = "balanced"


@dataclass(frozen=True)
class WatchlistModelBundle:
    model_name: str
    created_at: pd.Timestamp
    feature_columns: tuple[str, ...]
    label_mode: str
    winsor_bounds: dict[str, tuple[float, float]]
    impute_values: dict[str, float]
    scaler_mean: tuple[float, ...]
    scaler_scale: tuple[float, ...]
    artifact_path: Path
    metadata: dict[str, object]


def make_watchlist_model_spec(settings) -> WatchlistModelSpec:
    label_mode = settings.runtime.watchlist_label_mode.strip().lower()
    if label_mode not in LABEL_COLUMN_BY_MODE:
        raise ValueError(f"Unsupported watchlist_label_mode: {settings.runtime.watchlist_label_mode}")
    return WatchlistModelSpec(
        feature_columns=tuple(WATCHLIST_FEATURE_COLUMNS),
        label_mode=label_mode,
        shortlist_count=settings.runtime.shortlist_count,
        cv_folds=settings.runtime.watchlist_cv_folds,
        min_train_sessions=settings.runtime.watchlist_cv_min_train_sessions,
        embargo_sessions=settings.runtime.watchlist_cv_embargo_sessions,
    )


def build_legacy_watchlist(daily_features: pd.DataFrame, session_date: pd.Timestamp, shortlist_count: int) -> pd.DataFrame:
    feature_frame = extract_watchlist_feature_frame(daily_features)
    target_session = _normalize_date_series([session_date]).iloc[0]
    latest = feature_frame.loc[_normalize_date_series(feature_frame["session_date"]).eq(target_session)].copy()
    if latest.empty:
        return latest
    latest["shortlist_score"] = 0.0
    for feature_name, weight in LEGACY_SHORTLIST_WEIGHTS.items():
        latest["shortlist_score"] += latest[feature_name].rank(pct=True).fillna(0.0) * weight
    latest["shortlist_source"] = "legacy"
    latest = latest.sort_values(["shortlist_score", "daily_rs_score_prev"], ascending=[False, False]).head(shortlist_count).copy()
    latest["rank_order"] = range(1, len(latest) + 1)
    return latest


def _sorted_unique_sessions(frame: pd.DataFrame, column: str) -> list[pd.Timestamp]:
    return sorted(_normalize_date_series(frame[column]).dropna().unique().tolist())


def _build_session_map(daily_features: pd.DataFrame) -> pd.DataFrame:
    sessions = _sorted_unique_sessions(daily_features, "session_date")
    if len(sessions) < 2:
        return pd.DataFrame(columns=["feature_date", "next_session_date"])
    return pd.DataFrame(
        {
            "feature_date": _normalize_date_series(sessions[:-1]),
            "next_session_date": _normalize_date_series(sessions[1:]),
        }
    )


def extract_watchlist_feature_frame(daily_features: pd.DataFrame) -> pd.DataFrame:
    if daily_features.empty:
        return pd.DataFrame(columns=["session_date", "symbol", *WATCHLIST_FEATURE_COLUMNS])
    source_columns = ["session_date", "symbol", *WATCHLIST_FEATURE_SOURCE_COLUMNS.values()]
    frame = daily_features[source_columns].copy()
    rename_map = {source: target for target, source in WATCHLIST_FEATURE_SOURCE_COLUMNS.items()}
    frame = frame.rename(columns=rename_map)
    frame["session_date"] = _normalize_date_series(frame["session_date"])
    return frame


def build_stage2_intraday_panel(intraday_bars: pd.DataFrame, daily_features: pd.DataFrame, settings) -> pd.DataFrame:
    return build_stage2_labeled_panel(
        intraday_bars,
        daily_features,
        same_slot_lookback_sessions=settings.runtime.same_slot_lookback_sessions,
        min_minutes_from_open=settings.runtime.min_minutes_from_open,
        max_minutes_from_open=settings.runtime.max_minutes_from_open,
        commission_rate_one_way=settings.costs.commission_rate_one_way,
        slippage_bps_per_side=settings.costs.slippage_bps_per_side,
        spread_bps_round_trip=settings.costs.spread_bps_round_trip,
        adverse_fill_floor=settings.costs.extra_adverse_fill_floor,
        adverse_fill_cap=settings.costs.extra_adverse_fill_cap,
        symbol_chunk_size=settings.runtime.stage2_symbol_chunk_size,
        spill_parent_dir=settings.paths.parquet_dir / "tmp",
    )


def make_watchlist_labels(daily_features: pd.DataFrame, daily_bars: pd.DataFrame, intraday_labeled: pd.DataFrame) -> pd.DataFrame:
    if daily_features.empty:
        return pd.DataFrame()

    session_map = _build_session_map(daily_features)
    if session_map.empty:
        return pd.DataFrame()

    signal_labels = (
        intraday_labeled.groupby(["session_date", "symbol"], observed=True)
        .agg(
            label_watchlist_trade_any=("net_return_stress_exec", lambda s: int(s.notna().any())),
            label_watchlist_trade_and_profit=("net_return_stress_exec", lambda s: int((s > 0).any())),
            best_intraday_signal_return=("net_return_stress_exec", "max"),
            signal_row_count=("net_return_stress_exec", "size"),
        )
        .reset_index()
        .rename(columns={"session_date": "next_session_date"})
    )
    signal_labels["next_session_date"] = _normalize_date_series(signal_labels["next_session_date"])
    signal_labels = session_map.merge(signal_labels, on="next_session_date", how="left")

    daily_work = daily_bars.copy()
    daily_work["session_date"] = _daily_bar_session_dates(daily_work["ts"])
    daily_work = daily_work.dropna(subset=["session_date"]).sort_values(["symbol", "session_date"]).copy()
    daily_work["adj_close"] = daily_work["adj_close"].fillna(daily_work["close"])
    daily_work["next_adj_close"] = daily_work.groupby("symbol", sort=False)["adj_close"].shift(-1)
    daily_work["label_watchlist_nextday_close_up"] = (
        (daily_work["next_adj_close"] / daily_work["adj_close"]) - 1.0
    ) > 0
    close_labels = daily_work[["symbol", "session_date", "label_watchlist_nextday_close_up"]].copy()
    close_labels = close_labels.rename(columns={"session_date": "feature_date"})
    close_labels["label_watchlist_nextday_close_up"] = close_labels["label_watchlist_nextday_close_up"].fillna(False).astype("int8")

    signal_labels = signal_labels.merge(close_labels, on=["feature_date", "symbol"], how="left")
    signal_labels["label_watchlist_trade_any"] = signal_labels["label_watchlist_trade_any"].fillna(0).astype("int8")
    signal_labels["label_watchlist_trade_and_profit"] = signal_labels["label_watchlist_trade_and_profit"].fillna(0).astype("int8")
    signal_labels["label_watchlist_nextday_close_up"] = signal_labels["label_watchlist_nextday_close_up"].fillna(0).astype("int8")
    signal_labels["best_intraday_signal_return"] = signal_labels["best_intraday_signal_return"].fillna(0.0)
    signal_labels["signal_row_count"] = signal_labels["signal_row_count"].fillna(0).astype("int32")
    return signal_labels


def build_watchlist_training_panel(daily_features: pd.DataFrame, watchlist_labels: pd.DataFrame) -> pd.DataFrame:
    if daily_features.empty:
        return pd.DataFrame()

    session_map = _build_session_map(daily_features)
    if session_map.empty:
        return pd.DataFrame()

    frame = extract_watchlist_feature_frame(daily_features)
    frame = frame.rename(columns={"session_date": "feature_date"})
    frame["feature_date"] = _normalize_date_series(frame["feature_date"])
    frame = frame.merge(session_map, on="feature_date", how="inner")
    if not watchlist_labels.empty:
        frame = frame.merge(
            watchlist_labels,
            on=["feature_date", "next_session_date", "symbol"],
            how="left",
        )
    frame["has_intraday_label_coverage"] = frame.get("label_watchlist_trade_any", pd.Series(index=frame.index, dtype="float64")).notna().astype("int8")
    frame["has_close_label_coverage"] = frame.get("label_watchlist_nextday_close_up", pd.Series(index=frame.index, dtype="float64")).notna().astype("int8")
    for label_column in LABEL_COLUMN_BY_MODE.values():
        if label_column not in frame.columns:
            frame[label_column] = 0
        frame[label_column] = frame[label_column].fillna(0).astype("int8")
    if "best_intraday_signal_return" not in frame.columns:
        frame["best_intraday_signal_return"] = 0.0
    if "signal_row_count" not in frame.columns:
        frame["signal_row_count"] = 0
    frame["best_intraday_signal_return"] = frame["best_intraday_signal_return"].fillna(0.0)
    frame["signal_row_count"] = frame["signal_row_count"].fillna(0).astype("int32")
    return frame.sort_values(["feature_date", "symbol"]).reset_index(drop=True)


def filter_watchlist_training_panel_for_label_mode(frame: pd.DataFrame, label_mode: str) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    if label_mode in {"trade_and_profit", "trade_any"}:
        return frame.loc[frame["has_intraday_label_coverage"].eq(1)].copy()
    if label_mode == "nextday_close_up":
        return frame.loc[frame["has_close_label_coverage"].eq(1)].copy()
    return frame.copy()


def _cross_sectional_zscore(frame: pd.DataFrame, feature_columns: tuple[str, ...], date_column: str = "feature_date") -> pd.DataFrame:
    work = frame[list(feature_columns) + [date_column]].copy()
    grouped = work.groupby(date_column, sort=False, observed=True)
    means = grouped[list(feature_columns)].transform("mean")
    stds = grouped[list(feature_columns)].transform("std").replace(0, np.nan)
    zscores = (work[list(feature_columns)] - means) / stds
    zscores.columns = list(feature_columns)
    return zscores


def _fit_preprocessor(
    train_frame: pd.DataFrame,
    feature_columns: tuple[str, ...],
    winsor_lower: float,
    winsor_upper: float,
) -> tuple[dict[str, tuple[float, float]], dict[str, float], StandardScaler]:
    zscores = _cross_sectional_zscore(train_frame, feature_columns)
    winsor_bounds: dict[str, tuple[float, float]] = {}
    for column in feature_columns:
        series = pd.to_numeric(zscores[column], errors="coerce")
        lower = float(series.quantile(winsor_lower)) if series.notna().any() else -10.0
        upper = float(series.quantile(winsor_upper)) if series.notna().any() else 10.0
        winsor_bounds[column] = (lower, upper)
        zscores[column] = series.clip(lower=lower, upper=upper)
    medians = {
        column: float(pd.to_numeric(zscores[column], errors="coerce").median()) if zscores[column].notna().any() else 0.0
        for column in feature_columns
    }
    x_train = zscores.fillna(medians)
    scaler = StandardScaler()
    scaler.fit(x_train[list(feature_columns)])
    return winsor_bounds, medians, scaler


def _transform_with_preprocessor(
    frame: pd.DataFrame,
    feature_columns: tuple[str, ...],
    winsor_bounds: dict[str, tuple[float, float]],
    impute_values: dict[str, float],
    scaler: StandardScaler,
) -> pd.DataFrame:
    zscores = _cross_sectional_zscore(frame, feature_columns)
    for column in feature_columns:
        lower, upper = winsor_bounds[column]
        zscores[column] = pd.to_numeric(zscores[column], errors="coerce").clip(lower=lower, upper=upper)
    x_frame = zscores.fillna(impute_values)
    transformed = scaler.transform(x_frame[list(feature_columns)])
    return pd.DataFrame(transformed, columns=list(feature_columns), index=frame.index)


def train_watchlist_model(train_frame: pd.DataFrame, spec: WatchlistModelSpec) -> tuple[LogisticRegression, WatchlistModelBundle]:
    label_column = LABEL_COLUMN_BY_MODE[spec.label_mode]
    frame = filter_watchlist_training_panel_for_label_mode(train_frame, spec.label_mode)
    frame = frame.dropna(subset=["feature_date", "symbol", label_column]).copy()
    if frame.empty:
        raise ValueError("Watchlist training frame is empty.")
    y_train = frame[label_column].astype(int)
    if y_train.nunique() < 2:
        raise ValueError("Watchlist training label has fewer than two classes.")

    winsor_bounds, impute_values, scaler = _fit_preprocessor(
        frame,
        feature_columns=spec.feature_columns,
        winsor_lower=spec.winsor_lower,
        winsor_upper=spec.winsor_upper,
    )
    x_train = _transform_with_preprocessor(frame, spec.feature_columns, winsor_bounds, impute_values, scaler)
    model_kwargs = {
        "solver": spec.logistic_solver,
        "class_weight": spec.class_weight,
        "C": spec.logistic_c,
        "max_iter": spec.max_iter,
        "random_state": 42,
    }
    if spec.logistic_penalty != "l2":
        model_kwargs["penalty"] = spec.logistic_penalty
    model = LogisticRegression(**model_kwargs)
    model.fit(x_train, y_train)
    bundle = WatchlistModelBundle(
        model_name=f"watchlist_logreg_top{spec.shortlist_count}",
        created_at=pd.Timestamp.utcnow(),
        feature_columns=spec.feature_columns,
        label_mode=spec.label_mode,
        winsor_bounds=winsor_bounds,
        impute_values=impute_values,
        scaler_mean=tuple(float(value) for value in scaler.mean_),
        scaler_scale=tuple(float(value) for value in scaler.scale_),
        artifact_path=Path(),
        metadata={
            "training_rows": int(len(frame)),
            "positive_rate": float(y_train.mean()),
        },
    )
    return model, bundle


def score_watchlist_universe(model: LogisticRegression, bundle: WatchlistModelBundle, frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    scaler = StandardScaler()
    scaler.mean_ = np.asarray(bundle.scaler_mean, dtype="float64")
    scaler.scale_ = np.asarray(bundle.scaler_scale, dtype="float64")
    scaler.n_features_in_ = len(bundle.feature_columns)
    scaler.feature_names_in_ = np.asarray(bundle.feature_columns, dtype=object)
    transformed = _transform_with_preprocessor(
        frame,
        feature_columns=bundle.feature_columns,
        winsor_bounds=bundle.winsor_bounds,
        impute_values=bundle.impute_values,
        scaler=scaler,
    )
    scored = frame.copy()
    scored["watchlist_model_score"] = model.predict_proba(transformed[list(bundle.feature_columns)])[:, 1]
    return scored


def build_next_session_watchlist(scored_frame: pd.DataFrame, shortlist_count: int) -> pd.DataFrame:
    if scored_frame.empty:
        return scored_frame.copy()
    work = scored_frame.copy()
    sort_columns = ["feature_date", "watchlist_model_score", "daily_buy_pressure_prev", "daily_rs_score_prev"]
    work = work.sort_values(sort_columns, ascending=[True, False, False, False])
    work["rank_order"] = work.groupby("feature_date", sort=False).cumcount() + 1
    selected = work[work["rank_order"] <= shortlist_count].copy()
    selected["shortlist_score"] = selected["watchlist_model_score"]
    selected["shortlist_source"] = "watchlist_model"
    cutoff = selected.groupby("feature_date", sort=False)["shortlist_score"].min().rename("daily_cutoff")
    selected = selected.merge(cutoff, on="feature_date", how="left")
    return selected


def save_watchlist_model(model: LogisticRegression, bundle: WatchlistModelBundle, artifact_path: Path) -> WatchlistModelBundle:
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model,
        "bundle": {
            "model_name": bundle.model_name,
            "created_at": bundle.created_at,
            "feature_columns": bundle.feature_columns,
            "label_mode": bundle.label_mode,
            "winsor_bounds": bundle.winsor_bounds,
            "impute_values": bundle.impute_values,
            "scaler_mean": bundle.scaler_mean,
            "scaler_scale": bundle.scaler_scale,
            "metadata": bundle.metadata,
        },
    }
    with artifact_path.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return WatchlistModelBundle(
        model_name=bundle.model_name,
        created_at=bundle.created_at,
        feature_columns=bundle.feature_columns,
        label_mode=bundle.label_mode,
        winsor_bounds=bundle.winsor_bounds,
        impute_values=bundle.impute_values,
        scaler_mean=bundle.scaler_mean,
        scaler_scale=bundle.scaler_scale,
        artifact_path=artifact_path,
        metadata=bundle.metadata,
    )


def load_watchlist_model(artifact_path: Path) -> tuple[LogisticRegression, WatchlistModelBundle]:
    with artifact_path.open("rb") as handle:
        payload = pickle.load(handle)
    bundle_payload = payload["bundle"]
    bundle = WatchlistModelBundle(
        model_name=bundle_payload["model_name"],
        created_at=pd.Timestamp(bundle_payload["created_at"]),
        feature_columns=tuple(bundle_payload["feature_columns"]),
        label_mode=bundle_payload["label_mode"],
        winsor_bounds={key: tuple(value) for key, value in bundle_payload["winsor_bounds"].items()},
        impute_values={key: float(value) for key, value in bundle_payload["impute_values"].items()},
        scaler_mean=tuple(float(value) for value in bundle_payload["scaler_mean"]),
        scaler_scale=tuple(float(value) for value in bundle_payload["scaler_scale"]),
        artifact_path=artifact_path,
        metadata=dict(bundle_payload.get("metadata", {})),
    )
    return payload["model"], bundle


def _iter_purged_walk_forward_dates(
    dates: list[pd.Timestamp],
    n_splits: int,
    min_train_sessions: int,
    embargo_sessions: int,
) -> list[tuple[list[pd.Timestamp], list[pd.Timestamp]]]:
    if len(dates) <= min_train_sessions:
        return []
    remaining = len(dates) - min_train_sessions
    step = max(1, remaining // max(n_splits, 1))
    splits: list[tuple[list[pd.Timestamp], list[pd.Timestamp]]] = []
    val_start = min_train_sessions
    while val_start < len(dates) and len(splits) < n_splits:
        val_end = min(len(dates), val_start + step)
        if len(splits) == n_splits - 1:
            val_end = len(dates)
        train_end = max(0, val_start - embargo_sessions)
        train_dates = dates[:train_end]
        val_dates = dates[val_start:val_end]
        if len(train_dates) >= min_train_sessions and val_dates:
            splits.append((train_dates, val_dates))
        val_start = val_end
    return splits


def _auc_safe(y_true: pd.Series, scores: pd.Series) -> float:
    if y_true.nunique() < 2:
        return float("nan")
    return float(roc_auc_score(y_true, scores))


def _ap_safe(y_true: pd.Series, scores: pd.Series) -> float:
    if y_true.nunique() < 2:
        return float("nan")
    return float(average_precision_score(y_true, scores))


def _evaluate_topk_metrics(scored_frame: pd.DataFrame, label_column: str, shortlist_count: int) -> dict[str, float]:
    selected = build_next_session_watchlist(scored_frame, shortlist_count=shortlist_count)
    positives = float(scored_frame[label_column].sum())
    true_positives = float(selected[label_column].sum())
    precision = true_positives / max(float(len(selected)), 1.0)
    recall = true_positives / positives if positives > 0 else 0.0
    return {
        "topk_precision": float(precision),
        "topk_recall": float(recall),
        "selected_count": float(len(selected)),
    }


def _evaluate_trade_log(trades: pd.DataFrame, max_positions: int) -> dict[str, float]:
    if trades.empty:
        return {
            "trade_count": 0.0,
            "win_rate": 0.0,
            "total_return": 0.0,
            "avg_trade_return": 0.0,
            "max_drawdown": 0.0,
            "trigger_rate": 0.0,
        }
    work = trades.copy()
    work["session_date"] = _normalize_date_series(work["session_date"])
    daily_return = work.groupby("session_date", sort=True)["trade_return"].sum() / max(float(max_positions), 1.0)
    equity = (1.0 + daily_return).cumprod()
    drawdown = (equity / equity.cummax()) - 1.0
    active_days = float((daily_return != 0).sum())
    total_days = float(len(daily_return))
    return {
        "trade_count": float(len(work)),
        "win_rate": float((work["trade_return"] > 0).mean()),
        "total_return": float(work["trade_return"].sum()),
        "avg_trade_return": float(work["trade_return"].mean()),
        "max_drawdown": float(drawdown.min()),
        "trigger_rate": active_days / total_days if total_days > 0 else 0.0,
    }


def _run_stage2_for_shortlists(
    shortlist_frame: pd.DataFrame,
    intraday_labeled: pd.DataFrame,
    settings,
    train_execution_dates: list[pd.Timestamp],
) -> tuple[pd.DataFrame, float]:
    if shortlist_frame.empty:
        return pd.DataFrame(), float("nan")

    stage2_spec = StandardSystemSpec(
        min_minutes_from_open=settings.runtime.min_minutes_from_open,
        max_minutes_from_open=settings.runtime.max_minutes_from_open,
        max_positions=settings.runtime.max_positions,
        threshold_floor=settings.runtime.threshold_floor,
        threshold_quantile=settings.runtime.threshold_quantile,
    )
    train_mask = _normalize_date_series(intraday_labeled["session_date"]).isin(_normalize_date_series(train_execution_dates))
    train_intraday = intraday_labeled.loc[train_mask].copy()
    if train_intraday.empty:
        return pd.DataFrame(), float("nan")
    if len(train_intraday) > stage2_spec.max_train_rows:
        train_intraday = train_intraday.tail(stage2_spec.max_train_rows).copy()
    try:
        stage2_model, threshold = fit_hist_gbm(train_intraday, stage2_spec)
    except Exception:
        LOGGER.exception("Stage-2 model training failed during watchlist comparison")
        return pd.DataFrame(), float("nan")

    trade_rows: list[pd.DataFrame] = []
    shortlist_symbol_days = shortlist_frame[["next_session_date", "symbol"]].drop_duplicates()
    for next_session_date, session_shortlist in shortlist_symbol_days.groupby("next_session_date", sort=True):
        symbols = session_shortlist["symbol"].astype(str).tolist()
        frame = intraday_labeled[
            _normalize_date_series(intraday_labeled["session_date"]).eq(_normalize_date_series([next_session_date]).iloc[0])
            & intraday_labeled["symbol"].isin(symbols)
        ].copy()
        if frame.empty:
            continue
        scored = score_candidates(stage2_model, frame)
        chosen = select_candidates_for_session(scored, threshold=threshold, max_positions=stage2_spec.max_positions)
        if chosen.empty:
            continue
        chosen["trade_return"] = chosen["net_return_stress_exec"]
        trade_rows.append(chosen)
    trades = pd.concat(trade_rows, ignore_index=True) if trade_rows else pd.DataFrame()
    return trades, threshold


def _feature_distribution_table(training_panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for feature_name in WATCHLIST_FEATURE_COLUMNS:
        series = pd.to_numeric(training_panel[feature_name], errors="coerce")
        rows.append(
            {
                "feature_name": feature_name,
                "count": int(series.notna().sum()),
                "mean": float(series.mean()) if series.notna().any() else float("nan"),
                "std": float(series.std()) if series.notna().any() else float("nan"),
                "p01": float(series.quantile(0.01)) if series.notna().any() else float("nan"),
                "p50": float(series.quantile(0.50)) if series.notna().any() else float("nan"),
                "p99": float(series.quantile(0.99)) if series.notna().any() else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def evaluate_watchlist_model_cv(
    training_panel: pd.DataFrame,
    daily_features: pd.DataFrame,
    intraday_labeled: pd.DataFrame,
    settings,
    spec: WatchlistModelSpec,
) -> dict[str, pd.DataFrame | dict[str, float]]:
    label_column = LABEL_COLUMN_BY_MODE[spec.label_mode]
    training_panel = filter_watchlist_training_panel_for_label_mode(training_panel, spec.label_mode)
    dates = _sorted_unique_sessions(training_panel, "feature_date")
    splits = _iter_purged_walk_forward_dates(
        dates,
        n_splits=spec.cv_folds,
        min_train_sessions=spec.min_train_sessions,
        embargo_sessions=spec.embargo_sessions,
    )

    fold_rows: list[dict[str, object]] = []
    coefficient_rows: list[dict[str, object]] = []
    shortlist_rows: list[pd.DataFrame] = []
    legacy_shortlist_rows: list[pd.DataFrame] = []
    downstream_rows: list[dict[str, object]] = []
    all_validation_scored: list[pd.DataFrame] = []

    for fold_id, (train_dates, val_dates) in enumerate(splits, start=1):
        train_frame = training_panel[training_panel["feature_date"].isin(train_dates)].copy()
        val_frame = training_panel[training_panel["feature_date"].isin(val_dates)].copy()
        if train_frame.empty or val_frame.empty:
            continue
        try:
            model, bundle = train_watchlist_model(train_frame, spec)
        except Exception:
            LOGGER.exception("Watchlist model training failed on fold %s", fold_id)
            continue
        val_scored = score_watchlist_universe(model, bundle, val_frame)
        val_scored["fold_id"] = fold_id
        all_validation_scored.append(val_scored)
        fold_metrics = _evaluate_topk_metrics(val_scored, label_column, spec.shortlist_count)
        fold_rows.append(
            {
                "fold_id": fold_id,
                "train_start": str(min(train_dates).date()),
                "train_end": str(max(train_dates).date()),
                "validation_start": str(min(val_dates).date()),
                "validation_end": str(max(val_dates).date()),
                "validation_rows": int(len(val_scored)),
                "validation_positive_rate": float(val_scored[label_column].mean()),
                "roc_auc": _auc_safe(val_scored[label_column], val_scored["watchlist_model_score"]),
                "average_precision": _ap_safe(val_scored[label_column], val_scored["watchlist_model_score"]),
                **fold_metrics,
            }
        )

        for feature_name, coefficient in zip(spec.feature_columns, model.coef_[0]):
            coefficient_rows.append(
                {
                    "fold_id": fold_id,
                    "feature_name": feature_name,
                    "coefficient": float(coefficient),
                    "coefficient_sign": "positive" if coefficient > 0 else "negative" if coefficient < 0 else "zero",
                }
            )

        new_shortlist = build_next_session_watchlist(val_scored, shortlist_count=spec.shortlist_count)
        shortlist_rows.append(new_shortlist)

        legacy_frames = [build_legacy_watchlist(daily_features, session_date=date, shortlist_count=spec.shortlist_count) for date in val_dates]
        legacy_shortlist = pd.concat([frame for frame in legacy_frames if not frame.empty], ignore_index=True) if legacy_frames else pd.DataFrame()
        if not legacy_shortlist.empty:
            legacy_shortlist = legacy_shortlist.rename(columns={"session_date": "feature_date"})
            session_map = _build_session_map(daily_features)
            legacy_shortlist = legacy_shortlist.merge(session_map, on="feature_date", how="left")
            legacy_shortlist["shortlist_source"] = "legacy"
            legacy_shortlist_rows.append(legacy_shortlist)

        train_execution_dates = (
            training_panel.loc[training_panel["feature_date"].isin(train_dates), "next_session_date"]
            .dropna()
            .sort_values()
            .unique()
            .tolist()
        )
        new_trades, stage2_threshold = _run_stage2_for_shortlists(new_shortlist, intraday_labeled, settings, train_execution_dates)
        legacy_trades, _ = _run_stage2_for_shortlists(legacy_shortlist, intraday_labeled, settings, train_execution_dates)
        new_trade_metrics = _evaluate_trade_log(new_trades, settings.runtime.max_positions)
        legacy_trade_metrics = _evaluate_trade_log(legacy_trades, settings.runtime.max_positions)

        new_pairs = set(zip(pd.to_datetime(new_shortlist["feature_date"]).dt.date, new_shortlist["symbol"].astype(str)))
        legacy_pairs = set(zip(pd.to_datetime(legacy_shortlist["feature_date"]).dt.date, legacy_shortlist["symbol"].astype(str)))
        overlap_rate = len(new_pairs & legacy_pairs) / max(len(new_pairs | legacy_pairs), 1)
        downstream_rows.append(
            {
                "fold_id": fold_id,
                "stage2_threshold": stage2_threshold,
                "new_trade_count": new_trade_metrics["trade_count"],
                "new_win_rate": new_trade_metrics["win_rate"],
                "new_total_return": new_trade_metrics["total_return"],
                "new_max_drawdown": new_trade_metrics["max_drawdown"],
                "legacy_trade_count": legacy_trade_metrics["trade_count"],
                "legacy_win_rate": legacy_trade_metrics["win_rate"],
                "legacy_total_return": legacy_trade_metrics["total_return"],
                "legacy_max_drawdown": legacy_trade_metrics["max_drawdown"],
                "overlap_rate": overlap_rate,
                "trade_count_delta": new_trade_metrics["trade_count"] - legacy_trade_metrics["trade_count"],
                "total_return_delta": new_trade_metrics["total_return"] - legacy_trade_metrics["total_return"],
                "win_rate_delta": new_trade_metrics["win_rate"] - legacy_trade_metrics["win_rate"],
                "trigger_rate_delta": new_trade_metrics["trigger_rate"] - legacy_trade_metrics["trigger_rate"],
            }
        )

    fold_metrics_frame = pd.DataFrame(fold_rows)
    coefficient_frame = pd.DataFrame(coefficient_rows)
    coefficient_summary = (
        coefficient_frame.groupby("feature_name", observed=True)["coefficient"]
        .agg(["mean", "std", "median"])
        .reset_index()
        .rename(columns={"mean": "mean_coefficient", "std": "std_coefficient", "median": "median_coefficient"})
    ) if not coefficient_frame.empty else pd.DataFrame(columns=["feature_name", "mean_coefficient", "std_coefficient", "median_coefficient"])
    new_shortlists = pd.concat(shortlist_rows, ignore_index=True) if shortlist_rows else pd.DataFrame()
    legacy_shortlists = pd.concat(legacy_shortlist_rows, ignore_index=True) if legacy_shortlist_rows else pd.DataFrame()
    downstream_frame = pd.DataFrame(downstream_rows)
    validation_scored = pd.concat(all_validation_scored, ignore_index=True) if all_validation_scored else pd.DataFrame()
    shortlist_count_frame = (
        new_shortlists.groupby("feature_date", observed=True).size().rename("shortlist_count").reset_index()
        if not new_shortlists.empty
        else pd.DataFrame(columns=["feature_date", "shortlist_count"])
    )
    summary = {
        "cv_fold_count": int(len(fold_metrics_frame)),
        "mean_roc_auc": float(fold_metrics_frame["roc_auc"].mean()) if not fold_metrics_frame.empty else float("nan"),
        "mean_average_precision": float(fold_metrics_frame["average_precision"].mean()) if not fold_metrics_frame.empty else float("nan"),
        "mean_shortlist_precision": float(fold_metrics_frame["topk_precision"].mean()) if not fold_metrics_frame.empty else float("nan"),
        "mean_shortlist_recall": float(fold_metrics_frame["topk_recall"].mean()) if not fold_metrics_frame.empty else float("nan"),
        "mean_overlap_rate": float(downstream_frame["overlap_rate"].mean()) if not downstream_frame.empty else float("nan"),
        "mean_total_return_delta": float(downstream_frame["total_return_delta"].mean()) if not downstream_frame.empty else float("nan"),
    }
    return {
        "cv_metrics": fold_metrics_frame,
        "coefficients": coefficient_frame,
        "coefficient_summary": coefficient_summary,
        "new_shortlists": new_shortlists,
        "legacy_shortlists": legacy_shortlists,
        "downstream_comparison": downstream_frame,
        "validation_scored": validation_scored,
        "shortlist_counts": shortlist_count_frame,
        "feature_distribution": _feature_distribution_table(training_panel),
        "summary": summary,
    }


def write_watchlist_reports(report_dir: Path, cv_outputs: dict[str, pd.DataFrame | dict[str, float]], spec: WatchlistModelSpec) -> dict[str, Path]:
    report_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    csv_map = {
        "watchlist_cv_metrics_summary.csv": cv_outputs["cv_metrics"],
        "watchlist_legacy_vs_model_compare.csv": cv_outputs["downstream_comparison"],
        "watchlist_learned_coefficients.csv": cv_outputs["coefficients"],
        "watchlist_learned_coefficients_summary.csv": cv_outputs["coefficient_summary"],
        "watchlist_shortlist_counts.csv": cv_outputs["shortlist_counts"],
        "watchlist_feature_distribution.csv": cv_outputs["feature_distribution"],
    }
    for filename, frame in csv_map.items():
        path = report_dir / filename
        pd.DataFrame(frame).to_csv(path, index=False)
        paths[filename] = path

    summary = cv_outputs["summary"]
    lines = [
        "# Watchlist Model OOS Summary",
        "",
        f"- label_mode: `{spec.label_mode}`",
        f"- shortlist_count: {spec.shortlist_count}",
        f"- cv_fold_count: {summary['cv_fold_count']}",
        f"- mean_roc_auc: {summary['mean_roc_auc']:.4f}" if pd.notna(summary["mean_roc_auc"]) else "- mean_roc_auc: n/a",
        f"- mean_average_precision: {summary['mean_average_precision']:.4f}" if pd.notna(summary["mean_average_precision"]) else "- mean_average_precision: n/a",
        f"- mean_shortlist_precision: {summary['mean_shortlist_precision']:.4f}" if pd.notna(summary["mean_shortlist_precision"]) else "- mean_shortlist_precision: n/a",
        f"- mean_shortlist_recall: {summary['mean_shortlist_recall']:.4f}" if pd.notna(summary["mean_shortlist_recall"]) else "- mean_shortlist_recall: n/a",
        f"- mean_overlap_rate_vs_legacy: {summary['mean_overlap_rate']:.4f}" if pd.notna(summary["mean_overlap_rate"]) else "- mean_overlap_rate_vs_legacy: n/a",
        f"- mean_total_return_delta_vs_legacy: {summary['mean_total_return_delta']:.4f}" if pd.notna(summary["mean_total_return_delta"]) else "- mean_total_return_delta_vs_legacy: n/a",
        "",
        "## Notes",
        "",
        "- The 7 watchlist features are sourced from same-day EOD values at feature_date close.",
        "- Default label uses next-session open-drive rows and existing stage-2 accounting.",
        "- Legacy shortlist is retained only for OOS comparison and no longer drives runtime selection.",
    ]
    md_path = report_dir / "watchlist_model_report.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")
    paths["watchlist_model_report.md"] = md_path

    json_path = report_dir / "watchlist_model_summary.json"
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    paths["watchlist_model_summary.json"] = json_path
    return paths
