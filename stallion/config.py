from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()


@dataclass(frozen=True)
class CostConfig:
    commission_rate_one_way: float = 0.002
    slippage_bps_per_side: float = 5.0
    spread_bps_round_trip: float = 5.0
    extra_adverse_fill_floor: float = 0.0010
    extra_adverse_fill_cap: float = 0.0040


@dataclass(frozen=True)
class RuntimeConfig:
    market_timezone: str = "America/New_York"
    top_n_universe: int = 3000
    shortlist_count: int = 400
    monitor_count: int = 500
    daily_history_days: int = 400
    intraday_history_sessions: int = 40
    same_slot_lookback_sessions: int = 20
    training_sessions: int = 60
    max_positions: int = 4
    min_minutes_from_open: int = 5
    max_minutes_from_open: int = 90
    threshold_floor: float = 0.55
    threshold_quantile: float = 0.90
    min_price: float = 5.0
    min_dollar_volume: float = 5_000_000.0
    quote_poll_seconds: int = 15
    batch_quote_chunk_size: int = 200


@dataclass(frozen=True)
class PathsConfig:
    root_dir: Path
    data_dir: Path
    sqlite_path: Path
    parquet_dir: Path
    artifacts_dir: Path
    model_dir: Path
    reports_dir: Path
    watchlist_path: Path


@dataclass(frozen=True)
class Credentials:
    fmp_api_key: str
    webull_app_key: str | None
    webull_app_secret: str | None
    webull_account_id: str | None


@dataclass(frozen=True)
class Settings:
    credentials: Credentials
    runtime: RuntimeConfig
    costs: CostConfig
    paths: PathsConfig


def _build_paths(root_dir: Path) -> PathsConfig:
    data_dir = root_dir / "data"
    parquet_dir = data_dir / "parquet"
    artifacts_dir = data_dir / "artifacts"
    model_dir = artifacts_dir / "models"
    reports_dir = root_dir / "reports"
    return PathsConfig(
        root_dir=root_dir,
        data_dir=data_dir,
        sqlite_path=data_dir / "stallion_live.sqlite",
        parquet_dir=parquet_dir,
        artifacts_dir=artifacts_dir,
        model_dir=model_dir,
        reports_dir=reports_dir,
        watchlist_path=artifacts_dir / "next_session_shortlist.parquet",
    )


def load_settings(root_dir: str | Path | None = None) -> Settings:
    root = Path(root_dir or Path(__file__).resolve().parents[1]).resolve()
    fmp_api_key = os.getenv("FMP_API_KEY", "").strip()
    if not fmp_api_key:
        raise ValueError("FMP_API_KEY is required in .env or environment.")

    credentials = Credentials(
        fmp_api_key=fmp_api_key,
        webull_app_key=os.getenv("WEBULL_APP_KEY"),
        webull_app_secret=os.getenv("WEBULL_APP_SECRET"),
        webull_account_id=os.getenv("WEBULL_ACCOUNT_ID"),
    )
    settings = Settings(
        credentials=credentials,
        runtime=RuntimeConfig(),
        costs=CostConfig(),
        paths=_build_paths(root),
    )
    settings.paths.data_dir.mkdir(parents=True, exist_ok=True)
    settings.paths.parquet_dir.mkdir(parents=True, exist_ok=True)
    settings.paths.artifacts_dir.mkdir(parents=True, exist_ok=True)
    settings.paths.model_dir.mkdir(parents=True, exist_ok=True)
    settings.paths.reports_dir.mkdir(parents=True, exist_ok=True)
    return settings
