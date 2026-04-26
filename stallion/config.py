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
    shortlist_count: int = 100
    monitor_count: int = 100
    daily_history_days: int = 400
    intraday_history_sessions: int = 40
    same_slot_lookback_sessions: int = 20
    stage2_symbol_chunk_size: int = 100
    training_sessions: int = 60
    watchlist_label_mode: str = "trade_and_profit"
    watchlist_cv_folds: int = 5
    watchlist_cv_min_train_sessions: int = 15
    watchlist_cv_embargo_sessions: int = 1
    max_positions: int = 5
    fractional_shares_enabled: bool = False
    min_minutes_from_open: int = 5
    max_minutes_from_open: int = 90
    risk_pct_per_trade: float = 1.0
    max_alloc_pct_per_trade: float = 0.25
    ep_gap_exclusion_pct: float = 0.10
    stop_buffer_bps: float = 10.0
    atr_limit_mult: float = 1.0
    adr_limit_mult: float = 1.0
    fast_fail_days: int = 1
    day0_day1_pivot_fail_exit_all: bool = True
    dma10_reduce_to_frac: float = 0.25
    dma10_reduce_threshold: float = 55.0
    dma10_exit_threshold: float = 50.0
    dma21_exit_threshold: float = 45.0
    use_dma21_tight_low_volume_grace: bool = True
    use_intraday_trigger_time: bool = True
    entry_at: str = "trigger_close"
    allow_reentry_same_symbol: bool = False
    use_industry_composite_priority: bool = True
    enable_a_plus_replacement: bool = True
    replacement_score_margin: float = 18.0
    min_price: float = 5.0
    min_daily_volume: float = 1_000_000.0
    min_dollar_volume: float = 10_000_000.0
    quote_poll_seconds: int = 15
    batch_quote_chunk_size: int = 200
    no_new_orders_after_hour: int = 15
    no_new_orders_after_minute: int = 55
    eod_exit_hour: int = 15
    eod_exit_minute: int = 55
    shutdown_hour: int = 16
    shutdown_minute: int = 5
    order_cancel_after_seconds: int = 300
    marketable_limit_bps: float = 25.0
    marketable_limit_retry_count: int = 1
    demo_starting_buying_power: float = 100_000.0
    pre_market_notification_hour: int = 9
    pre_market_notification_minute: int = 25
    broker_sync_seconds: int = 60
    watchdog_stale_seconds: int = 180


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
    discord_bot_token: str | None
    discord_channel_id: str | None


@dataclass(frozen=True)
class Settings:
    credentials: Credentials
    runtime: RuntimeConfig
    costs: CostConfig
    paths: PathsConfig
    demo_mode: bool
    trade_mode: str
    discord_enabled: bool


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
        discord_bot_token=os.getenv("DISCORD_BOT_TOKEN"),
        discord_channel_id=os.getenv("DISCORD_CHANNEL_ID"),
    )
    webull_live_ready = all(
        str(value or "").strip()
        for value in (
            credentials.webull_app_key,
            credentials.webull_app_secret,
            credentials.webull_account_id,
        )
    )
    demo_mode = not webull_live_ready
    settings = Settings(
        credentials=credentials,
        runtime=RuntimeConfig(),
        costs=CostConfig(),
        paths=_build_paths(root),
        demo_mode=demo_mode,
        trade_mode="DEMO" if demo_mode else "LIVE",
        discord_enabled=bool(str(credentials.discord_bot_token or "").strip() and str(credentials.discord_channel_id or "").strip()),
    )
    settings.paths.data_dir.mkdir(parents=True, exist_ok=True)
    settings.paths.parquet_dir.mkdir(parents=True, exist_ok=True)
    settings.paths.artifacts_dir.mkdir(parents=True, exist_ok=True)
    settings.paths.model_dir.mkdir(parents=True, exist_ok=True)
    settings.paths.reports_dir.mkdir(parents=True, exist_ok=True)
    return settings
