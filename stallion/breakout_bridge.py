from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import pandas as pd

from qullamaggie_breakout_backtest import (
    BacktestConfig as ExitBacktestConfig,
    prepare_daily as _prepare_exit_daily,
    run_backtest as _run_breakout_backtest,
)
from breakout_signal_report import build_breakout_signal_report as _build_breakout_signal_report


@dataclass(frozen=True)
class BreakoutConfig:
    session_timezone: str = "America/New_York"
    initial_equity: float = 100_000.0
    max_positions: int = 5
    shortlist_count: int = 100
    risk_pct_per_trade: float = 1.0
    max_alloc_pct_per_trade: float = 0.25
    ep_gap_exclusion_pct: float = 0.10
    stop_buffer_bps: float = 10.0
    atr_limit_mult: float = 1.0
    adr_limit_mult: float = 1.0
    fast_fail_days: int = 1
    day0_day1_pivot_fail_exit_all: bool = True

    # Exit policy
    tp_partial_r: float = 1.75
    tp_partial_pct: float = 0.10

    dma10_reduce_to_frac: float = 0.25
    dma10_reduce_threshold: float = 55.0
    dma10_exit_threshold: float = 50.0

    dma21_reduce_to_frac: float = 0.10
    dma21_reduce_threshold: float = 45.0
    dma21_exit_after_reduce_threshold: float = 40.0
    use_dma21_tight_low_volume_grace: bool = True

    use_intraday_trigger_time: bool = True
    entry_at: str = "trigger_close"
    allow_reentry_same_symbol: bool = False

    @classmethod
    def from_settings(cls, settings: Any) -> "BreakoutConfig":
        runtime = settings.runtime
        return cls(
            session_timezone=runtime.market_timezone,
            initial_equity=float(getattr(runtime, "demo_starting_buying_power", 100_000.0)),
            max_positions=int(runtime.max_positions),
            shortlist_count=int(runtime.shortlist_count),
            risk_pct_per_trade=float(getattr(runtime, "risk_pct_per_trade", 1.0)),
            max_alloc_pct_per_trade=float(getattr(runtime, "max_alloc_pct_per_trade", 0.25)),
            ep_gap_exclusion_pct=float(getattr(runtime, "ep_gap_exclusion_pct", 0.10)),
            stop_buffer_bps=float(getattr(runtime, "stop_buffer_bps", 10.0)),
            atr_limit_mult=float(getattr(runtime, "atr_limit_mult", 1.0)),
            adr_limit_mult=float(getattr(runtime, "adr_limit_mult", 1.0)),
            fast_fail_days=int(getattr(runtime, "fast_fail_days", 1)),
            day0_day1_pivot_fail_exit_all=bool(getattr(runtime, "day0_day1_pivot_fail_exit_all", True)),
            tp_partial_r=float(getattr(runtime, "tp_partial_r", 1.75)),
            tp_partial_pct=float(getattr(runtime, "tp_partial_pct", 0.10)),
            dma10_reduce_to_frac=float(getattr(runtime, "dma10_reduce_to_frac", 0.25)),
            dma10_reduce_threshold=float(getattr(runtime, "dma10_reduce_threshold", 55.0)),
            dma10_exit_threshold=float(getattr(runtime, "dma10_exit_threshold", 50.0)),
            dma21_reduce_to_frac=float(getattr(runtime, "dma21_reduce_to_frac", 0.10)),
            dma21_reduce_threshold=float(getattr(runtime, "dma21_reduce_threshold", 45.0)),
            dma21_exit_after_reduce_threshold=float(
                getattr(runtime, "dma21_exit_after_reduce_threshold", 40.0)
            ),
            use_dma21_tight_low_volume_grace=bool(getattr(runtime, "use_dma21_tight_low_volume_grace", True)),
            use_intraday_trigger_time=bool(getattr(runtime, "use_intraday_trigger_time", True)),
            entry_at=str(getattr(runtime, "entry_at", "trigger_close")),
            allow_reentry_same_symbol=bool(getattr(runtime, "allow_reentry_same_symbol", False)),
        )


@dataclass
class BreakoutPositionState:
    symbol: str
    entry_date: pd.Timestamp
    entry_price: float
    initial_stop: float
    pivot_level: float
    breakout_day_low: float
    initial_shares: int
    shares: int
    initial_risk_per_share: float
    entry_bar_time: pd.Timestamp | None = None
    pending_dma21_grace: bool = False
    partial_profit_taken: bool = False
    reduced_on_dma21: bool = False
    entry_source: str = "standard_breakout"
    entry_lane: str = "none"


def _coerce_row(row: Mapping[str, Any] | pd.Series | Any) -> dict[str, Any]:
    if isinstance(row, pd.Series):
        return row.to_dict()
    if isinstance(row, Mapping):
        return dict(row)
    if hasattr(row, "_asdict"):
        return dict(row._asdict())
    raise TypeError("row must be a mapping, pandas Series, or namedtuple-like object")


def _normalize_bar_frame(obj: Any, *, time_col: str, session_timezone: str) -> pd.DataFrame:
    if isinstance(obj, dict):
        frames = []
        for symbol, frame in obj.items():
            if frame is None or len(frame) == 0:
                continue
            work = frame.copy()
            if "symbol" not in work.columns:
                work["symbol"] = symbol
            work = work.reset_index()
            work.columns = [str(col).lower() for col in work.columns]
            if time_col not in work.columns:
                index_col = str(work.columns[0]).lower()
                work = work.rename(columns={index_col: time_col})
            frames.append(work)
        if not frames:
            return pd.DataFrame()
        frame = pd.concat(frames, ignore_index=True)
    else:
        frame = obj.copy()
        frame.columns = [str(col).lower() for col in frame.columns]

    if "symbol" not in frame.columns:
        raise ValueError("bars must include a symbol column")
    if time_col not in frame.columns:
        if "ts" in frame.columns:
            frame = frame.rename(columns={"ts": time_col})
        elif "datetime" in frame.columns and time_col == "date":
            frame = frame.rename(columns={"datetime": "date"})
        else:
            raise ValueError(f"bars must include a {time_col} column")

    frame[time_col] = pd.to_datetime(frame[time_col], errors="coerce")
    if getattr(frame[time_col].dt, "tz", None) is not None:
        if time_col == "datetime":
            frame[time_col] = frame[time_col].dt.tz_convert(session_timezone).dt.tz_localize(None)
        else:
            frame[time_col] = frame[time_col].dt.tz_localize(None)
    frame[time_col] = frame[time_col].dt.normalize() if time_col == "date" else frame[time_col]

    expected = ["open", "high", "low", "close", "volume"]
    for column in expected:
        if column not in frame.columns:
            raise ValueError(f"bars must include a {column} column")
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def normalize_daily_bars(daily_bars: Any, *, session_timezone: str = "America/New_York") -> pd.DataFrame:
    frame = _normalize_bar_frame(daily_bars, time_col="date", session_timezone=session_timezone)
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    if getattr(frame["date"].dt, "tz", None) is not None:
        frame["date"] = frame["date"].dt.tz_localize(None)
    frame["date"] = frame["date"].dt.normalize()
    return frame[["symbol", "date", "open", "high", "low", "close", "volume"]].dropna(subset=["symbol", "date"]).copy()


def normalize_intraday_bars(intraday_bars: Any, *, session_timezone: str = "America/New_York") -> pd.DataFrame:
    frame = _normalize_bar_frame(intraday_bars, time_col="datetime", session_timezone=session_timezone)
    return frame[["symbol", "datetime", "open", "high", "low", "close", "volume"]].dropna(subset=["symbol", "datetime"]).copy()


def build_breakout_signal_report(
    daily_bars: Any,
    intraday_bars: Any,
    cfg: BreakoutConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cfg = cfg or BreakoutConfig()
    daily = normalize_daily_bars(daily_bars, session_timezone=cfg.session_timezone)
    intraday = normalize_intraday_bars(intraday_bars, session_timezone=cfg.session_timezone)
    report, summary = _build_breakout_signal_report(daily, intraday, session_tz=cfg.session_timezone)
    return report, summary


def select_breakout_candidates(
    frame: pd.DataFrame,
    *,
    session_date: pd.Timestamp | str | None = None,
    max_positions: int = 5,
) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()

    work = frame.copy()
    work["date"] = pd.to_datetime(work["date"]).dt.normalize()
    target_date = pd.Timestamp(session_date).normalize() if session_date is not None else pd.Timestamp(work["date"].max()).normalize()
    if "breakout_signal" in work.columns:
        work = work.loc[work["breakout_signal"].fillna(False)]
    work = work.loc[work["date"].eq(target_date)].copy()
    if work.empty:
        return work

    sort_cols: list[str] = []
    ascending: list[bool] = []
    if "entry_priority_bucket" in work.columns:
        sort_cols.append("entry_priority_bucket")
        ascending.append(True)
    if "priority_score_within_source" in work.columns:
        sort_cols.append("priority_score_within_source")
        ascending.append(False)
    if "leader_score" in work.columns:
        sort_cols.append("leader_score")
        ascending.append(False)
    if "trigger_time" in work.columns:
        work["trigger_time"] = pd.to_datetime(work["trigger_time"], errors="coerce")
        sort_cols.append("trigger_time")
        ascending.append(True)
    sort_cols.append("symbol")
    ascending.append(True)
    return work.sort_values(sort_cols, ascending=ascending, kind="mergesort").head(max_positions).reset_index(drop=True)


def prepare_exit_daily_frame(daily_bars: Any, *, session_timezone: str = "America/New_York") -> pd.DataFrame:
    daily = normalize_daily_bars(daily_bars, session_timezone=session_timezone)
    return _prepare_exit_daily(daily)


def build_position_state_from_signal(
    row: Mapping[str, Any] | pd.Series | Any,
    *,
    equity: float,
    cash: float,
    cfg: BreakoutConfig | None = None,
) -> BreakoutPositionState | None:
    cfg = cfg or BreakoutConfig()
    item = _coerce_row(row)

    prev_close = pd.to_numeric(item.get("prev_close"), errors="coerce")
    day_open = pd.to_numeric(item.get("open"), errors="coerce")
    if pd.notna(prev_close) and prev_close > 0 and pd.notna(day_open):
        gap_pct = float(day_open / prev_close - 1.0)
        if gap_pct >= cfg.ep_gap_exclusion_pct:
            return None

    # ZigZag 用の effective_pivot_level があればそれを使う
    pivot_level = pd.to_numeric(item.get("effective_pivot_level", item.get("pivot_high")), errors="coerce")
    breakout_day_low = pd.to_numeric(item.get("low"), errors="coerce")
    if not np.isfinite(pivot_level) or not np.isfinite(breakout_day_low):
        return None

    entry_price = pd.to_numeric(item.get("trigger_close", item.get("close")), errors="coerce")
    if not np.isfinite(entry_price) or entry_price <= 0:
        return None

    stop_buffer = cfg.stop_buffer_bps / 10_000.0
    initial_stop = min(float(breakout_day_low), float(pivot_level)) * (1.0 - stop_buffer)
    risk_per_share = float(entry_price) - float(initial_stop)
    if not np.isfinite(risk_per_share) or risk_per_share <= 0:
        return None

    # stop limit (ATR/ADR) チェック。zigzag 側で無視指定があればスキップ
    entry_stop_policy = str(item.get("entry_stop_policy", "respect_stop_limit"))
    if entry_stop_policy == "respect_stop_limit":
        atr20 = pd.to_numeric(item.get("atr20"), errors="coerce")
        adr20_pct = pd.to_numeric(item.get("adr20_pct"), errors="coerce")
        atr_limit = float(atr20) * cfg.atr_limit_mult if pd.notna(atr20) else np.nan
        adr_limit = float(adr20_pct) * float(entry_price) * cfg.adr_limit_mult if pd.notna(adr20_pct) else np.nan
        stop_limit = np.nanmin([atr_limit, adr_limit]) if not (pd.isna(atr_limit) and pd.isna(adr_limit)) else np.nan
        if pd.notna(stop_limit) and risk_per_share > stop_limit:
            return None

    risk_budget = float(equity) * cfg.risk_pct_per_trade
    alloc_budget = float(equity) * cfg.max_alloc_pct_per_trade
    shares_by_risk = math.floor(risk_budget / risk_per_share)
    shares_by_alloc = math.floor(alloc_budget / float(entry_price))
    shares_by_cash = math.floor(float(cash) / float(entry_price))
    shares = min(shares_by_risk, shares_by_alloc, shares_by_cash)
    if shares <= 0:
        return None

    trigger_time = pd.to_datetime(item.get("trigger_time"), errors="coerce")
    return BreakoutPositionState(
        symbol=str(item.get("symbol") or "").upper(),
        entry_date=pd.to_datetime(item.get("date")).normalize(),
        entry_price=float(entry_price),
        initial_stop=float(initial_stop),
        pivot_level=float(pivot_level),
        breakout_day_low=float(breakout_day_low),
        initial_shares=int(shares),
        shares=int(shares),
        initial_risk_per_share=float(risk_per_share),
        entry_bar_time=None if pd.isna(trigger_time) else trigger_time,
        entry_source=str(item.get("entry_source", "standard_breakout")),
        entry_lane=str(item.get("entry_lane", "none")),
    )


def evaluate_exit_action(
    state: BreakoutPositionState,
    latest_row: Mapping[str, Any] | pd.Series | Any,
    *,
    cfg: BreakoutConfig | None = None,
) -> dict[str, Any]:
    cfg = cfg or BreakoutConfig()
    row = _coerce_row(latest_row)
    cur_date = pd.to_datetime(row["date"]).normalize()
    close_price = float(pd.to_numeric(row.get("close"), errors="coerce"))
    low_price = float(pd.to_numeric(row.get("low"), errors="coerce"))
    high_price = float(pd.to_numeric(row.get("high"), errors="coerce"))
    hold_score = pd.to_numeric(row.get("hold_score"), errors="coerce")
    dma10 = pd.to_numeric(row.get("dma10"), errors="coerce")
    dma21 = pd.to_numeric(row.get("dma21"), errors="coerce")
    tight_low_volume = bool(row.get("tight_low_volume_day")) if pd.notna(row.get("tight_low_volume_day")) else False
    days_since_entry = int((cur_date - pd.Timestamp(state.entry_date).normalize()).days)

    if low_price <= state.initial_stop and state.shares > 0:
        return {
            "action": "exit_all",
            "reason": "hard_stop_lod",
            "price": state.initial_stop,
            "pending_dma21_grace": False,
            "partial_profit_taken": state.partial_profit_taken,
            "reduced_on_dma21": state.reduced_on_dma21,
        }

    if (
        cfg.day0_day1_pivot_fail_exit_all
        and days_since_entry <= cfg.fast_fail_days
        and close_price < state.pivot_level
        and state.shares > 0
    ):
        return {
            "action": "exit_all",
            "reason": "pivot_fail_exit_all",
            "price": close_price,
            "pending_dma21_grace": False,
            "partial_profit_taken": state.partial_profit_taken,
            "reduced_on_dma21": state.reduced_on_dma21,
        }

    # Partial TP は 1回だけ。高値到達で判定。
    if not state.partial_profit_taken and state.shares > 0:
        tp_price_r = state.entry_price + cfg.tp_partial_r * state.initial_risk_per_share
        tp_price_pct = state.entry_price * (1.0 + cfg.tp_partial_pct)
        tp_hit_price = min(tp_price_r, tp_price_pct)
        if high_price >= tp_hit_price:
            target_remaining = math.ceil(state.initial_shares * 0.50)
            return {
                "action": "reduce",
                "reason": "partial_tp_1st",
                "price": tp_hit_price,
                "target_remaining_shares": target_remaining,
                "pending_dma21_grace": state.pending_dma21_grace,
                "partial_profit_taken": True,
                "reduced_on_dma21": state.reduced_on_dma21,
            }

    # 21DMA: small runner に縮小 -> さらに悪化で final exit
    if pd.notna(dma21) and close_price < float(dma21) and state.shares > 0:
        if cfg.use_dma21_tight_low_volume_grace and tight_low_volume and not state.pending_dma21_grace:
            return {
                "action": "hold",
                "reason": "dma21_grace",
                "pending_dma21_grace": True,
                "partial_profit_taken": state.partial_profit_taken,
                "reduced_on_dma21": state.reduced_on_dma21,
            }

        if not state.reduced_on_dma21:
            if pd.notna(hold_score) and float(hold_score) < cfg.dma21_reduce_threshold:
                target_remaining = math.ceil(state.initial_shares * cfg.dma21_reduce_to_frac)
                return {
                    "action": "reduce",
                    "reason": "dma21_reduce_to_small_runner",
                    "price": close_price,
                    "target_remaining_shares": target_remaining,
                    "pending_dma21_grace": False,
                    "partial_profit_taken": state.partial_profit_taken,
                    "reduced_on_dma21": True,
                }
            return {
                "action": "hold",
                "reason": "dma21_hold",
                "pending_dma21_grace": False,
                "partial_profit_taken": state.partial_profit_taken,
                "reduced_on_dma21": False,
            }

        if pd.notna(hold_score) and float(hold_score) < cfg.dma21_exit_after_reduce_threshold:
            return {
                "action": "exit_all",
                "reason": "dma21_exit_after_reduce",
                "price": close_price,
                "pending_dma21_grace": False,
                "partial_profit_taken": state.partial_profit_taken,
                "reduced_on_dma21": True,
            }

        return {
            "action": "hold",
            "reason": "dma21_runner_hold",
            "pending_dma21_grace": False,
            "partial_profit_taken": state.partial_profit_taken,
            "reduced_on_dma21": True,
        }

    if pd.notna(dma10) and close_price < float(dma10) and state.shares > 0:
        if pd.notna(hold_score) and float(hold_score) < cfg.dma10_exit_threshold:
            return {
                "action": "exit_all",
                "reason": "dma10_holdscore_exit_all",
                "price": close_price,
                "pending_dma21_grace": state.pending_dma21_grace,
                "partial_profit_taken": state.partial_profit_taken,
                "reduced_on_dma21": state.reduced_on_dma21,
            }

        if pd.notna(hold_score) and float(hold_score) < cfg.dma10_reduce_threshold:
            target_remaining = math.ceil(state.initial_shares * cfg.dma10_reduce_to_frac)
            if state.shares > target_remaining:
                return {
                    "action": "reduce",
                    "reason": "dma10_holdscore_reduce",
                    "price": close_price,
                    "target_remaining_shares": target_remaining,
                    "pending_dma21_grace": state.pending_dma21_grace,
                    "partial_profit_taken": state.partial_profit_taken,
                    "reduced_on_dma21": state.reduced_on_dma21,
                }

    return {
        "action": "hold",
        "reason": "hold",
        "pending_dma21_grace": False,
        "partial_profit_taken": state.partial_profit_taken,
        "reduced_on_dma21": state.reduced_on_dma21,
    }


def signals_from_report(report: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "symbol",
        "date",
        "breakout_signal",
        "pivot_high",
        "effective_pivot_level",
        "trigger_time",
        "trigger_close",
        "trigger_score",
        "entry_source",
        "entry_lane",
        "entry_stop_policy",
        "entry_priority_bucket",
        "priority_score_within_source",
        "leader_score",
        "rs_rating",
    ]
    work = report.copy()
    work["breakout_signal"] = work["breakout_signal"].fillna(False).astype(bool)
    keep = [c for c in columns if c in work.columns]
    return work[keep].sort_values(["date", "symbol"], kind="mergesort").reset_index(drop=True)


def _to_exit_backtest_config(cfg: BreakoutConfig) -> ExitBacktestConfig:
    return ExitBacktestConfig(
        initial_equity=cfg.initial_equity,
        max_positions=cfg.max_positions,
        risk_pct_per_trade=cfg.risk_pct_per_trade,
        max_alloc_pct_per_trade=cfg.max_alloc_pct_per_trade,
        ep_gap_exclusion_pct=cfg.ep_gap_exclusion_pct,
        stop_buffer_bps=cfg.stop_buffer_bps,
        atr_limit_mult=cfg.atr_limit_mult,
        adr_limit_mult=cfg.adr_limit_mult,
        fast_fail_days=cfg.fast_fail_days,
        day0_day1_pivot_fail_exit_all=cfg.day0_day1_pivot_fail_exit_all,
        tp_partial_r=cfg.tp_partial_r,
        tp_partial_pct=cfg.tp_partial_pct,
        dma10_reduce_to_frac=cfg.dma10_reduce_to_frac,
        dma10_reduce_threshold=cfg.dma10_reduce_threshold,
        dma10_exit_threshold=cfg.dma10_exit_threshold,
        dma21_reduce_to_frac=cfg.dma21_reduce_to_frac,
        dma21_reduce_threshold=cfg.dma21_reduce_threshold,
        dma21_exit_after_reduce_threshold=cfg.dma21_exit_after_reduce_threshold,
        use_dma21_tight_low_volume_grace=cfg.use_dma21_tight_low_volume_grace,
        use_intraday_trigger_time=cfg.use_intraday_trigger_time,
        entry_at=cfg.entry_at,
        allow_reentry_same_symbol=cfg.allow_reentry_same_symbol,
    )


def run_breakout_backtest(
    daily_bars: Any,
    intraday_bars: Any,
    cfg: BreakoutConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float], pd.DataFrame]:
    cfg = cfg or BreakoutConfig()
    daily = normalize_daily_bars(daily_bars, session_timezone=cfg.session_timezone)
    intraday = normalize_intraday_bars(intraday_bars, session_timezone=cfg.session_timezone)
    report = build_breakout_signal_report(daily, intraday, cfg=cfg)
    signals = signals_from_report(report)
    equity_curve, fills_df, stats, _ = _run_breakout_backtest(
        daily=daily,
        signals=signals,
        intraday=intraday,
        cfg=_to_exit_backtest_config(cfg),
    )
    return equity_curve, fills_df, stats, report


def run_breakout_backtest_from_inputs(
    daily_bars: Any,
    intraday_bars: Any,
    signals: pd.DataFrame,
    cfg: BreakoutConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    cfg = cfg or BreakoutConfig()
    daily = normalize_daily_bars(daily_bars, session_timezone=cfg.session_timezone)
    intraday = normalize_intraday_bars(intraday_bars, session_timezone=cfg.session_timezone)
    signals_work = signals.copy()
    signals_work["date"] = pd.to_datetime(signals_work["date"]).dt.normalize()
    equity_curve, fills_df, stats, _ = _run_breakout_backtest(
        daily=daily,
        signals=signals_work,
        intraday=intraday,
        cfg=_to_exit_backtest_config(cfg),
    )
    return equity_curve, fills_df, stats
