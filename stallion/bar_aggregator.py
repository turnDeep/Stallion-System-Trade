from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


def _to_utc_timestamp(value: Any) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        raise ValueError("Naive timestamps are not allowed; pass an explicit UTC timestamp.")
    return ts.tz_convert("UTC")


def _bar_start_utc(observed_at_utc: pd.Timestamp, session_timezone: str) -> pd.Timestamp:
    local = observed_at_utc.tz_convert(session_timezone)
    floored = local.floor("5min")
    return floored.tz_convert("UTC")


@dataclass
class _BarState:
    symbol: str
    bar_start_utc: pd.Timestamp
    open: float
    high: float
    low: float
    close: float
    volume: float
    last_cumulative_volume: float
    last_observed_at_utc: pd.Timestamp


class QuoteBarAggregator:
    def __init__(self, session_timezone: str = "America/New_York") -> None:
        self.session_timezone = session_timezone
        self._state: dict[str, _BarState] = {}

    def ingest_quotes(self, quotes: pd.DataFrame, observed_at_utc: pd.Timestamp | None = None) -> pd.DataFrame:
        if quotes.empty:
            return pd.DataFrame(columns=["symbol", "ts", "open", "high", "low", "close", "volume", "cumulative_volume", "source"])
        work = quotes.copy()
        if observed_at_utc is None:
            observed_at_utc = pd.Timestamp.utcnow(tz="UTC")
        observed_at_utc = _to_utc_timestamp(observed_at_utc)
        finalized_rows: list[dict[str, Any]] = []

        for row in work.itertuples(index=False):
            symbol = str(getattr(row, "symbol")).upper()
            price = float(getattr(row, "price"))
            cumulative_volume = float(getattr(row, "cumulative_volume", getattr(row, "volume", 0.0)) or 0.0)
            bar_start = _bar_start_utc(observed_at_utc, self.session_timezone)
            current = self._state.get(symbol)

            if current is None:
                self._state[symbol] = _BarState(
                    symbol=symbol,
                    bar_start_utc=bar_start,
                    open=price,
                    high=price,
                    low=price,
                    close=price,
                    volume=0.0,
                    last_cumulative_volume=cumulative_volume,
                    last_observed_at_utc=observed_at_utc,
                )
                continue

            if bar_start != current.bar_start_utc:
                finalized_rows.append(
                    {
                        "symbol": current.symbol,
                        "ts": current.bar_start_utc.isoformat(),
                        "open": current.open,
                        "high": current.high,
                        "low": current.low,
                        "close": current.close,
                        "volume": current.volume,
                        "cumulative_volume": current.last_cumulative_volume,
                        "source": "fmp_quote_aggregate",
                    }
                )
                delta_volume = max(cumulative_volume - current.last_cumulative_volume, 0.0)
                self._state[symbol] = _BarState(
                    symbol=symbol,
                    bar_start_utc=bar_start,
                    open=price,
                    high=price,
                    low=price,
                    close=price,
                    volume=delta_volume,
                    last_cumulative_volume=cumulative_volume,
                    last_observed_at_utc=observed_at_utc,
                )
                continue

            delta_volume = max(cumulative_volume - current.last_cumulative_volume, 0.0)
            current.high = max(current.high, price)
            current.low = min(current.low, price)
            current.close = price
            current.volume += delta_volume
            current.last_cumulative_volume = cumulative_volume
            current.last_observed_at_utc = observed_at_utc

        return pd.DataFrame(finalized_rows)

    def bootstrap_from_snapshots(self, snapshots: pd.DataFrame) -> pd.DataFrame:
        if snapshots.empty:
            return pd.DataFrame(columns=["symbol", "ts", "open", "high", "low", "close", "volume", "cumulative_volume", "source"])
        work = snapshots.copy()
        work["ts"] = pd.to_datetime(work["ts"], utc=True, errors="coerce")
        work = work.dropna(subset=["ts"]).sort_values(["ts", "symbol"])
        finalized_frames: list[pd.DataFrame] = []
        for ts, frame in work.groupby("ts", sort=True):
            finalized = self.ingest_quotes(frame, observed_at_utc=ts)
            if not finalized.empty:
                finalized_frames.append(finalized)
        if not finalized_frames:
            return pd.DataFrame(columns=["symbol", "ts", "open", "high", "low", "close", "volume", "cumulative_volume", "source"])
        return pd.concat(finalized_frames, ignore_index=True)

    def flush_completed(self, as_of_utc: pd.Timestamp) -> pd.DataFrame:
        as_of_utc = _to_utc_timestamp(as_of_utc)
        rows: list[dict[str, Any]] = []
        for symbol, current in list(self._state.items()):
            if current.bar_start_utc + pd.Timedelta(minutes=5) > as_of_utc:
                continue
            rows.append(
                {
                    "symbol": current.symbol,
                    "ts": current.bar_start_utc.isoformat(),
                    "open": current.open,
                    "high": current.high,
                    "low": current.low,
                    "close": current.close,
                    "volume": current.volume,
                    "cumulative_volume": current.last_cumulative_volume,
                    "source": "fmp_quote_aggregate",
                }
            )
            del self._state[symbol]
        return pd.DataFrame(rows)
