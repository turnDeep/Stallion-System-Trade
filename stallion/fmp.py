from __future__ import annotations

import time
from typing import Any, Iterable

import pandas as pd
import requests
import yfinance as yf

from .config import Settings


FMP_STOCK_SCREENER_URL = "https://financialmodelingprep.com/api/v3/stock-screener"
FMP_BATCH_QUOTE_URL = "https://financialmodelingprep.com/api/v3/quote/{symbols}"


def _normalize_symbol(symbol: str) -> str:
    return str(symbol).strip().upper().replace(".", "-")


class FMPClient:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.session = requests.Session()
        self.request_timestamps: list[float] = []

    def _respect_rate_limit(self, max_per_minute: int = 700) -> None:
        now = time.time()
        self.request_timestamps = [ts for ts in self.request_timestamps if now - ts < 60]
        if len(self.request_timestamps) >= max_per_minute:
            sleep_for = 60 - (now - self.request_timestamps[0]) + 0.2
            if sleep_for > 0:
                time.sleep(sleep_for)
        self.request_timestamps.append(time.time())

    def _get_json(self, url: str, params: dict[str, Any] | None = None) -> Any:
        self._respect_rate_limit()
        payload = dict(params or {})
        payload["apikey"] = self.settings.credentials.fmp_api_key
        response = self.session.get(url, params=payload, timeout=60)
        response.raise_for_status()
        return response.json()

    def fetch_top_universe(self, top_n: int = 3000, exchanges: Iterable[str] = ("nasdaq", "nyse")) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for exchange in exchanges:
            data = self._get_json(
                FMP_STOCK_SCREENER_URL,
                {
                    "exchange": exchange.lower(),
                    "isEtf": "false",
                    "isFund": "false",
                    "isActivelyTrading": "true",
                    "limit": 10000,
                },
            )
            for item in data:
                symbol = _normalize_symbol(item.get("symbol", ""))
                if not symbol:
                    continue
                rows.append(
                    {
                        "symbol": symbol,
                        "yahoo_symbol": symbol,
                        "exchange": exchange.upper(),
                        "company_name": item.get("companyName"),
                        "market_cap": float(item.get("marketCap") or 0.0),
                        "sector": item.get("sector") or "Unknown",
                        "industry": item.get("industry") or "Unknown",
                        "country": item.get("country") or "Unknown",
                    }
                )
        universe = pd.DataFrame(rows)
        if universe.empty:
            raise RuntimeError("No rows returned from FMP stock screener.")
        universe = universe.sort_values(["market_cap", "symbol"], ascending=[False, True])
        universe = universe.drop_duplicates(subset=["yahoo_symbol"], keep="first").head(top_n).reset_index(drop=True)
        universe["rank_market_cap"] = range(1, len(universe) + 1)
        return universe

    def fetch_batch_quotes(self, symbols: list[str]) -> pd.DataFrame:
        if not symbols:
            return pd.DataFrame()
        url = FMP_BATCH_QUOTE_URL.format(symbols=",".join(symbols))
        data = self._get_json(url)
        frame = pd.DataFrame(data)
        if frame.empty:
            return frame
        frame["symbol"] = frame["symbol"].astype(str).str.upper()
        frame["fetched_at"] = pd.Timestamp.utcnow()
        return frame


def download_yfinance_bars(symbols: list[str], period: str, interval: str, auto_adjust: bool = False) -> pd.DataFrame:
    if not symbols:
        return pd.DataFrame()
    raw = yf.download(
        tickers=" ".join(symbols),
        period=period,
        interval=interval,
        auto_adjust=auto_adjust,
        group_by="ticker",
        progress=False,
        threads=True,
        prepost=False,
    )
    if raw.empty:
        return pd.DataFrame()

    frames: list[pd.DataFrame] = []
    if isinstance(raw.columns, pd.MultiIndex):
        for symbol in symbols:
            if symbol not in raw.columns.get_level_values(0):
                continue
            part = raw[symbol].dropna(how="all").copy()
            if part.empty:
                continue
            part.columns = [str(col).lower().replace(" ", "_") for col in part.columns]
            part["symbol"] = symbol
            part["ts"] = pd.to_datetime(part.index, utc=True, errors="coerce")
            frames.append(part.reset_index(drop=True))
    else:
        part = raw.dropna(how="all").copy()
        part.columns = [str(col).lower().replace(" ", "_") for col in part.columns]
        part["symbol"] = symbols[0]
        part["ts"] = pd.to_datetime(part.index, utc=True, errors="coerce")
        frames.append(part.reset_index(drop=True))

    if not frames:
        return pd.DataFrame()
    frame = pd.concat(frames, ignore_index=True)
    expected = ["open", "high", "low", "close", "adj_close", "volume", "symbol", "ts"]
    missing = set(expected).difference(frame.columns)
    for column in missing:
        frame[column] = pd.NA
    return frame[[*expected]]
