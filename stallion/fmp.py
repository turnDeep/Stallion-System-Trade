from __future__ import annotations

import logging
import time
from typing import Any, Iterable

import pandas as pd
import requests
import yfinance as yf

from .config import Settings

try:
    from curl_cffi import requests as curl_requests
except Exception:  # pragma: no cover - optional dependency
    curl_requests = None


FMP_STOCK_SCREENER_URL = "https://financialmodelingprep.com/api/v3/stock-screener"
FMP_BATCH_QUOTE_URL = "https://financialmodelingprep.com/api/v3/quote/{symbols}"
LOGGER = logging.getLogger(__name__)


def _normalize_symbol(symbol: str) -> str:
    return str(symbol).strip().upper().replace(".", "-")


def _make_yfinance_session():
    if curl_requests is None:
        return None
    return curl_requests.Session(impersonate="chrome")


def _chunk_symbols(symbols: list[str], size: int) -> list[list[str]]:
    return [symbols[start : start + size] for start in range(0, len(symbols), size)]


def _parse_yfinance_download(raw: pd.DataFrame, symbols: list[str]) -> tuple[pd.DataFrame, list[str]]:
    if raw.empty:
        return pd.DataFrame(), list(symbols)

    frames: list[pd.DataFrame] = []
    found_symbols: set[str] = set()
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
            found_symbols.add(symbol)
    else:
        part = raw.dropna(how="all").copy()
        if not part.empty:
            symbol = symbols[0]
            part.columns = [str(col).lower().replace(" ", "_") for col in part.columns]
            part["symbol"] = symbol
            part["ts"] = pd.to_datetime(part.index, utc=True, errors="coerce")
            frames.append(part.reset_index(drop=True))
            found_symbols.add(symbol)

    if not frames:
        return pd.DataFrame(), list(symbols)
    frame = pd.concat(frames, ignore_index=True)
    expected = ["open", "high", "low", "close", "adj_close", "volume", "symbol", "ts"]
    for column in set(expected).difference(frame.columns):
        frame[column] = pd.NA
    missing = [symbol for symbol in symbols if symbol not in found_symbols]
    return frame[[*expected]], missing


def _download_yfinance_batch(
    symbols: list[str],
    *,
    period: str,
    interval: str,
    auto_adjust: bool,
    session,
    timeout: int,
) -> tuple[pd.DataFrame, list[str]]:
    raw = yf.download(
        tickers=" ".join(symbols),
        period=period,
        interval=interval,
        auto_adjust=auto_adjust,
        group_by="ticker",
        progress=False,
        threads=False,
        prepost=False,
        timeout=timeout,
        session=session,
        multi_level_index=True,
    )
    return _parse_yfinance_download(raw, symbols)


def _download_yfinance_single_with_retry(
    symbol: str,
    *,
    period: str,
    interval: str,
    auto_adjust: bool,
    session,
    timeout: int,
    retry_delays: tuple[float, ...],
) -> pd.DataFrame:
    last_frame = pd.DataFrame()
    for attempt, delay in enumerate((0.0, *retry_delays), start=1):
        if delay > 0:
            time.sleep(delay)
        try:
            frame, missing = _download_yfinance_batch(
                [symbol],
                period=period,
                interval=interval,
                auto_adjust=auto_adjust,
                session=session,
                timeout=timeout,
            )
        except Exception as exc:  # pragma: no cover - network variance
            LOGGER.warning("yfinance single download failed for %s on attempt %s: %s", symbol, attempt, exc)
            session = _make_yfinance_session()
            continue
        last_frame = frame
        if not frame.empty and not missing:
            return frame
    return last_frame


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

    normalized_symbols = [_normalize_symbol(symbol) for symbol in symbols if str(symbol).strip()]
    deduped_symbols = list(dict.fromkeys(normalized_symbols))

    is_intraday = interval.endswith("m") or interval.endswith("h")
    chunk_size = 25 if is_intraday else 100
    inter_chunk_sleep = 1.5 if is_intraday else 0.5
    retry_delays = (5.0, 12.0, 25.0) if is_intraday else (2.0, 5.0, 10.0)
    timeout = 30 if is_intraday else 20

    session = _make_yfinance_session()
    frames: list[pd.DataFrame] = []
    missing_symbols: list[str] = []
    symbol_chunks = _chunk_symbols(deduped_symbols, chunk_size)

    LOGGER.info(
        "Downloading yfinance bars: interval=%s period=%s symbols=%s chunks=%s chunk_size=%s session=%s",
        interval,
        period,
        len(deduped_symbols),
        len(symbol_chunks),
        chunk_size,
        "curl_cffi" if session is not None else "default",
    )

    for chunk_idx, chunk_symbols in enumerate(symbol_chunks, start=1):
        last_missing = list(chunk_symbols)
        chunk_frame = pd.DataFrame()
        for attempt, delay in enumerate((0.0, *retry_delays), start=1):
            if delay > 0:
                time.sleep(delay)
            try:
                chunk_frame, last_missing = _download_yfinance_batch(
                    chunk_symbols,
                    period=period,
                    interval=interval,
                    auto_adjust=auto_adjust,
                    session=session,
                    timeout=timeout,
                )
            except Exception as exc:  # pragma: no cover - network variance
                LOGGER.warning(
                    "yfinance batch failed interval=%s chunk=%s/%s attempt=%s symbols=%s: %s",
                    interval,
                    chunk_idx,
                    len(symbol_chunks),
                    attempt,
                    len(chunk_symbols),
                    exc,
                )
                session = _make_yfinance_session()
                continue
            if not last_missing:
                break
            LOGGER.warning(
                "yfinance missing %s/%s symbols in chunk %s/%s attempt=%s interval=%s",
                len(last_missing),
                len(chunk_symbols),
                chunk_idx,
                len(symbol_chunks),
                attempt,
                interval,
            )
        if not chunk_frame.empty:
            frames.append(chunk_frame)
        if last_missing:
            missing_symbols.extend(last_missing)
        time.sleep(inter_chunk_sleep)

    recovered_frames: list[pd.DataFrame] = []
    still_missing: list[str] = []
    for symbol in dict.fromkeys(missing_symbols):
        frame = _download_yfinance_single_with_retry(
            symbol,
            period=period,
            interval=interval,
            auto_adjust=auto_adjust,
            session=session,
            timeout=timeout,
            retry_delays=retry_delays,
        )
        if frame.empty:
            still_missing.append(symbol)
        else:
            recovered_frames.append(frame)
        time.sleep(max(1.0, inter_chunk_sleep))

    if frames or recovered_frames:
        frame = pd.concat([*frames, *recovered_frames], ignore_index=True)
        frame = frame.dropna(subset=["ts"]).drop_duplicates(subset=["symbol", "ts"], keep="last")
        frame = frame.sort_values(["symbol", "ts"]).reset_index(drop=True)
    else:
        frame = pd.DataFrame(columns=["open", "high", "low", "close", "adj_close", "volume", "symbol", "ts"])

    if still_missing:
        preview = still_missing[:25]
        LOGGER.warning(
            "yfinance unresolved symbols interval=%s count=%s preview=%s",
            interval,
            len(still_missing),
            preview,
        )
    return frame
