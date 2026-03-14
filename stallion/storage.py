from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Iterable

import pandas as pd

from .config import Settings


class SQLiteParquetStore:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.sqlite_path = settings.paths.sqlite_path
        self.parquet_dir = settings.paths.parquet_dir
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.sqlite_path)
        connection.execute("PRAGMA journal_mode=WAL;")
        connection.execute("PRAGMA synchronous=NORMAL;")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS universe (
                    symbol TEXT PRIMARY KEY,
                    yahoo_symbol TEXT NOT NULL,
                    exchange TEXT,
                    company_name TEXT,
                    market_cap REAL,
                    sector TEXT,
                    industry TEXT,
                    country TEXT,
                    rank_market_cap INTEGER,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS bars_1d (
                    symbol TEXT NOT NULL,
                    ts TEXT NOT NULL,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    adj_close REAL,
                    volume REAL,
                    source TEXT,
                    PRIMARY KEY(symbol, ts)
                );

                CREATE TABLE IF NOT EXISTS bars_5m (
                    symbol TEXT NOT NULL,
                    ts TEXT NOT NULL,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    volume REAL,
                    cumulative_volume REAL,
                    source TEXT,
                    PRIMARY KEY(symbol, ts)
                );

                CREATE TABLE IF NOT EXISTS daily_features (
                    symbol TEXT NOT NULL,
                    session_date TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    PRIMARY KEY(symbol, session_date)
                );

                CREATE TABLE IF NOT EXISTS nightly_shortlist (
                    session_date TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    rank_order INTEGER NOT NULL,
                    shortlist_score REAL,
                    payload_json TEXT NOT NULL,
                    PRIMARY KEY(session_date, symbol)
                );

                CREATE TABLE IF NOT EXISTS model_registry (
                    model_name TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    threshold REAL NOT NULL,
                    artifact_path TEXT NOT NULL,
                    metadata_json TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS live_signals (
                    session_date TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    score REAL,
                    threshold REAL,
                    selected INTEGER NOT NULL,
                    payload_json TEXT NOT NULL,
                    PRIMARY KEY(session_date, timestamp, symbol)
                );

                CREATE TABLE IF NOT EXISTS live_fills (
                    fill_id TEXT PRIMARY KEY,
                    session_date TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    quantity INTEGER NOT NULL,
                    price REAL NOT NULL,
                    payload_json TEXT NOT NULL
                );
                """
            )

    def _upsert_frame(self, frame: pd.DataFrame, table_name: str) -> None:
        if frame.empty:
            return
        with self._connect() as connection:
            frame.to_sql(table_name, connection, if_exists="append", index=False, method="multi")

    def save_universe(self, frame: pd.DataFrame) -> None:
        work = frame.copy()
        work["updated_at"] = pd.Timestamp.utcnow().isoformat()
        with self._connect() as connection:
            connection.execute("DELETE FROM universe")
            work.to_sql("universe", connection, if_exists="append", index=False, method="multi")
        self.write_parquet_snapshot(work, "universe/latest.parquet")

    def save_bars(self, frame: pd.DataFrame, timeframe: str) -> None:
        if frame.empty:
            return
        table_name = "bars_1d" if timeframe == "1d" else "bars_5m"
        work = frame.copy()
        work["ts"] = pd.to_datetime(work["ts"], utc=True, errors="coerce").astype(str)
        if timeframe == "1d":
            expected_columns = ["symbol", "ts", "open", "high", "low", "close", "adj_close", "volume", "source"]
        else:
            expected_columns = ["symbol", "ts", "open", "high", "low", "close", "volume", "cumulative_volume", "source"]
        for column in expected_columns:
            if column not in work.columns:
                work[column] = None
        work = work[expected_columns].copy()
        columns = list(work.columns)
        placeholders = ",".join(["?"] * len(columns))
        sql = f"INSERT OR REPLACE INTO {table_name} ({','.join(columns)}) VALUES ({placeholders})"
        rows = [tuple(row[column] for column in columns) for row in work.to_dict(orient="records")]
        with self._connect() as connection:
            connection.executemany(sql, rows)
            connection.commit()
        partition_label = "daily" if timeframe == "1d" else "intraday_5m"
        self.write_parquet_snapshot(work, f"raw/{partition_label}/latest.parquet")

    def save_daily_features(self, frame: pd.DataFrame) -> None:
        if frame.empty:
            return
        records = []
        for row in frame.to_dict(orient="records"):
            payload = dict(row)
            symbol = str(payload.pop("symbol"))
            session_date = str(pd.Timestamp(payload.pop("session_date")).date())
            records.append(
                {
                    "symbol": symbol,
                    "session_date": session_date,
                    "payload_json": json.dumps(payload, ensure_ascii=True, default=str),
                }
            )
        work = pd.DataFrame(records)
        with self._connect() as connection:
            connection.execute("DELETE FROM daily_features")
            work.to_sql("daily_features", connection, if_exists="append", index=False, method="multi")
        self.write_parquet_snapshot(frame, "features/daily/latest.parquet")

    def save_shortlist(self, session_date: pd.Timestamp, frame: pd.DataFrame) -> None:
        day = str(pd.Timestamp(session_date).date())
        work = frame.copy()
        if "rank_order" not in work.columns:
            work["rank_order"] = range(1, len(work) + 1)
        records = []
        for row in work.to_dict(orient="records"):
            payload = dict(row)
            symbol = str(payload.get("symbol"))
            rank_order = int(payload.get("rank_order", 0))
            shortlist_score = float(payload.get("shortlist_score", payload.get("daily_rs_score_prev", 0.0)))
            records.append(
                {
                    "session_date": day,
                    "symbol": symbol,
                    "rank_order": rank_order,
                    "shortlist_score": shortlist_score,
                    "payload_json": json.dumps(payload, ensure_ascii=True, default=str),
                }
            )
        shortlist = pd.DataFrame(records)
        with self._connect() as connection:
            connection.execute("DELETE FROM nightly_shortlist WHERE session_date = ?", (day,))
            shortlist.to_sql("nightly_shortlist", connection, if_exists="append", index=False, method="multi")
        self.write_parquet_snapshot(frame, f"artifacts/shortlist/{day}.parquet")

    def save_model_registry(self, model_name: str, created_at: pd.Timestamp, threshold: float, artifact_path: Path, metadata: dict) -> None:
        row = pd.DataFrame(
            [
                {
                    "model_name": model_name,
                    "created_at": created_at.isoformat(),
                    "threshold": threshold,
                    "artifact_path": str(artifact_path),
                    "metadata_json": json.dumps(metadata, ensure_ascii=True, default=str),
                }
            ]
        )
        with self._connect() as connection:
            connection.execute("DELETE FROM model_registry WHERE model_name = ?", (model_name,))
            row.to_sql("model_registry", connection, if_exists="append", index=False, method="multi")

    def append_live_signals(self, frame: pd.DataFrame) -> None:
        if frame.empty:
            return
        work = frame.copy()
        if "payload_json" not in work.columns:
            work["payload_json"] = work.apply(lambda row: json.dumps(row.to_dict(), ensure_ascii=True, default=str), axis=1)
        with self._connect() as connection:
            work.to_sql("live_signals", connection, if_exists="append", index=False, method="multi")

    def append_live_fill(self, row: dict) -> None:
        frame = pd.DataFrame([row])
        with self._connect() as connection:
            frame.to_sql("live_fills", connection, if_exists="append", index=False, method="multi")

    def write_parquet_snapshot(self, frame: pd.DataFrame, relative_path: str) -> Path:
        destination = self.parquet_dir / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(destination, index=False)
        return destination

    def load_universe(self) -> pd.DataFrame:
        query = "SELECT * FROM universe ORDER BY rank_market_cap ASC"
        with self._connect() as connection:
            return pd.read_sql_query(query, connection)

    def load_bars(self, timeframe: str, symbols: Iterable[str] | None = None) -> pd.DataFrame:
        table_name = "bars_1d" if timeframe == "1d" else "bars_5m"
        query = f"SELECT * FROM {table_name}"
        params: tuple = ()
        if symbols:
            symbol_list = tuple(sorted(set(symbols)))
            placeholders = ",".join(["?"] * len(symbol_list))
            query += f" WHERE symbol IN ({placeholders})"
            params = symbol_list
        with self._connect() as connection:
            frame = pd.read_sql_query(query, connection, params=params)
        if frame.empty:
            return frame
        frame["ts"] = pd.to_datetime(frame["ts"], utc=True, errors="coerce")
        return frame.sort_values(["symbol", "ts"]).reset_index(drop=True)

    def load_daily_features(self, session_date: pd.Timestamp | None = None, symbols: Iterable[str] | None = None) -> pd.DataFrame:
        query = "SELECT symbol, session_date, payload_json FROM daily_features"
        clauses = []
        params: list[str] = []
        if session_date is not None:
            clauses.append("session_date = ?")
            params.append(str(pd.Timestamp(session_date).date()))
        if symbols:
            symbol_list = tuple(sorted(set(symbols)))
            placeholders = ",".join(["?"] * len(symbol_list))
            clauses.append(f"symbol IN ({placeholders})")
            params.extend(symbol_list)
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        with self._connect() as connection:
            frame = pd.read_sql_query(query, connection, params=tuple(params))
        if frame.empty:
            return frame
        payload = frame["payload_json"].apply(json.loads).apply(pd.Series)
        merged = pd.concat([frame.drop(columns=["payload_json"]), payload], axis=1)
        merged["session_date"] = pd.to_datetime(merged["session_date"])
        return merged

    def load_shortlist(self, session_date: pd.Timestamp | None = None) -> pd.DataFrame:
        query = "SELECT session_date, symbol, rank_order, shortlist_score, payload_json FROM nightly_shortlist"
        params: tuple = ()
        if session_date is not None:
            query += " WHERE session_date = ?"
            params = (str(pd.Timestamp(session_date).date()),)
        query += " ORDER BY session_date DESC, rank_order ASC"
        with self._connect() as connection:
            frame = pd.read_sql_query(query, connection, params=params)
        if frame.empty:
            return frame
        payload = frame["payload_json"].apply(json.loads).apply(pd.Series)
        merged = pd.concat([frame.drop(columns=["payload_json"]), payload], axis=1)
        merged["session_date"] = pd.to_datetime(merged["session_date"])
        return merged
