from __future__ import annotations

import json
import logging
import sqlite3
import time
from pathlib import Path
from typing import Iterable

import pandas as pd

from .config import Settings


LOGGER = logging.getLogger(__name__)


class SQLiteParquetStore:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.sqlite_path = settings.paths.sqlite_path
        self.parquet_dir = settings.paths.parquet_dir
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.sqlite_path, timeout=60.0)
        connection.execute("PRAGMA journal_mode=WAL;")
        connection.execute("PRAGMA synchronous=NORMAL;")
        connection.execute("PRAGMA busy_timeout=60000;")
        connection.execute("PRAGMA temp_store=MEMORY;")
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

                CREATE TABLE IF NOT EXISTS quote_snapshots (
                    symbol TEXT NOT NULL,
                    ts TEXT NOT NULL,
                    price REAL,
                    cumulative_volume REAL,
                    payload_json TEXT NOT NULL,
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

                CREATE TABLE IF NOT EXISTS live_orders (
                    client_order_id TEXT PRIMARY KEY,
                    session_date TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    quantity INTEGER NOT NULL,
                    filled_quantity INTEGER NOT NULL,
                    requested_price REAL,
                    status TEXT NOT NULL,
                    broker_order_id TEXT,
                    placed_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS open_positions (
                    symbol TEXT PRIMARY KEY,
                    session_date TEXT NOT NULL,
                    quantity INTEGER NOT NULL,
                    avg_price REAL,
                    entry_time TEXT NOT NULL,
                    broker_order_id TEXT,
                    status TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL
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

                CREATE TABLE IF NOT EXISTS system_state (
                    state_key TEXT PRIMARY KEY,
                    state_value TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS heartbeats (
                    component TEXT PRIMARY KEY,
                    heartbeat_at TEXT NOT NULL,
                    status TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS alerts (
                    alert_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    level TEXT NOT NULL,
                    component TEXT NOT NULL,
                    message TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                );
                """
            )

    def _sqlite_max_variable_number(self, connection: sqlite3.Connection) -> int:
        try:
            rows = connection.execute("PRAGMA compile_options;").fetchall()
        except sqlite3.DatabaseError:
            return 999
        for (option,) in rows:
            option_text = str(option)
            if option_text.startswith("MAX_VARIABLE_NUMBER="):
                try:
                    return int(option_text.split("=", 1)[1])
                except ValueError:
                    return 999
        return 999

    def _frame_chunksize(self, connection: sqlite3.Connection, column_count: int, hard_cap: int = 2_000) -> int:
        if column_count <= 0:
            return 1
        max_variables = max(100, self._sqlite_max_variable_number(connection) - 16)
        return max(1, min(hard_cap, max_variables // column_count))

    def _append_frame_chunked(self, connection: sqlite3.Connection, table_name: str, frame: pd.DataFrame) -> None:
        if frame.empty:
            return
        chunksize = self._frame_chunksize(connection, len(frame.columns))
        frame.to_sql(table_name, connection, if_exists="append", index=False, method="multi", chunksize=chunksize)

    def _executemany_chunked(
        self,
        connection: sqlite3.Connection,
        sql: str,
        rows: list[tuple],
        row_width: int,
        *,
        commit_every_chunks: int = 25,
        retry_attempts: int = 3,
    ) -> None:
        if not rows:
            return
        chunk_size = self._frame_chunksize(connection, row_width, hard_cap=5_000)
        total_chunks = (len(rows) + chunk_size - 1) // chunk_size
        for chunk_index, start in enumerate(range(0, len(rows), chunk_size), start=1):
            chunk_rows = rows[start : start + chunk_size]
            for attempt in range(1, retry_attempts + 1):
                try:
                    connection.executemany(sql, chunk_rows)
                    break
                except sqlite3.OperationalError as exc:
                    if "disk i/o error" not in str(exc).lower() or attempt >= retry_attempts:
                        raise
                    try:
                        connection.rollback()
                    except sqlite3.DatabaseError:
                        pass
                    LOGGER.warning(
                        "SQLite disk I/O error on chunk %s/%s, retrying attempt %s/%s",
                        chunk_index,
                        total_chunks,
                        attempt,
                        retry_attempts,
                    )
                    time.sleep(float(attempt))
            if chunk_index % max(1, commit_every_chunks) == 0:
                connection.commit()
                try:
                    connection.execute("PRAGMA wal_checkpoint(PASSIVE);")
                except sqlite3.DatabaseError:
                    pass

    def _upsert_frame(self, frame: pd.DataFrame, table_name: str) -> None:
        if frame.empty:
            return
        with self._connect() as connection:
            self._append_frame_chunked(connection, table_name, frame)

    def save_universe(self, frame: pd.DataFrame) -> None:
        work = frame.copy()
        work["updated_at"] = pd.Timestamp.utcnow().isoformat()
        with self._connect() as connection:
            connection.execute("DELETE FROM universe")
            self._append_frame_chunked(connection, "universe", work)
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
            self._executemany_chunked(
                connection,
                sql,
                rows,
                len(columns),
                commit_every_chunks=5 if timeframe == "5m" else 25,
                retry_attempts=5 if timeframe == "5m" else 3,
            )
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
            self._append_frame_chunked(connection, "daily_features", work)
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
            self._append_frame_chunked(connection, "nightly_shortlist", shortlist)
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
            self._append_frame_chunked(connection, "model_registry", row)

    def append_live_signals(self, frame: pd.DataFrame) -> None:
        if frame.empty:
            return
        work = frame.copy()
        if "payload_json" not in work.columns:
            work["payload_json"] = work.apply(lambda row: json.dumps(row.to_dict(), ensure_ascii=True, default=str), axis=1)
        with self._connect() as connection:
            self._append_frame_chunked(connection, "live_signals", work)

    def append_live_fill(self, row: dict) -> None:
        frame = pd.DataFrame([row])
        with self._connect() as connection:
            self._append_frame_chunked(connection, "live_fills", frame)

    def append_quote_snapshots(self, frame: pd.DataFrame) -> None:
        if frame.empty:
            return
        work = frame.copy()
        work["ts"] = pd.to_datetime(work["ts"], utc=True, errors="coerce").astype(str)
        if "payload_json" not in work.columns:
            work["payload_json"] = work.apply(lambda row: json.dumps(row.to_dict(), ensure_ascii=True, default=str), axis=1)
        expected = ["symbol", "ts", "price", "cumulative_volume", "payload_json"]
        for column in expected:
            if column not in work.columns:
                work[column] = None
        sql = "INSERT OR REPLACE INTO quote_snapshots (symbol, ts, price, cumulative_volume, payload_json) VALUES (?, ?, ?, ?, ?)"
        rows = [tuple(item[column] for column in expected) for item in work[expected].to_dict(orient="records")]
        with self._connect() as connection:
            self._executemany_chunked(connection, sql, rows, len(expected))
            connection.commit()

    def load_quote_snapshots(self, session_date: pd.Timestamp | None = None, symbols: Iterable[str] | None = None) -> pd.DataFrame:
        query = "SELECT symbol, ts, price, cumulative_volume, payload_json FROM quote_snapshots"
        clauses: list[str] = []
        params: list[str] = []
        if session_date is not None:
            start = pd.Timestamp(session_date).tz_localize("America/New_York").tz_convert("UTC")
            end = start + pd.Timedelta(days=1)
            clauses.append("ts >= ? AND ts < ?")
            params.extend([start.isoformat(), end.isoformat()])
        if symbols:
            symbol_list = tuple(sorted(set(symbols)))
            placeholders = ",".join(["?"] * len(symbol_list))
            clauses.append(f"symbol IN ({placeholders})")
            params.extend(symbol_list)
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY ts ASC, symbol ASC"
        with self._connect() as connection:
            frame = pd.read_sql_query(query, connection, params=tuple(params))
        if frame.empty:
            return frame
        frame["ts"] = pd.to_datetime(frame["ts"], utc=True, errors="coerce")
        return frame

    def upsert_live_order(self, row: dict) -> None:
        work = pd.DataFrame([row]).copy()
        for column in ["placed_at", "updated_at"]:
            if column not in work.columns:
                work[column] = pd.Timestamp.utcnow().isoformat()
        if "payload_json" not in work.columns:
            work["payload_json"] = work.apply(lambda item: json.dumps(item.to_dict(), ensure_ascii=True, default=str), axis=1)
        columns = [
            "client_order_id",
            "session_date",
            "symbol",
            "side",
            "quantity",
            "filled_quantity",
            "requested_price",
            "status",
            "broker_order_id",
            "placed_at",
            "updated_at",
            "payload_json",
        ]
        for column in columns:
            if column not in work.columns:
                work[column] = None
        sql = (
            "INSERT OR REPLACE INTO live_orders "
            "(client_order_id, session_date, symbol, side, quantity, filled_quantity, requested_price, status, broker_order_id, placed_at, updated_at, payload_json) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
        )
        rows = [tuple(item[column] for column in columns) for item in work[columns].to_dict(orient="records")]
        with self._connect() as connection:
            connection.executemany(sql, rows)
            connection.commit()

    def load_live_orders(self, session_date: pd.Timestamp | None = None) -> pd.DataFrame:
        query = "SELECT * FROM live_orders"
        params: tuple = ()
        if session_date is not None:
            query += " WHERE session_date = ?"
            params = (str(pd.Timestamp(session_date).date()),)
        query += " ORDER BY placed_at ASC"
        with self._connect() as connection:
            return pd.read_sql_query(query, connection, params=params)

    def replace_open_positions(self, frame: pd.DataFrame) -> None:
        expected = ["symbol", "session_date", "quantity", "avg_price", "entry_time", "broker_order_id", "status", "payload_json", "updated_at"]
        work = frame.copy()
        for column in expected:
            if column not in work.columns:
                work[column] = None
        work = work[expected].copy()
        with self._connect() as connection:
            connection.execute("DELETE FROM open_positions")
            if not work.empty:
                self._append_frame_chunked(connection, "open_positions", work)

    def load_open_positions(self) -> pd.DataFrame:
        with self._connect() as connection:
            return pd.read_sql_query("SELECT * FROM open_positions ORDER BY symbol ASC", connection)

    def put_system_state(self, state_key: str, state_value: str) -> None:
        row = pd.DataFrame(
            [
                {
                    "state_key": state_key,
                    "state_value": state_value,
                    "updated_at": pd.Timestamp.utcnow().isoformat(),
                }
            ]
        )
        with self._connect() as connection:
            connection.execute("DELETE FROM system_state WHERE state_key = ?", (state_key,))
            self._append_frame_chunked(connection, "system_state", row)

    def get_system_state(self, state_key: str) -> str | None:
        with self._connect() as connection:
            row = connection.execute("SELECT state_value FROM system_state WHERE state_key = ?", (state_key,)).fetchone()
        return row[0] if row else None

    def write_heartbeat(self, component: str, status: str, payload: dict | None = None) -> None:
        row = pd.DataFrame(
            [
                {
                    "component": component,
                    "heartbeat_at": pd.Timestamp.utcnow().isoformat(),
                    "status": status,
                    "payload_json": json.dumps(payload or {}, ensure_ascii=True, default=str),
                }
            ]
        )
        with self._connect() as connection:
            connection.execute("DELETE FROM heartbeats WHERE component = ?", (component,))
            self._append_frame_chunked(connection, "heartbeats", row)

    def append_alert(self, *, level: str, component: str, message: str, payload: dict | None = None) -> None:
        row = pd.DataFrame(
            [
                {
                    "alert_id": json.dumps([component, message, pd.Timestamp.utcnow().isoformat()]),
                    "created_at": pd.Timestamp.utcnow().isoformat(),
                    "level": level,
                    "component": component,
                    "message": message,
                    "payload_json": json.dumps(payload or {}, ensure_ascii=True, default=str),
                }
            ]
        )
        with self._connect() as connection:
            self._append_frame_chunked(connection, "alerts", row)

    def load_heartbeats(self) -> pd.DataFrame:
        with self._connect() as connection:
            return pd.read_sql_query("SELECT * FROM heartbeats ORDER BY heartbeat_at DESC", connection)

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
