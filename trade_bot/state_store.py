from __future__ import annotations

import json
import os
import sqlite3
from typing import Any, Dict

from .events import BotEvent


class SQLiteStateStore:
    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        conn = self._connect()
        try:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS bot_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    trace_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS bot_snapshots (
                    snapshot_key TEXT PRIMARY KEY,
                    updated_at TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS research_events (
                    event_id TEXT PRIMARY KEY,
                    source TEXT NOT NULL,
                    category TEXT NOT NULL,
                    symbol TEXT,
                    status TEXT NOT NULL,
                    published_at TEXT,
                    payload_json TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS research_event_outcomes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_id TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    evaluated_at TEXT NOT NULL,
                    outcome_json TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS operational_metrics (
                    metric_key TEXT PRIMARY KEY,
                    metric_value TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS trade_learning_observations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    observed_at TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    strategy TEXT NOT NULL,
                    side TEXT NOT NULL,
                    success INTEGER NOT NULL,
                    r_multiple REAL NOT NULL,
                    payload_json TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS trade_learning_patterns (
                    pattern_key TEXT PRIMARY KEY,
                    updated_at TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                );
                """
            )
        finally:
            conn.close()

    def append_event(self, event: BotEvent) -> None:
        conn = self._connect()
        try:
            conn.execute(
                """
                INSERT INTO bot_events (trace_id, event_type, created_at, payload_json)
                VALUES (?, ?, ?, ?)
                """,
                (
                    event.trace_id,
                    event.event_type,
                    event.created_at,
                    json.dumps(event.payload, ensure_ascii=True, default=str),
                ),
            )
            conn.commit()
        finally:
            conn.close()

    def persist_snapshot(self, snapshot: Dict[str, Any]) -> None:
        snapshot_key = str(snapshot.get("snapshot_key", "runtime"))
        updated_at = str(snapshot.get("updated_at"))
        conn = self._connect()
        try:
            conn.execute(
                """
                INSERT INTO bot_snapshots (snapshot_key, updated_at, payload_json)
                VALUES (?, ?, ?)
                ON CONFLICT(snapshot_key) DO UPDATE SET
                    updated_at=excluded.updated_at,
                    payload_json=excluded.payload_json
                """,
                (
                    snapshot_key,
                    updated_at,
                    json.dumps(snapshot, ensure_ascii=True, default=str),
                ),
            )
            conn.commit()
        finally:
            conn.close()

    def load_snapshot(self, snapshot_key: str = "runtime") -> Dict[str, Any]:
        if not os.path.exists(self.path):
            return {}
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT payload_json FROM bot_snapshots WHERE snapshot_key = ?",
                (snapshot_key,),
            ).fetchone()
        finally:
            conn.close()
        if not row:
            return {}
        return json.loads(row["payload_json"])

    def upsert_research_event(
        self,
        *,
        event_id: str,
        source: str,
        category: str,
        symbol: str | None,
        status: str,
        published_at: str | None,
        payload: Dict[str, Any],
    ) -> None:
        conn = self._connect()
        try:
            conn.execute(
                """
                INSERT INTO research_events (event_id, source, category, symbol, status, published_at, payload_json)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(event_id) DO UPDATE SET
                    source=excluded.source,
                    category=excluded.category,
                    symbol=excluded.symbol,
                    status=excluded.status,
                    published_at=excluded.published_at,
                    payload_json=excluded.payload_json
                """,
                (
                    event_id,
                    source,
                    category,
                    symbol,
                    status,
                    published_at,
                    json.dumps(payload, ensure_ascii=True, default=str),
                ),
            )
            conn.commit()
        finally:
            conn.close()

    def append_research_outcome(self, *, event_id: str, symbol: str, evaluated_at: str, outcome: Dict[str, Any]) -> None:
        conn = self._connect()
        try:
            conn.execute(
                """
                INSERT INTO research_event_outcomes (event_id, symbol, evaluated_at, outcome_json)
                VALUES (?, ?, ?, ?)
                """,
                (
                    event_id,
                    symbol,
                    evaluated_at,
                    json.dumps(outcome, ensure_ascii=True, default=str),
                ),
            )
            conn.commit()
        finally:
            conn.close()

    def load_recent_research_events(self, limit: int = 20) -> list[Dict[str, Any]]:
        conn = self._connect()
        try:
            rows = conn.execute(
                """
                SELECT event_id, source, category, symbol, status, published_at, payload_json
                FROM research_events
                ORDER BY COALESCE(published_at, '') DESC, event_id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        finally:
            conn.close()
        results: list[Dict[str, Any]] = []
        for row in rows:
            payload = json.loads(row["payload_json"])
            results.append(
                {
                    "event_id": row["event_id"],
                    "source": row["source"],
                    "category": row["category"],
                    "symbol": row["symbol"],
                    "status": row["status"],
                    "published_at": row["published_at"],
                    "payload": payload,
                }
            )
        return results

    def set_operational_metric(self, key: str, value: Any) -> None:
        conn = self._connect()
        try:
            conn.execute(
                """
                INSERT INTO operational_metrics (metric_key, metric_value)
                VALUES (?, ?)
                ON CONFLICT(metric_key) DO UPDATE SET
                    metric_value=excluded.metric_value
                """,
                (key, json.dumps(value, ensure_ascii=True, default=str)),
            )
            conn.commit()
        finally:
            conn.close()

    def increment_operational_metric(self, key: str, amount: float = 1.0) -> None:
        metrics = self.load_operational_metrics()
        current = float(metrics.get(key, 0.0))
        self.set_operational_metric(key, current + amount)

    def load_operational_metrics(self) -> Dict[str, Any]:
        conn = self._connect()
        try:
            rows = conn.execute("SELECT metric_key, metric_value FROM operational_metrics").fetchall()
        finally:
            conn.close()
        result: Dict[str, Any] = {}
        for row in rows:
            result[row["metric_key"]] = json.loads(row["metric_value"])
        return result

    def append_learning_observation(self, observation: Dict[str, Any]) -> None:
        conn = self._connect()
        try:
            conn.execute(
                """
                INSERT INTO trade_learning_observations (observed_at, symbol, strategy, side, success, r_multiple, payload_json)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(observation.get("observed_at")),
                    str(observation.get("symbol")),
                    str(observation.get("strategy")),
                    str(observation.get("side")),
                    1 if bool(observation.get("success")) else 0,
                    float(observation.get("r_multiple", 0.0) or 0.0),
                    json.dumps(observation, ensure_ascii=True, default=str),
                ),
            )
            conn.commit()
        finally:
            conn.close()

    def upsert_learning_pattern(self, pattern_key: str, payload: Dict[str, Any]) -> None:
        conn = self._connect()
        try:
            conn.execute(
                """
                INSERT INTO trade_learning_patterns (pattern_key, updated_at, payload_json)
                VALUES (?, ?, ?)
                ON CONFLICT(pattern_key) DO UPDATE SET
                    updated_at=excluded.updated_at,
                    payload_json=excluded.payload_json
                """,
                (
                    pattern_key,
                    str(payload.get("updated_at")),
                    json.dumps(payload, ensure_ascii=True, default=str),
                ),
            )
            conn.commit()
        finally:
            conn.close()

    def load_learning_pattern(self, pattern_key: str) -> Dict[str, Any]:
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT payload_json FROM trade_learning_patterns WHERE pattern_key = ?",
                (pattern_key,),
            ).fetchone()
        finally:
            conn.close()
        if not row:
            return {}
        return json.loads(row["payload_json"])

    def load_learning_patterns(self, pattern_keys: Dict[str, float]) -> list[Dict[str, Any]]:
        if not pattern_keys:
            return []
        conn = self._connect()
        try:
            rows = conn.execute(
                f"SELECT pattern_key, payload_json FROM trade_learning_patterns WHERE pattern_key IN ({','.join('?' for _ in pattern_keys)})",
                tuple(pattern_keys.keys()),
            ).fetchall()
        finally:
            conn.close()
        results: list[Dict[str, Any]] = []
        for row in rows:
            payload = json.loads(row["payload_json"])
            payload["pattern_key"] = row["pattern_key"]
            payload["weight"] = float(pattern_keys.get(row["pattern_key"], payload.get("weight", 0.0)) or 0.0)
            results.append(payload)
        return results
