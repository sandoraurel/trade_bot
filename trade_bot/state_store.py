from __future__ import annotations

import datetime as dt
import json
import os
import sqlite3
from dataclasses import is_dataclass
from typing import Any, Dict

from .events import BotEvent


def _json_safe(value: Any, *, depth: int = 0, max_depth: int = 8, seen: set[int] | None = None) -> Any:
    seen = set(seen or set())
    if depth >= max_depth:
        return str(type(value).__name__)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dt.datetime):
        return value.isoformat()
    if isinstance(value, dt.date):
        return value.isoformat()
    if isinstance(value, (dict, list, tuple, set)) or is_dataclass(value) or hasattr(value, "__dict__"):
        marker = id(value)
        if marker in seen:
            return "<recursive-ref>"
        seen.add(marker)
    if isinstance(value, dict):
        return {str(key): _json_safe(item, depth=depth + 1, max_depth=max_depth, seen=seen) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item, depth=depth + 1, max_depth=max_depth, seen=seen) for item in value]
    if is_dataclass(value):
        return _json_safe(getattr(value, "__dict__", str(value)), depth=depth + 1, max_depth=max_depth, seen=seen)
    if hasattr(value, "__dict__"):
        return _json_safe(vars(value), depth=depth + 1, max_depth=max_depth, seen=seen)
    return str(value)


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

                CREATE TABLE IF NOT EXISTS trade_learning_models (
                    model_key TEXT PRIMARY KEY,
                    updated_at TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS trade_learning_decisions (
                    decision_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    status TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS learning_store_imports (
                    source_id TEXT PRIMARY KEY,
                    imported_at TEXT NOT NULL,
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
                    json.dumps(_json_safe(event.payload), ensure_ascii=True, default=str),
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
                    json.dumps(_json_safe(snapshot), ensure_ascii=True, default=str),
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
                    json.dumps(_json_safe(payload), ensure_ascii=True, default=str),
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
                    json.dumps(_json_safe(outcome), ensure_ascii=True, default=str),
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
                (key, json.dumps(_json_safe(value), ensure_ascii=True, default=str)),
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

    def load_recent_learning_observations(self, limit: int = 50) -> list[Dict[str, Any]]:
        conn = self._connect()
        try:
            rows = conn.execute(
                """
                SELECT payload_json
                FROM trade_learning_observations
                ORDER BY observed_at DESC, id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        finally:
            conn.close()
        return [json.loads(row["payload_json"]) for row in rows]

    def load_top_learning_patterns(self, limit: int = 20) -> list[Dict[str, Any]]:
        conn = self._connect()
        try:
            rows = conn.execute(
                """
                SELECT pattern_key, payload_json
                FROM trade_learning_patterns
                ORDER BY updated_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        finally:
            conn.close()
        results: list[Dict[str, Any]] = []
        for row in rows:
            payload = json.loads(row["payload_json"])
            payload["pattern_key"] = row["pattern_key"]
            results.append(payload)
        return results

    def upsert_learning_model(self, model_key: str, payload: Dict[str, Any]) -> None:
        conn = self._connect()
        try:
            conn.execute(
                """
                INSERT INTO trade_learning_models (model_key, updated_at, payload_json)
                VALUES (?, ?, ?)
                ON CONFLICT(model_key) DO UPDATE SET
                    updated_at=excluded.updated_at,
                    payload_json=excluded.payload_json
                """,
                (
                    model_key,
                    str(payload.get("updated_at")),
                    json.dumps(payload, ensure_ascii=True, default=str),
                ),
            )
            conn.commit()
        finally:
            conn.close()

    def load_learning_model(self, model_key: str) -> Dict[str, Any]:
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT payload_json FROM trade_learning_models WHERE model_key = ?",
                (model_key,),
            ).fetchone()
        finally:
            conn.close()
        if not row:
            return {}
        return json.loads(row["payload_json"])

    def list_learning_models(self, *, prefix: str, limit: int = 50) -> list[tuple[str, Dict[str, Any]]]:
        conn = self._connect()
        try:
            rows = conn.execute(
                """
                SELECT model_key, payload_json
                FROM trade_learning_models
                WHERE model_key LIKE ?
                ORDER BY updated_at DESC
                LIMIT ?
                """,
                (f"{prefix}%", limit),
            ).fetchall()
        finally:
            conn.close()
        return [(row["model_key"], json.loads(row["payload_json"])) for row in rows]

    def append_learning_decision(self, decision_id: str, created_at: str, status: str, payload: Dict[str, Any]) -> None:
        conn = self._connect()
        try:
            conn.execute(
                """
                INSERT INTO trade_learning_decisions (decision_id, created_at, status, payload_json)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(decision_id) DO UPDATE SET
                    status=excluded.status,
                    payload_json=excluded.payload_json
                """,
                (
                    decision_id,
                    created_at,
                    status,
                    json.dumps(payload, ensure_ascii=True, default=str),
                ),
            )
            conn.commit()
        finally:
            conn.close()

    def load_learning_decision(self, decision_id: str) -> Dict[str, Any]:
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT payload_json FROM trade_learning_decisions WHERE decision_id = ?",
                (decision_id,),
            ).fetchone()
        finally:
            conn.close()
        if not row:
            return {}
        return json.loads(row["payload_json"])

    def load_pending_learning_decisions(self, limit: int = 200) -> list[Dict[str, Any]]:
        conn = self._connect()
        try:
            rows = conn.execute(
                """
                SELECT decision_id, payload_json
                FROM trade_learning_decisions
                WHERE status = 'pending'
                ORDER BY created_at ASC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        finally:
            conn.close()
        results: list[Dict[str, Any]] = []
        for row in rows:
            payload = json.loads(row["payload_json"])
            payload["decision_id"] = row["decision_id"]
            results.append(payload)
        return results

    def load_learning_inventory(self) -> Dict[str, Any]:
        conn = self._connect()
        try:
            observations = int(conn.execute("SELECT COUNT(*) FROM trade_learning_observations").fetchone()[0])
            patterns = int(conn.execute("SELECT COUNT(*) FROM trade_learning_patterns").fetchone()[0])
            models = int(conn.execute("SELECT COUNT(*) FROM trade_learning_models").fetchone()[0])
            decisions = int(conn.execute("SELECT COUNT(*) FROM trade_learning_decisions").fetchone()[0])
            imports = int(conn.execute("SELECT COUNT(*) FROM learning_store_imports").fetchone()[0])
        finally:
            conn.close()
        return {
            "path": self.path,
            "observations": observations,
            "patterns": patterns,
            "models": models,
            "pending_decisions": decisions,
            "imports": imports,
        }

    def has_learning_import(self, source_id: str) -> bool:
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT 1 FROM learning_store_imports WHERE source_id = ?",
                (source_id,),
            ).fetchone()
        finally:
            conn.close()
        return row is not None

    def record_learning_import(self, source_id: str, payload: Dict[str, Any]) -> None:
        conn = self._connect()
        try:
            conn.execute(
                """
                INSERT INTO learning_store_imports (source_id, imported_at, payload_json)
                VALUES (?, ?, ?)
                ON CONFLICT(source_id) DO UPDATE SET
                    imported_at=excluded.imported_at,
                    payload_json=excluded.payload_json
                """,
                (
                    source_id,
                    str(payload.get("imported_at")),
                    json.dumps(payload, ensure_ascii=True, default=str),
                ),
            )
            conn.commit()
        finally:
            conn.close()


class PersistentLearningStore:
    """
    Durable learning memory backed by a global SQLite store.

    - Pattern/model/observation memory is shared across simulations and live runs.
    - Pending shadow decisions remain local to the current run/runtime store.
    """

    def __init__(self, local_store: SQLiteStateStore | None, global_store: SQLiteStateStore | None):
        self.local_store = local_store
        self.global_store = global_store or local_store

    def _learning_store(self) -> SQLiteStateStore | None:
        return self.global_store or self.local_store

    def append_learning_observation(self, observation: Dict[str, Any]) -> None:
        store = self._learning_store()
        if store is not None:
            store.append_learning_observation(observation)

    def upsert_learning_pattern(self, pattern_key: str, payload: Dict[str, Any]) -> None:
        store = self._learning_store()
        if store is not None:
            store.upsert_learning_pattern(pattern_key, payload)

    def load_learning_pattern(self, pattern_key: str) -> Dict[str, Any]:
        store = self._learning_store()
        return store.load_learning_pattern(pattern_key) if store is not None else {}

    def load_learning_patterns(self, pattern_keys: Dict[str, float]) -> list[Dict[str, Any]]:
        store = self._learning_store()
        return store.load_learning_patterns(pattern_keys) if store is not None else []

    def load_recent_learning_observations(self, limit: int = 50) -> list[Dict[str, Any]]:
        store = self._learning_store()
        return store.load_recent_learning_observations(limit=limit) if store is not None else []

    def load_top_learning_patterns(self, limit: int = 20) -> list[Dict[str, Any]]:
        store = self._learning_store()
        return store.load_top_learning_patterns(limit=limit) if store is not None else []

    def upsert_learning_model(self, model_key: str, payload: Dict[str, Any]) -> None:
        store = self._learning_store()
        if store is not None:
            store.upsert_learning_model(model_key, payload)

    def load_learning_model(self, model_key: str) -> Dict[str, Any]:
        store = self._learning_store()
        return store.load_learning_model(model_key) if store is not None else {}

    def list_learning_models(self, *, prefix: str, limit: int = 50) -> list[tuple[str, Dict[str, Any]]]:
        store = self._learning_store()
        return store.list_learning_models(prefix=prefix, limit=limit) if store is not None else []

    def append_learning_decision(self, decision_id: str, created_at: str, status: str, payload: Dict[str, Any]) -> None:
        store = self.local_store or self._learning_store()
        if store is not None:
            store.append_learning_decision(decision_id, created_at, status, payload)

    def load_learning_decision(self, decision_id: str) -> Dict[str, Any]:
        store = self.local_store or self._learning_store()
        return store.load_learning_decision(decision_id) if store is not None else {}

    def load_pending_learning_decisions(self, limit: int = 200) -> list[Dict[str, Any]]:
        store = self.local_store or self._learning_store()
        return store.load_pending_learning_decisions(limit=limit) if store is not None else []

    def describe(self) -> Dict[str, Any]:
        local_inventory = self.local_store.load_learning_inventory() if self.local_store is not None else {}
        global_inventory = self.global_store.load_learning_inventory() if self.global_store is not None else {}
        return {
            "local": local_inventory,
            "global": global_inventory,
            "shared_learning_path": str(global_inventory.get("path") or local_inventory.get("path") or ""),
        }


def backfill_learning_from_sqlite_artifacts(base_dir: str, target_store: SQLiteStateStore) -> Dict[str, Any]:
    data_dir = os.path.join(os.path.abspath(base_dir), "data")
    target_path = os.path.abspath(target_store.path)
    summary = {
        "scanned_files": 0,
        "imported_files": 0,
        "imported_observations": 0,
        "imported_patterns": 0,
        "imported_models": 0,
        "target_path": target_path,
    }
    if not os.path.isdir(data_dir):
        return summary

    for root, _, files in os.walk(data_dir):
        for name in sorted(files):
            if not name.endswith(".sqlite3"):
                continue
            source_path = os.path.abspath(os.path.join(root, name))
            if source_path == target_path:
                continue
            summary["scanned_files"] += 1
            source_id = source_path
            if target_store.has_learning_import(source_id):
                continue
            source_store = SQLiteStateStore(source_path)
            imported = _merge_learning_store(source_store, target_store)
            imported["source_path"] = source_path
            imported["imported_at"] = imported.get("imported_at") or ""
            target_store.record_learning_import(source_id, imported)
            summary["imported_files"] += 1
            summary["imported_observations"] += int(imported.get("observations", 0))
            summary["imported_patterns"] += int(imported.get("patterns", 0))
            summary["imported_models"] += int(imported.get("models", 0))
    return summary


def _merge_learning_store(source_store: SQLiteStateStore, target_store: SQLiteStateStore) -> Dict[str, Any]:
    conn = source_store._connect()
    try:
        observation_rows = conn.execute(
            """
            SELECT payload_json
            FROM trade_learning_observations
            ORDER BY observed_at ASC, id ASC
            """
        ).fetchall()
        pattern_rows = conn.execute(
            """
            SELECT pattern_key, payload_json
            FROM trade_learning_patterns
            """
        ).fetchall()
        model_rows = conn.execute(
            """
            SELECT model_key, payload_json
            FROM trade_learning_models
            """
        ).fetchall()
    finally:
        conn.close()

    for row in observation_rows:
        target_store.append_learning_observation(json.loads(row["payload_json"]))
    for row in pattern_rows:
        target_store.upsert_learning_pattern(row["pattern_key"], json.loads(row["payload_json"]))
    for row in model_rows:
        target_store.upsert_learning_model(row["model_key"], json.loads(row["payload_json"]))
    return {
        "imported_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "observations": len(observation_rows),
        "patterns": len(pattern_rows),
        "models": len(model_rows),
    }
