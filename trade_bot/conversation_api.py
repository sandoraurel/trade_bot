from __future__ import annotations

import base64
import datetime as dt
import hashlib
import hmac
import json
import os
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib.parse import parse_qs
from wsgiref.simple_server import make_server

from trade_bot.main import BotConfig, TradeBot


INSECURE_DEFAULTS = {
    "CONVERSATION_API_JWT_SECRET": "change-me-in-production",
    "CONVERSATION_API_DEFAULT_TENANT_ID": "default-tenant",
    "CONVERSATION_API_DEFAULT_CLIENT_ID": "local-dev-client",
    "CONVERSATION_API_DEFAULT_CLIENT_SECRET": "local-dev-secret",
}


def _require_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    if value == INSECURE_DEFAULTS.get(name):
        raise RuntimeError(f"Refusing to start with insecure default for {name}")
    return value


def _load_bootstrap_settings() -> Dict[str, str]:
    return {
        "jwt_secret": _require_env("CONVERSATION_API_JWT_SECRET"),
        "tenant_id": _require_env("CONVERSATION_API_DEFAULT_TENANT_ID"),
        "client_id": _require_env("CONVERSATION_API_DEFAULT_CLIENT_ID"),
        "client_secret": _require_env("CONVERSATION_API_DEFAULT_CLIENT_SECRET"),
    }


def _b64url_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _b64url_decode(data: str) -> bytes:
    padding = "=" * (-len(data) % 4)
    return base64.urlsafe_b64decode(data + padding)


class JWTManager:
    def __init__(self, secret: str, issuer: str = "trade-bot-conversation-api") -> None:
        self.secret = secret.encode("utf-8")
        self.issuer = issuer

    def issue_token(self, subject: str, tenant_id: str, expires_in_seconds: int = 3600) -> str:
        now = int(time.time())
        header = {"alg": "HS256", "typ": "JWT"}
        payload = {
            "sub": subject,
            "tenant_id": tenant_id,
            "iss": self.issuer,
            "iat": now,
            "exp": now + expires_in_seconds,
            "jti": str(uuid.uuid4()),
        }
        header_part = _b64url_encode(json.dumps(header, separators=(",", ":"), sort_keys=True).encode("utf-8"))
        payload_part = _b64url_encode(json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8"))
        signature = hmac.new(
            self.secret,
            f"{header_part}.{payload_part}".encode("ascii"),
            hashlib.sha256,
        ).digest()
        return f"{header_part}.{payload_part}.{_b64url_encode(signature)}"

    def verify_token(self, token: str) -> Dict[str, Any]:
        try:
            header_part, payload_part, signature_part = token.split(".")
        except ValueError as exc:
            raise PermissionError("Malformed token") from exc

        expected = hmac.new(
            self.secret,
            f"{header_part}.{payload_part}".encode("ascii"),
            hashlib.sha256,
        ).digest()
        if not hmac.compare_digest(expected, _b64url_decode(signature_part)):
            raise PermissionError("Invalid token signature")

        payload = json.loads(_b64url_decode(payload_part).decode("utf-8"))
        now = int(time.time())
        if payload.get("iss") != self.issuer:
            raise PermissionError("Invalid token issuer")
        if now >= int(payload.get("exp", 0)):
            raise PermissionError("Token expired")
        return payload


class Redactor:
    SENSITIVE_KEYS = {
        "authorization",
        "api_key",
        "api_secret",
        "password",
        "token",
        "secret",
        "access_token",
        "refresh_token",
    }

    @classmethod
    def redact(cls, value: Any, key: str = "") -> Any:
        normalized_key = key.lower()
        if normalized_key in cls.SENSITIVE_KEYS:
            return "[REDACTED]"

        if isinstance(value, dict):
            return {k: cls.redact(v, k) for k, v in value.items()}

        if isinstance(value, list):
            return [cls.redact(item) for item in value]

        if isinstance(value, str):
            if normalized_key in cls.SENSITIVE_KEYS:
                return "[REDACTED]"
            lowered = value.lower()
            if lowered.startswith("bearer "):
                return "Bearer [REDACTED]"
            if len(value) > 80 and any(marker in lowered for marker in ("sk-", "token", "secret")):
                return "[REDACTED]"
        return value


class SQLiteStore:
    def __init__(self, path: str) -> None:
        self.path = path
        self._lock = threading.Lock()
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        conn = self._connect()
        try:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS tenants (
                    tenant_id TEXT PRIMARY KEY,
                    tenant_name TEXT NOT NULL,
                    client_id TEXT NOT NULL UNIQUE,
                    client_secret_hash TEXT NOT NULL,
                    rpm_limit INTEGER NOT NULL,
                    tpm_limit INTEGER NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    tenant_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS messages (
                    message_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    tenant_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS request_logs (
                    request_id TEXT PRIMARY KEY,
                    tenant_id TEXT,
                    session_id TEXT,
                    path TEXT NOT NULL,
                    method TEXT NOT NULL,
                    request_json TEXT NOT NULL,
                    response_json TEXT NOT NULL,
                    status_code INTEGER NOT NULL,
                    token_count INTEGER NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS rate_limits (
                    tenant_id TEXT NOT NULL,
                    minute_bucket TEXT NOT NULL,
                    request_count INTEGER NOT NULL,
                    token_count INTEGER NOT NULL,
                    PRIMARY KEY (tenant_id, minute_bucket)
                );
                """
            )
        finally:
            conn.close()

    @staticmethod
    def _now() -> str:
        return dt.datetime.utcnow().isoformat()

    @staticmethod
    def _hash_secret(secret: str) -> str:
        return hashlib.sha256(secret.encode("utf-8")).hexdigest()

    def upsert_tenant(
        self,
        tenant_id: str,
        tenant_name: str,
        client_id: str,
        client_secret: str,
        rpm_limit: int = 60,
        tpm_limit: int = 12000,
    ) -> None:
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    """
                    INSERT INTO tenants (tenant_id, tenant_name, client_id, client_secret_hash, rpm_limit, tpm_limit, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(tenant_id) DO UPDATE SET
                        tenant_name=excluded.tenant_name,
                        client_id=excluded.client_id,
                        client_secret_hash=excluded.client_secret_hash,
                        rpm_limit=excluded.rpm_limit,
                        tpm_limit=excluded.tpm_limit
                    """,
                    (
                        tenant_id,
                        tenant_name,
                        client_id,
                        self._hash_secret(client_secret),
                        rpm_limit,
                        tpm_limit,
                        self._now(),
                    ),
                )
                conn.commit()
            finally:
                conn.close()

    def authenticate_client(self, client_id: str, client_secret: str) -> Optional[sqlite3.Row]:
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT * FROM tenants WHERE client_id = ?",
                (client_id,),
            ).fetchone()
        finally:
            conn.close()
        if not row:
            return None
        if row["client_secret_hash"] != self._hash_secret(client_secret):
            return None
        return row

    def get_tenant(self, tenant_id: str) -> Optional[sqlite3.Row]:
        conn = self._connect()
        try:
            return conn.execute("SELECT * FROM tenants WHERE tenant_id = ?", (tenant_id,)).fetchone()
        finally:
            conn.close()

    def create_or_touch_session(self, tenant_id: str, session_id: str, user_id: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        now = self._now()
        payload = json.dumps(metadata, ensure_ascii=True, sort_keys=True)
        with self._lock:
            conn = self._connect()
            try:
                existing = conn.execute(
                    "SELECT session_id FROM sessions WHERE session_id = ? AND tenant_id = ?",
                    (session_id, tenant_id),
                ).fetchone()
                if existing:
                    conn.execute(
                        "UPDATE sessions SET user_id = ?, metadata_json = ?, updated_at = ? WHERE session_id = ?",
                        (user_id, payload, now, session_id),
                    )
                else:
                    conn.execute(
                        """
                        INSERT INTO sessions (session_id, tenant_id, user_id, metadata_json, created_at, updated_at)
                        VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        (session_id, tenant_id, user_id, payload, now, now),
                    )
                conn.commit()
            finally:
                conn.close()
        return {"session_id": session_id, "tenant_id": tenant_id, "user_id": user_id, "metadata": metadata}

    def append_message(self, tenant_id: str, session_id: str, role: str, content: str) -> None:
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    """
                    INSERT INTO messages (message_id, session_id, tenant_id, role, content, created_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (str(uuid.uuid4()), session_id, tenant_id, role, content, self._now()),
                )
                conn.execute(
                    "UPDATE sessions SET updated_at = ? WHERE session_id = ?",
                    (self._now(), session_id),
                )
                conn.commit()
            finally:
                conn.close()

    def get_session_messages(self, tenant_id: str, session_id: str, limit: int = 20) -> List[Dict[str, Any]]:
        conn = self._connect()
        try:
            rows = conn.execute(
                """
                SELECT role, content, created_at
                FROM messages
                WHERE tenant_id = ? AND session_id = ?
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (tenant_id, session_id, limit),
            ).fetchall()
        finally:
            conn.close()
        messages = [dict(row) for row in reversed(rows)]
        return messages

    def log_request(
        self,
        request_id: str,
        tenant_id: Optional[str],
        session_id: Optional[str],
        path: str,
        method: str,
        request_payload: Dict[str, Any],
        response_payload: Dict[str, Any],
        status_code: int,
        token_count: int,
    ) -> None:
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    """
                    INSERT INTO request_logs
                    (request_id, tenant_id, session_id, path, method, request_json, response_json, status_code, token_count, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        request_id,
                        tenant_id,
                        session_id,
                        path,
                        method,
                        json.dumps(Redactor.redact(request_payload), ensure_ascii=True, sort_keys=True),
                        json.dumps(Redactor.redact(response_payload), ensure_ascii=True, sort_keys=True),
                        status_code,
                        token_count,
                        self._now(),
                    ),
                )
                conn.commit()
            finally:
                conn.close()

    def enforce_rate_limit(self, tenant_id: str, rpm_limit: int, tpm_limit: int, token_count: int) -> Tuple[bool, Dict[str, int]]:
        minute_bucket = dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M")
        with self._lock:
            conn = self._connect()
            try:
                row = conn.execute(
                    """
                    SELECT request_count, token_count
                    FROM rate_limits
                    WHERE tenant_id = ? AND minute_bucket = ?
                    """,
                    (tenant_id, minute_bucket),
                ).fetchone()

                current_requests = int(row["request_count"]) if row else 0
                current_tokens = int(row["token_count"]) if row else 0
                projected_requests = current_requests + 1
                projected_tokens = current_tokens + token_count
                allowed = projected_requests <= rpm_limit and projected_tokens <= tpm_limit

                if allowed:
                    if row:
                        conn.execute(
                            """
                            UPDATE rate_limits
                            SET request_count = ?, token_count = ?
                            WHERE tenant_id = ? AND minute_bucket = ?
                            """,
                            (projected_requests, projected_tokens, tenant_id, minute_bucket),
                        )
                    else:
                        conn.execute(
                            """
                            INSERT INTO rate_limits (tenant_id, minute_bucket, request_count, token_count)
                            VALUES (?, ?, ?, ?)
                            """,
                            (tenant_id, minute_bucket, projected_requests, projected_tokens),
                        )
                    conn.commit()
            finally:
                conn.close()

        return allowed, {
            "minute_bucket": minute_bucket,
            "request_count": projected_requests,
            "token_count": projected_tokens,
            "rpm_limit": rpm_limit,
            "tpm_limit": tpm_limit,
        }


@dataclass
class ConversationAuthContext:
    tenant_id: str
    subject: str
    token_payload: Dict[str, Any]


class ConversationService:
    def __init__(self, bot: TradeBot, store: SQLiteStore, jwt_manager: JWTManager):
        self.bot = bot
        self.store = store
        self.jwt_manager = jwt_manager

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        return max(1, len(text.split()) + (len(text) // 16))

    def authenticate_bearer(self, authorization_header: str, tenant_header: Optional[str]) -> ConversationAuthContext:
        if not authorization_header or not authorization_header.lower().startswith("bearer "):
            raise PermissionError("Missing bearer token")
        token = authorization_header.split(" ", 1)[1]
        payload = self.jwt_manager.verify_token(token)
        tenant_id = payload["tenant_id"]
        if tenant_header and tenant_header != tenant_id:
            raise PermissionError("Tenant mismatch")
        return ConversationAuthContext(
            tenant_id=tenant_id,
            subject=payload["sub"],
            token_payload=payload,
        )

    def issue_access_token(self, client_id: str, client_secret: str) -> Dict[str, Any]:
        tenant = self.store.authenticate_client(client_id, client_secret)
        if not tenant:
            raise PermissionError("Invalid client credentials")
        token = self.jwt_manager.issue_token(
            subject=tenant["client_id"],
            tenant_id=tenant["tenant_id"],
            expires_in_seconds=3600,
        )
        return {
            "access_token": token,
            "token_type": "Bearer",
            "expires_in": 3600,
            "tenant_id": tenant["tenant_id"],
        }

    def create_session(self, tenant_id: str, user_id: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        session_id = str(uuid.uuid4())
        return self.store.create_or_touch_session(tenant_id, session_id, user_id, metadata)

    def get_history(self, tenant_id: str, session_id: str) -> List[Dict[str, Any]]:
        return self.store.get_session_messages(tenant_id, session_id)

    def stream_conversation(
        self,
        tenant_id: str,
        session_id: str,
        prompt: str,
        metadata: Dict[str, Any],
    ) -> Tuple[Iterable[bytes], Dict[str, Any], int]:
        tenant = self.store.get_tenant(tenant_id)
        if not tenant:
            raise PermissionError("Unknown tenant")

        token_estimate = self._estimate_tokens(prompt)
        allowed, limits = self.store.enforce_rate_limit(
            tenant_id=tenant_id,
            rpm_limit=int(tenant["rpm_limit"]),
            tpm_limit=int(tenant["tpm_limit"]),
            token_count=token_estimate,
        )
        if not allowed:
            raise RuntimeError(json.dumps({"error": "rate_limit_exceeded", "limits": limits}))

        self.store.append_message(tenant_id, session_id, "user", prompt)
        answer_payload = self.bot.answer_operator_question(prompt)
        answer_text = answer_payload["answer"]
        self.store.append_message(tenant_id, session_id, "assistant", answer_text)

        response_payload = {
            "session_id": session_id,
            "answer": answer_text,
            "retrieved_context": answer_payload["retrieved_context"],
            "tool_results": answer_payload["tool_results"],
            "limits": limits,
            "metadata": metadata,
        }

        def event_stream() -> Iterable[bytes]:
            yield _sse_event("session", {"session_id": session_id, "tenant_id": tenant_id})
            chunk_size = 160
            for index in range(0, len(answer_text), chunk_size):
                chunk = answer_text[index:index + chunk_size]
                yield _sse_event("message.delta", {"delta": chunk})
            yield _sse_event(
                "message.completed",
                {
                    "answer": answer_text,
                    "retrieved_context": answer_payload["retrieved_context"],
                    "tool_results": answer_payload["tool_results"],
                    "limits": limits,
                },
            )

        return event_stream(), response_payload, token_estimate


def _sse_event(event: str, payload: Dict[str, Any]) -> bytes:
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=True)}\n\n".encode("utf-8")


class ConversationAPIApp:
    def __init__(self, service: ConversationService, store: SQLiteStore):
        self.service = service
        self.store = store

    @staticmethod
    def _json_response(status: str, payload: Dict[str, Any]) -> Tuple[str, List[Tuple[str, str]], List[bytes]]:
        body = json.dumps(payload, ensure_ascii=True).encode("utf-8")
        headers = [
            ("Content-Type", "application/json"),
            ("Content-Length", str(len(body))),
        ]
        return status, headers, [body]

    @staticmethod
    def _read_json_body(environ: Dict[str, Any]) -> Dict[str, Any]:
        length = int(environ.get("CONTENT_LENGTH") or 0)
        body = environ["wsgi.input"].read(length) if length > 0 else b"{}"
        return json.loads(body.decode("utf-8") or "{}")

    @staticmethod
    def _internal_error_response() -> Dict[str, Any]:
        return {
            "error": "internal_error",
            "message": "Internal server error",
        }

    def __call__(self, environ: Dict[str, Any], start_response: Any) -> Iterable[bytes]:
        path = environ.get("PATH_INFO", "")
        method = environ.get("REQUEST_METHOD", "GET").upper()
        request_id = str(uuid.uuid4())
        tenant_id: Optional[str] = None
        session_id: Optional[str] = None
        request_payload: Dict[str, Any] = {}
        response_payload: Dict[str, Any] = {}
        token_count = 0
        status_code = 500

        try:
            if path == "/healthz" and method == "GET":
                status_code = 200
                response_payload = {"status": "ok"}
                status, headers, body = self._json_response("200 OK", response_payload)
                start_response(status, headers)
                return body

            if path == "/v1/auth/token" and method == "POST":
                request_payload = self._read_json_body(environ)
                response_payload = self.service.issue_access_token(
                    client_id=request_payload.get("client_id", ""),
                    client_secret=request_payload.get("client_secret", ""),
                )
                status_code = 200
                status, headers, body = self._json_response("200 OK", response_payload)
                start_response(status, headers)
                return body

            if path == "/v1/sessions" and method == "POST":
                request_payload = self._read_json_body(environ)
                auth = self.service.authenticate_bearer(
                    environ.get("HTTP_AUTHORIZATION", ""),
                    environ.get("HTTP_X_TENANT_ID"),
                )
                tenant_id = auth.tenant_id
                response_payload = self.service.create_session(
                    tenant_id=tenant_id,
                    user_id=request_payload.get("user_id", auth.subject),
                    metadata=request_payload.get("metadata", {}),
                )
                session_id = response_payload["session_id"]
                status_code = 201
                status, headers, body = self._json_response("201 Created", response_payload)
                start_response(status, headers)
                return body

            if path == "/v1/conversations/stream" and method == "POST":
                request_payload = self._read_json_body(environ)
                auth = self.service.authenticate_bearer(
                    environ.get("HTTP_AUTHORIZATION", ""),
                    environ.get("HTTP_X_TENANT_ID"),
                )
                tenant_id = auth.tenant_id
                session_id = request_payload.get("session_id") or str(uuid.uuid4())
                self.store.create_or_touch_session(
                    tenant_id=tenant_id,
                    session_id=session_id,
                    user_id=request_payload.get("user_id", auth.subject),
                    metadata=request_payload.get("metadata", {}),
                )
                event_stream, response_payload, token_count = self.service.stream_conversation(
                    tenant_id=tenant_id,
                    session_id=session_id,
                    prompt=request_payload.get("message", ""),
                    metadata=request_payload.get("metadata", {}),
                )
                status_code = 200
                start_response(
                    "200 OK",
                    [
                        ("Content-Type", "text/event-stream"),
                        ("Cache-Control", "no-cache"),
                        ("Connection", "keep-alive"),
                        ("X-Accel-Buffering", "no"),
                    ],
                )
                return event_stream

            if path.startswith("/v1/sessions/") and path.endswith("/messages") and method == "GET":
                parts = path.split("/")
                session_id = parts[3]
                auth = self.service.authenticate_bearer(
                    environ.get("HTTP_AUTHORIZATION", ""),
                    environ.get("HTTP_X_TENANT_ID"),
                )
                tenant_id = auth.tenant_id
                response_payload = {
                    "session_id": session_id,
                    "messages": self.service.get_history(tenant_id, session_id),
                }
                status_code = 200
                status, headers, body = self._json_response("200 OK", response_payload)
                start_response(status, headers)
                return body

            status_code = 404
            response_payload = {"error": "not_found"}
            status, headers, body = self._json_response("404 Not Found", response_payload)
            start_response(status, headers)
            return body

        except PermissionError as exc:
            status_code = 401
            response_payload = {"error": "unauthorized", "message": str(exc)}
            status, headers, body = self._json_response("401 Unauthorized", response_payload)
            start_response(status, headers)
            return body
        except RuntimeError as exc:
            status_code = 429 if "rate_limit_exceeded" in str(exc) else 400
            try:
                response_payload = json.loads(str(exc))
            except json.JSONDecodeError:
                response_payload = {"error": "bad_request", "message": str(exc)}
            status, headers, body = self._json_response(
                "429 Too Many Requests" if status_code == 429 else "400 Bad Request",
                response_payload,
            )
            start_response(status, headers)
            return body
        except Exception as exc:
            status_code = 500
            response_payload = self._internal_error_response()
            status, headers, body = self._json_response("500 Internal Server Error", response_payload)
            start_response(status, headers)
            return body
        finally:
            if path != "/healthz":
                self.store.log_request(
                    request_id=request_id,
                    tenant_id=tenant_id,
                    session_id=session_id,
                    path=path,
                    method=method,
                    request_payload=request_payload,
                    response_payload=response_payload,
                    status_code=status_code,
                    token_count=token_count,
                )


def build_conversation_api(base_dir: str | None = None) -> ConversationAPIApp:
    base_dir = base_dir or os.getcwd()
    os.makedirs(os.path.join(base_dir, "data"), exist_ok=True)
    store = SQLiteStore(os.path.join(base_dir, "data", "conversation_api.sqlite3"))
    bootstrap = _load_bootstrap_settings()
    jwt_manager = JWTManager(bootstrap["jwt_secret"])

    store.upsert_tenant(
        tenant_id=bootstrap["tenant_id"],
        tenant_name="Default Tenant",
        client_id=bootstrap["client_id"],
        client_secret=bootstrap["client_secret"],
        rpm_limit=int(os.getenv("CONVERSATION_API_DEFAULT_RPM", "60")),
        tpm_limit=int(os.getenv("CONVERSATION_API_DEFAULT_TPM", "12000")),
    )

    bot = TradeBot(
        BotConfig(
            starting_balance=50.0,
            use_paper_trading=True,
            telegram_bot_token=os.getenv("TELEGRAM_BOT_TOKEN"),
            telegram_chat_id=os.getenv("TELEGRAM_CHAT_ID"),
            api_key=os.getenv("BINANCE_API_KEY"),
            api_secret=os.getenv("BINANCE_API_SECRET"),
        )
    )
    service = ConversationService(bot=bot, store=store, jwt_manager=jwt_manager)
    return ConversationAPIApp(service=service, store=store)


def run_server(host: str = "0.0.0.0", port: int = 8080) -> None:
    app = build_conversation_api()
    with make_server(host, port, app) as httpd:
        print(f"Conversation API listening on http://{host}:{port}")
        httpd.serve_forever()


if __name__ == "__main__":
    run_server(
        host=os.getenv("CONVERSATION_API_HOST", "0.0.0.0"),
        port=int(os.getenv("CONVERSATION_API_PORT", "8080")),
    )
