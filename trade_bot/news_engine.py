from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import re
import time
from dataclasses import dataclass, asdict
from html import unescape
from typing import Any, Dict, List, Optional, Tuple

import requests


def utcnow_naive() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)


BINANCE_ANNOUNCEMENT_SOURCES = [
    {
        "name": "binance_new_listings",
        "catalog_id": 48,
        "event_bias": "listing",
    },
    {
        "name": "binance_delistings",
        "catalog_id": 161,
        "event_bias": "delisting",
    },
    {
        "name": "binance_futures",
        "catalog_id": 57,
        "event_bias": "derivatives",
    },
    {
        "name": "binance_system",
        "catalog_id": 7,
        "event_bias": "risk",
    },
]


ASSET_RE = re.compile(r"\b([A-Z]{2,10})\b")
WORD_RE = re.compile(r"[a-z0-9]{3,}")
TITLE_RE = re.compile(r'"title":"(.*?)"')
CODE_RE = re.compile(r'"code":"(.*?)"')
LINK_RE = re.compile(r'"url":"(.*?)"')
TIME_RE = re.compile(r'"releaseDate":(\d{10,13})')
ANCHOR_RE = re.compile(
    r'href="(?P<href>/en[^"]*?/support/announcement/[^"]+|/[^"]*?/support/announcement/[^"]+)"[^>]*>(?P<title>[^<]{8,200})</a>',
    re.IGNORECASE,
)
DATE_RE = re.compile(r"\b20\d{2}-\d{2}-\d{2}\b")


@dataclass
class NewsEvent:
    event_id: str
    source: str
    title: str
    url: str
    published_at: str
    category: str
    assets: List[str]
    summary: str


@dataclass
class TradeCommand:
    command_id: str
    event_id: str
    action: str
    symbol: str
    side: str
    confidence: float
    urgency: str
    ttl_minutes: int
    rationale: str
    event_category: str
    created_at: str
    metadata: Dict[str, Any]


@dataclass
class EventOutcome:
    event_id: str
    symbol: str
    event_category: str
    side: str
    horizon_minutes: int
    observed_return: float
    max_upside: float
    max_drawdown: float
    relative_return_vs_benchmark: float
    beta_adjusted_return: float
    estimated_beta_to_benchmark: float
    realized_volatility: float
    volume_ratio: float
    drift_ratio: float
    success: bool
    evaluated_at: str
    metadata: Dict[str, Any]


@dataclass
class EventResearchRecord:
    event_id: str
    source: str
    category: str
    symbol: str
    status: str
    published_at: str
    title: str
    url: str
    semantic_cluster: str
    market_context: Dict[str, Any]
    command_summary: Dict[str, Any]
    labels: Dict[str, Any]
    notes: List[str]


class NewsStateStore:
    def __init__(self, path: str):
        self.path = path
        self.state = self._load()

    def _load(self) -> Dict[str, Any]:
        if not os.path.exists(self.path):
            return {
                "seen_event_ids": [],
                "seen_event_fingerprints": [],
                "commands": [],
                "event_outcomes": [],
                "policy_stats": {},
                "last_scan_at": None,
            }
        try:
            with open(self.path, "r", encoding="utf-8") as handle:
                return json.load(handle)
        except Exception:
            return {
                "seen_event_ids": [],
                "seen_event_fingerprints": [],
                "commands": [],
                "event_outcomes": [],
                "policy_stats": {},
                "last_scan_at": None,
            }

    def save(self) -> None:
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as handle:
            json.dump(self.state, handle, indent=2)

    def mark_seen(self, event_id: str) -> None:
        seen = set(self.state.get("seen_event_ids", []))
        seen.add(event_id)
        self.state["seen_event_ids"] = sorted(seen)

    def has_seen(self, event_id: str) -> bool:
        return event_id in set(self.state.get("seen_event_ids", []))

    def mark_seen_fingerprint(self, fingerprint: str) -> None:
        seen = set(self.state.get("seen_event_fingerprints", []))
        seen.add(fingerprint)
        self.state["seen_event_fingerprints"] = sorted(seen)

    def has_seen_fingerprint(self, fingerprint: str) -> bool:
        return fingerprint in set(self.state.get("seen_event_fingerprints", []))

    def append_commands(self, commands: List[TradeCommand]) -> None:
        serialized = [asdict(command) for command in commands]
        self.state.setdefault("commands", []).extend(serialized)
        self.state["last_scan_at"] = utcnow_naive().isoformat()

    def append_outcomes(self, outcomes: List[EventOutcome]) -> None:
        serialized = [asdict(outcome) for outcome in outcomes]
        self.state.setdefault("event_outcomes", []).extend(serialized)

    def get_policy_stats(self) -> Dict[str, Any]:
        return self.state.setdefault("policy_stats", {})

    def update_policy_stat(self, key: str, stat: Dict[str, Any]) -> None:
        self.state.setdefault("policy_stats", {})[key] = stat


class BinanceNewsEngine:
    """
    Official-source Binance announcement watcher.

    Design:
    - fetch official Binance support announcement pages
    - extract announcement metadata
    - classify likely trade impact
    - enrich with current market context
    - emit structured trade-center commands
    """

    def __init__(
        self,
        exchange_client: Any,
        state_path: str,
        symbols: List[str],
        requests_session: Optional[requests.Session] = None,
        research_store: Optional[Any] = None,
    ) -> None:
        self.exchange_client = exchange_client
        self.symbols = symbols
        self.session = requests_session or requests.Session()
        self.store = NewsStateStore(state_path)
        self.research_store = research_store

    def scan(self) -> List[TradeCommand]:
        self.learn_from_realized_impacts()
        commands: List[TradeCommand] = []
        for source in BINANCE_ANNOUNCEMENT_SOURCES:
            events = self._fetch_source(source)
            for event in events:
                fingerprint = self._event_fingerprint(event)
                if self.store.has_seen(event.event_id) or self.store.has_seen_fingerprint(fingerprint):
                    continue
                generated = self._commands_for_event(event)
                if generated:
                    commands.extend(generated)
                    for command in generated:
                        self._record_research_event(event, command)
                self.store.mark_seen(event.event_id)
                self.store.mark_seen_fingerprint(fingerprint)

        if commands:
            self.store.append_commands(commands)
        else:
            self.store.state["last_scan_at"] = utcnow_naive().isoformat()
        self.store.save()
        return commands

    def learn_from_realized_impacts(self) -> List[EventOutcome]:
        pending_commands = self.store.state.get("commands", [])
        learned_outcomes: List[EventOutcome] = []
        remaining_commands: List[Dict[str, Any]] = []

        for command in pending_commands:
            if command.get("learning_evaluated"):
                remaining_commands.append(command)
                continue

            created_at = self._parse_iso(command.get("created_at"))
            ttl_minutes = int(command.get("ttl_minutes", 180))
            if (utcnow_naive() - created_at).total_seconds() < ttl_minutes * 60:
                remaining_commands.append(command)
                continue

            outcome = self._evaluate_command_impact(command)
            command["learning_evaluated"] = True
            if outcome:
                learned_outcomes.append(outcome)
                self._update_policy_from_outcome(command, outcome)
                self._record_research_outcome(command, outcome)
            remaining_commands.append(command)

        self.store.state["commands"] = remaining_commands
        if learned_outcomes:
            self.store.append_outcomes(learned_outcomes)
            self.store.save()
        return learned_outcomes

    def _fetch_source(self, source: Dict[str, str]) -> List[NewsEvent]:
        events = self._fetch_from_cms_api(source)
        if events:
            return events

        response = self.session.get(
            f"https://www.binance.com/en/support/announcement?c={source['catalog_id']}",
            headers={
                "User-Agent": "trade-bot-news-engine/1.0",
                "Accept-Language": "en-US,en;q=0.9",
            },
            timeout=20,
        )
        response.raise_for_status()
        html = response.text
        return self._extract_events_from_html(html, source)

    def _fetch_from_cms_api(self, source: Dict[str, str]) -> List[NewsEvent]:
        url = "https://www.binance.com/bapi/composite/v1/public/cms/article/list/query"
        response = self.session.get(
            url,
            params={
                "type": 1,
                "catalogId": source["catalog_id"],
                "pageNo": 1,
                "pageSize": 20,
            },
            headers={
                "User-Agent": "trade-bot-news-engine/1.0",
                "Accept": "application/json",
                "Accept-Language": "en-US,en;q=0.9",
            },
            timeout=20,
        )
        if response.status_code != 200:
            return []

        try:
            payload = response.json()
        except Exception:
            return []

        data = payload.get("data", {})
        articles = data.get("articles", [])
        if not articles:
            catalogs = data.get("catalogs", [])
            for catalog in catalogs:
                articles.extend(catalog.get("articles", []))

        events: List[NewsEvent] = []
        for article in articles:
            title = str(article.get("title", "")).strip()
            if not title:
                continue
            article_code = article.get("code") or article.get("id") or title
            release_time = article.get("releaseDate") or article.get("publishDate") or int(time.time() * 1000)
            if isinstance(release_time, str) and release_time.isdigit():
                release_time = int(release_time)
            if isinstance(release_time, int) and release_time > 10_000_000_000:
                published_at = dt.datetime.utcfromtimestamp(release_time / 1000).isoformat()
            elif isinstance(release_time, int):
                published_at = dt.datetime.utcfromtimestamp(release_time).isoformat()
            else:
                published_at = utcnow_naive().isoformat()

            link = article.get("webLink") or article.get("link") or ""
            if link and link.startswith("/"):
                link = "https://www.binance.com" + link
            if not link:
                link = f"https://www.binance.com/en/support/announcement/{article_code}"

            events.append(
                NewsEvent(
                    event_id=f"{source['name']}:{article_code}",
                    source=source["name"],
                    title=title,
                    url=link,
                    published_at=published_at,
                    category=source["event_bias"],
                    assets=self._extract_assets(title),
                    summary=self._summarize_title(title),
                )
            )
        return events

    def _extract_events_from_html(self, html: str, source: Dict[str, str]) -> List[NewsEvent]:
        titles = [unescape(item) for item in TITLE_RE.findall(html)]
        urls = [item.replace("\\/", "/") for item in LINK_RE.findall(html)]
        codes = CODE_RE.findall(html)
        times = TIME_RE.findall(html)

        events: List[NewsEvent] = []
        for idx, title in enumerate(titles[:25]):
            if not title or len(title) < 8:
                continue

            link = ""
            for raw_url in urls:
                if "/support/announcement/" in raw_url:
                    link = raw_url
                    break

            if link and link.startswith("/"):
                link = "https://www.binance.com" + link

            code = codes[idx] if idx < len(codes) else f"{source['name']}-{idx}"
            release_ts = times[idx] if idx < len(times) else str(int(time.time()))
            release_ms = int(release_ts[:13]) if len(release_ts) >= 13 else int(release_ts) * 1000
            published_at = dt.datetime.utcfromtimestamp(release_ms / 1000).isoformat()
            assets = self._extract_assets(title)

            events.append(
                NewsEvent(
                    event_id=f"{source['name']}:{code}",
                    source=source["name"],
                    title=title,
                    url=link or f"https://www.binance.com/en/support/announcement?c={source['catalog_id']}",
                    published_at=published_at,
                    category=source["event_bias"],
                    assets=assets,
                    summary=self._summarize_title(title),
                )
            )

        deduped: Dict[str, NewsEvent] = {}
        for event in events:
            deduped[event.event_id] = event
        if deduped:
            return list(deduped.values())

        return self._extract_events_from_anchors(html, source)

    def _extract_events_from_anchors(self, html: str, source: Dict[str, str]) -> List[NewsEvent]:
        events: List[NewsEvent] = []
        for idx, match in enumerate(ANCHOR_RE.finditer(html)):
            raw_title = unescape(match.group("title")).strip()
            if raw_title.lower() in {"latest articles", "announcement"}:
                continue
            href = match.group("href").replace("\\/", "/")
            if href.startswith("/"):
                url = "https://www.binance.com" + href
            else:
                url = href

            date_match = DATE_RE.search(raw_title)
            published_at = utcnow_naive().isoformat()
            if date_match:
                try:
                    published_at = dt.datetime.strptime(date_match.group(0), "%Y-%m-%d").isoformat()
                except ValueError:
                    published_at = utcnow_naive().isoformat()

            event = NewsEvent(
                event_id=self._stable_event_id(source["name"], raw_title, url),
                source=source["name"],
                title=raw_title,
                url=url,
                published_at=published_at,
                category=source["event_bias"],
                assets=self._extract_assets(raw_title),
                summary=self._summarize_title(raw_title),
            )
            events.append(event)

        deduped: Dict[str, NewsEvent] = {}
        for event in events:
            deduped[event.title] = event
        return list(deduped.values())[:25]

    def _extract_assets(self, title: str) -> List[str]:
        known_bases = {symbol.split("/")[0] for symbol in self.symbols}
        assets: List[str] = []
        normalized_title = title.replace("USDⓈ", "USD").replace("USDT", " USDT ")
        pair_matches = re.findall(r"\b([A-Z]{2,10})USDT\b", normalized_title)
        for token in pair_matches:
            if token in known_bases:
                assets.append(token)
        for token in ASSET_RE.findall(normalized_title):
            if token in known_bases:
                assets.append(token)
        return sorted(set(assets))

    def _summarize_title(self, title: str) -> str:
        if len(title) <= 220:
            return title
        return title[:217] + "..."

    def _commands_for_event(self, event: NewsEvent) -> List[TradeCommand]:
        commands: List[TradeCommand] = []
        impacted_symbols = self._map_event_to_symbols(event)
        fallback_symbols = self._fallback_symbols_for_event(event)
        if not impacted_symbols and not fallback_symbols and event.category != "risk":
            return []

        for symbol in impacted_symbols or fallback_symbols:
            market_context = self._market_context(symbol)
            command = self._command_from_event(event, symbol, market_context)
            if command:
                commands.append(command)
        return commands

    def _map_event_to_symbols(self, event: NewsEvent) -> List[str]:
        base_to_symbol = {symbol.split("/")[0]: symbol for symbol in self.symbols}
        return [base_to_symbol[asset] for asset in event.assets if asset in base_to_symbol]

    def _fallback_symbols_for_event(self, event: NewsEvent) -> List[str]:
        if not self.symbols:
            return []
        if event.category == "risk":
            return list(self.symbols)
        return []

    def _select_benchmark_symbol(self, target_symbol: str) -> str:
        if target_symbol in self.symbols and len(self.symbols) > 1:
            for candidate in self.symbols:
                if candidate != target_symbol:
                    return candidate
        return self.symbols[0] if self.symbols else target_symbol

    def _market_context(self, symbol: str) -> Dict[str, Any]:
        candles = self.exchange_client.fetch_ohlcv(symbol, "15m", limit=24) or []
        benchmark_symbol = self._select_benchmark_symbol(symbol)
        benchmark = self.exchange_client.fetch_ohlcv(benchmark_symbol, "15m", limit=24) or []
        order_book = self.exchange_client.get_order_book(symbol)
        last_close = candles[-1][4] if candles else None
        previous_close = candles[-2][4] if len(candles) > 1 else last_close
        momentum = 0.0
        if last_close and previous_close:
            momentum = (last_close - previous_close) / previous_close
        benchmark_momentum = 0.0
        if len(benchmark) > 1 and benchmark[-2][4]:
            benchmark_momentum = (benchmark[-1][4] - benchmark[-2][4]) / benchmark[-2][4]
        spread = 0.0
        try:
            bid = order_book["bid"]
            ask = order_book["ask"]
            mid = (bid + ask) / 2
            if mid > 0:
                spread = (ask - bid) / mid
        except Exception:
            spread = 0.0

        return {
            "last_close": last_close,
            "momentum_15m": momentum,
            "benchmark_symbol": benchmark_symbol,
            "benchmark_momentum_15m": benchmark_momentum,
            "spread": spread,
        }

    def _command_from_event(self, event: NewsEvent, symbol: str, market_context: Dict[str, Any]) -> Optional[TradeCommand]:
        title = event.title.lower()
        action = "HOLD"
        side = "flat"
        confidence = 0.0
        urgency = "low"
        ttl_minutes = 180
        rationale = event.summary

        if event.category == "listing":
            action = "ENTER"
            side = "buy"
            confidence = 0.74
            urgency = "high"
            ttl_minutes = 240
            rationale = f"Listing-related announcement detected for {symbol}. Positive exchange listing flow can create short-term upside."

        elif event.category == "delisting":
            action = "EXIT_OR_SHORT"
            side = "sell"
            confidence = 0.86
            urgency = "high"
            ttl_minutes = 720
            rationale = f"Delisting-related announcement detected for {symbol}. Delisting risk is usually strongly bearish for spot liquidity."

        elif event.category == "derivatives":
            if "usd-margined" in title or "perpetual" in title or "futures" in title:
                action = "ENTER"
                side = "buy"
                confidence = 0.68
                urgency = "medium"
                ttl_minutes = 180
                rationale = f"Derivatives-market expansion announcement for {symbol}. New futures support often increases visibility and trading interest."

        elif event.category == "risk":
            action = "RISK_OFF"
            side = "flat"
            confidence = 0.82
            urgency = "high"
            ttl_minutes = 120
            rationale = "System or operational risk announcement detected. Reduce or halt new exposure until operational conditions normalize."

        if action == "HOLD":
            return None

        semantic_cluster = self._semantic_cluster(event)
        learned_prior = self._learned_prior(event.category, symbol, side, semantic_cluster)
        confidence += learned_prior["confidence_delta"]
        rationale += learned_prior["rationale_suffix"]

        if market_context["spread"] and market_context["spread"] > 0.01:
            confidence -= 0.12
            rationale += " Confidence reduced because current spread is wide."

        if side == "buy" and market_context["momentum_15m"] < -0.03:
            confidence -= 0.10
            rationale += " Confidence reduced because short-term momentum is sharply negative."

        if side == "sell" and market_context["momentum_15m"] > 0.05:
            confidence -= 0.08
            rationale += " Confidence reduced because short-term momentum is strongly positive."

        confidence = max(0.0, min(confidence, 0.95))
        if confidence < 0.55:
            return None

        expected_return = learned_prior["expected_return"]
        edge_gate = self._execution_edge_gate(learned_prior)
        if side == "buy" and expected_return < -0.01:
            return None
        if side == "sell" and expected_return > 0.01:
            return None
        if not edge_gate["allow"]:
            return None

        return TradeCommand(
            command_id=str(uuid_like(event.event_id, symbol, action)),
            event_id=event.event_id,
            action=action,
            symbol=symbol,
            side=side,
            confidence=round(confidence, 3),
            urgency=urgency,
            ttl_minutes=ttl_minutes,
            rationale=rationale,
            event_category=event.category,
            created_at=utcnow_naive().isoformat(),
            metadata={
                "event_title": event.title,
                "event_url": event.url,
                "published_at": event.published_at,
                "market_context": market_context,
                "semantic_cluster": semantic_cluster,
                "learned_prior": learned_prior,
                "execution_gate": edge_gate,
                "event_labels": self._build_event_labels(event, symbol, market_context, learned_prior),
            },
        )

    def event_risk_snapshot(self) -> Dict[str, Any]:
        commands = self.store.state.get("commands", [])
        active = []
        for command in commands:
            if command.get("learning_evaluated"):
                continue
            active.append(
                {
                    "event_id": command.get("event_id"),
                    "symbol": command.get("symbol"),
                    "event_category": command.get("event_category"),
                    "urgency": command.get("urgency"),
                    "confidence": command.get("confidence"),
                    "created_at": command.get("created_at"),
                }
            )
        high_risk = [item for item in active if item["event_category"] == "risk" or item["urgency"] == "high"]
        return {
            "active_event_commands": active[:25],
            "high_risk_events": high_risk[:10],
            "active_count": len(active),
            "high_risk_count": len(high_risk),
            "last_scan_at": self.store.state.get("last_scan_at"),
        }

    def active_commands(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        active: List[Dict[str, Any]] = []
        now = utcnow_naive()
        for command in self.store.state.get("commands", []):
            if command.get("learning_evaluated"):
                continue
            created_at = self._parse_iso(command.get("created_at"))
            ttl_minutes = int(command.get("ttl_minutes", 180))
            if (now - created_at).total_seconds() > ttl_minutes * 60:
                continue
            if symbol is not None and command.get("symbol") != symbol:
                continue
            active.append(command)
        return active

    def research_signal_context(self, symbol: str) -> Dict[str, Any]:
        commands = self.active_commands(symbol)
        bullish = 0.0
        bearish = 0.0
        risk_off = 0.0
        reasons: List[str] = []

        for command in commands:
            confidence = float(command.get("confidence", 0.0) or 0.0)
            action = str(command.get("action", "")).upper()
            side = str(command.get("side", "")).lower()
            category = str(command.get("event_category", "unknown"))
            if action == "RISK_OFF" or category == "risk":
                risk_off = max(risk_off, confidence)
                reasons.append(f"{category}:risk_off:{confidence:.2f}")
                continue
            if side == "buy":
                bullish += confidence
                reasons.append(f"{category}:bullish:{confidence:.2f}")
            elif side == "sell":
                bearish += confidence
                reasons.append(f"{category}:bearish:{confidence:.2f}")

        bullish = min(bullish, 1.0)
        bearish = min(bearish, 1.0)
        net_bias = bullish - bearish
        return {
            "symbol": symbol,
            "bullish_confidence": bullish,
            "bearish_confidence": bearish,
            "net_bias": max(-1.0, min(net_bias, 1.0)),
            "risk_off_confidence": min(risk_off, 1.0),
            "has_conflict": bullish > 0 and bearish > 0,
            "active_commands": [
                {
                    "event_id": command.get("event_id"),
                    "action": command.get("action"),
                    "side": command.get("side"),
                    "confidence": command.get("confidence"),
                    "event_category": command.get("event_category"),
                    "urgency": command.get("urgency"),
                }
                for command in commands
            ],
            "reasons": reasons[:10],
        }

    def _event_fingerprint(self, event: NewsEvent) -> str:
        basis = "|".join(
            [
                event.source,
                event.category,
                self._normalize_title(event.title),
                event.url.strip().lower(),
                event.published_at[:19],
            ]
        )
        return hashlib.sha256(basis.encode("utf-8")).hexdigest()

    def _stable_event_id(self, source_name: str, title: str, url: str) -> str:
        basis = f"{source_name}|{self._normalize_title(title)}|{url.strip().lower()}"
        return f"{source_name}:anchor:{hashlib.sha256(basis.encode('utf-8')).hexdigest()[:24]}"

    @staticmethod
    def _normalize_title(title: str) -> str:
        return " ".join(title.strip().lower().split())

    def recent_research_records(self, limit: int = 20) -> List[Dict[str, Any]]:
        if self.research_store is None:
            return []
        return self.research_store.load_recent_research_events(limit=limit)

    def _build_event_labels(
        self,
        event: NewsEvent,
        symbol: str,
        market_context: Dict[str, Any],
        learned_prior: Dict[str, Any],
    ) -> Dict[str, Any]:
        return {
            "semantic_cluster": self._semantic_cluster(event),
            "direction_bias": "bullish" if event.category in {"listing", "derivatives"} else "bearish" if event.category == "delisting" else "risk_off",
            "spread_bucket": "wide" if market_context.get("spread", 0.0) > 0.004 else "normal",
            "momentum_bucket": "negative" if market_context.get("momentum_15m", 0.0) < -0.01 else "positive" if market_context.get("momentum_15m", 0.0) > 0.01 else "flat",
            "samples": learned_prior.get("samples", 0),
        }

    def _record_research_event(self, event: NewsEvent, command: TradeCommand) -> None:
        if self.research_store is None:
            return
        metadata = command.metadata or {}
        record = EventResearchRecord(
            event_id=event.event_id,
            source=event.source,
            category=event.category,
            symbol=command.symbol,
            status="pending_outcome",
            published_at=event.published_at,
            title=event.title,
            url=event.url,
            semantic_cluster=str(metadata.get("semantic_cluster", "generic")),
            market_context=dict(metadata.get("market_context", {})),
            command_summary={
                "action": command.action,
                "side": command.side,
                "confidence": command.confidence,
                "urgency": command.urgency,
                "ttl_minutes": command.ttl_minutes,
            },
            labels=dict(metadata.get("event_labels", {})),
            notes=[command.rationale],
        )
        self.research_store.upsert_research_event(
            event_id=record.event_id,
            source=record.source,
            category=record.category,
            symbol=record.symbol,
            status=record.status,
            published_at=record.published_at,
            payload=asdict(record),
        )

    def _record_research_outcome(self, command: Dict[str, Any], outcome: EventOutcome) -> None:
        if self.research_store is None:
            return
        event_id = command["event_id"]
        payload = asdict(outcome)
        self.research_store.append_research_outcome(
            event_id=event_id,
            symbol=outcome.symbol,
            evaluated_at=outcome.evaluated_at,
            outcome=payload,
        )
        self.research_store.upsert_research_event(
            event_id=event_id,
            source=str(command.get("metadata", {}).get("event_source", "binance")),
            category=str(command.get("event_category", "unknown")),
            symbol=outcome.symbol,
            status="evaluated",
            published_at=str(command.get("metadata", {}).get("published_at") or command.get("created_at")),
            payload={
                "event_id": event_id,
                "symbol": outcome.symbol,
                "latest_outcome": payload,
                "command": command,
            },
        )

    def _learned_prior(self, category: str, symbol: str, side: str, semantic_cluster: str) -> Dict[str, Any]:
        keys = [
            self._policy_key(category, symbol, side),
            self._policy_key(f"{category}:{semantic_cluster}", symbol, side),
            self._policy_key(f"{category}:{semantic_cluster}", "GLOBAL", side),
            self._policy_key(category, "GLOBAL", side),
        ]
        stats = self.store.get_policy_stats()
        for key in keys:
            if key not in stats:
                continue
            stat = stats[key]
            samples = int(stat.get("samples", 0))
            avg_return = float(stat.get("avg_return", 0.0))
            hit_rate = float(stat.get("hit_rate", 0.5))
            avg_relative_return = float(stat.get("avg_relative_return_vs_benchmark", 0.0))
            avg_beta_adjusted_return = float(stat.get("avg_beta_adjusted_return", avg_return))
            avg_realized_volatility = float(stat.get("avg_realized_volatility", 0.0))
            avg_volume_ratio = float(stat.get("avg_volume_ratio", 1.0))
            avg_drift_ratio = float(stat.get("avg_drift_ratio", 0.0))
            confidence_delta = min(max((hit_rate - 0.5) * 0.35, -0.12), 0.12)
            confidence_delta += min(max(avg_return * 2.0, -0.08), 0.08)
            confidence_delta += min(max(avg_beta_adjusted_return * 3.0, -0.08), 0.08)
            if avg_volume_ratio > 1.1:
                confidence_delta += 0.03
            if avg_drift_ratio < 0:
                confidence_delta -= 0.03

            learned_edge_score = (
                (avg_beta_adjusted_return * 4.0)
                + (avg_relative_return * 2.0)
                + ((hit_rate - 0.5) * 0.8)
                + min(max(avg_volume_ratio - 1.0, -0.2), 0.2)
                - min(avg_realized_volatility * 2.0, 0.25)
            )
            return {
                "samples": samples,
                "avg_return": avg_return,
                "avg_relative_return_vs_benchmark": avg_relative_return,
                "avg_beta_adjusted_return": avg_beta_adjusted_return,
                "avg_realized_volatility": avg_realized_volatility,
                "avg_volume_ratio": avg_volume_ratio,
                "avg_drift_ratio": avg_drift_ratio,
                "hit_rate": hit_rate,
                "confidence_delta": confidence_delta,
                "expected_return": avg_return,
                "learned_edge_score": learned_edge_score,
                "rationale_suffix": (
                    f" Learned prior from {samples} similar events: "
                    f"avg_return={avg_return:.3%}, beta_adj={avg_beta_adjusted_return:.3%}, "
                    f"hit_rate={hit_rate:.1%}."
                ),
            }

        return {
            "samples": 0,
            "avg_return": 0.0,
            "avg_relative_return_vs_benchmark": 0.0,
            "avg_beta_adjusted_return": 0.0,
            "avg_realized_volatility": 0.0,
            "avg_volume_ratio": 1.0,
            "avg_drift_ratio": 0.0,
            "hit_rate": 0.5,
            "confidence_delta": 0.0,
            "expected_return": 0.0,
            "learned_edge_score": 0.0,
            "rationale_suffix": "",
        }

    def _execution_edge_gate(self, learned_prior: Dict[str, Any]) -> Dict[str, Any]:
        samples = int(learned_prior.get("samples", 0))
        hit_rate = float(learned_prior.get("hit_rate", 0.5))
        beta_adjusted = float(learned_prior.get("avg_beta_adjusted_return", 0.0))
        edge_score = float(learned_prior.get("learned_edge_score", 0.0))

        if samples == 0:
            return {"allow": True, "reason": "cold_start"}

        allow = (
            samples >= 2
            and hit_rate >= 0.52
            and beta_adjusted >= 0.001
            and edge_score >= 0.02
        )
        return {
            "allow": allow,
            "reason": "learned_edge_threshold",
            "samples": samples,
            "hit_rate": hit_rate,
            "beta_adjusted_return": beta_adjusted,
            "edge_score": edge_score,
        }

    def _policy_key(self, category: str, symbol: str, side: str) -> str:
        return f"{category}|{symbol}|{side}"

    def _update_policy_from_outcome(self, command: Dict[str, Any], outcome: EventOutcome) -> None:
        stats = self.store.get_policy_stats()
        semantic_cluster = command.get("metadata", {}).get("semantic_cluster", "generic")
        keys = [
            self._policy_key(outcome.event_category, outcome.symbol, outcome.side),
            self._policy_key(f"{outcome.event_category}:{semantic_cluster}", outcome.symbol, outcome.side),
            self._policy_key(f"{outcome.event_category}:{semantic_cluster}", "GLOBAL", outcome.side),
            self._policy_key(outcome.event_category, "GLOBAL", outcome.side),
        ]
        for key in keys:
            existing = stats.get(
                key,
                {
                    "samples": 0,
                    "avg_return": 0.0,
                    "hit_rate": 0.5,
                    "avg_upside": 0.0,
                    "avg_drawdown": 0.0,
                    "avg_relative_return_vs_benchmark": 0.0,
                    "avg_beta_adjusted_return": 0.0,
                    "avg_estimated_beta_to_benchmark": 1.0,
                    "avg_realized_volatility": 0.0,
                    "avg_volume_ratio": 1.0,
                    "avg_drift_ratio": 0.0,
                },
            )
            samples = int(existing["samples"]) + 1
            existing["samples"] = samples
            existing["avg_return"] = self._ema(existing["avg_return"], outcome.observed_return, samples, alpha=0.25)
            existing["hit_rate"] = self._ema(existing["hit_rate"], 1.0 if outcome.success else 0.0, samples, alpha=0.25)
            existing["avg_upside"] = self._ema(existing["avg_upside"], outcome.max_upside, samples, alpha=0.25)
            existing["avg_drawdown"] = self._ema(existing["avg_drawdown"], outcome.max_drawdown, samples, alpha=0.25)
            existing["avg_relative_return_vs_benchmark"] = self._ema(
                existing["avg_relative_return_vs_benchmark"], outcome.relative_return_vs_benchmark, samples, alpha=0.25
            )
            existing["avg_beta_adjusted_return"] = self._ema(
                existing["avg_beta_adjusted_return"], outcome.beta_adjusted_return, samples, alpha=0.25
            )
            existing["avg_estimated_beta_to_benchmark"] = self._ema(
                existing["avg_estimated_beta_to_benchmark"], outcome.estimated_beta_to_benchmark, samples, alpha=0.25
            )
            existing["avg_realized_volatility"] = self._ema(
                existing["avg_realized_volatility"], outcome.realized_volatility, samples, alpha=0.25
            )
            existing["avg_volume_ratio"] = self._ema(
                existing["avg_volume_ratio"], outcome.volume_ratio, samples, alpha=0.25
            )
            existing["avg_drift_ratio"] = self._ema(
                existing["avg_drift_ratio"], outcome.drift_ratio, samples, alpha=0.25
            )
            self.store.update_policy_stat(key, existing)

    def _evaluate_command_impact(self, command: Dict[str, Any]) -> Optional[EventOutcome]:
        symbol = command.get("symbol")
        side = command.get("side", "flat")
        if side == "flat":
            return None

        published_at = self._parse_iso(command.get("metadata", {}).get("published_at") or command.get("created_at"))
        horizon_minutes = int(command.get("ttl_minutes", 180))
        market_window = self._fetch_market_window(symbol, published_at, horizon_minutes)
        if len(market_window["prices"]) < 2:
            return None

        prices = market_window["prices"]
        entry = prices[0]
        exit_price = prices[-1]
        path_returns = [(price - entry) / entry for price in prices]
        if side == "buy":
            observed_return = (exit_price - entry) / entry
            max_upside = max(path_returns)
            max_drawdown = min(path_returns)
        else:
            observed_return = (entry - exit_price) / entry
            max_upside = max((-ret) for ret in path_returns)
            max_drawdown = min((-ret) for ret in path_returns)

        benchmark_prices = market_window["benchmark_prices"]
        benchmark_return = 0.0
        if len(benchmark_prices) >= 2 and benchmark_prices[0]:
            benchmark_return = (benchmark_prices[-1] - benchmark_prices[0]) / benchmark_prices[0]
        relative_return = observed_return - benchmark_return
        estimated_beta = self._estimate_beta(prices, benchmark_prices)
        beta_adjusted_return = observed_return - (estimated_beta * benchmark_return)

        realized_volatility = 0.0
        if len(path_returns) > 1:
            mean_return = sum(path_returns) / len(path_returns)
            variance = sum((ret - mean_return) ** 2 for ret in path_returns) / len(path_returns)
            realized_volatility = variance ** 0.5

        volumes = market_window["volumes"]
        baseline_volumes = volumes[: max(1, len(volumes) // 3)]
        event_volumes = volumes[max(1, len(volumes) // 3):] or volumes
        base_avg = (sum(baseline_volumes) / len(baseline_volumes)) if baseline_volumes else 0.0
        event_avg = (sum(event_volumes) / len(event_volumes)) if event_volumes else 0.0
        volume_ratio = (event_avg / base_avg) if base_avg > 0 else 1.0

        drift_ratio = 0.0
        if max_upside != 0:
            drift_ratio = observed_return / max_upside if side == "buy" else observed_return / max_upside

        success = observed_return > 0
        return EventOutcome(
            event_id=command["event_id"],
            symbol=symbol,
            event_category=command["event_category"],
            side=side,
            horizon_minutes=horizon_minutes,
            observed_return=observed_return,
            max_upside=max_upside,
            max_drawdown=max_drawdown,
            relative_return_vs_benchmark=relative_return,
            beta_adjusted_return=beta_adjusted_return,
            estimated_beta_to_benchmark=estimated_beta,
            realized_volatility=realized_volatility,
            volume_ratio=volume_ratio,
            drift_ratio=drift_ratio,
            success=success,
            evaluated_at=utcnow_naive().isoformat(),
            metadata={
                "command_id": command["command_id"],
                "created_at": command["created_at"],
                "semantic_cluster": command.get("metadata", {}).get("semantic_cluster", "generic"),
            },
        )

    def _fetch_market_window(self, symbol: str, start: dt.datetime, horizon_minutes: int) -> Dict[str, List[float]]:
        try:
            import ccxt
        except Exception:
            return {"prices": [], "benchmark_prices": [], "volumes": []}

        exchange = ccxt.binance({"enableRateLimit": True})
        since = int(start.timestamp() * 1000)
        limit = max(2, min((horizon_minutes // 15) + 2, 200))
        benchmark_symbol = self._select_benchmark_symbol(symbol)
        try:
            candles = exchange.fetch_ohlcv(symbol, "15m", since=since, limit=limit)
            benchmark = exchange.fetch_ohlcv(benchmark_symbol, "15m", since=since, limit=limit)
        except Exception:
            return {"prices": [], "benchmark_prices": [], "volumes": []}
        return {
            "prices": [float(candle[4]) for candle in candles if len(candle) >= 5],
            "benchmark_prices": [float(candle[4]) for candle in benchmark if len(candle) >= 5],
            "volumes": [float(candle[5]) for candle in candles if len(candle) >= 6],
        }

    def _estimate_beta(self, asset_prices: List[float], benchmark_prices: List[float]) -> float:
        if len(asset_prices) < 3 or len(benchmark_prices) < 3:
            return 1.0
        length = min(len(asset_prices), len(benchmark_prices))
        asset_returns: List[float] = []
        benchmark_returns: List[float] = []
        for idx in range(1, length):
            prev_asset = asset_prices[idx - 1]
            prev_bench = benchmark_prices[idx - 1]
            if prev_asset and prev_bench:
                asset_returns.append((asset_prices[idx] - prev_asset) / prev_asset)
                benchmark_returns.append((benchmark_prices[idx] - prev_bench) / prev_bench)
        if len(asset_returns) < 2:
            return 1.0
        mean_asset = sum(asset_returns) / len(asset_returns)
        mean_bench = sum(benchmark_returns) / len(benchmark_returns)
        covariance = sum(
            (asset - mean_asset) * (bench - mean_bench)
            for asset, bench in zip(asset_returns, benchmark_returns)
        ) / len(asset_returns)
        variance = sum((bench - mean_bench) ** 2 for bench in benchmark_returns) / len(benchmark_returns)
        if variance <= 1e-12:
            return 1.0
        return covariance / variance

    def _semantic_cluster(self, event: NewsEvent) -> str:
        title = event.title.lower()
        tokens = set(WORD_RE.findall(title))
        if {"delist", "delisting", "removal"} & tokens:
            return "delist_removal"
        if {"listing", "launch", "launches"} & tokens:
            return "listing_launch"
        if {"margin", "pairs", "pair"} & tokens:
            return "margin_pairs"
        if {"futures", "perpetual", "contract", "contracts"} & tokens:
            return "futures_contract"
        if {"system", "maintenance", "upgrade"} & tokens:
            return "system_risk"
        if {"bot", "bots"} & tokens:
            return "trading_bots"
        if {"support", "swap", "rebranding"} & tokens:
            return "token_event"
        return "generic"

    def _parse_iso(self, value: Optional[str]) -> dt.datetime:
        if not value:
            return utcnow_naive()
        parsed = dt.datetime.fromisoformat(value)
        return parsed.replace(tzinfo=None) if parsed.tzinfo else parsed

    def _ema(self, current: float, observed: float, samples: int, alpha: float = 0.25) -> float:
        if samples <= 1:
            return observed
        return (alpha * observed) + ((1 - alpha) * current)


def uuid_like(*parts: str) -> str:
    seed = "|".join(parts)
    digest = json.dumps(seed).encode("utf-8")
    return f"cmd-{abs(hash(digest))}"
