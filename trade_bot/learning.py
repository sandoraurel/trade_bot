from __future__ import annotations

import datetime as dt
import json
from typing import Any, Dict, List


def utcnow_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


class TradeLearningEngine:
    """
    Outcome-driven learning layer.

    The engine does not place trades and does not hard-disable a strategy after
    a single loss. Instead it:
    - stores the bot's decision context at entry time
    - records realized outcomes at exit time
    - aggregates recurring failure/success patterns
    - returns evidence-weighted adjustments for future signals
    """

    def __init__(self, config: Any, state_store: Any):
        self.config = config
        self.state_store = state_store

    def build_trade_context(self, signal: Dict[str, Any], regime: Any | None = None) -> Dict[str, Any]:
        metadata = dict(signal.get("metadata", {}) or {})
        research = dict(signal.get("research_context", {}) or metadata.get("research_context", {}) or {})
        regime_name = str(signal.get("regime") or getattr(regime, "regime", "unknown"))
        regime_confidence = float(metadata.get("regime_confidence", getattr(regime, "confidence", 0.0)) or 0.0)
        trend_direction = str(metadata.get("trend_direction", getattr(regime, "metadata", {}).get("trend_direction", "flat")))
        rr_ratio = float(signal.get("rr_ratio", 0.0) or 0.0)
        confidence = float(signal.get("signal_quality", 0.0) or 0.0)
        expected_edge_bps = float(signal.get("expected_edge_bps", 0.0) or 0.0)
        hurst = float(signal.get("hurst_exponent", metadata.get("hurst_exponent", 0.5)) or 0.5)
        pattern = self._pattern_features(
            strategy=str(signal.get("strategy", "unknown")),
            side=str(signal.get("side", "flat")),
            regime=regime_name,
            trend_direction=trend_direction,
            confidence=confidence,
            rr_ratio=rr_ratio,
            expected_edge_bps=expected_edge_bps,
            research=research,
            hurst=hurst,
        )
        return {
            "captured_at": utcnow_iso(),
            "strategy": str(signal.get("strategy", "unknown")),
            "side": str(signal.get("side", "flat")),
            "regime": regime_name,
            "regime_confidence": regime_confidence,
            "trend_direction": trend_direction,
            "signal_quality": confidence,
            "expected_edge_bps": expected_edge_bps,
            "rr_ratio": rr_ratio,
            "hurst_exponent": hurst,
            "fast_move": bool(signal.get("fast_move", False)),
            "research_context": research,
            "pattern": pattern,
        }

    def learning_context_for_signal(self, signal: Dict[str, Any], regime: Any | None = None) -> Dict[str, Any]:
        if self.state_store is None:
            return {}
        trade_context = self.build_trade_context(signal, regime)
        pattern = trade_context["pattern"]
        keys = self._pattern_keys(pattern)
        stats = self.state_store.load_learning_patterns(keys)
        if not stats:
            return {
                "score_delta": 0.0,
                "confidence_delta": 0.0,
                "veto": False,
                "reasons": [],
                "evidence_samples": 0,
                "pattern": pattern,
            }

        aggregate = self._aggregate_stats(stats)
        score_delta = aggregate["score_delta"]
        confidence_delta = aggregate["confidence_delta"]
        veto = bool(
            aggregate["samples"] >= 4
            and aggregate["loss_rate"] >= 0.80
            and aggregate["avg_r_multiple"] <= -0.75
            and aggregate["matched_key_strength"] >= 1.0
        )
        return {
            "score_delta": score_delta,
            "confidence_delta": confidence_delta,
            "veto": veto,
            "reasons": aggregate["reasons"],
            "evidence_samples": aggregate["samples"],
            "pattern": pattern,
        }

    def record_closed_trade(
        self,
        *,
        symbol: str,
        position: Any,
        close_price: float,
        profit_loss: float,
        exit_reason: str,
        closed_at: str | None = None,
    ) -> None:
        if self.state_store is None:
            return
        metadata = dict(getattr(position, "metadata", {}) or {})
        trade_context = dict(metadata.get("decision_context", {}) or {})
        if not trade_context:
            trade_context = {
                "strategy": getattr(position, "strategy", "unknown"),
                "side": getattr(position, "side", "flat"),
                "regime": "unknown",
                "regime_confidence": 0.0,
                "trend_direction": "flat",
                "signal_quality": 0.0,
                "expected_edge_bps": 0.0,
                "rr_ratio": 0.0,
                "hurst_exponent": 0.5,
                "fast_move": False,
                "research_context": {},
                "pattern": self._pattern_features(
                    strategy=getattr(position, "strategy", "unknown"),
                    side=getattr(position, "side", "flat"),
                    regime="unknown",
                    trend_direction="flat",
                    confidence=0.0,
                    rr_ratio=0.0,
                    expected_edge_bps=0.0,
                    research={},
                    hurst=0.5,
                ),
            }

        entry_price = float(getattr(position, "entry_price", 0.0) or 0.0)
        initial_stop_loss = float(
            getattr(position, "initial_stop_loss", None)
            or getattr(position, "stop_loss", 0.0)
            or 0.0
        )
        size = float(getattr(position, "size", 0.0) or 0.0)
        side = str(getattr(position, "side", "flat"))
        risk_per_unit = abs(entry_price - initial_stop_loss)
        initial_risk = risk_per_unit * size
        r_multiple = (profit_loss / initial_risk) if initial_risk > 0 else 0.0
        failure_reasons = self._failure_reasons(trade_context, profit_loss, exit_reason)
        observation = {
            "observed_at": closed_at or utcnow_iso(),
            "symbol": symbol,
            "strategy": str(trade_context.get("strategy", getattr(position, "strategy", "unknown"))),
            "side": side,
            "entry_price": entry_price,
            "close_price": close_price,
            "profit_loss": profit_loss,
            "r_multiple": r_multiple,
            "exit_reason": exit_reason,
            "success": profit_loss > 0,
            "pattern": trade_context.get("pattern", {}),
            "research_context": trade_context.get("research_context", {}),
            "failure_reasons": failure_reasons,
            "decision_context": trade_context,
        }
        self.state_store.append_learning_observation(observation)
        self._update_patterns(observation)

    def _failure_reasons(self, trade_context: Dict[str, Any], profit_loss: float, exit_reason: str) -> List[str]:
        reasons: List[str] = []
        if profit_loss >= 0:
            reasons.append("positive_outcome")
            return reasons

        if exit_reason.upper() == "SL":
            reasons.append("thesis_invalidated")
        if bool(trade_context.get("fast_move")):
            reasons.append("fast_move_entry")
        if float(trade_context.get("signal_quality", 0.0) or 0.0) < 0.62:
            reasons.append("weak_signal_quality")
        if float(trade_context.get("expected_edge_bps", 0.0) or 0.0) < 12.0:
            reasons.append("thin_expected_edge")
        if float(trade_context.get("regime_confidence", 0.0) or 0.0) < 0.65:
            reasons.append("weak_regime_confidence")

        research = dict(trade_context.get("research_context", {}) or {})
        bullish = float(research.get("bullish_confidence", 0.0) or 0.0)
        bearish = float(research.get("bearish_confidence", 0.0) or 0.0)
        risk_off = float(research.get("risk_off_confidence", 0.0) or 0.0)
        side = str(trade_context.get("side", "flat")).lower()
        if risk_off >= float(getattr(self.config, "research_risk_off_veto_confidence", 0.75)):
            reasons.append("ignored_risk_off")
        if side in {"long", "buy"} and bearish > bullish:
            reasons.append("traded_against_bearish_research")
        if side in {"short", "sell"} and bullish > bearish:
            reasons.append("traded_against_bullish_research")
        if not reasons:
            reasons.append("generic_loss_pattern")
        return reasons

    def _update_patterns(self, observation: Dict[str, Any]) -> None:
        pattern = dict(observation.get("pattern", {}) or {})
        if not pattern:
            return
        keys = self._pattern_keys(pattern)
        for key, weight in keys.items():
            current = self.state_store.load_learning_pattern(key) or {
                "samples": 0,
                "wins": 0,
                "losses": 0,
                "total_r_multiple": 0.0,
                "stop_loss_exits": 0,
                "failure_reasons": {},
                "weight": weight,
            }
            current["samples"] = int(current.get("samples", 0)) + 1
            if observation["success"]:
                current["wins"] = int(current.get("wins", 0)) + 1
            else:
                current["losses"] = int(current.get("losses", 0)) + 1
            current["total_r_multiple"] = float(current.get("total_r_multiple", 0.0)) + float(observation.get("r_multiple", 0.0) or 0.0)
            if str(observation.get("exit_reason", "")).upper() == "SL":
                current["stop_loss_exits"] = int(current.get("stop_loss_exits", 0)) + 1
            reason_counts = dict(current.get("failure_reasons", {}) or {})
            for reason in observation.get("failure_reasons", []):
                reason_counts[reason] = int(reason_counts.get(reason, 0)) + 1
            current["failure_reasons"] = reason_counts
            current["updated_at"] = observation["observed_at"]
            current["weight"] = weight
            self.state_store.upsert_learning_pattern(key, current)

    def _aggregate_stats(self, stats: List[Dict[str, Any]]) -> Dict[str, Any]:
        weighted_samples = 0.0
        weighted_r = 0.0
        weighted_loss = 0.0
        weighted_stop = 0.0
        strongest_weight = 0.0
        reasons: Dict[str, float] = {}
        max_samples = 0
        for item in stats:
            samples = int(item.get("samples", 0))
            if samples <= 0:
                continue
            weight = float(item.get("weight", 0.0) or 0.0)
            weighted = samples * weight
            max_samples = max(max_samples, samples)
            weighted_samples += weighted
            avg_r = float(item.get("total_r_multiple", 0.0)) / samples
            loss_rate = float(item.get("losses", 0)) / samples
            stop_rate = float(item.get("stop_loss_exits", 0)) / samples
            weighted_r += avg_r * weighted
            weighted_loss += loss_rate * weighted
            weighted_stop += stop_rate * weighted
            strongest_weight = max(strongest_weight, weight)
            for reason, count in dict(item.get("failure_reasons", {}) or {}).items():
                reasons[reason] = reasons.get(reason, 0.0) + (float(count) * weight)
        if weighted_samples <= 0:
            return {
                "samples": 0,
                "avg_r_multiple": 0.0,
                "loss_rate": 0.0,
                "score_delta": 0.0,
                "confidence_delta": 0.0,
                "matched_key_strength": 0.0,
                "reasons": [],
            }
        avg_r_multiple = weighted_r / weighted_samples
        loss_rate = weighted_loss / weighted_samples
        stop_rate = weighted_stop / weighted_samples
        score_delta = 0.0
        confidence_delta = 0.0
        if max_samples >= 3:
            score_delta += max(min(avg_r_multiple * 8.0, 8.0), -12.0)
            score_delta += max(min((0.5 - loss_rate) * 10.0, 5.0), -7.0)
            score_delta += max(min((0.45 - stop_rate) * 6.0, 4.0), -5.0)
            confidence_delta += max(min(avg_r_multiple * 0.06, 0.06), -0.10)
        top_reasons = [reason for reason, _ in sorted(reasons.items(), key=lambda item: item[1], reverse=True)[:3]]
        return {
            "samples": max_samples,
            "avg_r_multiple": avg_r_multiple,
            "loss_rate": loss_rate,
            "score_delta": score_delta,
            "confidence_delta": confidence_delta,
            "matched_key_strength": strongest_weight,
            "reasons": top_reasons,
        }

    def _pattern_features(
        self,
        *,
        strategy: str,
        side: str,
        regime: str,
        trend_direction: str,
        confidence: float,
        rr_ratio: float,
        expected_edge_bps: float,
        research: Dict[str, Any],
        hurst: float,
    ) -> Dict[str, Any]:
        bullish = float(research.get("bullish_confidence", 0.0) or 0.0)
        bearish = float(research.get("bearish_confidence", 0.0) or 0.0)
        risk_off = float(research.get("risk_off_confidence", 0.0) or 0.0)
        research_state = "risk_off" if risk_off >= 0.75 else "conflict" if bullish > 0 and bearish > 0 else "aligned_bullish" if bullish > bearish else "aligned_bearish" if bearish > bullish else "neutral"
        return {
            "strategy": strategy,
            "side": side,
            "regime": regime,
            "trend_direction": trend_direction,
            "confidence_bucket": self._bucket(confidence, [(0.6, "low"), (0.75, "mid"), (10.0, "high")]),
            "rr_bucket": self._bucket(rr_ratio, [(1.35, "thin"), (2.2, "good"), (10.0, "strong")]),
            "edge_bucket": self._bucket(expected_edge_bps, [(12.0, "thin"), (25.0, "good"), (10_000.0, "strong")]),
            "hurst_bucket": self._bucket(hurst, [(0.25, "low"), (0.55, "mid"), (1.0, "high")]),
            "research_state": research_state,
        }

    @staticmethod
    def _bucket(value: float, levels: List[tuple[float, str]]) -> str:
        for threshold, label in levels:
            if value < threshold:
                return label
        return levels[-1][1]

    def _pattern_keys(self, pattern: Dict[str, Any]) -> Dict[str, float]:
        compact = json.dumps(pattern, sort_keys=True, separators=(",", ":"))
        broad = json.dumps(
            {
                "strategy": pattern.get("strategy"),
                "side": pattern.get("side"),
                "regime": pattern.get("regime"),
                "research_state": pattern.get("research_state"),
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        family = json.dumps(
            {
                "strategy": pattern.get("strategy"),
                "side": pattern.get("side"),
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return {
            compact: 1.0,
            broad: 0.6,
            family: 0.35,
        }
