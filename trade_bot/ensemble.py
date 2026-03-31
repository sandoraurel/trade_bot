from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .models import Signal


@dataclass
class EnsembleDecision:
    signal: Optional[Signal]
    proposals: List[Dict[str, Any]]
    selected_strategy: Optional[str]
    regime: Dict[str, Any]
    rejected_reasons: List[str]


class EnsembleAllocator:
    def __init__(self, config: Any):
        self.config = config

    def choose(
        self,
        symbol: str,
        regime: Any,
        proposals: List[Any],
        portfolio_snapshot: Optional[Any] = None,
        research_context: Optional[Dict[str, Any]] = None,
    ) -> EnsembleDecision:
        rejected: List[str] = []
        proposal_rows: List[Dict[str, Any]] = []
        best_signal: Optional[Signal] = None
        best_score = float("-inf")

        spread_cost_bps = float(getattr(self.config, "backtest_spread_bps", 4.0))
        slippage_cost_bps = float(getattr(self.config, "backtest_slippage_bps", 5.0))
        minimum_edge_bps = float(getattr(self.config, "min_expected_edge_bps", 8.0))
        exposure_penalty = 0.0
        if portfolio_snapshot is not None and getattr(portfolio_snapshot, "balance", 0.0) > 0:
            exposure_penalty = (float(getattr(portfolio_snapshot, "gross_exposure", 0.0)) / float(portfolio_snapshot.balance)) * 10.0

        for proposal in proposals:
            signal = proposal.signal
            cost_buffer = spread_cost_bps + slippage_cost_bps + exposure_penalty
            rr_ratio = self._reward_risk_ratio(signal)
            regime_fit_bonus = self._regime_fit_bonus(signal, regime)
            research_adjustment = self._research_adjustment(signal, research_context)
            learning_adjustment = dict(getattr(signal, "metadata", {}).get("learning_context", {}) or {})
            event_penalty = 12.0 if getattr(regime, "event_risk", False) else 0.0
            liquidity_penalty = max(0.0, (0.9 - float(getattr(regime, "liquidity_score", 1.0))) * 30.0)
            ensemble_score = (
                (signal.confidence * 100.0)
                + proposal.expected_edge_bps
                + (rr_ratio * 8.0)
                + regime_fit_bonus
                + research_adjustment["score_delta"]
                + float(learning_adjustment.get("score_delta", 0.0) or 0.0)
                - cost_buffer
                - event_penalty
                - liquidity_penalty
            )
            proposal_rows.append(
                {
                    "strategy": signal.strategy,
                    "confidence": signal.confidence,
                    "expected_edge_bps": proposal.expected_edge_bps,
                    "rr_ratio": rr_ratio,
                    "regime_fit_bonus": regime_fit_bonus,
                    "research_score_delta": research_adjustment["score_delta"],
                    "research_reason": research_adjustment["reason"],
                    "learning_score_delta": float(learning_adjustment.get("score_delta", 0.0) or 0.0),
                    "learning_reasons": list(learning_adjustment.get("reasons", []) or []),
                    "cost_buffer_bps": cost_buffer,
                    "ensemble_score": ensemble_score,
                }
            )
            if bool(learning_adjustment.get("veto", False)):
                rejected.append(f"{signal.strategy}:learning_veto")
                continue
            if research_adjustment["veto"]:
                rejected.append(f"{signal.strategy}:research_conflict")
                continue
            if regime.unstable:
                rejected.append(f"{signal.strategy}:unstable_regime")
                continue
            if proposal.expected_edge_bps < minimum_edge_bps:
                rejected.append(f"{signal.strategy}:edge_below_threshold")
                continue
            if signal.strategy == "mean_reversion" and regime.regime in {"high_volatility", "unstable"}:
                rejected.append(f"{signal.strategy}:regime_blocked")
                continue
            if rr_ratio < 1.05:
                rejected.append(f"{signal.strategy}:rr_too_low")
                continue
            if ensemble_score > best_score:
                best_score = ensemble_score
                best_signal = signal

        if best_signal is None:
            return EnsembleDecision(
                signal=None,
                proposals=proposal_rows,
                selected_strategy=None,
                regime={"regime": regime.regime, "confidence": regime.confidence},
                rejected_reasons=rejected,
            )

        best_signal.metadata["ensemble_score"] = best_score
        best_signal.metadata["proposal_count"] = len(proposals)
        return EnsembleDecision(
            signal=best_signal,
            proposals=proposal_rows,
            selected_strategy=best_signal.strategy,
            regime={"regime": regime.regime, "confidence": regime.confidence},
            rejected_reasons=rejected,
        )

    @staticmethod
    def _reward_risk_ratio(signal: Signal) -> float:
        risk = abs(float(signal.entry_price) - float(signal.stop_loss))
        reward = abs(float(signal.take_profit) - float(signal.entry_price))
        if risk <= 0:
            return 0.0
        return reward / risk

    @staticmethod
    def _regime_fit_bonus(signal: Signal, regime: Any) -> float:
        regime_name = getattr(regime, "regime", "unknown")
        trend_direction = getattr(regime, "metadata", {}).get("trend_direction", "flat")
        if signal.strategy == "trend_breakout":
            if regime_name == "trending":
                if signal.side == "long" and trend_direction == "bullish":
                    return 12.0
                if signal.side == "short" and trend_direction == "bearish":
                    return 12.0
            if regime_name == "high_volatility":
                return 5.0
        if signal.strategy == "trend_pullback":
            if regime_name == "trending":
                if signal.side == "long" and trend_direction == "bullish":
                    return 11.0
                if signal.side == "short" and trend_direction == "bearish":
                    return 11.0
            if regime_name == "high_volatility":
                return 4.0
        if signal.strategy == "mean_reversion":
            if regime_name == "mean_reverting":
                return 10.0
            if regime_name == "choppy":
                return 4.0
        return 0.0

    def _research_adjustment(self, signal: Signal, research_context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        if not research_context:
            return {"score_delta": 0.0, "veto": False, "reason": "none"}

        risk_off = float(research_context.get("risk_off_confidence", 0.0) or 0.0)
        bullish = float(research_context.get("bullish_confidence", 0.0) or 0.0)
        bearish = float(research_context.get("bearish_confidence", 0.0) or 0.0)
        if risk_off >= float(getattr(self.config, "research_risk_off_veto_confidence", 0.75)):
            return {"score_delta": -25.0, "veto": True, "reason": "risk_off_veto"}

        alignment_bonus = float(getattr(self.config, "research_alignment_bonus", 10.0))
        conflict_penalty = float(getattr(self.config, "research_conflict_penalty", 14.0))
        conflict_veto_confidence = float(getattr(self.config, "research_conflict_veto_confidence", 0.72))

        if signal.side == "long":
            if bearish >= conflict_veto_confidence:
                return {"score_delta": -(conflict_penalty * bearish), "veto": True, "reason": "bearish_veto"}
            return {"score_delta": (alignment_bonus * bullish) - (conflict_penalty * bearish), "veto": False, "reason": "long_bias"}
        if signal.side == "short":
            if bullish >= conflict_veto_confidence:
                return {"score_delta": -(conflict_penalty * bullish), "veto": True, "reason": "bullish_veto"}
            return {"score_delta": (alignment_bonus * bearish) - (conflict_penalty * bullish), "veto": False, "reason": "short_bias"}
        return {"score_delta": 0.0, "veto": False, "reason": "flat"}
