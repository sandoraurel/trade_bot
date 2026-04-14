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
        frequency_context: Optional[Dict[str, Any]] = None,
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
            metadata = dict(getattr(signal, "metadata", {}) or {})
            learning_adjustment = dict(metadata.get("learning_context", {}) or {})
            calibration = dict(learning_adjustment.get("calibration", {}) or {})
            regime_fit_bonus = self._regime_fit_bonus(signal, regime)
            rotation_adjustment = self._rotation_adjustment(signal, regime)
            research_adjustment = self._research_adjustment(signal, research_context)
            rr_ratio = self._reward_risk_ratio(signal)
            expected_win_probability = self._expected_win_probability(signal, regime, calibration, learning_adjustment)
            expected_r = (expected_win_probability * rr_ratio) - (1.0 - expected_win_probability)
            expected_reward_bps = self._reward_bps(signal)
            momentum_crash_risk = float(getattr(regime, "metadata", {}).get("momentum_crash_risk", 0.0) or 0.0)
            cost_buffer = (
                spread_cost_bps
                + slippage_cost_bps
                + exposure_penalty
                + float(metadata.get("entry_deviation_bps", 0.0) or 0.0) * 0.35
            )
            if str(metadata.get("preferred_order_type", "")).lower() == "market":
                cost_buffer += 4.0
            strategy_variant = str(metadata.get("strategy_variant", "base"))
            breakout_style_bonus = 0.0
            if signal.strategy == "trend_breakout" and strategy_variant == "confirmed_market":
                breakout_style_bonus += 2.5
            if signal.strategy == "trend_breakout" and str(getattr(regime, "metadata", {}).get("trend_direction", "flat")) == ("bullish" if signal.side == "long" else "bearish"):
                breakout_style_bonus += 2.0
            diversification_bonus = 0.0
            if signal.strategy == "trend_pullback":
                diversification_bonus += 4.0
            elif signal.strategy == "mean_reversion":
                diversification_bonus += 5.0
            frequency_adjustment = self._frequency_adjustment(signal.strategy, frequency_context)
            event_penalty = 12.0 if getattr(regime, "event_risk", False) else 0.0
            liquidity_penalty = max(0.0, (0.9 - float(getattr(regime, "liquidity_score", 1.0))) * 30.0)
            if signal.strategy == "trend_breakout" and strategy_variant == "confirmed_market":
                liquidity_penalty += 4.0
            crash_penalty = 0.0
            if signal.strategy == "trend_breakout":
                crash_penalty += momentum_crash_risk * 18.0
            elif signal.strategy == "trend_pullback":
                crash_penalty += momentum_crash_risk * 5.0
            elif signal.strategy == "mean_reversion":
                crash_penalty -= min(momentum_crash_risk * 4.0, 3.0)
            net_expectancy_bps = (expected_reward_bps * expected_win_probability) - (cost_buffer + event_penalty + liquidity_penalty)
            ensemble_score = (
                (expected_r * 95.0)
                + net_expectancy_bps
                + breakout_style_bonus
                + diversification_bonus
                + frequency_adjustment["score_delta"]
                + regime_fit_bonus
                + rotation_adjustment["score_delta"]
                - crash_penalty
                + research_adjustment["score_delta"]
                + float(learning_adjustment.get("score_delta", 0.0) or 0.0)
                + (signal.confidence * 35.0)
            )
            proposal_rows.append(
                {
                    "strategy": signal.strategy,
                    "confidence": signal.confidence,
                    "calibrated_confidence": expected_win_probability,
                    "expected_edge_bps": proposal.expected_edge_bps,
                    "rr_ratio": rr_ratio,
                    "expected_r": expected_r,
                    "net_expectancy_bps": net_expectancy_bps,
                    "regime_fit_bonus": regime_fit_bonus,
                    "rotation_score_delta": rotation_adjustment["score_delta"],
                    "rotation_reason": rotation_adjustment["reason"],
                    "research_score_delta": research_adjustment["score_delta"],
                    "research_reason": research_adjustment["reason"],
                    "learning_score_delta": float(learning_adjustment.get("score_delta", 0.0) or 0.0),
                    "learning_reasons": list(learning_adjustment.get("reasons", []) or []),
                    "cost_buffer_bps": cost_buffer,
                    "breakout_style_bonus": breakout_style_bonus,
                    "diversification_bonus": diversification_bonus,
                    "frequency_score_delta": frequency_adjustment["score_delta"],
                    "frequency_min_net_expectancy_delta": frequency_adjustment["min_net_expectancy_delta"],
                    "frequency_reason": frequency_adjustment["reason"],
                    "momentum_crash_risk": momentum_crash_risk,
                    "crash_penalty": crash_penalty,
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
            if signal.strategy == "trend_breakout" and momentum_crash_risk >= 0.72:
                rejected.append(f"{signal.strategy}:momentum_crash_risk")
                continue
            if rr_ratio < 1.05:
                rejected.append(f"{signal.strategy}:rr_too_low")
                continue
            minimum_net_expectancy = float(getattr(self.config, "min_expected_edge_bps", 8.0))
            if signal.strategy == "trend_breakout" and strategy_variant == "confirmed_market":
                minimum_net_expectancy = minimum_net_expectancy + 4.0
            elif signal.strategy == "trend_pullback":
                minimum_net_expectancy = max(minimum_net_expectancy - 1.0, 6.0)
            elif signal.strategy == "mean_reversion":
                minimum_net_expectancy = max(minimum_net_expectancy - 1.5, 5.5)
            minimum_net_expectancy += frequency_adjustment["min_net_expectancy_delta"]
            portfolio_gross = float(getattr(portfolio_snapshot, "gross_exposure", 0.0) or 0.0) if portfolio_snapshot is not None else 0.0
            portfolio_balance = float(getattr(portfolio_snapshot, "balance", 0.0) or 0.0) if portfolio_snapshot is not None else 0.0
            if portfolio_balance > 0:
                gross_fraction = portfolio_gross / portfolio_balance
                if gross_fraction >= 0.45:
                    minimum_net_expectancy += min((gross_fraction - 0.45) * 20.0, 6.0)
            if expected_r <= 0.12 or net_expectancy_bps <= minimum_net_expectancy:
                rejected.append(f"{signal.strategy}:negative_net_expectancy")
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
        best_signal.metadata["cross_sectional_score"] = best_score
        best_signal.metadata["proposal_count"] = len(proposals)
        return EnsembleDecision(
            signal=best_signal,
            proposals=proposal_rows,
            selected_strategy=best_signal.strategy,
            regime={"regime": regime.regime, "confidence": regime.confidence},
            rejected_reasons=rejected,
        )

    @staticmethod
    def _frequency_adjustment(strategy: str, frequency_context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        if not frequency_context:
            return {"score_delta": 0.0, "min_net_expectancy_delta": 0.0, "reason": "inactive"}
        status = str(frequency_context.get("status", "inactive") or "inactive")
        dominant_strategy = str(frequency_context.get("dominant_strategy", "") or "")
        strategy_share = float(frequency_context.get("strategy_trade_share", {}).get(strategy, 0.0) or 0.0)
        if status in {"far_below_target", "below_target"}:
            if strategy == "trend_pullback":
                if status == "far_below_target":
                    return {"score_delta": 5.0, "min_net_expectancy_delta": -2.0, "reason": "below_target_pullback_relax"}
                return {"score_delta": 3.0, "min_net_expectancy_delta": -1.0, "reason": "soft_below_target_pullback_relax"}
            if strategy == "mean_reversion":
                if status == "far_below_target":
                    return {"score_delta": 6.0, "min_net_expectancy_delta": -2.5, "reason": "below_target_mean_reversion_relax"}
                return {"score_delta": 3.5, "min_net_expectancy_delta": -1.25, "reason": "soft_below_target_mean_reversion_relax"}
            if strategy == "trend_breakout":
                return {"score_delta": -1.0, "min_net_expectancy_delta": 0.5, "reason": "below_target_breakout_not_preferred"}
        if status in {"above_target", "far_above_target"}:
            severity = 2.0 if status == "above_target" else 4.0
            if strategy == dominant_strategy or (strategy_share >= 0.5 and strategy_share > 0.0):
                return {
                    "score_delta": -severity,
                    "min_net_expectancy_delta": severity * 0.75,
                    "reason": "over_target_dominant_strategy_tighten",
                }
            if strategy == "trend_breakout":
                return {
                    "score_delta": -(severity + 1.0),
                    "min_net_expectancy_delta": (severity * 0.75) + 0.5,
                    "reason": "over_target_breakout_tighten",
                }
        return {"score_delta": 0.0, "min_net_expectancy_delta": 0.0, "reason": "neutral"}

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

    @staticmethod
    def _rotation_adjustment(signal: Signal, regime: Any) -> Dict[str, Any]:
        metadata = dict(getattr(regime, "metadata", {}) or {})
        policy = dict(metadata.get("rotation_policy", {}) or {})
        preferred_family = str(policy.get("preferred_family", metadata.get("preferred_family", "")) or "")
        suppressed_family = str(policy.get("suppressed_family", metadata.get("suppressed_family", "")) or "")
        confidence = float(policy.get("confidence", metadata.get("rotation_confidence", 0.0)) or 0.0)
        if confidence <= 0.0:
            return {"score_delta": 0.0, "reason": "rotation_inactive"}
        if signal.strategy == preferred_family:
            return {"score_delta": 16.0 * confidence, "reason": f"preferred_family:{preferred_family}"}
        if signal.strategy == suppressed_family:
            return {"score_delta": -16.5 * confidence, "reason": f"suppressed_family:{suppressed_family}"}
        return {"score_delta": 0.0, "reason": "rotation_neutral"}

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

    @staticmethod
    def _reward_bps(signal: Signal) -> float:
        entry = float(signal.entry_price)
        if entry <= 0:
            return 0.0
        if signal.side == "short":
            return max((entry - float(signal.take_profit)) / entry * 10000.0, 0.0)
        return max((float(signal.take_profit) - entry) / entry * 10000.0, 0.0)

    @staticmethod
    def _expected_win_probability(
        signal: Signal,
        regime: Any,
        calibration: Dict[str, Any],
        learning_context: Dict[str, Any],
    ) -> float:
        calibrated = float(calibration.get("calibrated_confidence", signal.confidence) or signal.confidence)
        regime_conf = float(getattr(regime, "confidence", 0.0) or 0.0)
        drift = dict(learning_context.get("drift", {}) or {})
        opportunity = dict(learning_context.get("opportunity", {}) or {})
        probability = calibrated
        probability += min(max(regime_conf - 0.6, -0.08), 0.08) * 0.35
        probability += min(max(float(opportunity.get("avg_forward_r", 0.0) or 0.0), -0.35), 0.35) * 0.08
        probability -= min(max(float(drift.get("severity", 0.0) or 0.0), 0.0), 1.5) * 0.02
        return max(min(probability, 0.82), 0.18)
