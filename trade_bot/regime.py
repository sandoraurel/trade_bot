from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np


@dataclass
class RegimeAssessment:
    regime: str
    confidence: float
    volatility_ratio: float
    trend_strength: float
    liquidity_score: float
    event_risk: bool
    unstable: bool
    metadata: Dict[str, Any]


class MarketRegimeEngine:
    def __init__(self, config: Any, exch: Any):
        self.config = config
        self.exch = exch

    def classify(self, symbol: str) -> RegimeAssessment:
        candles = self.exch.fetch_ohlcv(symbol, "1h", limit=60) or []
        entry_candles = self.exch.fetch_ohlcv(symbol, self.config.timeframes.get("entry", "15m"), limit=60) or []
        if len(candles) < 25 or len(entry_candles) < 25:
            return RegimeAssessment(
                regime="unstable",
                confidence=0.2,
                volatility_ratio=0.0,
                trend_strength=0.0,
                liquidity_score=0.0,
                event_risk=False,
                unstable=True,
                metadata={"reason": "insufficient_data"},
            )

        closes = np.array([float(c[4]) for c in candles], dtype=float)
        entry_closes = np.array([float(c[4]) for c in entry_candles], dtype=float)
        volumes = np.array([float(c[5]) for c in entry_candles], dtype=float)
        hourly_returns = self._returns(closes)
        entry_returns = self._returns(entry_closes)

        atr = self._compute_atr(entry_candles, period=14) or 0.0
        current_price = float(closes[-1]) if len(closes) else 0.0
        volatility_ratio = (atr / current_price) if current_price > 0 else 0.0

        sma_fast = closes[-10:].mean()
        sma_slow = closes[-25:].mean()
        trend_strength = ((sma_fast - sma_slow) / sma_slow) if sma_slow > 0 else 0.0
        rolling_mean = float(closes[-20:].mean()) if len(closes) >= 20 else current_price
        entry_mean = float(entry_closes[-20:].mean()) if len(entry_closes) >= 20 else float(entry_closes[-1])
        entry_std = float(entry_closes[-20:].std()) if len(entry_closes) >= 20 else 0.0
        entry_zscore = ((float(entry_closes[-1]) - entry_mean) / entry_std) if entry_std > 1e-9 else 0.0
        directional_efficiency = self._efficiency_ratio(entry_closes, lookback=min(20, len(entry_closes) - 1))
        short_trend_strength = 0.0
        if len(entry_closes) >= 21 and entry_mean > 0:
            short_trend_strength = (
                (float(entry_closes[-8:].mean()) - float(entry_closes[-21:].mean())) / float(entry_closes[-21:].mean())
            )
        stretch_from_mean = ((current_price - rolling_mean) / rolling_mean) if rolling_mean else 0.0
        trailing_peak = float(np.max(closes[-20:])) if len(closes) >= 20 else current_price
        recent_trough = float(np.min(closes[-8:])) if len(closes) >= 8 else current_price
        recent_drawdown = ((current_price - trailing_peak) / trailing_peak) if trailing_peak else 0.0
        rebound_from_trough = ((current_price - recent_trough) / recent_trough) if recent_trough else 0.0
        realized_vol_percentile = self._volatility_percentile(entry_returns, window=18)
        semivariance_skew = self._semivariance_skew(entry_returns, lookback=20)
        exhaustion_score = self._exhaustion_score(entry_returns, entry_zscore, semivariance_skew)
        trend_persistence = self._trend_persistence(entry_returns, lookback=12)
        ema_fast_entry = self._ema(entry_closes, 8)
        ema_slow_entry = self._ema(entry_closes, 21)
        ema_alignment = 0.0
        if len(ema_fast_entry) >= 2 and len(ema_slow_entry) >= 2:
            ema_alignment = (
                np.sign(ema_fast_entry[-1] - ema_slow_entry[-1])
                * min(abs(float(ema_fast_entry[-1] - ema_slow_entry[-1])) / max(float(entry_closes[-1]), 1e-9) * 150.0, 1.0)
            )
        direction_bias = (trend_strength * 95.0) + (short_trend_strength * 160.0) + (ema_alignment * 0.8)
        if direction_bias > 0.18:
            trend_direction = "bullish"
        elif direction_bias < -0.18:
            trend_direction = "bearish"
        else:
            trend_direction = "flat"

        avg_volume = volumes.mean() if len(volumes) else 0.0
        recent_volume = volumes[-6:].mean() if len(volumes) >= 6 else avg_volume
        liquidity_score = (recent_volume / avg_volume) if avg_volume > 0 else 1.0
        volume_impulse = liquidity_score

        spread = 0.0
        try:
            order_book = self.exch.get_order_book(symbol)
            mid = (order_book["bid"] + order_book["ask"]) / 2.0
            if mid > 0:
                spread = (order_book["ask"] - order_book["bid"]) / mid
        except Exception:
            spread = 0.0

        event_risk = spread > float(getattr(self.config, "spread_shock_halt_fraction", 0.006))
        unstable = volatility_ratio > float(getattr(self.config, "emergency_volatility_threshold", 0.08)) or event_risk
        momentum_crash_risk = 0.0
        if recent_drawdown <= -0.045 and rebound_from_trough >= 0.012:
            momentum_crash_risk += 0.6
        if realized_vol_percentile >= 0.82:
            momentum_crash_risk += 0.2
        if abs(entry_zscore) >= 1.0:
            momentum_crash_risk += 0.1
        if trend_persistence < 0.42:
            momentum_crash_risk += 0.1
        momentum_crash_risk = min(momentum_crash_risk, 1.0)

        trend_score = (
            abs(trend_strength) * 120.0
            + directional_efficiency * 1.5
            + trend_persistence * 1.15
            + max(abs(ema_alignment) - 0.1, 0.0) * 0.7
            + max(volume_impulse - 0.95, 0.0) * 0.35
            - max(abs(entry_zscore) - 1.9, 0.0) * 0.3
        )
        continuation_score = (
            trend_score
            + max(0.45 - abs(semivariance_skew), 0.0) * 0.2
            - max(realized_vol_percentile - 0.92, 0.0) * 0.8
            - max(abs(stretch_from_mean) - 0.045, 0.0) * 12.0
        )
        breakout_score = (
            continuation_score
            + max(realized_vol_percentile - 0.48, 0.0) * 0.55
            + max(abs(semivariance_skew) - 0.08, 0.0) * 0.3
        )
        pullback_score = (
            continuation_score
            + max(0.03 - abs(stretch_from_mean), 0.0) * 8.0
            + max(0.45 - abs(entry_zscore), 0.0) * 0.15
        )
        mean_reversion_score = (
            (1.0 - min(directional_efficiency, 1.0)) * 1.5
            + abs(entry_zscore) * 0.38
            + exhaustion_score * 0.95
            + max(0.48 - trend_persistence, 0.0) * 0.8
            - abs(trend_strength) * 85.0
            - max(realized_vol_percentile - 0.88, 0.0) * 1.2
        )

        if unstable:
            regime = "unstable"
            confidence = 0.9
        elif (
            liquidity_score < 0.55
            or (
                liquidity_score < 0.72
                and spread >= max(float(getattr(self.config, "max_spread_fraction", 0.002)) * 0.75, 0.0008)
            )
        ):
            regime = "low_liquidity"
            confidence = min(0.82, 0.58 + max(0.72 - liquidity_score, 0.0) * 0.65 + min(spread * 140.0, 0.18))
        elif (
            trend_direction != "flat"
            and trend_score >= float(getattr(self.config, "regime_trend_score_threshold", 1.5))
            and continuation_score >= float(getattr(self.config, "regime_continuation_score_threshold", 1.4))
        ):
            regime = "trending"
            confidence = min(0.95, 0.55 + (continuation_score - 1.0) * 0.18 + min(abs(direction_bias), 0.18))
        elif (
            mean_reversion_score >= float(getattr(self.config, "regime_mean_reversion_score_threshold", 1.3))
            and abs(entry_zscore) >= float(getattr(self.config, "regime_mean_reversion_zscore_threshold", 1.0))
            and abs(trend_strength) <= 0.014
            and realized_vol_percentile <= 0.86
        ):
            regime = "mean_reverting"
            confidence = min(0.9, 0.52 + (mean_reversion_score - 1.0) * 0.18 + min(abs(entry_zscore) * 0.04, 0.12))
        elif realized_vol_percentile >= float(getattr(self.config, "regime_high_volatility_percentile", 0.78)):
            regime = "high_volatility"
            confidence = min(0.88, 0.58 + max(realized_vol_percentile - 0.75, 0.0) * 0.55 + min(abs(direction_bias), 0.12))
        else:
            regime = "choppy"
            confidence = min(0.72, 0.52 + max(0.42 - directional_efficiency, 0.0) * 0.35 + max(0.02 - abs(trend_strength), 0.0) * 4.0)

        rotation_policy = self._rotation_policy(
            regime=regime,
            trend_direction=trend_direction,
            continuation_score=continuation_score,
            mean_reversion_score=mean_reversion_score,
            breakout_score=breakout_score,
            pullback_score=pullback_score,
            momentum_crash_risk=momentum_crash_risk,
            liquidity_score=liquidity_score,
            directional_efficiency=directional_efficiency,
            realized_vol_percentile=realized_vol_percentile,
            stretch_from_mean=stretch_from_mean,
        )

        return RegimeAssessment(
            regime=regime,
            confidence=confidence,
            volatility_ratio=volatility_ratio,
            trend_strength=trend_strength,
            liquidity_score=liquidity_score,
            event_risk=event_risk,
            unstable=unstable,
            metadata={
                "spread": spread,
                "direction_bias": direction_bias,
                "trend_score": trend_score,
                "continuation_score": continuation_score,
                "breakout_score": breakout_score,
                "pullback_score": pullback_score,
                "mean_reversion_score": mean_reversion_score,
                "trend_direction": trend_direction,
                "stretch_from_mean": stretch_from_mean,
                "volume_impulse": volume_impulse,
                "atr": atr,
                "entry_zscore": entry_zscore,
                "directional_efficiency": directional_efficiency,
                "short_trend_strength": short_trend_strength,
                "trend_persistence": trend_persistence,
                "realized_vol_percentile": realized_vol_percentile,
                "semivariance_skew": semivariance_skew,
                "exhaustion_score": exhaustion_score,
                "ema_alignment": ema_alignment,
                "recent_drawdown": recent_drawdown,
                "rebound_from_trough": rebound_from_trough,
                "momentum_crash_risk": momentum_crash_risk,
                "rotation_policy": rotation_policy,
                "preferred_family": rotation_policy["preferred_family"],
                "suppressed_family": rotation_policy["suppressed_family"],
                "rotation_confidence": rotation_policy["confidence"],
            },
        )

    @staticmethod
    def _rotation_policy(
        *,
        regime: str,
        trend_direction: str,
        continuation_score: float,
        mean_reversion_score: float,
        breakout_score: float,
        pullback_score: float,
        momentum_crash_risk: float,
        liquidity_score: float,
        directional_efficiency: float,
        realized_vol_percentile: float,
        stretch_from_mean: float,
    ) -> Dict[str, Any]:
        preferred_family = "trend_pullback"
        suppressed_family = "trend_breakout"
        reason = "default_pullback_bias"
        confidence = 0.55

        if regime == "low_liquidity":
            preferred_family = "mean_reversion"
            suppressed_family = "trend_breakout"
            reason = "low_liquidity_passive_bias"
            confidence = 0.72
        elif regime == "mean_reverting":
            preferred_family = "mean_reversion"
            suppressed_family = "trend_breakout"
            reason = "explicit_mean_reversion_regime"
            confidence = 0.78
        elif regime == "choppy":
            if mean_reversion_score >= 1.15 and abs(stretch_from_mean) >= 0.003:
                preferred_family = "mean_reversion"
                suppressed_family = "trend_breakout"
                reason = "choppy_stretched_reversal_bias"
                confidence = 0.72
            else:
                preferred_family = "trend_pullback"
                suppressed_family = "trend_breakout"
                reason = "choppy_pullback_bias"
                confidence = 0.60
        elif regime == "trending":
            if momentum_crash_risk >= 0.6:
                preferred_family = "trend_pullback"
                suppressed_family = "trend_breakout"
                reason = "trend_crash_risk_pullback_bias"
                confidence = min(0.88, 0.66 + (momentum_crash_risk - 0.6) * 0.40)
            elif breakout_score > pullback_score + 0.18 and directional_efficiency >= 0.38 and realized_vol_percentile <= 0.82:
                preferred_family = "trend_breakout"
                suppressed_family = "mean_reversion"
                reason = "clean_trend_breakout_bias"
                confidence = min(0.86, 0.62 + max(breakout_score - pullback_score, 0.0) * 0.12)
            else:
                preferred_family = "trend_pullback"
                suppressed_family = "mean_reversion"
                reason = "trend_pullback_bias"
                confidence = min(0.84, 0.60 + max(pullback_score - 1.0, 0.0) * 0.10)
        elif regime == "high_volatility":
            preferred_family = "trend_pullback"
            suppressed_family = "trend_breakout"
            reason = "high_volatility_pullback_bias"
            confidence = 0.74
        elif regime == "unstable":
            preferred_family = "mean_reversion"
            suppressed_family = "trend_breakout"
            reason = "unstable_avoid_breakout"
            confidence = 0.70

        if liquidity_score < 0.70 and preferred_family == "trend_breakout":
            preferred_family = "trend_pullback"
            suppressed_family = "trend_breakout"
            reason = "breakout_degraded_by_liquidity"
            confidence = max(confidence, 0.70)

        return {
            "preferred_family": preferred_family,
            "suppressed_family": suppressed_family,
            "confidence": float(min(max(confidence, 0.0), 1.0)),
            "reason": reason,
            "trend_direction": trend_direction,
            "regime": regime,
        }

    @staticmethod
    def _efficiency_ratio(values: np.ndarray, lookback: int = 10) -> float:
        if len(values) <= lookback or lookback <= 0:
            return 0.0
        window = values[-(lookback + 1):]
        path = float(np.abs(np.diff(window)).sum())
        if path <= 0:
            return 0.0
        direct = abs(float(window[-1]) - float(window[0]))
        return direct / path

    @staticmethod
    def _returns(values: np.ndarray) -> np.ndarray:
        if len(values) < 2:
            return np.array([], dtype=float)
        safe = np.maximum(values, 1e-9)
        return np.diff(np.log(safe))

    @staticmethod
    def _ema(values: np.ndarray, period: int) -> np.ndarray:
        if len(values) == 0:
            return values
        alpha = 2.0 / (period + 1.0)
        ema = np.empty(len(values), dtype=float)
        ema[0] = float(values[0])
        for idx in range(1, len(values)):
            ema[idx] = (alpha * float(values[idx])) + ((1.0 - alpha) * ema[idx - 1])
        return ema

    @staticmethod
    def _volatility_percentile(returns: np.ndarray, window: int = 18) -> float:
        if len(returns) < window + 3:
            return 0.5
        realized = np.array(
            [float(np.std(returns[idx - window:idx])) for idx in range(window, len(returns) + 1)],
            dtype=float,
        )
        current = float(realized[-1])
        if current <= 0:
            return 0.0
        return float(np.mean(realized <= current))

    @staticmethod
    def _semivariance_skew(returns: np.ndarray, lookback: int = 20) -> float:
        if len(returns) < max(lookback, 3):
            return 0.0
        window = returns[-lookback:]
        upside = float(np.square(np.clip(window, 0.0, None)).mean())
        downside = float(np.square(np.clip(window, None, 0.0)).mean())
        total = upside + downside
        if total <= 1e-12:
            return 0.0
        return (upside - downside) / total

    @staticmethod
    def _trend_persistence(returns: np.ndarray, lookback: int = 12) -> float:
        if len(returns) < max(lookback, 3):
            return 0.0
        window = returns[-lookback:]
        direction = np.sign(window)
        effective = float(np.mean(direction == np.sign(window.mean())))
        drift = abs(float(window.mean())) / max(float(window.std()), 1e-6)
        return min(max((effective * 0.65) + min(drift * 0.18, 0.35), 0.0), 1.0)

    @staticmethod
    def _exhaustion_score(returns: np.ndarray, zscore: float, semivariance_skew: float) -> float:
        if len(returns) == 0:
            return min(abs(zscore) * 0.4, 1.0)
        recent = returns[-min(len(returns), 8):]
        tail_move = abs(float(np.sum(recent)))
        impulse = abs(float(recent[-1])) / max(float(np.std(recent)), 1e-6)
        return min((abs(zscore) * 0.35) + (abs(semivariance_skew) * 0.35) + min(tail_move * 35.0, 0.3) + min(impulse * 0.08, 0.25), 1.5)

    @staticmethod
    def _compute_atr(candles: list[list[float]], period: int = 14) -> Optional[float]:
        if len(candles) < period + 1:
            return None
        trs = []
        prev_close = candles[0][4]
        for candle in candles[1:]:
            high = candle[2]
            low = candle[3]
            close = candle[4]
            trs.append(max(high - low, abs(high - prev_close), abs(low - prev_close)))
            prev_close = close
        if not trs:
            return None
        return float(sum(trs[-period:]) / period)
