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
        highs = np.array([float(c[2]) for c in candles], dtype=float)
        lows = np.array([float(c[3]) for c in candles], dtype=float)
        volumes = np.array([float(c[5]) for c in entry_candles], dtype=float)

        atr = self._compute_atr(entry_candles, period=14) or 0.0
        current_price = float(closes[-1]) if len(closes) else 0.0
        volatility_ratio = (atr / current_price) if current_price > 0 else 0.0

        sma_fast = closes[-10:].mean()
        sma_slow = closes[-25:].mean()
        trend_strength = ((sma_fast - sma_slow) / sma_slow) if sma_slow > 0 else 0.0
        rolling_mean = float(closes[-20:].mean()) if len(closes) >= 20 else current_price
        mean_reversion_score = abs((closes[-1] - rolling_mean) / rolling_mean) if rolling_mean else 0.0
        trend_direction = "bullish" if trend_strength > 0.003 else "bearish" if trend_strength < -0.003 else "flat"
        stretch_from_mean = ((current_price - rolling_mean) / rolling_mean) if rolling_mean else 0.0

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

        if unstable:
            regime = "unstable"
            confidence = 0.9
        elif liquidity_score < 0.75:
            regime = "low_liquidity"
            confidence = 0.7
        elif abs(trend_strength) > 0.01:
            regime = "trending"
            confidence = min(0.95, 0.55 + abs(trend_strength) * 8.0)
        elif mean_reversion_score > 0.012 and volatility_ratio < 0.02 and abs(trend_strength) < 0.008:
            regime = "mean_reverting"
            confidence = min(0.9, 0.55 + mean_reversion_score * 12.0)
        elif volatility_ratio > 0.018:
            regime = "high_volatility"
            confidence = 0.7
        else:
            regime = "choppy"
            confidence = 0.6

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
                "mean_reversion_score": mean_reversion_score,
                "trend_direction": trend_direction,
                "stretch_from_mean": stretch_from_mean,
                "volume_impulse": volume_impulse,
                "atr": atr,
            },
        )

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
