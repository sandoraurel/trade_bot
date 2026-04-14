from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional

import numpy as np

from .models import Signal


@dataclass
class StrategyProposal:
    signal: Signal
    expected_edge_bps: float
    rationale: str


class StrategyBase:
    name = "base"

    def __init__(self, config: Any, exch: Any, helpers: Any):
        self.config = config
        self.exch = exch
        self.helpers = helpers

    def evaluate(self, symbol: str, regime: Any) -> Optional[StrategyProposal]:
        raise NotImplementedError


def _ema(values: np.ndarray, period: int) -> np.ndarray:
    if len(values) == 0:
        return values
    alpha = 2.0 / (period + 1.0)
    result = np.empty(len(values), dtype=float)
    result[0] = float(values[0])
    for idx in range(1, len(values)):
        result[idx] = (alpha * float(values[idx])) + ((1.0 - alpha) * result[idx - 1])
    return result


def _efficiency_ratio(values: np.ndarray, lookback: int = 10) -> float:
    if len(values) <= lookback:
        return 0.0
    window = values[-(lookback + 1):]
    path = float(np.abs(np.diff(window)).sum())
    if path <= 0:
        return 0.0
    direct = abs(float(window[-1]) - float(window[0]))
    return direct / path


def _candle_body_fraction(candle: list[float]) -> float:
    high = float(candle[2])
    low = float(candle[3])
    open_price = float(candle[1])
    close_price = float(candle[4])
    rng = max(high - low, 1e-9)
    return abs(close_price - open_price) / rng


def _close_location(candle: list[float]) -> float:
    high = float(candle[2])
    low = float(candle[3])
    close_price = float(candle[4])
    rng = max(high - low, 1e-9)
    return (close_price - low) / rng


def _reward_risk_ratio(entry_price: float, stop_loss: float, take_profit: float) -> float:
    risk = abs(float(entry_price) - float(stop_loss))
    reward = abs(float(take_profit) - float(entry_price))
    if risk <= 0:
        return 0.0
    return reward / risk


def _expected_edge_bps(side: str, entry_price: float, stop_loss: float, take_profit: float, win_probability: float) -> float:
    entry_price = float(entry_price)
    if entry_price <= 0:
        return 0.0
    side = (side or "").lower()
    if side == "short":
        reward_bps = max((entry_price - float(take_profit)) / entry_price * 10000.0, 0.0)
        risk_bps = max((float(stop_loss) - entry_price) / entry_price * 10000.0, 0.0)
    else:
        reward_bps = max((float(take_profit) - entry_price) / entry_price * 10000.0, 0.0)
        risk_bps = max((entry_price - float(stop_loss)) / entry_price * 10000.0, 0.0)
    return max((reward_bps * win_probability) - (risk_bps * (1.0 - win_probability)), 0.0)


class TrendBreakoutStrategy(StrategyBase):
    name = "trend_breakout"

    def evaluate(self, symbol: str, regime: Any) -> Optional[StrategyProposal]:
        if regime.regime not in {"trending", "high_volatility"}:
            return None

        candles_15m = self.exch.fetch_ohlcv(symbol, "15m", limit=48)
        lookback = max(int(getattr(self.config, "breakout_swing_lookback", 5)), 3)
        if not candles_15m or len(candles_15m) < max(lookback + 4, 20):
            return None
        prior_candles = candles_15m[:-1]
        swing_high, swing_low = self.helpers.get_recent_swing_high_low(prior_candles, lookback=lookback)
        if swing_high is None or swing_low is None:
            return None

        last_candle = candles_15m[-1]
        last_close = float(last_candle[4])
        last_open = float(last_candle[1])
        atr_15m = self.helpers.compute_atr(symbol, "15m", period=14)
        if atr_15m is None or atr_15m <= 0:
            return None

        closes = np.array([float(c[4]) for c in candles_15m], dtype=float)
        highs = np.array([float(c[2]) for c in candles_15m], dtype=float)
        lows = np.array([float(c[3]) for c in candles_15m], dtype=float)
        ema_fast = _ema(closes, 8)
        ema_slow = _ema(closes, 21)
        ema_anchor = _ema(closes, 34)
        trend_efficiency = _efficiency_ratio(closes, lookback=min(12, len(closes) - 1))
        body_fraction = _candle_body_fraction(last_candle)
        close_location = _close_location(last_candle)
        micro_trend_up = len(ema_fast) > 1 and ema_fast[-1] > ema_slow[-1] and ema_fast[-1] >= ema_fast[-2]
        micro_trend_down = len(ema_fast) > 1 and ema_fast[-1] < ema_slow[-1] and ema_fast[-1] <= ema_fast[-2]

        direction = str(regime.metadata.get("trend_direction", "flat"))
        volume_impulse = float(regime.metadata.get("volume_impulse", 1.0))
        continuation_score = float(regime.metadata.get("continuation_score", 0.0))
        if continuation_score <= 0:
            continuation_score = 1.0 + (float(regime.confidence) * 0.6) + (trend_efficiency * 0.8)
        breakout_score = float(regime.metadata.get("breakout_score", 0.0))
        if breakout_score <= 0:
            breakout_score = continuation_score + max(trend_efficiency - 0.20, 0.0)
        trend_persistence = float(regime.metadata.get("trend_persistence", 0.0))
        if trend_persistence <= 0:
            trend_persistence = min(0.35 + (trend_efficiency * 0.4), 0.75)
        volatility_percentile = float(regime.metadata.get("realized_vol_percentile", 0.5))
        entry_zscore = float(regime.metadata.get("entry_zscore", 0.0))
        semivariance_skew = float(
            regime.metadata.get(
                "semivariance_skew",
                -0.08 if direction == "bearish" else 0.08 if direction == "bullish" else 0.0,
            )
        )
        stretch = float(regime.metadata.get("stretch_from_mean", 0.0))
        recent_drawdown = float(regime.metadata.get("recent_drawdown", 0.0))
        rebound_from_trough = float(regime.metadata.get("rebound_from_trough", 0.0))
        momentum_crash_risk = float(regime.metadata.get("momentum_crash_risk", 0.0))

        breakout_buffer = atr_15m * 0.10
        confirm_buffer = atr_15m * float(getattr(self.config, "breakout_confirm_close_atr_fraction", 0.05))
        retest_buffer = atr_15m * float(getattr(self.config, "breakout_retest_entry_buffer_atr_fraction", 0.04))
        market_breakout_min_distance = atr_15m * float(getattr(self.config, "breakout_confirmed_market_min_breakout_atr", 0.12))
        market_body_floor = float(getattr(self.config, "breakout_confirmed_market_min_body_fraction", 0.58))
        breakout_continuation_floor = float(getattr(self.config, "breakout_confirmed_market_min_continuation_score", 1.58))
        breakout_score_floor = float(getattr(self.config, "breakout_confirmed_market_min_breakout_score", 1.62))
        breakout_persistence_floor = float(getattr(self.config, "breakout_confirmed_market_min_trend_persistence", 0.44))
        breakout_volume_floor = float(getattr(self.config, "breakout_confirmed_market_min_volume_impulse", 1.08))
        breakout_efficiency_floor = float(getattr(self.config, "breakout_confirmed_market_min_trend_efficiency", 0.34))
        post_shock_drawdown_floor = float(getattr(self.config, "breakout_post_shock_drawdown_floor", -0.045))
        post_shock_rebound_floor = float(getattr(self.config, "breakout_post_shock_rebound_floor", 0.012))
        post_shock_crash_risk_floor = float(getattr(self.config, "breakout_post_shock_crash_risk_floor", 0.60))
        weak_confirmation_penalty_bps = float(getattr(self.config, "breakout_weak_confirmation_penalty_bps", 5.0))
        stressed_rebound = (
            recent_drawdown <= post_shock_drawdown_floor
            and rebound_from_trough >= post_shock_rebound_floor
            and momentum_crash_risk >= post_shock_crash_risk_floor
        )

        if direction == "bearish":
            if not self.helpers.is_4h_bearish(symbol) or not self.helpers.is_1h_downtrend(symbol):
                return None
            if not micro_trend_down or ema_slow[-1] >= ema_anchor[-1]:
                return None
            if last_close - breakout_buffer > float(swing_low):
                return None
            if body_fraction < 0.45 or close_location > 0.36:
                return None
            if trend_efficiency < 0.28 or volume_impulse < 1.0:
                return None
            if stretch < -0.022 or entry_zscore < -1.35 or semivariance_skew > -0.02:
                return None

            breakout_distance = max(float(swing_low) - last_close, 0.0)
            if stressed_rebound and breakout_distance > 0.0:
                return None
            confirmed_market = (
                breakout_distance >= max(confirm_buffer, market_breakout_min_distance)
                and continuation_score >= max(float(getattr(self.config, "regime_continuation_score_threshold", 1.4)), breakout_continuation_floor)
                and breakout_score >= breakout_score_floor
                and trend_persistence >= breakout_persistence_floor
                and trend_efficiency >= breakout_efficiency_floor
                and body_fraction >= max(market_body_floor, 0.58)
                and close_location <= min(float(getattr(self.config, "breakout_confirmed_market_max_close_location_short", 0.28)) + 0.04, 0.34)
                and stretch >= -(float(getattr(self.config, "breakout_confirmed_market_max_stretch", 0.032)) + 0.008)
                and entry_zscore >= -0.92
                and volume_impulse >= breakout_volume_floor
                and regime.regime == "trending"
            )
            confirmed_close = (
                breakout_distance >= confirm_buffer
                and continuation_score >= max(float(getattr(self.config, "regime_continuation_score_threshold", 1.4)), 1.55)
                and breakout_score >= 1.70
                and trend_persistence >= 0.38
            )
            retest_ready = (
                float(candles_15m[-2][3]) <= float(swing_low) + (atr_15m * 0.10)
                or breakout_distance > 0.0
            )
            if not confirmed_market and not confirmed_close and not retest_ready:
                return None

            if confirmed_market:
                variant = "confirmed_market"
                preferred_order_type = "market"
                fast_move = True
                entry_price = last_close
                bounce_reference = min(float(np.max(highs[-4:])) + (atr_15m * 0.10), float(ema_slow[-1]) + (atr_15m * 0.16))
                stop_loss = min(entry_price + (atr_15m * 0.92), bounce_reference)
            else:
                variant = "confirmed_retest" if confirmed_close else "breakout_retest"
                preferred_order_type = "limit"
                fast_move = breakout_distance > 0.0
                entry_price = max(last_close, float(swing_low) + (retest_buffer * (0.55 if confirmed_close else 1.0)))
                bounce_reference = min(float(np.max(highs[-4:])) + (atr_15m * 0.10), float(ema_slow[-1]) + (atr_15m * 0.18))
                stop_loss = max(
                    entry_price + (atr_15m * 0.82),
                    bounce_reference,
                )
            if stop_loss <= entry_price:
                return None
            sl_distance = stop_loss - entry_price
            if sl_distance < atr_15m * 0.30 or sl_distance > atr_15m * 1.90:
                return None

            rr_ratio = (
                float(getattr(self.config, "breakout_confirmed_market_rr_ratio", 1.55))
                if preferred_order_type == "market"
                else float(getattr(self.config, "breakout_retest_rr_ratio", 1.7))
            )
            take_profit = entry_price - (sl_distance * rr_ratio)
            if take_profit >= entry_price or take_profit <= 0:
                return None

            weak_confirmation_penalty = 0.0
            weak_confirmation_penalty += max(breakout_volume_floor - volume_impulse, 0.0) * 12.0
            weak_confirmation_penalty += max(breakout_efficiency_floor - trend_efficiency, 0.0) * 18.0
            weak_confirmation_penalty += max((-stretch) - 0.014, 0.0) * 120.0
            weak_confirmation_penalty += max(abs(entry_zscore) - 0.72, 0.0) * 2.5
            weak_confirmation_penalty = min(weak_confirmation_penalty, weak_confirmation_penalty_bps)
            win_probability = min(
                0.63,
                max(
                    0.47,
                    0.45
                    + (continuation_score * 0.05)
                    + min(max(volume_impulse - breakout_volume_floor, 0.0), 0.03)
                    + min(trend_persistence * 0.05, 0.035),
                    - (weak_confirmation_penalty / 100.0),
                ),
            )
            expected_edge_bps = _expected_edge_bps("short", entry_price, stop_loss, take_profit, win_probability)
            expected_edge_bps = max(expected_edge_bps - weak_confirmation_penalty, 0.0)
            confidence = min(0.78, max(0.51, 0.47 + (continuation_score * 0.07) + min(max(volume_impulse - breakout_volume_floor, 0.0), 0.03) - (weak_confirmation_penalty / 100.0)))
            signal = Signal(
                symbol=symbol,
                side="short",
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                strategy=self.name,
                confidence=confidence,
                timeframe="15m",
                expected_holding_minutes=6 * 60 if preferred_order_type == "market" else 8 * 60,
                expected_edge_bps=expected_edge_bps,
                fast_move=fast_move,
                is_futures=False,
                regime=regime.regime,
                metadata={
                    "regime_confidence": regime.confidence,
                    "trend_direction": direction,
                    "volume_impulse": volume_impulse,
                    "trend_efficiency": trend_efficiency,
                    "body_fraction": body_fraction,
                    "stretch_from_mean": stretch,
                    "breakout_level": float(swing_low),
                    "breakout_score": breakout_score,
                    "trend_persistence": trend_persistence,
                    "entry_zscore": entry_zscore,
                    "realized_vol_percentile": volatility_percentile,
                    "volatility_ratio": float(regime.volatility_ratio),
                    "weak_confirmation_penalty_bps": weak_confirmation_penalty,
                    "strategy_variant": variant,
                    "preferred_order_type": preferred_order_type,
                    "force_limit_entry": preferred_order_type == "limit" and fast_move,
                    "order_expiry_bars": int(getattr(self.config, "breakout_retest_expiry_bars", 6)) if preferred_order_type == "limit" else 2,
                    "stale_cancel_distance_bps": 32.0 if preferred_order_type == "limit" else 18.0,
                },
            )
            return StrategyProposal(signal=signal, expected_edge_bps=expected_edge_bps, rationale=f"trend continuation breakdown ({variant})")

        if direction != "bullish":
            return None
        if not self.helpers.is_4h_bullish(symbol) or not self.helpers.is_1h_uptrend(symbol):
            return None
        if not micro_trend_up or ema_slow[-1] <= ema_anchor[-1]:
            return None
        if last_close + breakout_buffer < float(swing_high):
            return None
        if body_fraction < 0.45 or close_location < 0.64:
            return None
        if trend_efficiency < 0.28 or volume_impulse < 1.0:
            return None
        if stretch > 0.022 or entry_zscore > 1.35 or semivariance_skew < 0.02:
            return None
        if last_close <= last_open:
            return None

        breakout_distance = max(last_close - float(swing_high), 0.0)
        if stressed_rebound and breakout_distance > 0.0:
            return None
        confirmed_market = (
            breakout_distance >= max(confirm_buffer, market_breakout_min_distance)
            and continuation_score >= max(float(getattr(self.config, "regime_continuation_score_threshold", 1.4)), breakout_continuation_floor)
            and breakout_score >= breakout_score_floor
            and trend_persistence >= breakout_persistence_floor
            and trend_efficiency >= breakout_efficiency_floor
            and body_fraction >= max(market_body_floor, 0.58)
            and close_location >= max(float(getattr(self.config, "breakout_confirmed_market_min_close_location_long", 0.72)) - 0.04, 0.68)
            and stretch <= float(getattr(self.config, "breakout_confirmed_market_max_stretch", 0.032)) + 0.008
            and entry_zscore <= 0.92
            and volume_impulse >= breakout_volume_floor
            and regime.regime == "trending"
        )
        confirmed_close = (
            breakout_distance >= confirm_buffer
            and continuation_score >= max(float(getattr(self.config, "regime_continuation_score_threshold", 1.4)), 1.55)
            and breakout_score >= 1.70
            and trend_persistence >= 0.48
        )
        retest_ready = (
            float(candles_15m[-2][2]) >= float(swing_high) - (atr_15m * 0.10)
            or breakout_distance > 0.0
        )
        if not confirmed_market and not confirmed_close and not retest_ready:
            return None

        if confirmed_market:
            variant = "confirmed_market"
            preferred_order_type = "market"
            fast_move = True
            entry_price = last_close
            support_reference = max(float(np.min(lows[-4:])) - (atr_15m * 0.10), float(ema_slow[-1]) - (atr_15m * 0.16))
            stop_loss = max(
                entry_price - (atr_15m * 0.92),
                support_reference,
            )
        else:
            variant = "confirmed_retest" if confirmed_close else "breakout_retest"
            preferred_order_type = "limit"
            fast_move = breakout_distance > 0.0
            entry_price = min(last_close, float(swing_high) - (retest_buffer * (0.55 if confirmed_close else 1.0)))
            support_reference = max(float(np.min(lows[-4:])) - (atr_15m * 0.10), float(ema_slow[-1]) - (atr_15m * 0.18))
            stop_loss = min(
                entry_price - (atr_15m * 0.82),
                support_reference,
            )
        if stop_loss <= 0 or stop_loss >= entry_price:
            return None
        sl_distance = entry_price - stop_loss
        if sl_distance < atr_15m * 0.30 or sl_distance > atr_15m * 1.90:
            return None

        rr_ratio = (
            float(getattr(self.config, "breakout_confirmed_market_rr_ratio", 1.55))
            if preferred_order_type == "market"
            else float(getattr(self.config, "breakout_retest_rr_ratio", 1.7))
        )
        take_profit = entry_price + (sl_distance * rr_ratio)
        if take_profit <= entry_price:
            return None

        weak_confirmation_penalty = 0.0
        weak_confirmation_penalty += max(breakout_volume_floor - volume_impulse, 0.0) * 12.0
        weak_confirmation_penalty += max(breakout_efficiency_floor - trend_efficiency, 0.0) * 18.0
        weak_confirmation_penalty += max(stretch - 0.014, 0.0) * 120.0
        weak_confirmation_penalty += max(abs(entry_zscore) - 0.72, 0.0) * 2.5
        weak_confirmation_penalty = min(weak_confirmation_penalty, weak_confirmation_penalty_bps)
        win_probability = min(
            0.63,
            max(
                0.47,
                0.45
                + (continuation_score * 0.05)
                + min(max(volume_impulse - breakout_volume_floor, 0.0), 0.03)
                + min(trend_persistence * 0.05, 0.035),
                - (weak_confirmation_penalty / 100.0),
            ),
        )
        expected_edge_bps = _expected_edge_bps("long", entry_price, stop_loss, take_profit, win_probability)
        expected_edge_bps = max(expected_edge_bps - weak_confirmation_penalty, 0.0)
        confidence = min(0.78, max(0.51, 0.47 + (continuation_score * 0.07) + min(max(volume_impulse - breakout_volume_floor, 0.0), 0.03) - (weak_confirmation_penalty / 100.0)))
        signal = Signal(
            symbol=symbol,
            side="long",
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            strategy=self.name,
            confidence=confidence,
            timeframe="15m",
            expected_holding_minutes=6 * 60 if preferred_order_type == "market" else 8 * 60,
            expected_edge_bps=expected_edge_bps,
            fast_move=fast_move,
            is_futures=False,
            regime=regime.regime,
            metadata={
                "regime_confidence": regime.confidence,
                "trend_direction": direction,
                "volume_impulse": volume_impulse,
                "trend_efficiency": trend_efficiency,
                "body_fraction": body_fraction,
                "stretch_from_mean": stretch,
                "breakout_level": float(swing_high),
                "breakout_score": breakout_score,
                "trend_persistence": trend_persistence,
                "entry_zscore": entry_zscore,
                "realized_vol_percentile": volatility_percentile,
                "volatility_ratio": float(regime.volatility_ratio),
                "weak_confirmation_penalty_bps": weak_confirmation_penalty,
                "strategy_variant": variant,
                "preferred_order_type": preferred_order_type,
                "force_limit_entry": preferred_order_type == "limit" and fast_move,
                "order_expiry_bars": int(getattr(self.config, "breakout_retest_expiry_bars", 6)) if preferred_order_type == "limit" else 2,
                "stale_cancel_distance_bps": 32.0 if preferred_order_type == "limit" else 18.0,
            },
        )
        return StrategyProposal(signal=signal, expected_edge_bps=expected_edge_bps, rationale=f"trend continuation breakout ({variant})")


class TrendPullbackStrategy(StrategyBase):
    name = "trend_pullback"

    def evaluate(self, symbol: str, regime: Any) -> Optional[StrategyProposal]:
        if regime.regime not in {"trending", "high_volatility"}:
            return None

        candles_15m = self.exch.fetch_ohlcv(symbol, "15m", limit=80)
        if len(candles_15m) < 40:
            return None

        closes = np.array([float(c[4]) for c in candles_15m], dtype=float)
        highs = np.array([float(c[2]) for c in candles_15m], dtype=float)
        lows = np.array([float(c[3]) for c in candles_15m], dtype=float)
        atr_15m = self.helpers.compute_atr(symbol, "15m", period=14)
        if atr_15m is None or atr_15m <= 0:
            return None

        fast_period = max(int(getattr(self.config, "pullback_ema_fast_period", 8)), 3)
        slow_period = max(int(getattr(self.config, "pullback_ema_slow_period", 21)), fast_period + 1)
        ema_fast = _ema(closes, fast_period)
        ema_slow = _ema(closes, slow_period)
        ema_anchor = _ema(closes, 34)
        trend_efficiency = _efficiency_ratio(closes, lookback=min(14, len(closes) - 1))
        trend_efficiency = max(trend_efficiency, float(regime.metadata.get("directional_efficiency", trend_efficiency)))

        volume_impulse = float(regime.metadata.get("volume_impulse", 1.0))
        stretch = float(regime.metadata.get("stretch_from_mean", 0.0))
        direction = str(regime.metadata.get("trend_direction", "flat"))
        sparse_regime_metadata = "continuation_score" not in regime.metadata
        continuation_score = float(regime.metadata.get("continuation_score", 0.0))
        if continuation_score <= 0:
            continuation_score = 1.0 + (float(regime.confidence) * 0.55) + (trend_efficiency * 0.6)
        pullback_score = float(regime.metadata.get("pullback_score", 0.0))
        if pullback_score <= 0:
            pullback_score = continuation_score + 0.1
        trend_persistence = float(regime.metadata.get("trend_persistence", 0.0))
        if trend_persistence <= 0:
            trend_persistence = min(0.35 + (trend_efficiency * 0.45), 0.75)
        entry_zscore = float(regime.metadata.get("entry_zscore", 0.0))
        volatility_percentile = float(regime.metadata.get("realized_vol_percentile", 0.5))

        pullback_fraction = float(getattr(self.config, "pullback_entry_atr_fraction", 0.35))
        pullback_threshold = atr_15m * pullback_fraction
        confirmation_buffer = atr_15m * float(getattr(self.config, "pullback_confirmation_buffer_atr_fraction", 0.08))

        last_candle = candles_15m[-1]
        prev_candle = candles_15m[-2]
        last_close = float(last_candle[4])
        prev_close = float(prev_candle[4])
        prev_high = float(prev_candle[2])
        prev_low = float(prev_candle[3])
        body_fraction = _candle_body_fraction(last_candle)
        close_location = _close_location(last_candle)
        recent_high = float(np.max(highs[-10:]))
        recent_low = float(np.min(lows[-10:]))
        recent_low_buffer = float(np.min(lows[-4:]))
        recent_high_buffer = float(np.max(highs[-4:]))

        shallow_pullback_score_floor = float(getattr(self.config, "pullback_shallow_min_pullback_score", 1.24))
        shallow_persistence_floor = float(getattr(self.config, "pullback_shallow_min_trend_persistence", 0.38))
        deep_continuation_floor = float(getattr(self.config, "pullback_deep_min_continuation_score", 1.58))
        shallow_volatility_ceiling = float(getattr(self.config, "pullback_shallow_max_volatility_percentile", 0.72))
        deep_volatility_ceiling = float(getattr(self.config, "pullback_deep_max_volatility_percentile", 0.88))
        deep_efficiency_floor = float(getattr(self.config, "pullback_deep_min_trend_efficiency", 0.16))
        deep_volume_floor = float(getattr(self.config, "pullback_deep_min_volume_impulse", 0.92))
        reclaim_body_fraction_floor = float(getattr(self.config, "pullback_reclaim_body_fraction_min", 0.18))
        reclaim_close_location_min = float(getattr(self.config, "pullback_reclaim_close_location_min", 0.24))
        reclaim_close_location_max = float(getattr(self.config, "pullback_reclaim_close_location_max", 0.76))
        reclaim_buffer = atr_15m * float(getattr(self.config, "pullback_reclaim_buffer_atr_fraction", 0.04))

        if direction == "bullish":
            if not self.helpers.is_4h_bullish(symbol) or not self.helpers.is_1h_uptrend(symbol):
                return None
            if ema_fast[-1] <= ema_slow[-1] or ema_slow[-1] <= ema_anchor[-1]:
                return None
            if trend_efficiency < 0.10 or volume_impulse < 0.86:
                return None
            if stretch > float(getattr(self.config, "pullback_max_stretch", 0.035)):
                return None
            pullback_depth = max(ema_fast[-1] - last_close, 0.0)
            shallow_pullback = (
                pullback_score >= shallow_pullback_score_floor
                and pullback_depth >= pullback_threshold * 0.08
                and pullback_depth <= atr_15m * 0.90
                and recent_low_buffer <= ema_fast[-1] + (atr_15m * 0.10)
                and last_close <= ema_fast[-1] + (atr_15m * 0.05)
                and last_close >= ema_fast[-1] - (atr_15m * 0.30)
                and trend_persistence >= shallow_persistence_floor
                and entry_zscore <= 0.35
                and volatility_percentile <= shallow_volatility_ceiling
            )
            deep_pullback = (
                continuation_score >= deep_continuation_floor
                and pullback_depth >= atr_15m * 0.45
                and pullback_depth <= atr_15m * 1.35
                and recent_low_buffer <= ema_slow[-1] + (atr_15m * 0.18)
                and last_close >= ema_slow[-1] - (atr_15m * 0.06)
                and entry_zscore <= 0.25
                and trend_persistence >= 0.40
                and volatility_percentile <= deep_volatility_ceiling
                and trend_efficiency >= deep_efficiency_floor
                and volume_impulse >= deep_volume_floor
            )
            legacy_pullback = (
                sparse_regime_metadata
                and pullback_depth >= pullback_threshold * 0.08
                and pullback_depth <= atr_15m * 1.30
                and last_close >= ema_slow[-1]
                and last_close > prev_close
                and body_fraction >= 0.30
                and close_location >= 0.30
            )
            if not shallow_pullback and not deep_pullback and not legacy_pullback:
                return None
            bullish_continuation_confirmation = (
                last_close > prev_close
                and last_close >= max(prev_close + (confirmation_buffer * 0.20), ema_fast[-1] - (pullback_threshold * 0.75))
            )
            bullish_reclaim_confirmation = (
                body_fraction >= reclaim_body_fraction_floor
                and close_location >= max(reclaim_close_location_min, 0.28 if shallow_pullback else reclaim_close_location_min)
                and last_close >= (
                    ema_fast[-1] - (pullback_threshold * 0.78)
                    if shallow_pullback
                    else ema_slow[-1] - (atr_15m * 0.14)
                )
                and last_close >= prev_close - reclaim_buffer
                and float(last_candle[3]) >= ema_slow[-1] - (atr_15m * 0.10)
            )
            if not bullish_continuation_confirmation and not bullish_reclaim_confirmation:
                return None

            variant = "shallow_pullback" if shallow_pullback else "deep_pullback" if deep_pullback else "legacy_pullback"
            entry_price = min(last_close, float(ema_fast[-1]) + (atr_15m * 0.02))
            stop_loss = min(
                last_close - (atr_15m * (1.08 if shallow_pullback else 1.22)),
                (recent_low_buffer if shallow_pullback else recent_low) - (atr_15m * 0.12),
                prev_low - (atr_15m * 0.05),
            )
            if stop_loss <= 0 or stop_loss >= entry_price:
                return None
            risk = entry_price - stop_loss
            if risk < atr_15m * 0.24 or risk > atr_15m * 1.90:
                return None

            rr_ratio = (
                float(getattr(self.config, "pullback_shallow_rr_ratio", 1.35))
                if shallow_pullback
                else float(getattr(self.config, "pullback_deep_rr_ratio", 1.2))
            )
            take_profit = max(recent_high, entry_price + (risk * rr_ratio))
            if take_profit <= entry_price:
                return None

            profile_bonus = 0.02 if shallow_pullback and volatility_percentile <= shallow_volatility_ceiling else 0.0
            profile_penalty = 0.02 if deep_pullback and volatility_percentile > 0.74 else 0.0
            win_probability = min(
                0.71,
                max(
                    0.51,
                    0.49 + (pullback_score * 0.08) + min(trend_persistence * 0.08, 0.05) + profile_bonus - profile_penalty,
                ),
            )
            expected_edge_bps = _expected_edge_bps("long", entry_price, stop_loss, take_profit, win_probability)
            confidence = min(
                0.86,
                max(
                    0.56,
                    0.50
                    + (pullback_score * 0.10)
                    + min(max(volume_impulse - 1.0, 0.0), 0.06)
                    + (0.02 if shallow_pullback and volatility_percentile <= shallow_volatility_ceiling else 0.0)
                    - (0.02 if deep_pullback and volatility_percentile > 0.74 else 0.0)
                    + (0.02 if bullish_reclaim_confirmation and not bullish_continuation_confirmation else 0.0),
                ),
            )
            signal = Signal(
                symbol=symbol,
                side="long",
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                strategy=self.name,
                confidence=confidence,
                timeframe="15m",
                expected_holding_minutes=5 * 60,
                expected_edge_bps=expected_edge_bps,
                fast_move=False,
                is_futures=False,
                regime=regime.regime,
                metadata={
                    "trend_direction": direction,
                    "pullback_depth": pullback_depth,
                    "trend_efficiency": trend_efficiency,
                    "volume_impulse": volume_impulse,
                    "continuation_score": continuation_score,
                    "pullback_score": pullback_score,
                    "entry_zscore": entry_zscore,
                    "trend_persistence": trend_persistence,
                    "realized_vol_percentile": volatility_percentile,
                    "volatility_ratio": float(regime.volatility_ratio),
                    "strategy_variant": variant,
                    "profile_preference": "shallow_preferred" if shallow_pullback else "deep_selective" if deep_pullback else "legacy",
                    "confirmation_variant": "continuation" if bullish_continuation_confirmation else "reclaim_hold",
                    "preferred_order_type": "limit",
                    "order_expiry_bars": 4,
                },
            )
            return StrategyProposal(signal=signal, expected_edge_bps=expected_edge_bps, rationale=f"bull trend pullback continuation ({variant})")

        if direction != "bearish":
            return None
        if not self.helpers.is_4h_bearish(symbol) or not self.helpers.is_1h_downtrend(symbol):
            return None
        if ema_fast[-1] >= ema_slow[-1] or ema_slow[-1] >= ema_anchor[-1]:
            return None
            if trend_efficiency < 0.10 or volume_impulse < 0.86:
                return None
        if stretch < -float(getattr(self.config, "pullback_max_stretch", 0.035)):
            return None

        pullback_depth = max(last_close - ema_fast[-1], 0.0)
        shallow_pullback = (
            pullback_score >= shallow_pullback_score_floor
            and pullback_depth >= pullback_threshold * 0.08
            and pullback_depth <= atr_15m * 0.90
            and recent_high_buffer >= ema_fast[-1] - (atr_15m * 0.10)
            and last_close >= ema_fast[-1] - (atr_15m * 0.05)
            and last_close <= ema_fast[-1] + (atr_15m * 0.24)
            and trend_persistence >= shallow_persistence_floor
            and entry_zscore >= -0.25
            and entry_zscore <= 0.60
            and volatility_percentile <= shallow_volatility_ceiling
        )
        deep_pullback = (
            continuation_score >= deep_continuation_floor
            and pullback_depth >= atr_15m * 0.45
            and pullback_depth <= atr_15m * 1.35
            and recent_high_buffer >= ema_slow[-1] - (atr_15m * 0.18)
            and last_close <= ema_slow[-1] + (atr_15m * 0.06)
            and entry_zscore >= -0.20
            and entry_zscore <= 0.45
            and trend_persistence >= 0.40
            and volatility_percentile <= deep_volatility_ceiling
            and trend_efficiency >= deep_efficiency_floor
            and volume_impulse >= deep_volume_floor
        )
        legacy_pullback = (
            sparse_regime_metadata
            and pullback_depth >= pullback_threshold * 0.08
            and pullback_depth <= atr_15m * 1.30
            and last_close <= ema_slow[-1]
            and last_close < prev_close
            and body_fraction >= 0.30
            and close_location <= 0.70
        )
        if not shallow_pullback and not deep_pullback and not legacy_pullback:
            return None
        bearish_continuation_confirmation = (
            last_close < prev_close
            and last_close <= min(prev_close - (confirmation_buffer * 0.20), ema_fast[-1] + (pullback_threshold * 0.75))
        )
        bearish_reclaim_confirmation = (
            body_fraction >= reclaim_body_fraction_floor
            and close_location <= min(reclaim_close_location_max, 0.72 if shallow_pullback else reclaim_close_location_max)
            and last_close <= (
                ema_fast[-1] + (pullback_threshold * 0.78)
                if shallow_pullback
                else ema_slow[-1] + (atr_15m * 0.14)
            )
            and last_close <= prev_close + reclaim_buffer
            and float(last_candle[2]) <= ema_slow[-1] + (atr_15m * 0.10)
        )
        if not bearish_continuation_confirmation and not bearish_reclaim_confirmation:
            return None

        variant = "shallow_pullback" if shallow_pullback else "deep_pullback" if deep_pullback else "legacy_pullback"
        entry_price = max(last_close, float(ema_fast[-1]) - (atr_15m * 0.02))
        stop_loss = max(
            last_close + (atr_15m * (1.08 if shallow_pullback else 1.22)),
            (recent_high_buffer if shallow_pullback else recent_high) + (atr_15m * 0.12),
            prev_high + (atr_15m * 0.05),
        )
        if stop_loss <= entry_price:
            return None
        risk = stop_loss - entry_price
        if risk < atr_15m * 0.24 or risk > atr_15m * 1.90:
            return None

        rr_ratio = (
            float(getattr(self.config, "pullback_shallow_rr_ratio", 1.35))
            if shallow_pullback
            else float(getattr(self.config, "pullback_deep_rr_ratio", 1.2))
        )
        take_profit = min(recent_low, entry_price - (risk * rr_ratio))
        if take_profit >= entry_price or take_profit <= 0:
            return None

        profile_bonus = 0.02 if shallow_pullback and volatility_percentile <= shallow_volatility_ceiling else 0.0
        profile_penalty = 0.02 if deep_pullback and volatility_percentile > 0.74 else 0.0
        win_probability = min(
            0.71,
            max(
                0.51,
                0.49 + (pullback_score * 0.08) + min(trend_persistence * 0.08, 0.05) + profile_bonus - profile_penalty,
            ),
        )
        expected_edge_bps = _expected_edge_bps("short", entry_price, stop_loss, take_profit, win_probability)
        confidence = min(
            0.86,
            max(
                0.56,
                0.50
                + (pullback_score * 0.10)
                + min(max(volume_impulse - 1.0, 0.0), 0.06)
                + (0.02 if shallow_pullback and volatility_percentile <= shallow_volatility_ceiling else 0.0)
                - (0.02 if deep_pullback and volatility_percentile > 0.74 else 0.0)
                + (0.02 if bearish_reclaim_confirmation and not bearish_continuation_confirmation else 0.0),
            ),
        )
        signal = Signal(
            symbol=symbol,
            side="short",
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            strategy=self.name,
            confidence=confidence,
            timeframe="15m",
            expected_holding_minutes=5 * 60,
            expected_edge_bps=expected_edge_bps,
            fast_move=False,
            is_futures=False,
            regime=regime.regime,
            metadata={
                "trend_direction": direction,
                "pullback_depth": pullback_depth,
                "trend_efficiency": trend_efficiency,
                "volume_impulse": volume_impulse,
                "continuation_score": continuation_score,
                "pullback_score": pullback_score,
                "entry_zscore": entry_zscore,
                "trend_persistence": trend_persistence,
                "realized_vol_percentile": volatility_percentile,
                "volatility_ratio": float(regime.volatility_ratio),
                "strategy_variant": variant,
                "profile_preference": "shallow_preferred" if shallow_pullback else "deep_selective" if deep_pullback else "legacy",
                "confirmation_variant": "continuation" if bearish_continuation_confirmation else "reclaim_hold",
                "preferred_order_type": "limit",
                "order_expiry_bars": 4,
            },
        )
        return StrategyProposal(signal=signal, expected_edge_bps=expected_edge_bps, rationale=f"bear trend pullback continuation ({variant})")


class MeanReversionStrategy(StrategyBase):
    name = "mean_reversion"

    def evaluate(self, symbol: str, regime: Any) -> Optional[StrategyProposal]:
        if regime.regime not in {"mean_reverting", "choppy"}:
            return None
        spread_fraction = float(regime.metadata.get("spread", 0.0))
        max_spread_fraction = float(getattr(self.config, "max_spread_fraction", 0.002))
        if regime.event_risk or regime.unstable:
            return None
        if regime.liquidity_score < 0.52 or spread_fraction >= max(max_spread_fraction * 0.95, 0.0014):
            return None

        candles = self.exch.fetch_ohlcv(symbol, "15m", limit=100)
        if len(candles) < 50:
            return None

        closes = np.array([float(c[4]) for c in candles], dtype=float)
        rsi = self.helpers.rsi(closes, 14)
        bb_upper, bb_middle, bb_lower = self.helpers.bollinger_bands(closes, 20, 2)
        atr_15m = self.helpers.compute_atr(symbol, "15m", period=14)
        if atr_15m is None or atr_15m <= 0:
            return None

        last_close = float(closes[-1])
        prev_close = float(closes[-2])
        last_rsi = float(rsi[-1])
        prev_rsi = float(rsi[-2])
        bb_mid = float(bb_middle[-1])
        bb_upper_last = float(bb_upper[-1])
        bb_lower_last = float(bb_lower[-1])
        stretch = float(regime.metadata.get("stretch_from_mean", 0.0))
        trend_direction = str(regime.metadata.get("trend_direction", "flat"))
        ema_fast = _ema(closes, 8)
        mean_reversion_quality = _efficiency_ratio(closes, lookback=min(10, len(closes) - 1))
        regime_efficiency = float(regime.metadata.get("directional_efficiency", mean_reversion_quality))
        mean_reversion_quality = min(mean_reversion_quality, regime_efficiency)
        recent_range = max(float(np.max(closes[-6:]) - np.min(closes[-6:])), 1e-9)
        rebound_strength = (last_close - float(np.min(closes[-6:]))) / recent_range
        fade_strength = (float(np.max(closes[-6:])) - last_close) / recent_range
        continuation_score = float(regime.metadata.get("continuation_score", 0.0))
        mean_reversion_score = float(regime.metadata.get("mean_reversion_score", 0.0))
        semivariance_skew = float(regime.metadata.get("semivariance_skew", -0.20 if float(regime.metadata.get("entry_zscore", 0.0)) < 0 else 0.20))
        exhaustion_score = float(regime.metadata.get("exhaustion_score", 0.0))
        entry_zscore = float(regime.metadata.get("entry_zscore", 0.0))
        volatility_percentile = float(regime.metadata.get("realized_vol_percentile", 0.5))
        if mean_reversion_score <= 0:
            mean_reversion_score = 1.0 + max(abs(entry_zscore) - 1.0, 0.0) * 0.2 + max(0.42 - mean_reversion_quality, 0.0) * 1.5
        if exhaustion_score <= 0:
            exhaustion_score = min(0.45 + max(abs(entry_zscore) - 1.0, 0.0) * 0.25, 1.2)
        max_efficiency = float(getattr(self.config, "mean_reversion_max_efficiency", 0.48))
        entry_zscore_threshold = float(getattr(self.config, "mean_reversion_entry_zscore", 1.10))
        min_exhaustion_score = float(getattr(self.config, "mean_reversion_min_exhaustion_score", 0.42))
        min_rebound_strength = float(getattr(self.config, "mean_reversion_min_rebound_strength", 0.12))
        min_fade_strength = float(getattr(self.config, "mean_reversion_min_fade_strength", 0.12))
        liquid_relaxed_liquidity_floor = float(getattr(self.config, "mean_reversion_liquid_relaxed_liquidity_score", 0.78))
        liquid_relaxed_efficiency_bonus = float(getattr(self.config, "mean_reversion_liquid_relaxed_efficiency_bonus", 0.06))
        liquid_relaxed_exhaustion_delta = float(getattr(self.config, "mean_reversion_liquid_relaxed_exhaustion_delta", 0.05))
        liquid_relaxed_rebound_delta = float(getattr(self.config, "mean_reversion_liquid_relaxed_rebound_delta", 0.03))
        liquid_relaxed_fade_delta = float(getattr(self.config, "mean_reversion_liquid_relaxed_fade_delta", 0.03))
        countertrend_continuation_floor = float(getattr(self.config, "mean_reversion_countertrend_continuation_score", 1.28))
        countertrend_efficiency_floor = float(getattr(self.config, "mean_reversion_countertrend_efficiency_floor", 0.34))
        countertrend_trend_strength_floor = float(getattr(self.config, "mean_reversion_countertrend_trend_strength_floor", 0.010))
        liquid_relaxed = (
            regime.liquidity_score >= liquid_relaxed_liquidity_floor
            and mean_reversion_quality <= max(max_efficiency - 0.04, 0.0)
            and volatility_percentile <= 0.82
            and continuation_score <= 1.25
        )
        effective_max_efficiency = max_efficiency + (liquid_relaxed_efficiency_bonus if liquid_relaxed else 0.0)
        effective_min_exhaustion_score = max(min_exhaustion_score - (liquid_relaxed_exhaustion_delta if liquid_relaxed else 0.0), 0.0)
        effective_min_rebound_strength = max(min_rebound_strength - (liquid_relaxed_rebound_delta if liquid_relaxed else 0.0), 0.0)
        effective_min_fade_strength = max(min_fade_strength - (liquid_relaxed_fade_delta if liquid_relaxed else 0.0), 0.0)
        countertrend_bearish_veto = (
            trend_direction == "bearish"
            and continuation_score >= countertrend_continuation_floor
            and regime_efficiency >= countertrend_efficiency_floor
            and abs(float(regime.trend_strength)) >= countertrend_trend_strength_floor
        )
        countertrend_bullish_veto = (
            trend_direction == "bullish"
            and continuation_score >= countertrend_continuation_floor
            and regime_efficiency >= countertrend_efficiency_floor
            and abs(float(regime.trend_strength)) >= countertrend_trend_strength_floor
        )
        close_location = _close_location(candles[-1])
        inside_band_reclaim = last_close >= bb_lower_last or close_location >= 0.42
        inside_band_fade = last_close <= bb_upper_last or close_location <= 0.58

        if (
            (entry_zscore <= -max(entry_zscore_threshold - 0.10, 1.10) or last_close <= bb_lower_last * 1.003)
            and last_rsi <= 37
            and prev_rsi <= last_rsi + 5.0
            and (last_close > prev_close or inside_band_reclaim)
            and stretch < -0.003
            and mean_reversion_quality <= (effective_max_efficiency + 0.02)
            and mean_reversion_score >= 1.0
            and exhaustion_score >= effective_min_exhaustion_score
            and semivariance_skew <= 0.12
            and volatility_percentile <= 0.88
            and not countertrend_bearish_veto
            and not (trend_direction == "bearish" and continuation_score >= 1.45)
            and ema_fast[-1] >= ema_fast[-2] - (atr_15m * 0.005)
            and rebound_strength >= effective_min_rebound_strength
        ):
            entry_price = min(last_close, bb_lower_last + (atr_15m * 0.12))
            stop_loss = min(last_close - (atr_15m * 0.70), bb_lower_last - (atr_15m * 0.45))
            take_profit = max(bb_mid, entry_price + ((entry_price - stop_loss) * 1.18))
            if stop_loss <= 0 or stop_loss >= entry_price or take_profit <= entry_price:
                return None
            expected_edge_bps = _expected_edge_bps("long", entry_price, stop_loss, take_profit, min(0.69, max(0.52, 0.50 + mean_reversion_score * 0.07)))
            signal = Signal(
                symbol=symbol,
                side="long",
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                strategy=self.name,
                confidence=min(0.79, max(0.52, 0.50 + regime.confidence * 0.10 + min((abs(entry_zscore) - 1.0) * 0.05, 0.08))),
                timeframe="15m",
                expected_holding_minutes=3 * 60,
                expected_edge_bps=expected_edge_bps,
                fast_move=False,
                is_futures=False,
                regime=regime.regime,
                metadata={
                    "rsi": last_rsi,
                    "stretch_from_mean": stretch,
                    "rebound_strength": rebound_strength,
                    "mean_reversion_score": mean_reversion_score,
                    "entry_zscore": entry_zscore,
                    "directional_efficiency": mean_reversion_quality,
                    "realized_vol_percentile": volatility_percentile,
                    "volatility_ratio": float(regime.volatility_ratio),
                    "liquid_relaxed": liquid_relaxed,
                    "countertrend_veto_active": countertrend_bearish_veto,
                    "strategy_variant": "exhaustion_reversal",
                    "preferred_order_type": "limit",
                    "order_expiry_bars": 3,
                },
            )
            return StrategyProposal(signal=signal, expected_edge_bps=expected_edge_bps, rationale="oversold mean reversion rebound")

        if (
            (entry_zscore >= max(entry_zscore_threshold - 0.10, 1.10) or last_close >= bb_upper_last * 0.997)
            and last_rsi >= 63
            and prev_rsi >= last_rsi - 5.0
            and (last_close < prev_close or inside_band_fade)
            and stretch > 0.003
            and mean_reversion_quality <= (effective_max_efficiency + 0.02)
            and mean_reversion_score >= 1.0
            and exhaustion_score >= effective_min_exhaustion_score
            and semivariance_skew >= -0.12
            and volatility_percentile <= 0.88
            and not countertrend_bullish_veto
            and not (trend_direction == "bullish" and continuation_score >= 1.45)
            and ema_fast[-1] <= ema_fast[-2] + (atr_15m * 0.005)
            and fade_strength >= effective_min_fade_strength
        ):
            entry_price = max(last_close, bb_upper_last - (atr_15m * 0.12))
            stop_loss = max(last_close + (atr_15m * 0.70), bb_upper_last + (atr_15m * 0.45))
            take_profit = min(bb_mid, entry_price - ((stop_loss - entry_price) * 1.18))
            if stop_loss <= entry_price or take_profit >= entry_price or take_profit <= 0:
                return None
            expected_edge_bps = _expected_edge_bps("short", entry_price, stop_loss, take_profit, min(0.69, max(0.52, 0.50 + mean_reversion_score * 0.07)))
            signal = Signal(
                symbol=symbol,
                side="short",
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                strategy=self.name,
                confidence=min(0.79, max(0.52, 0.50 + regime.confidence * 0.10 + min((abs(entry_zscore) - 1.0) * 0.05, 0.08))),
                timeframe="15m",
                expected_holding_minutes=3 * 60,
                expected_edge_bps=expected_edge_bps,
                fast_move=False,
                is_futures=False,
                regime=regime.regime,
                metadata={
                    "rsi": last_rsi,
                    "stretch_from_mean": stretch,
                    "fade_strength": fade_strength,
                    "mean_reversion_score": mean_reversion_score,
                    "entry_zscore": entry_zscore,
                    "directional_efficiency": mean_reversion_quality,
                    "realized_vol_percentile": volatility_percentile,
                    "volatility_ratio": float(regime.volatility_ratio),
                    "liquid_relaxed": liquid_relaxed,
                    "countertrend_veto_active": countertrend_bullish_veto,
                    "strategy_variant": "exhaustion_reversal",
                    "preferred_order_type": "limit",
                    "order_expiry_bars": 3,
                },
            )
            return StrategyProposal(signal=signal, expected_edge_bps=expected_edge_bps, rationale="overbought mean reversion fade")
        return None


def build_strategies(config: Any, exch: Any, helpers: Any) -> List[StrategyBase]:
    strategies: List[StrategyBase] = []
    if getattr(config, "enable_trend_breakout", True):
        strategies.append(TrendBreakoutStrategy(config, exch, helpers))
    if getattr(config, "enable_trend_pullback", True):
        strategies.append(TrendPullbackStrategy(config, exch, helpers))
    if getattr(config, "enable_mean_reversion", True):
        strategies.append(MeanReversionStrategy(config, exch, helpers))
    return strategies
