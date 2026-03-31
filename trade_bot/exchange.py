from __future__ import annotations

import datetime as dt
import random
import time
from typing import Any, Dict, List, Optional

from .config import BotConfig
from .state import BotState


class MockExchange:
    def __init__(self, config: BotConfig, state: BotState):
        self.config = config
        self.state = state
        self.market_data: Dict[str, List[List[float]]] = {}
        self.last_update: Dict[str, dt.datetime] = {}
        self.base_prices = {
            "BTC/USDT": 45000.0,
            "ETH/USDT": 2800.0,
            "BNB/USDT": 320.0,
            "SOL/USDT": 95.0,
            "XRP/USDT": 0.55,
            "ADA/USDT": 0.45,
            "AVAX/USDT": 35.0,
            "DOT/USDT": 7.20,
            "LINK/USDT": 14.50,
            "TON/USDT": 2.10,
        }

        now = dt.datetime.now()
        for symbol in config.symbols:
            self._initialize_symbol_data(symbol, now)

        self.order_books: Dict[str, Dict[str, float]] = {}
        for symbol in config.symbols:
            base_price = self.base_prices.get(symbol.split("/")[0] + "/USDT", 100.0)
            self.order_books[symbol] = {
                "bid": base_price * 0.9995,
                "ask": base_price * 1.0005,
            }

        print("[MOCK] Paper trading exchange initialized with realistic market simulation.")

    def _initialize_symbol_data(self, symbol: str, start_time: dt.datetime) -> None:
        base_price = self.base_prices.get(symbol.split("/")[0] + "/USDT", 100.0)
        candles: List[List[float]] = []
        current_price = float(base_price)
        current_time = start_time - dt.timedelta(hours=200)
        for _ in range(200):
            trend = random.uniform(-0.002, 0.002)
            volatility = random.gauss(0, 0.015)
            open_price = current_price
            high_price = open_price * (1 + abs(random.gauss(0, 0.01)))
            low_price = open_price * (1 - abs(random.gauss(0, 0.01)))
            close_price = open_price * (1 + trend + volatility)
            high_price = max(high_price, open_price, close_price)
            low_price = min(low_price, open_price, close_price)
            candle = [
                float(int(current_time.timestamp() * 1000)),
                round(float(open_price), 4),
                round(float(high_price), 4),
                round(float(low_price), 4),
                round(float(close_price), 4),
                float(random.randint(100000, 1000000)),
            ]
            candles.append(candle)
            current_price = close_price
            current_time += dt.timedelta(hours=1)
        self.market_data[symbol] = candles
        self.last_update[symbol] = current_time

    def _generate_new_candle(self, symbol: str, timeframe: str) -> Optional[List[float]]:
        if symbol not in self.market_data:
            return None
        candles = self.market_data[symbol]
        if not candles:
            return None

        last_candle = candles[-1]
        last_close = last_candle[4]
        last_time = dt.datetime.fromtimestamp(last_candle[0] / 1000)
        minutes = {"15m": 15, "1h": 60, "4h": 240}.get(timeframe, 60)
        new_time = last_time + dt.timedelta(minutes=minutes)
        change = random.uniform(-0.005, 0.005) + random.gauss(0, 0.012) + random.choice([-1, 1]) * random.uniform(0, 0.003)
        open_price = last_close
        close_price = last_close * (1 + change)
        noise = random.gauss(0, 0.005)
        high_price = max(open_price, close_price) * (1 + abs(noise))
        low_price = min(open_price, close_price) * (1 - abs(noise))
        high_price = max(high_price, open_price, close_price)
        low_price = min(low_price, open_price, close_price)
        return [
            float(int(new_time.timestamp() * 1000)),
            round(float(open_price), 4),
            round(float(high_price), 4),
            round(float(low_price), 4),
            round(float(close_price), 4),
            float(random.randint(50000, 500000)),
        ]

    def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int = 200) -> List[List[float]]:
        if symbol not in self.market_data:
            return []
        candles = self.market_data[symbol]
        now = dt.datetime.now()
        last_update = self.last_update.get(symbol, now - dt.timedelta(hours=1))
        tf_minutes = {"15m": 15, "1h": 60, "4h": 240}.get(timeframe, 60)
        while (now - last_update).total_seconds() > (tf_minutes * 60):
            new_candle = self._generate_new_candle(symbol, timeframe)
            if new_candle:
                candles.append(new_candle)
                last_update = dt.datetime.fromtimestamp(new_candle[0] / 1000)
                self.last_update[symbol] = last_update
        return candles[-limit:] if len(candles) >= limit else candles

    def get_order_book(self, symbol: str) -> Dict[str, float]:
        if symbol not in self.order_books:
            base_price = self.base_prices.get(symbol.split("/")[0] + "/USDT", 100.0)
            self.order_books[symbol] = {
                "bid": base_price * 0.9995,
                "ask": base_price * 1.0005,
            }
        current = self.order_books[symbol]
        spread = current["ask"] - current["bid"]
        mid = (current["ask"] + current["bid"]) / 2
        movement = random.gauss(0, 0.001)
        new_mid = mid * (1 + movement)
        self.order_books[symbol] = {"bid": new_mid - spread / 2, "ask": new_mid + spread / 2}
        return self.order_books[symbol]

    def place_order(
        self,
        symbol: str,
        side: str,
        size: float,
        order_type: str,
        price: Optional[float] = None,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        is_futures: bool = False,
    ) -> bool:
        print(
            f"[MOCK ORDER] {order_type.upper()} {side.upper()} {symbol} "
            f"size={size}, price={price}, SL={stop_loss}, TP={take_profit}, futures={is_futures}"
        )
        return True


class ExchangeClient:
    def __init__(self, config: BotConfig, state: BotState) -> None:
        self.config = config
        self.state = state
        self.public_client: Any = None
        self.trade_client: Any = None
        self.futures_trade_client: Any = None
        self.last_order: Any = None

        if state.paper_mode:
            print("[INFO] Paper mode: skipping real exchange client initialization.")
            return

        import ccxt as ccxt_module

        self.public_client = ccxt_module.binance({"enableRateLimit": True, "timeout": 5000})
        if getattr(self.config, "use_testnet_public", True):
            self.public_client.set_sandbox_mode(True)
            print("[INFO] Public client set to Binance TESTNET (testnet.binance.vision).")
        else:
            print("[INFO] Public client using Binance MAINNET (api.binance.com).")

        self.trade_client = ccxt_module.binance(
            {
                "apiKey": config.api_key,
                "secret": config.api_secret,
                "enableRateLimit": True,
                "timeout": 5000,
            }
        )
        self.trade_client.set_sandbox_mode(True)
        print("[INFO] Spot trade client set to Binance TESTNET (sandbox mode).")

        trading_mode = getattr(self.config, "trading_mode", "spot")
        if trading_mode in ("futures", "mixed"):
            try:
                self.futures_trade_client = ccxt_module.binance(
                    {
                        "apiKey": config.api_key,
                        "secret": config.api_secret,
                        "enableRateLimit": True,
                        "timeout": 5000,
                        "options": {"defaultType": "future"},
                    }
                )
                self.futures_trade_client.set_sandbox_mode(True)
                print("[INFO] Futures trade client prepared for Binance FUTURES TESTNET (sandbox mode).")
            except Exception as exc:
                print(f"[WARN] Failed to initialize futures trade client: {type(exc).__name__}: {exc}")
                self.futures_trade_client = None

        for client, label in (
            (self.public_client, "public"),
            (self.trade_client, "spot trade"),
            (self.futures_trade_client, "futures trade"),
        ):
            if client is None:
                continue
            try:
                client.load_markets()
            except Exception as exc:
                print(f"[WARN] Failed to load {label} markets: {type(exc).__name__}: {exc}")

    def safe_fetch(self, func: Any, *args: Any, retries: int = 3, fallback: Any = None, **kwargs: Any) -> Any:
        func_name = getattr(func, "__name__", str(func))
        for attempt in range(retries):
            try:
                return func(*args, **kwargs)
            except Exception as exc:
                print(f"[WARN] Temporary API error ({attempt + 1}/{retries}) in {func_name}: {type(exc).__name__}: {exc}")
                time.sleep(0.5)
        print("[ERROR] API failed after retries, using fallback.")
        return fallback

    def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int = 200) -> List[List[float]]:
        if self.public_client is None:
            return [[0.0, 100.0, 105.0, 95.0, 101.0, 1000.0] for _ in range(limit)]
        return self.safe_fetch(self.public_client.fetch_ohlcv, symbol, timeframe=timeframe, limit=limit, fallback=[]) or []

    def get_order_book(self, symbol: str) -> Dict[str, float]:
        if self.public_client is None:
            return {"bid": 100.0, "ask": 100.1}
        ob: Dict[str, List[List[float]]] = self.safe_fetch(
            self.public_client.fetch_order_book,
            symbol,
            limit=5,
            fallback={"bids": [[100.0, 10.0]], "asks": [[100.1, 10.0]]},
        ) or {}
        try:
            return {"bid": ob["bids"][0][0], "ask": ob["asks"][0][0]}
        except Exception:
            return {"bid": 100.0, "ask": 100.1}

    def place_order(
        self,
        symbol: str,
        side: str,
        size: float,
        order_type: str,
        price: Optional[float] = None,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        is_futures: bool = False,
    ) -> bool:
        if self.state.paper_mode or self.trade_client is None:
            print(
                f"[PAPER] {order_type.upper()} {side.upper()} {symbol} "
                f"size={size}, price={price}, SL={stop_loss}, TP={take_profit}, futures={is_futures}"
            )
            return True

        trading_mode = getattr(self.config, "trading_mode", "spot")
        if is_futures and trading_mode in ("futures", "mixed"):
            return self._place_futures_order(symbol, side, size, order_type, price, stop_loss, take_profit)
        if is_futures and trading_mode == "spot":
            print("[WARN] place_order called with is_futures=True but config.trading_mode='spot'. Using SPOT client.")
        return self._place_spot_order(symbol, side, size, order_type, price)

    def _place_spot_order(self, symbol: str, side: str, size: float, order_type: str, price: Optional[float] = None) -> bool:
        if self.trade_client is None:
            print("[ERROR] Spot trade client nem elérhető.")
            return False
        try:
            if order_type == "market":
                order = self.trade_client.create_order(symbol=symbol, type="market", side=side, amount=size)
            else:
                order = self.trade_client.create_order(symbol=symbol, type="limit", side=side, amount=size, price=price)
            self.last_order = order
            print(f"[LIVE-SPOT] Order elküldve: {order}")
            return True
        except Exception as exc:
            print(f"[ERROR] Spot order sikertelen: {type(exc).__name__}: {exc}")
            return False

    def _place_futures_order(
        self,
        symbol: str,
        side: str,
        size: float,
        order_type: str,
        price: Optional[float] = None,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
    ) -> bool:
        client = self.futures_trade_client
        if client is None:
            print("[ERROR] Futures trade client nem elérhető.")
            return False

        leverage = self.config.symbol_leverage.get(symbol, self.config.default_leverage)
        margin_type = self.config.margin_type.upper()
        leverage_ok = self.safe_fetch(client.set_leverage, leverage, symbol, fallback=None)
        if leverage_ok is None:
            print(f"[WARN] Leverage beállítás sikertelen: {symbol} → {leverage}x")
        print(f"[FUTURES] Leverage beállítva: {symbol} → {leverage}x")

        try:
            client.set_margin_mode(marginMode=margin_type, symbol=symbol)
            print(f"[FUTURES] Margin mód beállítva: {symbol} → {margin_type}")
        except Exception as exc:
            error_msg = str(exc).lower()
            if "no need" in error_msg or "already" in error_msg or "-4046" in error_msg:
                print(f"[FUTURES] Margin mód már be van állítva ({margin_type}), folytatás.")
            else:
                print(f"[WARN] Margin type beállítás sikertelen: {type(exc).__name__}: {exc}")

        opposite_side = "sell" if side.lower() == "buy" else "buy"
        position_mode = getattr(self.config, "futures_position_mode", "one_way")
        base_params: Dict[str, Any] = {}
        if position_mode == "hedge":
            base_params["positionSide"] = "LONG" if side.lower() == "buy" else "SHORT"

        try:
            if order_type == "market":
                main_order = client.create_order(symbol=symbol, type="market", side=side, amount=size, params=base_params)
            else:
                main_order = client.create_order(
                    symbol=symbol,
                    type="limit",
                    side=side,
                    amount=size,
                    price=price,
                    params={**base_params, "timeInForce": "GTC"},
                )
            self.last_order = main_order
            print(f"[FUTURES] Főmegbízás elküldve: {main_order}")
        except Exception as exc:
            print(f"[ERROR] Futures főmegbízás sikertelen: {type(exc).__name__}: {exc}")
            return False

        if stop_loss is not None:
            sl_type = getattr(self.config, "futures_sl_order_type", "STOP_MARKET")
            sl_params: Dict[str, Any] = {
                **base_params,
                "stopPrice": stop_loss,
                "reduceOnly": True,
                "workingType": "MARK_PRICE",
            }
            sl_ok = self.safe_fetch(client.create_order, symbol, sl_type, opposite_side, size, fallback=None, params=sl_params)
            if sl_ok is None:
                print(f"[WARN] Stop-Loss megbízás sikertelen: {symbol} SL={stop_loss}")
            else:
                print(f"[FUTURES] Stop-Loss beállítva: {symbol} → {stop_loss}")

        if take_profit is not None:
            tp_type = getattr(self.config, "futures_tp_order_type", "TAKE_PROFIT_MARKET")
            tp_params: Dict[str, Any] = {
                **base_params,
                "stopPrice": take_profit,
                "reduceOnly": True,
                "workingType": "MARK_PRICE",
            }
            tp_ok = self.safe_fetch(client.create_order, symbol, tp_type, opposite_side, size, fallback=None, params=tp_params)
            if tp_ok is None:
                print(f"[WARN] Take-Profit megbízás sikertelen: {symbol} TP={take_profit}")
            else:
                print(f"[FUTURES] Take-Profit beállítva: {symbol} → {take_profit}")
        return True
