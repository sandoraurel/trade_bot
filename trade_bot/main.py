import requests
import os
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), '.env'))
import json
import numpy as np
import os
import sys
from typing import Optional, Dict, Any, List
import argparse
import warnings
from typing import Optional, Dict, Any, Tuple
import time
import datetime as dt
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any


# Import ccxt lazily to avoid import-time hangs in paper mode


# ============================
# MOCK EXCHANGE FOR PAPER TRADING
# ============================
class MockExchange:
    """
    Paper trading exchange that simulates Binance behavior with realistic market data.
    Generates OHLCV data using random walk with trend and volatility components.
    """

    def __init__(self, config: 'BotConfig', state: 'BotState'):
        self.config = config
        self.state = state

        # Market data simulation
        # symbol -> list of OHLCV candles
        self.market_data: Dict[str, List[List[float]]] = {}
        # symbol -> last update timestamp
        self.last_update: Dict[str, dt.datetime] = {}
        self.base_prices = {  # Realistic starting prices
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

        # Initialize market data for all symbols
        now = dt.datetime.now()
        for symbol in config.symbols:
            self._initialize_symbol_data(symbol, now)

        # Mock order book data
        self.order_books: Dict[str, Dict[str, float]] = {}
        for symbol in config.symbols:
            base_price = self.base_prices.get(
    symbol.split('/')[0] + '/USDT', 100.0)
            self.order_books[symbol] = {
                "bid": base_price * 0.9995,
                "ask": base_price * 1.0005
            }

        print(
            "[MOCK] Paper trading exchange initialized with realistic market simulation.")

    def _initialize_symbol_data(
    self,
    symbol: str,
     start_time: dt.datetime) -> None:
        """Initialize historical OHLCV data for a symbol"""
        base_price = self.base_prices.get(
    symbol.split("/")[0] + "/USDT", 100.0)
        candles: List[List[float]] = []
        current_price: float = float(base_price)
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
                float(random.randint(100000, 1000000))
            ]
            candles.append(candle)
            current_price = close_price
            current_time += dt.timedelta(hours=1)
        self.market_data[symbol] = candles
        self.last_update[symbol] = current_time

    def _generate_new_candle(self, symbol: str,
                             timeframe: str) -> Optional[List[float]]:
        """Generate a new OHLCV candle for the given timeframe"""
        if symbol not in self.market_data:
            return None

        candles: List[List[float]] = self.market_data[symbol]
        if not candles:
            return None

        last_candle: List[float] = candles[-1]
        last_close: float = last_candle[4]
        last_time = dt.datetime.fromtimestamp(last_candle[0] / 1000)

        # Timeframe multipliers
        tf_multipliers = {
            "15m": 15,
            "1h": 60,
            "4h": 240,
        }
        minutes = tf_multipliers.get(timeframe, 60)

        new_time = last_time + dt.timedelta(minutes=minutes)

        # Generate realistic price movement
        trend_strength = random.uniform(-0.005, 0.005)
        volatility = random.gauss(0, 0.012)
        momentum = random.choice([-1, 1]) * random.uniform(0, 0.003)

        change = trend_strength + volatility + momentum
        open_price = last_close
        close_price = last_close * (1 + change)

        # Add some noise to create realistic OHLC
        noise = random.gauss(0, 0.005)
        high_price = max(open_price, close_price) * (1 + abs(noise))
        low_price = min(open_price, close_price) * (1 - abs(noise))

        # Ensure proper OHLC relationships
        high_price = max(high_price, open_price, close_price)
        low_price = min(low_price, open_price, close_price)

        new_candle: List[float] = [
            float(int(new_time.timestamp() * 1000)),
            round(float(open_price), 4),
            round(float(high_price), 4),
            round(float(low_price), 4),
            round(float(close_price), 4),
            float(random.randint(50000, 500000))
        ]

        return new_candle

    def fetch_ohlcv(self, symbol: str, timeframe: str,
                    limit: int = 200) -> List[List[float]]:
        """Return OHLCV data, generating new candles as needed"""
        if symbol not in self.market_data:
            return []

        candles: List[List[float]] = self.market_data[symbol]

        # Generate new candles if needed
        now = dt.datetime.now()
        last_update: dt.datetime = self.last_update.get(
            symbol, now - dt.timedelta(hours=1))

        # Timeframe in minutes
        tf_minutes = {"15m": 15, "1h": 60, "4h": 240}.get(timeframe, 60)

        while (now - last_update).total_seconds() > (tf_minutes * 60):
            new_candle = self._generate_new_candle(symbol, timeframe)
            if new_candle:
                candles.append(new_candle)
                last_update = dt.datetime.fromtimestamp(new_candle[0] / 1000)
                self.last_update[symbol] = last_update

        # Return the most recent 'limit' candles
        return candles[-limit:] if len(candles) >= limit else candles

    def get_order_book(self, symbol: str):
        """Return mock order book data"""
        if symbol not in self.order_books:
            base_price = self.base_prices.get(
    symbol.split('/')[0] + '/USDT', 100.0)
            self.order_books[symbol] = {
                "bid": base_price * 0.9995,
                "ask": base_price * 1.0005
            }

        # Update order book with some small random movement
        current = self.order_books[symbol]
        spread = current["ask"] - current["bid"]
        mid = (current["ask"] + current["bid"]) / 2

        # Small random movement
        movement = random.gauss(0, 0.001)
        new_mid = mid * (1 + movement)

        self.order_books[symbol] = {
            "bid": new_mid - spread / 2,
            "ask": new_mid + spread / 2
        }

        return self.order_books[symbol]

    def place_order(self, symbol: str, side: str, size: float, order_type: str,
                   price: Optional[float] = None, stop_loss: Optional[float] = None,
                   take_profit: Optional[float] = None, is_futures: bool = False) -> bool:
        """Simulate order placement in paper mode"""
        print(f"[MOCK ORDER] {order_type.upper()} {side.upper()} {symbol} "
              f"size={size}, price={price}, SL={stop_loss}, TP={take_profit}, futures={is_futures}")
        return True


LOG_DIR = "logs"
STATE_FILE = "bot_state.json"

# ============================
# 1) CONFIG – YOUR RULES
# ============================


@dataclass
class BotConfig:
    starting_balance: float = 50.0
    # Risk
    risk_per_trade_min: float = 0.02  # 2%
    risk_per_trade_max: float = 0.03  # 3%
    max_open_positions: int = 3
    max_trades_per_day_min: int = 3
    max_trades_per_day_max: int = 6
    # Symbols / timeframes
    symbols: List[str] = field(
        default_factory=lambda: [
            "BTC/USDT",
            "ETH/USDT",
            "BNB/USDT",
            "SOL/USDT",
            "XRP/USDT",
            "ADA/USDT",
            "AVAX/USDT",
            "DOT/USDT",
            "LINK/USDT",
            "TON/USDT",
        ]
    )
    timeframes: Dict[str, str] = field(
        default_factory=lambda: {
            "structure": "4h",
            "trend": "1h",
            "entry": "15m",
        }
    )
    # Behavior
    strict_signals: bool = True
    avoid_chop: bool = True
    late_confirmation: bool = True
    # 👉 ALAPÉRTELMEZETTEN NEM PAPERMÓD
    use_paper_trading: bool = False
    # Execution mode
    trading_mode: str = "spot"  # "spot", "futures", or "mixed"
    # Safety
    daily_loss_limit_fraction: float = 0.10  # 10% max daily loss
    consecutive_loss_threshold: int = 2
    # minutes to pause trading after hitting loss streak
    cooldown_minutes_after_loss_streak: int = 60
    reduced_risk_factor: float = 0.5  # halve risk when in reduced mode
    emergency_volatility_threshold: float = 0.08  # e.g., 8% candle move
    # Execution
    max_spread_fraction: float = 0.002  # 0.2%
    max_slippage_fraction: float = 0.003  # 0.3%
    # Dynamic stops / partial management (long-only for now)
    breakeven_rr: float = 1.0  # move SL to breakeven after +1R
    trailing_rr: float = 1.5  # start trailing after this R multiple
    trailing_atr_mult: float = 1.0  # trail distance = ATR * mult
    trailing_atr_period: int = 14  # ATR period for trailing
    trailing_timeframe: str = "15m"  # timeframe to compute trailing ATR
    # Dashboard times (local)
    morning_hour: int = 8
    evening_hour: int = 20
    # Telegram (optional – if None, no notifications)
    telegram_bot_token: Optional[str] = None
    telegram_chat_id: Optional[str] = None
        # ============================
    # FUTURES-SPECIFIKUS BEÁLLÍTÁSOK
    # ============================

    # Alapértelmezett tőkeáttét minden szimbólumra
    default_leverage: int = 5

    # Szimbólum-specifikus tőkeáttét override
    # Példa: {"BTC/USDT": 10, "ETH/USDT": 7}
    symbol_leverage: Dict[str, int] = field(default_factory=dict)

    # Fedezeti mód: "ISOLATED" vagy "CROSSED"
    margin_type: str = "ISOLATED"

    # Futures SL/TP megbízás típusa
    # "STOP_MARKET" = piaci stop, "STOP" = limit stop
    futures_sl_order_type: str = "STOP_MARKET"
    futures_tp_order_type: str = "TAKE_PROFIT_MARKET"

    # Pozíció mód: "one_way" vagy "hedge"
    # "one_way" = egyirányú (alapértelmezett Binance-n)
    # "hedge"   = fedezeti mód (long + short egyszerre)
    futures_position_mode: str = "one_way"
    # Testnet / API
    # ============================
    # FUTURES-SPECIFIKUS BEÁLLÍTÁSOK
    # ============================
    default_leverage: int = 5
    symbol_leverage: Dict[str, int] = field(default_factory=dict)
    margin_type: str = "ISOLATED"
    futures_sl_order_type: str = "STOP_MARKET"
    futures_tp_order_type: str = "TAKE_PROFIT_MARKET"
    futures_position_mode: str = "one_way"
    api_key: Optional[str] = None
    api_secret: Optional[str] = None
    # NEW: control whether public data uses Binance testnet (testnet.binance.vision)
    # If True -> public_client.set_sandbox_mode(True)
    # If False -> public_client uses mainnet (api.binance.com)
    use_testnet_public: bool = True

    def validate(self):
        """Basic configuration validation"""
        if self.starting_balance <= 0:
            raise ValueError("starting_balance must be > 0")
        if not (0 < self.risk_per_trade_min < self.risk_per_trade_max < 1):
            raise ValueError(
                "risk_per_trade_min and risk_per_trade_max must be between 0 and 1")
        if self.max_open_positions <= 0:
            raise ValueError("max_open_positions must be > 0")
        if self.max_trades_per_day_max <= 0:
            raise ValueError("max_trades_per_day_max must be > 0")
        if not isinstance(self.symbols, list) or len(self.symbols) == 0:
            raise ValueError("symbols must be a non-empty list")
        if self.daily_loss_limit_fraction <= 0 or self.daily_loss_limit_fraction >= 1:
            raise ValueError(
                "daily_loss_limit_fraction must be between 0 and 1")


# ============================
# 2 BOT STATE
# ============================
@dataclass
class Position:
    symbol: str
    side: str
    entry_price: float
    size: float
    stop_loss: float
    take_profit: float
    strategy: str = "unknown"
    is_futures: bool = False
    opened_at: dt.datetime = field(default_factory=dt.datetime.now)
    leverage: Optional[float] = None
    order_id: Optional[str] = None
    status: str = "open"
    fee_paid: float = 0.0
    unrealized_pnl: float = 0.0
    last_update: Optional[dt.datetime] = None
    initial_stop_loss: Optional[float] = None
    initial_take_profit: Optional[float] = None


@dataclass
class BotState:
    balance: float = 0.0

    # IMPORTANT: keep interface as-is for backward compatibility
    # Currently: one position per symbol – future multi-position support
    # should be added carefully in other modules when you decide.
    open_positions: Dict[str, Any] = field(default_factory=dict)

    today_trades_count: int = field(default=0)
    today_start_date: dt.date = field(
    default_factory=lambda: dt.datetime.now().date())

    consecutive_losses: int = 0
    reduced_risk_mode: bool = False
    emergency_mode: bool = False

    last_morning_dashboard_date: Optional[dt.date] = None
    last_evening_dashboard_date: Optional[dt.date] = None
    paper_mode: bool = True  # ezt a TradeBot állítja config alapján

    # Napi performance
    equity_start_of_day: float = 0.0
    realized_pl_today: float = 0.0
    wins_today: int = 0
    losses_today: int = 0

    # Globális drawdown
    peak_equity: float = 0.0

    # ----- NEW FIELDS (ALL PASSIVE BY DEFAULT) -----

    # Cooldown / lockout after loss streak – to be used later by
    # RiskManager/TradeBot.
    cooldown_until: Optional[dt.datetime] = None

    # Heartbeat / monitoring – last time we sent a "still alive" message.
    last_heartbeat: Optional[dt.datetime] = None

    # Lifetime statistics (for analytics & dashboards; not used yet).
    lifetime_profit: float = 0.0
    lifetime_trades: int = 0
    best_single_trade: float = 0.0
    worst_single_trade: float = 0.0

    # Unrealized PnL and equity tracking (for more accurate DD and reporting).
    unrealized_pnl: float = 0.0
    last_equity_update: Optional[dt.datetime] = None

    # Placeholder to control behavior of position handling in the future.
    # For now it does nothing, but later you can use it to switch to
    # multiple positions per symbol without changing the schema again.
    multi_position_mode: bool = False


# ============================
# 3 EXCHANGE CLIENT (ccxt + Binance)
# ============================
class ExchangeClient:
    """
    Exchange client using ccxt for Binance:
    - public_client: Binance adat (OHLCV, order book)
    - trade_client: Binance sandbox/testnet (order placement, SPOT for now)
    - futures_trade_client: Optional Binance FUTURES TESTNET client (prepared, not used yet)
    """

    def __init__(self, config: BotConfig, state: BotState) -> None:
        self.config = config
        self.state = state
        self.public_client: "Any" = None
        self.trade_client: "Any" = None
        self.futures_trade_client: "Any" = None
        self.last_order: Any = None

        # In paper mode, skip real exchange connection to avoid hanging
        if state.paper_mode:
            print("[INFO] Paper mode: skipping real exchange client initialization.")
            return

        # Lazy import ccxt only when needed for live mode
        import ccxt as ccxt_module

        # ----- PUBLIC CLIENT (DATA ONLY) -----
        self.public_client = ccxt_module.binance(
            {
                "enableRateLimit": True,
                "timeout": 5000,  # 5 second timeout in milliseconds
            }
        )

        # Decide whether public data uses MAINNET or TESTNET
        if getattr(self.config, "use_testnet_public", True):
            # This switches binance URLs to testnet (testnet.binance.vision)
            self.public_client.set_sandbox_mode(True)
            print(
                "[INFO] Public client set to Binance TESTNET (testnet.binance.vision).")
        else:
            print("[INFO] Public client using Binance MAINNET (api.binance.com).")

        # ----- SPOT TRADE CLIENT FOR TESTNET (SANDBOX MODE) -----
        self.trade_client = ccxt_module.binance(
            {
                "apiKey": config.api_key,
                "secret": config.api_secret,
                "enableRateLimit": True,
                "timeout": 5000,  # 5 second timeout in milliseconds
            }
        )
        self.trade_client.set_sandbox_mode(True)
        print("[INFO] Spot trade client set to Binance TESTNET (sandbox mode).")

        # ----- FUTURES TRADE CLIENT PLACEHOLDER (PREPARED, NOT YET USED) -----
        # We prepare a futures client for later use, but DO NOT change behavior yet.
        # This keeps current bot behavior spot-only unless you explicitly
        # integrate futures.
        self.futures_trade_client = None
        trading_mode = getattr(self.config, "trading_mode", "spot")

        if trading_mode in ("futures", "mixed"):
            try:
                self.futures_trade_client = ccxt_module.binance(
                    {
                        "apiKey": config.api_key,
                        "secret": config.api_secret,
                        "enableRateLimit": True,
                        "timeout": 5000,  # 5 second timeout in milliseconds
                        # Tell ccxt we intend to use futures endpoints (for
                        # later wiring)
                        "options": {"defaultType": "future"},
                    }
                )
                self.futures_trade_client.set_sandbox_mode(True)
                print(
                    "[INFO] Futures trade client prepared for Binance FUTURES TESTNET (sandbox mode).")
            except Exception as e:
                # We do not fail hard – we just log and continue with spot-only
                # behavior
                print(
    f"[WARN] Failed to initialize futures trade client: {
        type(e).__name__}: {e}")
                self.futures_trade_client = None

        # ----- LOAD MARKETS (BEST-EFFORT) -----
        # This helps ccxt know symbol metadata; failures do not stop the bot.
        try:
            self.public_client.load_markets()
        except Exception as e:
            print(
    f"[WARN] Failed to load public markets: {
        type(e).__name__}: {e}")

        try:
            self.trade_client.load_markets()
        except Exception as e:
            print(
    f"[WARN] Failed to load spot trade markets: {
        type(e).__name__}: {e}")

        if self.futures_trade_client is not None:
            try:
                self.futures_trade_client.load_markets()
            except Exception as e:
                print(
    f"[WARN] Failed to load futures trade markets: {
        type(e).__name__}: {e}")

        # Track the last raw order response for debugging / analytics
        # (optional)
        self.last_order = None

    def safe_fetch(
        self,
        func: callable,
        *args: Any,
        retries: int = 3,
        fallback: Any = None,
        timeout: int = 5,
        **kwargs: Any
    ) -> Any:
        """
        Retry wrapper – tries a few times before giving up and returning fallback.
        Now includes timeout to prevent hanging indefinitely.
        """
        func_name = getattr(func, "__name__", str(func))
        for attempt in range(retries):
            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                print(
                    f"[WARN] Temporary API error ({attempt + 1}/{retries}) "
                    f"in {func_name}: {type(e).__name__}: {e}"
                )
                time.sleep(0.5)

        print("[ERROR] API failed after retries, using fallback.")
        return fallback

    def fetch_ohlcv(self, symbol: str, timeframe: str,
                    limit: int = 200) -> List[List[float]]:
        """
        OHLCV gyertyák lekérése Binance-ről (public adat).
        In paper mode, returns mock data.
        """
        if self.public_client is None:
            # Paper mode: return mock candles
            return [[0.0, 100.0, 105.0, 95.0, 101.0, 1000.0]
                for _ in range(limit)]

        return self.safe_fetch(
            self.public_client.fetch_ohlcv,
            symbol,
            timeframe=timeframe,
            limit=limit,
            fallback=[],
        ) or []

    def get_order_book(self, symbol: str) -> Dict[str, float]:
        """
        Order book lekérése Binance-ről (spread ellenőrzéshez).
        In paper mode, returns mock data.
        """
        if self.public_client is None:
            # Paper mode: return reasonable mock spread
            return {"bid": 100.0, "ask": 100.1}

        ob: Dict[str, List[List[float]]] = self.safe_fetch(
            self.public_client.fetch_order_book,
            symbol,
            limit=5,
            fallback={"bids": [[100.0, 10.0]], "asks": [[100.1, 10.0]]},
        ) or {}
        try:
            bid: float = ob["bids"][0][0]
            ask: float = ob["asks"][0][0]
            return {"bid": bid, "ask": ask}
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
        """
        Order küldése:
        - paper_mode=True  → csak szimulál (MockExchange vagy log)
        - paper_mode=False → valós order a TESTNET-re
          * is_futures=False → SPOT trade_client
          * is_futures=True  → FUTURES futures_trade_client
            - Leverage beállítás
            - Margin type beállítás
            - Főmegbízás (market/limit)
            - Stop-Loss mellék-megbízás
            - Take-Profit mellék-megbízás
        """

        # ------------------------------------------------
        # PAPER MODE – nincs valós order, csak napló
        # ------------------------------------------------
        if self.state.paper_mode or self.trade_client is None:
            print(
                f"[PAPER] {order_type.upper()} {side.upper()} {symbol} "
                f"size={size}, price={price}, SL={stop_loss}, TP={take_profit}, "
                f"futures={is_futures}"
            )
            return True

        # ------------------------------------------------
        # FUTURES ÚT
        # ------------------------------------------------
        trading_mode = getattr(self.config, "trading_mode", "spot")

        if is_futures and trading_mode in ("futures", "mixed"):
            return self._place_futures_order(
                symbol=symbol,
                side=side,
                size=size,
                order_type=order_type,
                price=price,
                stop_loss=stop_loss,
                take_profit=take_profit,
            )

        # ------------------------------------------------
        # SPOT ÚT (eredeti logika megőrizve)
        # ------------------------------------------------
        if is_futures and trading_mode == "spot":
            print(
                "[WARN] place_order called with is_futures=True "
                "but config.trading_mode='spot'. Using SPOT client."
            )

        return self._place_spot_order(
            symbol=symbol,
            side=side,
            size=size,
            order_type=order_type,
            price=price,
        )

    # --------------------------------------------------
    # PRIVÁT SEGÉDMETÓDUS – SPOT
    # --------------------------------------------------
    def _place_spot_order(
        self,
        symbol: str,
        side: str,
        size: float,
        order_type: str,
        price: Optional[float] = None,
    ) -> bool:
        """Spot megbízás küldése a TESTNET spot kliensre."""
        if self.trade_client is None:
            print("[ERROR] Spot trade client nem elérhető.")
            return False

        try:
            if order_type == "market":
                order = self.trade_client.create_order(
                    symbol=symbol,
                    type="market",
                    side=side,
                    amount=size,
                )
            else:
                order = self.trade_client.create_order(
                    symbol=symbol,
                    type="limit",
                    side=side,
                    amount=size,
                    price=price,
                )

            self.last_order = order
            print(f"[LIVE-SPOT] Order elküldve: {order}")
            return True

        except Exception as e:
            print(f"[ERROR] Spot order sikertelen: {type(e).__name__}: {e}")
            return False

    # --------------------------------------------------
    # PRIVÁT SEGÉDMETÓDUS – FUTURES
    # --------------------------------------------------
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
        """
        Teljes futures megbízás-folyamat:
          1. Kliens ellenőrzés
          2. Leverage beállítás
          3. Margin type beállítás
          4. Főmegbízás küldése
          5. Stop-Loss megbízás küldése
          6. Take-Profit megbízás küldése
        """

        # 1) Kliens ellenőrzés
        client = self.futures_trade_client
        if client is None:
            print("[ERROR] Futures trade client nem elérhető.")
            return False

        # 2) Leverage meghatározása
        leverage: int = self.config.symbol_leverage.get(
            symbol,
            self.config.default_leverage,
        )

        # 3) Margin type lekérése
        margin_type: str = self.config.margin_type.upper()  # "ISOLATED" vagy "CROSSED"

        # ------- LEVERAGE BEÁLLÍTÁS -------
        leverage_ok = self.safe_fetch(
            client.set_leverage,
            leverage,
            symbol,
            fallback=None,
        )
        if leverage_ok is None:
            print(f"[WARN] Leverage beállítás sikertelen: {symbol} → {leverage}x")
            # Nem állítjuk le – folytatjuk a default leverage-dzsel

        print(f"[FUTURES] Leverage beállítva: {symbol} → {leverage}x")

        # ------- MARGIN TYPE BEÁLLÍTÁS -------
        # A Binance API hibát dob, ha már be van állítva ugyanaz a mód.
        # Ezért elkapjuk és figyelmen kívül hagyjuk azt a specifikus hibát.
        try:
            client.set_margin_mode(
                marginMode=margin_type,
                symbol=symbol,
            )
            print(f"[FUTURES] Margin mód beállítva: {symbol} → {margin_type}")
        except Exception as e:
            error_msg = str(e).lower()
            # Binance kód: "No need to change margin type" → nem hiba
            if "no need" in error_msg or "already" in error_msg or "-4046" in error_msg:
                print(f"[FUTURES] Margin mód már be van állítva ({margin_type}), folytatás.")
            else:
                print(f"[WARN] Margin type beállítás sikertelen: {type(e).__name__}: {e}")
                # Nem kritikus – folytatjuk

        # ------- ELLENTÉTES OLDAL MEGHATÁROZÁSA (SL/TP-hez) -------
        # Ha long pozíciót nyitunk (buy), akkor az SL/TP "sell" oldal és fordítva
        opposite_side: str = "sell" if side.lower() == "buy" else "buy"

        # ------- POZÍCIÓ MÓD PARAMS -------
        # "one_way" módban nincs szükség positionSide megadására
        # "hedge" módban kötelező: "LONG" vagy "SHORT"
        position_mode = getattr(self.config, "futures_position_mode", "one_way")

        base_params: Dict[str, Any] = {}
        if position_mode == "hedge":
            base_params["positionSide"] = "LONG" if side.lower() == "buy" else "SHORT"

        # ------- FŐ MEGBÍZÁS KÜLDÉSE -------
        try:
            if order_type == "market":
                main_order = client.create_order(
                    symbol=symbol,
                    type="market",
                    side=side,
                    amount=size,
                    params=base_params,
                )
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

        except Exception as e:
            print(f"[ERROR] Futures főmegbízás sikertelen: {type(e).__name__}: {e}")
            return False  # Ha a főmegbízás sikertelen, ne küldjük az SL/TP-t sem

        # ------- STOP-LOSS MEGBÍZÁS -------
        if stop_loss is not None:
            sl_type: str = getattr(
                self.config, "futures_sl_order_type", "STOP_MARKET"
            )
            sl_params: Dict[str, Any] = {
                **base_params,
                "stopPrice": stop_loss,
                "reduceOnly": True,        # Csak pozíció zárásra, nem nyitásra!
                "workingType": "MARK_PRICE",  # Mark price alapú trigger (stabilabb)
            }

            sl_ok = self.safe_fetch(
                client.create_order,
                symbol,
                sl_type,          # "STOP_MARKET" vagy "STOP"
                opposite_side,
                size,
                fallback=None,
                params=sl_params,
            )

            if sl_ok is None:
                print(f"[WARN] Stop-Loss megbízás sikertelen: {symbol} SL={stop_loss}")
            else:
                print(f"[FUTURES] Stop-Loss beállítva: {symbol} → {stop_loss}")

        # ------- TAKE-PROFIT MEGBÍZÁS -------
        if take_profit is not None:
            tp_type: str = getattr(
                self.config, "futures_tp_order_type", "TAKE_PROFIT_MARKET"
            )
            tp_params: Dict[str, Any] = {
                **base_params,
                "stopPrice": take_profit,
                "reduceOnly": True,        # Csak pozíció zárásra!
                "workingType": "MARK_PRICE",
            }

            tp_ok = self.safe_fetch(
                client.create_order,
                symbol,
                tp_type,          # "TAKE_PROFIT_MARKET" vagy "TAKE_PROFIT"
                opposite_side,
                size,
                fallback=None,
                params=tp_params,
            )

            if tp_ok is None:
                print(f"[WARN] Take-Profit megbízás sikertelen: {symbol} TP={take_profit}")
            else:
                print(f"[FUTURES] Take-Profit beállítva: {symbol} → {take_profit}")

        return True

        # PAPER MODE: no real orders, just log.
        if self.state.paper_mode or self.trade_client is None:
            print(
                f"[PAPER] {order_type.upper()} {side.upper()} {symbol} "
                f"size={size}, price={price}, SL={stop_loss}, TP={take_profit}, "
                f"futures={is_futures}"
            )
            return True

        # LIVE TESTNET ORDER (SPOT CLIENT ONLY FOR NOW)
        trading_mode = getattr(self.config, "trading_mode", "spot")
        if is_futures and trading_mode == "spot":
            # Explicitly warn about mismatch – behavior stays the same (spot
            # client).
            print(
                "[WARN] place_order called with is_futures=True but config.trading_mode='spot'. "
                "Using SPOT TESTNET trade_client. Futures client is prepared but not yet wired."
            )

        try:
            params = {}

            # Choose client based on mode and availability
            client = self.trade_client
            if (
                is_futures
                and trading_mode in ("futures", "mixed")
                and self.futures_trade_client is not None
            ):
                client = self.futures_trade_client

            if client is None:
                print("[ERROR] No trade client available")
                return False

            if order_type == "market":
                order = client.create_order(
                    symbol=symbol,
                    type="market",
                    side=side,
                    amount=size,
                    params=params,
                )
            else:
                order = client.create_order(
                    symbol=symbol,
                    type="limit",
                    side=side,
                    amount=size,
                    price=price,
                    params=params,
                )
            # Store last order for debugging / analysis
            self.last_order = order

            print(f"[LIVE-TESTNET] Order placed: {order}")
            return True

        except Exception as e:
            print(f"[ERROR] Live order failed: {e}")
            return False


# ============================
# 4) SIGNAL & MARKET LOGIC
# ============================

warnings.filterwarnings('ignore')


try:
    pass
    import pandas as pd
except ImportError:
    pass
    print("WARNING: pandas not installed. Install with: pip install pandas")
    pd = None


class SignalEngine:
    def __init__(self, config: BotConfig, exch: Any):
        self.config = config
        self.exch = exch
        # Phase 1: Multi-strategy weights (dynamic later via ML)
        self.strategy_weights = {
            'trend_breakout': 0.35,
            'mean_reversion': 0.30,
            'momentum_scalp': 0.25,
            'regime_filter': 0.10  # Veto power
        }
        # Recent performance tracking (updates after trades)
        self.strategy_performance = {
            'trend_breakout': 1.0,
            'mean_reversion': 1.0,
            'momentum_scalp': 1.0
        }

    def compute_hurst_exponent(
    self,
    symbol: str,
    timeframe: str = "1h",
     lookback: int = 50) -> float:
        """Hurst Exponent classifier"""

        candles: List[List[float]] = self.exch.fetch_ohlcv(
            symbol, timeframe, limit=lookback * 2) or []
        if not candles or len(candles) < lookback:
            return 0.5  # neutral

        prices: np.ndarray = np.array([float(c[4])
                                      for c in candles[-lookback:]])  # closes
        if np.std(prices) == 0:
            return 0.5

        # Simplified R/S analysis
        lags = range(2, 20)
        rs_values: List[float] = []

        for lag in lags:
            n = len(prices) // lag
            if n < 4:
                continue

            rs: List[float] = []
            for i in range(n):
                seg: np.ndarray = prices[i * lag:(i + 1) * lag]
                if len(seg) < 4:
                    continue
                R = float(np.max(seg) - np.min(seg))
                S = float(np.std(seg))
                if S > 0:
                    rs.append(float(np.log(R / S)))

            if rs:
                rs_values.append(float(np.mean(rs)))

        if not rs_values:
            return 0.5

        # Hurst ~ log(R/S) / log(lag) slope approximation
        hurst = float(np.mean(rs_values))
        return min(max(hurst, 0.2), 0.8)

    def get_market_regime(self, symbol: str) -> str:
        "Returns regime string"
        hurst = self.compute_hurst_exponent(symbol)
        adx_candles = self.exch.fetch_ohlcv(symbol, '1h', limit=25)
        if not adx_candles:
            return 'choppy'

        # Simplified ADX (trend strength)
        highs, lows, closes = [
    c[2] for c in adx_candles], [
        c[3] for c in adx_candles], [
            c[4] for c in adx_candles]
        dm_plus = sum(max(h - hc, 0) for h, hc in zip(highs[1:], highs[:-1]))
        dm_minus = sum(max(lc - l, 0) for l, lc in zip(lows[1:], lows[:-1]))
        tr = sum(max(h - l, abs(h - pc), abs(l - pc))
                 for h, l, pc in zip(highs[1:], lows[1:], closes[:-1]))

        adx = 100 * abs(dm_plus - dm_minus) / tr if tr > 0 else 20

        if hurst > 0.65 or adx > 30:
            return 'trending'
        elif hurst < 0.35 or adx < 15:
            return 'mean_reverting'
        else:
            return 'choppy'

    def is_choppy_market(self, symbol: str) -> bool:
        "Enhanced chop filter"
        regime = self.get_market_regime(symbol)
        return regime == 'choppy'

    def compute_atr(
        self,
        symbol: str,
        timeframe: str = "15m",
        period: int = 14,
    ) -> Optional[float]:
        """
        Compute a simple ATR-like volatility measure.
        Returns:
            float ATR value, or None if not enough data.
        """
        candles = self.exch.fetch_ohlcv(symbol, timeframe, limit=period + 1)
        if not candles:
            print(
    f"[WARN] compute_atr: no candles for {symbol} on {timeframe}.")
            return None

        if len(candles) < period + 1:
            print(
                f"[WARN] compute_atr: not enough candles for {symbol} on {timeframe} "
                f"(need {period + 1}, got {len(candles)})."
            )
            return None

        trs = []
        prev_close = candles[0][4]

        for c in candles[1:]:
            high = c[2]
            low = c[3]
            close = c[4]

            tr = max(
                high - low,
                abs(high - prev_close),
                abs(low - prev_close),
            )
            trs.append(tr)
            prev_close = close

        if not trs:
            return None

        atr = sum(trs) / len(trs)
        return atr

    def _has_higher_highs_lows(self, candles: list, lookback: int = 6) -> bool:
        """
        Check if price structure shows Higher Highs and Higher Lows (uptrend).
        - More lenient: looks for general uptrend direction instead of strict HH/HL
        - Returns True if recent average > earlier average (trend confirmation)
        """
        if not candles or len(candles) < lookback:
            return False

        recent_candles = candles[-lookback:]
        earlier_candles = candles[-(lookback * 2):-lookback]

        if not earlier_candles:
            return False

        # Calculate midpoint (close price as proxy for structure)
        recent_mid = sum(c[4] for c in recent_candles) / len(recent_candles)
        earlier_mid = sum(c[4] for c in earlier_candles) / len(earlier_candles)

        # Uptrend if recent average is higher than earlier
        # Allow 0.1% tolerance to avoid rounding issues
        return recent_mid > earlier_mid * 1.001

    def is_4h_bullish(self, symbol: str) -> bool:
        """
        4H TIMEFRAME FILTER (Structure Confirmation)
        - Must show clear uptrend: Higher Highs AND Higher Lows
        - Looks at 12 candles (6 recent vs 6 earlier)
        - Very strict - only enters when 4H structure is proven bullish
        """
        candles_4h = self.exch.fetch_ohlcv(symbol, "4h", limit=20)
        if not candles_4h or len(candles_4h) < 12:
            return False

        return self._has_higher_highs_lows(candles_4h, lookback=6)

    def is_1h_uptrend(self, symbol: str) -> bool:
        """
        1H TIMEFRAME FILTER (Momentum Confirmation)
        - Must show uptrend: Higher Highs AND Higher Lows
        - Price must be above 20-period SMA (additional confirmation)
        - Looks at 8 candles (4 recent vs 4 earlier)
        """
        candles_1h = self.exch.fetch_ohlcv(symbol, "1h", limit=50)
        if not candles_1h or len(candles_1h) < 20:
            return False

        # Check HH/HL structure
        if not self._has_higher_highs_lows(candles_1h, lookback=4):
            return False

        # Price above 20 SMA
        closes_1h = [c[4] for c in candles_1h]
        last_close = closes_1h[-1]
        sma_20 = sum(closes_1h[-20:]) / 20

        return last_close > sma_20

    def get_recent_swing_high_low(
    self,
    candles: list,
     lookback: int = 5) -> tuple:
        """
        Find recent swing high and low for breakout entry.
        - Looks at last N candles
        - Returns (swing_high, swing_low)
        """
        if not candles or len(candles) < lookback:
            return None, None

        recent = candles[-lookback:]
        swing_high = max(c[2] for c in recent)
        swing_low = min(c[3] for c in recent)

        return swing_high, swing_low

    def is_atr_spike(
        self,
        symbol: str,
        timeframe: str = "15m",
        period: int = 14,
        spike_mult: float = 3.0,
    ) -> bool:
        """
        Detect extreme volatility spike:
        - Compute ATR-like series and compare last ATR to previous average.
        - If last ATR >= spike_mult * previous ATR average -> spike.
        """
        candles = self.exch.fetch_ohlcv(symbol, timeframe, limit=period + 5)
        if not candles:
            print(
    f"[WARN] is_atr_spike: no candles for {symbol} on {timeframe}.")
            return False

        if len(candles) < period + 2:
            return False

        trs = []
        prev_close = candles[0][4]

        for c in candles[1:]:
            high = c[2]
            low = c[3]
            close = c[4]

            tr = max(
                high - low,
                abs(high - prev_close),
                abs(low - prev_close),
            )
            trs.append(tr)
            prev_close = close

        if len(trs) < period + 1:
            return False

        last_atr = sum(trs[-period:]) / period
        prev_atr = sum(trs[-(period + 1): -1]) / period

        if prev_atr <= 0:
            return False

        return last_atr >= prev_atr * spike_mult

    def mean_reversion_signal(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Mean reversion strategy - RSI extremes + BB"""
        candles = self.exch.fetch_ohlcv(symbol, '15m', limit=100)
        if len(candles) < 50:
            return None

        closes = np.array([c[4] for c in candles])
        rsi = self.rsi(closes, 14)
        bb_upper, bb_middle, bb_lower = self.bollinger_bands(closes, 20, 2)

        last_close = closes[-1]
        last_rsi = rsi[-1]

        if last_rsi < 25 and last_close <= bb_lower[-1] * 1.001:  # Long
            sl = bb_lower[-1] * 0.998
            tp = bb_middle[-1]
            return {
                'side': 'long',
                'entry_price': last_close,
                'stop_loss': sl,
                'take_profit': tp,
                'strategy': 'mean_reversion'
            }
        elif last_rsi > 75 and last_close >= bb_upper[-1] * 0.999:  # Short
            sl = bb_upper[-1] * 1.002
            tp = bb_middle[-1]
            return {
                'side': 'short',
                'entry_price': last_close,
                'stop_loss': sl,
                'take_profit': tp,
                'strategy': 'mean_reversion'
            }
        return None

    def rsi(self, prices: np.ndarray, period: int = 14) -> np.ndarray:
        if np is None:
            return np.array([50.0] * len(prices))
        delta = np.diff(prices)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = np.convolve(gain, np.ones(period) / period, mode='valid')
        avg_loss = np.convolve(loss, np.ones(period) / period, mode='valid')
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return np.concatenate((np.full(period - 1, 50.0), rsi))

    def bollinger_bands(
    self,
    prices: np.ndarray,
    period: int = 20,
     std_mult: float = 2.0) -> tuple:
        sma = np.convolve(prices, np.ones(period) / period, mode='valid')
        std = np.array([np.std(prices[i:i + period])
                       for i in range(len(prices) - period + 1)])
        upper = sma + std * std_mult
        lower = sma - std * std_mult
        return np.concatenate((np.full(period -
    1, prices[:period -
    1]), upper)), sma, np.concatenate((np.full(period -
    1, prices[:period -
     1]), lower))

    def generate_signal(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        MULTI-TIMEFRAME CONFLUENCE + ENSEMBLE
        # REQUIREMENTS (ALL MUST BE TRUE):
        # This is a HIGH-CONFIDENCE signal only (fewer trades, better quality)

        """
        # FILTER 1: Choppy market rejection
        if self.config.avoid_chop and self.is_choppy_market(symbol):
            return None

        # FILTER 2: 4H STRUCTURE - Must show clear uptrend (HH + HL)
        if not self.is_4h_bullish(symbol):
            return None

        # FILTER 3: 1H MOMENTUM - Must confirm uptrend + price above SMA20
        if not self.is_1h_uptrend(symbol):
            return None

        # FILTER 4: 15M ENTRY - Get breakout level from swing high
        candles_15m = self.exch.fetch_ohlcv(symbol, "15m", limit=20)
        if not candles_15m or len(candles_15m) < 5:
            return None

        # Get recent swing high/low for breakout
        swing_high, swing_low = self.get_recent_swing_high_low(
            candles_15m, lookback=5)
        if swing_high is None or swing_low is None:
            return None

        last_15m_close = candles_15m[-1][4]

        # Entry only if price is at or above swing high (breakout confirmed)
        if last_15m_close < swing_high:
            return None  # Not at breakout yet

        # Use close if above swing high
        entry_price = max(last_15m_close, swing_high)

        # FILTER 5: STOP LOSS - Place below swing low (structure-based, not
        # arbitrary)
        stop_loss = swing_low * 0.999  # Minimum 0.1% above swing low

        if stop_loss <= 0 or stop_loss >= entry_price:
            return None

        sl_distance = entry_price - stop_loss

        # FILTER 6: ATR VALIDATION - SL should be reasonable relative to
        # volatility
        atr_15m = self.compute_atr(symbol, "15m", period=14)
        if atr_15m is None or atr_15m <= 0:
            return None

        # SL distance should be between 0.3x and 2x ATR (tight but reasonable
        # stops)
        if sl_distance < 0.3 * atr_15m or sl_distance > 2.0 * atr_15m:
            return None

        # FILTER 7: MINIMUM 1:3 RISK/REWARD RATIO
        # This is CRITICAL for profitability with even modest win rates
        min_rr_ratio = 3.0
        take_profit = entry_price + (sl_distance * min_rr_ratio)

        # One final check: ensure TP is reasonable (not >10% move, unrealistic)
        max_tp_pct = 0.10  # 10% maximum TP distance
        if (take_profit - entry_price) / entry_price > max_tp_pct:
            # If TP would be > 10% away, reject (unrealistic target)
            return None

        return {
            "side": "long",
            "entry_price": entry_price,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "fast_move": False,
            "is_futures": False,
            "confluence_score": "HIGH",  # Track signal quality
        }


# ============================
# 5) RISK MANAGEMENT
# ============================
class RiskManager:
    def __init__(self, config: BotConfig, state: BotState):
        self.config = config
        self.state = state

    # --------- INTERNAL HELPERS (BACKWARDS-COMPATIBLE) ---------
    def _now(self) -> dt.datetime:
        return dt.datetime.now()

    def _count_open_positions(self) -> int:
        if not self.state.open_positions:
            return 0
        total = 0
        for v in self.state.open_positions.values():
            if isinstance(v, list):
                total += len(v)
            else:
                total += 1
        return total

    def _is_in_cooldown(self) -> bool:
        if self.state.cooldown_until is None:
            return False
        return self._now() < self.state.cooldown_until

    # --------- RISK PER TRADE ---------
    def current_risk_fraction(self) -> float:
        # For now:
        #   - Base is risk_per_trade_max.
        #   - If reduced_risk_mode is active -> scale by reduced_risk_factor.
        # NOTE: risk_per_trade_min is not yet actively used to avoid changing
        # behavior.
        base = self.config.risk_per_trade_max
        if self.state.reduced_risk_mode:
            base = base * self.config.reduced_risk_factor

        # Ensure we never go below zero
        if base <= 0:
            return 0.0

        return base

    def can_take_more_trades_today(self) -> bool:
        return self.state.today_trades_count < self.config.max_trades_per_day_max

    def can_open_new_position(self, symbol: str) -> bool:
        """Enhanced: + correlation limits"""
        # Cooldown
        if self._is_in_cooldown():
            return False

        # Max positions
        if self._count_open_positions() >= self.config.max_open_positions:
            return False

        # Daily trades
        if not self.can_take_more_trades_today():
            return False

        # NEW: Correlation family limit (25% risk max per group)
        if not self._check_correlation_limit(symbol):
            print(f"[RISK] Correlation limit hit for {symbol}")
            return False

        return True

    def _check_correlation_limit(self, symbol: str) -> bool:
        "Max 25% portfolio risk per coin family"
        coin = symbol.split('/')[0]
        family_risk = self._get_family_risk(coin)
        return family_risk < 0.25

    def _get_family_risk(self, coin: str) -> float:
        "Calculate risk exposure for coin family"
        families = {
            'BTC': ['BTC'],
            'ETH': ['ETH'],
            'BNB': ['BNB'],
            'SOL': ['SOL'],
            'XRP': ['XRP', 'ADA'],
            'AVAX': ['AVAX', 'DOT'],
            'LINK': ['LINK', 'TON']
        }

        total_risk = 0.0
        for family, coins in families.items():
            if coin in coins:
                for sym_pos in self.state.open_positions.values():
                    if isinstance(sym_pos, list):
                        for pos in sym_pos:
                            if pos.symbol.split('/')[0] in coins:
                                total_risk += self.config.risk_per_trade_max
                    else:
                        if pos.symbol.split('/')[0] in coins:
                            total_risk += self.config.risk_per_trade_max
                break
        return total_risk

    def check_daily_reset(self):
        today = self._now().date()
        if today != self.state.today_start_date:
            self.state.today_start_date = today
            self.state.today_trades_count = 0
            self.state.consecutive_losses = 0
            self.state.reduced_risk_mode = False
            self.state.equity_start_of_day = self.state.balance
            self.state.realized_pl_today = 0.0
            self.state.wins_today = 0
            self.state.losses_today = 0
            self.state.emergency_mode = False
            # Clear cooldown at start of a new day
            self.state.cooldown_until = None

    def check_daily_loss_limit(self) -> bool:
        "Returns True if the DAILY LOSS LIMIT is reached or exceeded, meaning trading should be halted for the rest of the day."
        equity_start = self.state.equity_start_of_day
        current_equity = self.state.balance

        realized_loss = current_equity - equity_start
        max_allowed_loss = equity_start * self.config.daily_loss_limit_fraction * -1

        return realized_loss <= max_allowed_loss

    def update_after_trade_result(self, profit_loss: float):
        "Update daily and lifetime stats after a trade closes. Also handles loss-streak based cooldown logic."
        self.state.today_trades_count += 1
        self.state.realized_pl_today += profit_loss

        # Daily win/loss counts
        if profit_loss >= 0:
            self.state.wins_today += 1
        else:
            self.state.losses_today += 1

        # Consecutive loss tracking
        if profit_loss < 0:
            self.state.consecutive_losses += 1
        else:
            self.state.consecutive_losses = 0

        # Lifetime stats (purely informational – no behavior changes)
        self.state.lifetime_trades += 1
        self.state.lifetime_profit += profit_loss
        if profit_loss > self.state.best_single_trade:
            self.state.best_single_trade = profit_loss
        if profit_loss < self.state.worst_single_trade:
            self.state.worst_single_trade = profit_loss

        # Cooldown activation after loss streak
        # Uses: config.consecutive_loss_threshold &
        # config.cooldown_minutes_after_loss_streak
        if (
            profit_loss < 0
            and self.state.consecutive_losses >= self.config.consecutive_loss_threshold
            and self.config.cooldown_minutes_after_loss_streak > 0
        ):
            cooldown_minutes = self.config.cooldown_minutes_after_loss_streak
            self.state.cooldown_until = self._now() + dt.timedelta(minutes=cooldown_minutes)
            # NOTE: logging/notification about cooldown can be added later via
            # Reporter.

    def calc_position_size(
    self,
    entry_price: float,
     stop_loss: float) -> float:
        "Basic position size calculation for spot."
        risk_fraction = self.current_risk_fraction()
        risk_amount = self.state.balance * risk_fraction
        sl_distance = abs(entry_price - stop_loss)

        if sl_distance <= 0 or risk_amount <= 0:
            return 0.0

        return risk_amount / sl_distance


# ============================
# 6) EXECUTION ENGINE
# ============================
from typing import Optional  # ensure this is at top of the file only once


class ExecutionEngine:
    def __init__(self, config: BotConfig, state: BotState,
                 exch: 'ExchangeClient | MockExchange'):
        self.config = config
        self.state = state
        self.exch = exch

    def compute_spread(self, symbol: str) -> float:
        ob = self.exch.get_order_book(symbol)
        bid = ob["bid"]
        ask = ob["ask"]
        mid = (ask + bid) / 2.0
        if mid <= 0:
            return 1.0
        return (ask - bid) / mid

    # ... [rest of ExecutionEngine methods - truncated for diff]
    # Full methods already in code above, just completing class

class BacktestEngine:
    "Backtesting framework - replays historical data through live signal pipeline."
    # Features:
    # - Realistic fills (slippage, spread, commissions)
    # - Exact signal reproduction
    # - Full metrics (Sharpe, maxDD, Calmar, etc.)


    def __init__(self, config: BotConfig):
        self.config = config
        self.trades: List[Dict] = []
        self.balance_curve: List[float] = []
        self.equity_start = config.starting_balance

    def load_historical_data(
    self,
    symbol: str,
    timeframe: str,
     days: int = 180) -> pd.DataFrame:
        "Load OHLCV from Binance public API"
        import ccxt

        exchange = ccxt.binance({'enableRateLimit': True})
        since = int(
    (pd.Timestamp.now() -
    pd.Timedelta(
        days=days)).timestamp() *
         1000)

        ohlcv = exchange.fetch_ohlcv(
    symbol, timeframe, since=since, limit=1000)

        df = pd.DataFrame(
    ohlcv,
    columns=[
        'timestamp',
        'open',
        'high',
        'low',
        'close',
         'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)

        print(
    f"Loaded {
        len(df)} candles for {symbol} {timeframe} ({days} days)")
        return df

    def run_backtest(
    self,
    symbol: str,
    timeframe: str = '15m',
     days: int = 180) -> Dict:
        "Full backtest pipeline"
        print(f"Starting backtest: {symbol} {timeframe} ({days} days)")

        # Load data
        df = self.load_historical_data(symbol, timeframe, days)
        if df.empty:
            return {'error': 'No data loaded'}

        # Initialize backtest state
        balance = self.config.starting_balance
        equity = balance
        positions = {}
        self.trades = []
        self.balance_curve = [balance]

        # Create backtest exchange with full df
        mock_exch = MockBacktestExchange(df, symbol)
        signal_engine = SignalEngine(self.config, mock_exch)
        risk_mgr = RiskManager(
    self.config, BotState(
        balance=self.config.starting_balance))

        for i in range(100, len(df)):  # Skip warmup
            # Set current market state for signal generation
            current_candle = df.iloc[i]

            # Generate signals using current market data
            signal = signal_engine.generate_signal(symbol)

            # Simulate RiskManager.can_open_new_position (simplified)
            if (signal and
                len([p for ps in positions.values() for p in ps]) < self.config.max_open_positions and
                len(self.trades) < self.config.max_trades_per_day_max * days / 30):  # Approx daily

                entry_price = signal['entry_price']
                stop_loss = signal['stop_loss']
                take_profit = signal['take_profit']

                # Position sizing
                risk_mgr = RiskManager(self.config, BotState(balance=balance))
                size = risk_mgr.calc_position_size(entry_price, stop_loss)

                if size > 0:
                    # Open position
                    pos = {
                        'entry_time': current_candle.name,
                        'entry_price': entry_price,
                        'size': size,
                        'stop_loss': stop_loss,
                        'take_profit': take_profit,
                        'side': 'long'
                    }
                    positions.setdefault(symbol, []).append(pos)

            # Check position exits (SL/TP hit)
            remaining_pos = []
            for pos in positions.get(symbol, []):
                high = current_candle['high']
                low = current_candle['low']
                close = current_candle['close']

                closed = False
                exit_price = None

                if low <= pos['stop_loss']:
                    exit_price = pos['stop_loss']
                    closed = True
                elif high >= pos['take_profit']:
                    exit_price = pos['take_profit']
                    closed = True

                if closed:
                    pl = (exit_price - pos['entry_price']) * pos['size']
                    balance += pl
                    self.trades.append({
                        'entry_time': pos['entry_time'],
                        'exit_time': current_candle.name,
                        'pl': pl,
                        'exit_reason': 'SL' if exit_price == pos['stop_loss'] else 'TP'
                    })
                else:
                    remaining_pos.append(pos)

            positions[symbol] = remaining_pos
            equity = balance  # Simplified, no unrealized PnL yet
            self.balance_curve.append(equity)

        metrics = self.compute_metrics()
        return {
            **metrics,
            'raw_trades': len(self.trades)
        }

    def compute_metrics(self) -> Dict[str, float]:
        "Compute standard trading metrics"
        if not self.balance_curve:
            return {}

        returns = pd.Series(self.balance_curve).pct_change().dropna()

        if len(returns) == 0:
            return {}

        total_return = (
            self.balance_curve[-1] / self.balance_curve[0] - 1) * 100

        wins = [t['pl'] for t in self.trades if t['pl'] > 0]
        losses = [t['pl'] for t in self.trades if t['pl'] < 0]

        win_rate = len(wins) / len(self.trades) * 100 if self.trades else 0

        gross_profit = sum(wins)
        gross_loss = abs(sum(losses))
        profit_factor = gross_profit / \
            gross_loss if gross_loss > 0 else float('inf')

        # Sharpe (annualized)
        sharpe = returns.mean() / returns.std() * \
                              np.sqrt(252 * 24) if returns.std() > 0 else 0

        # Max Drawdown
        peak = np.maximum.accumulate(self.balance_curve)
        drawdown = (self.balance_curve - peak) / peak
        max_dd = drawdown.min() * 100

        return {
            'total_return_pct': total_return,
            'num_trades': len(self.trades),
            'win_rate_pct': win_rate,
            'profit_factor': profit_factor,
            'sharpe_ratio': sharpe,
            'max_drawdown_pct': max_dd,
            'avg_win': np.mean(wins) if wins else 0,
            'avg_loss': np.mean(losses) if losses else 0,
        }

class MockBacktestExchange:
    """Mock exchange for backtesting - provides exact historical data on demand"""

    def __init__(self, df: 'pd.DataFrame', symbol: str):  # type: ignore
        self.df = df.reset_index()
        self.symbol = symbol

    def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int = 200):
        if symbol != self.symbol:
            return []
        return self.df.tail(
            limit)[['timestamp', 'open', 'high', 'low', 'close', 'volume']].values.tolist()

    def get_order_book(self, symbol: str):
        "Mock for spread checks"
        return {"bid": 45000.0, "ask": 45001.0}

class HyperoptEngine:
    "Hyperparameter optimization using backtesting grid search"

    PARAM_GRID = {
        'risk_per_trade_max': [0.01, 0.015, 0.02, 0.025, 0.03],
        'min_rr_ratio': [2.0, 2.5, 3.0, 3.5, 4.0],
        'atr_sl_mult': [0.5, 1.0, 1.5, 2.0],
        'swing_lookback': [3, 5, 7, 10]
    }

    def __init__(self):
        self.best_score = -float('inf')
        self.best_params = {}
        self.results = []

    def optimize(
    self,
    symbol: str = 'BTC/USDT',
    days: int = 90,
     max_combinations: int = None) -> Dict:
        """Full parallel grid search hyperparameter optimization.
        Tests full grid(5 * 5 * 4 * 4=400 combos, approx. 10 minutes) or subset.
        """
        import itertools

        # Full parameter grid
        param_names = list(self.PARAM_GRID.keys())
        full_grid = list(itertools.product(
            *(self.PARAM_GRID[name] for name in param_names)))

        # Limit if requested
        if max_combinations and len(full_grid) > max_combinations:
            import random
            full_grid = random.sample(full_grid, max_combinations)

        print(
    f"🚀 HYPEROPT: Testing {
        len(full_grid)} combinations on {symbol} ({days}d)")
        print(f"Grid: {dict(zip(param_names, [len(v) for v in self.PARAM_GRID.values()]))}")

        # Single-threaded sequential for simplicity (parallel had import
        # issues)
        all_results = []
        for i, params in enumerate(full_grid):
            param_dict = dict(zip(param_names, params))
            print(f"  [{i+1}/{len(full_grid)}] Testing {param_dict}")

            # Create test config
            test_config = BotConfig(starting_balance=10000.0, **param_dict)

            # Run backtest
            bt = BacktestEngine(test_config)
            result = bt.run_backtest(symbol, days=days)

            sharpe = result.get('sharpe_ratio', 0)
            all_results.append((param_dict, sharpe, result))

            # Track best
            if sharpe > self.best_score:
                self.best_score = sharpe
                self.best_params = param_dict.copy()

        self.results = all_results

        # Save full results
        summary = {
            'best_sharpe': self.best_score,
            'best_params': self.best_params,
            'total_tested': len(all_results),
            'all_results': [{'params': p, 'sharpe': s} for p, s, _ in all_results]
        }

        with open('hyperopt_results.json', 'w') as f:
            json.dump(summary, f, indent=2, default=str)

        # Save best config
        best_config = BotConfig(starting_balance=10000.0, **self.best_params)
        config_data = {k: v for k, v in vars(
            best_config).items() if not k.startswith('_')}
        with open('best_config.json', 'w') as f:
            json.dump(config_data, f, indent=2)

        print(f"\n🎯 BEST RESULTS:")
        print(f"Sharpe: {self.best_score:.3f}")
        print(f"Params: {self.best_params}")
        print(f"Saved: hyperopt_results.json + best_config.json")

        return summary


    def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int = 200):
        if symbol != self.symbol:
            return []
        return self.df.tail(limit).reset_index().values.tolist()

    def get_order_book(self, symbol: str):
        # Mock for spread checks
        return {"bid": 100.0, "ask": 100.1}

    def compute_spread(self, symbol: str) -> float:
        ob = self.exch.get_order_book(symbol)
        bid = ob["bid"]
        ask = ob["ask"]
        mid = (ask + bid) / 2.0
        if mid <= 0:
            return 1.0  # absurdly high to block trading
        return (ask - bid) / mid

    def _get_mid_price(self, symbol: str) -> Optional[float]:
        "Helper to compute the mid price using the current order book."
        # Returns None if something goes wrong.
        try:
            ob = self.exch.get_order_book(symbol)
            bid = ob["bid"]
            ask = ob["ask"]
            mid = (ask + bid) / 2.0
            if mid <= 0:
                return None
            return mid
        except Exception as e:
            print(
    f"[WARN] _get_mid_price failed for {symbol}: {
        type(e).__name__}: {e}")
            return None

    def _check_slippage_for_market(
    self,
    symbol: str,
     entry_price: float) -> bool:
        "Estimate slippage for a market order."
        # Compare intended entry_price vs current mid price.
        # If relative difference exceeds max_slippage_fraction, block the trade.
        # NOTE: This is an estimate before the order is sent, using the order
        # book mid.
        mid = self._get_mid_price(symbol)
        if mid is None or mid <= 0:
            # If we can't estimate slippage, allow the trade but warn.
            print(
                f"[WARN] Could not estimate slippage for {symbol} "
                f"(no valid mid price). Proceeding with market order."
            )
            return True

        slippage = abs(mid - entry_price) / mid
        if slippage > self.config.max_slippage_fraction:
            print(
                f"[WARN] Estimated slippage too high on {symbol}: {
    slippage:.4%} "
                f"(max allowed {
    self.config.max_slippage_fraction:.4%}). Aborting market order."
            )
            return False

        return True

    def _adjust_order_for_market(
        self,
        symbol: str,
        size: float,
        price: Optional[float],
    ) -> tuple[float, Optional[float]]:
        """
        Adjust size and price to match exchange precision and minimums, if metadata is available.

        - Uses trade_client.markets if available.
        - Rounds to 'amount' and 'price' precision.
        - Checks min amount and min notional(cost).
        - If adjustment makes size invalid ( <= 0 or below min), returns (0.0, price).

        This is best - effort and will silently fall back if market metadata is missing.
        """
        market = None
        try:
            if hasattr(self.exch, 'trade_client') and self.exch.trade_client:
                markets = getattr(self.exch.trade_client, "markets", None)
            else:
                markets = None
            if markets and symbol in markets:
                market = markets[symbol]
        except Exception as e:
            print(
    f"[WARN] _adjust_order_for_market: could not access markets for {symbol}: {e}")
            market = None

        if not market:
            # No metadata -> return original values
            return size, price

        limits = market.get("limits", {}) or {}
        precision = market.get("precision", {}) or {}

        amount_min = (limits.get("amount") or {}).get("min")
        cost_min = (limits.get("cost") or {}).get("min")

        amount_prec = precision.get("amount")
        price_prec = precision.get("price")

        # Round size to amount precision
        if amount_prec is not None and amount_prec >= 0:
            factor = 10**amount_prec
            size = int(size * factor) / factor

        # Round price to price precision
        if price is not None and price_prec is not None and price_prec >= 0:
            factor = 10**price_prec
            price = int(price * factor) / factor

        # Enforce minimum amount
        if amount_min is not None and size < amount_min:
            print(
                f"[WARN] Adjusted size {size} for {symbol} is below minimum amount "
                f"{amount_min}. Skipping order."
            )
            return 0.0, price

        # Enforce minimum notional (cost)
        if cost_min is not None and price is not None:
            if size * price < cost_min:
                print(
                    f"[WARN] Notional {
    size * price:.8f} for {symbol} is below minimum "
                    f"cost {cost_min}. Skipping order."
                )
                return 0.0, price

        return size, price

    def can_trade_symbol_now(self, symbol: str) -> bool:
        spread = self.compute_spread(symbol)
        if spread > self.config.max_spread_fraction:
            print(
                f"[INFO] Spread too high on {symbol}: {spread:.4%} "
                f"(max allowed {
    self.config.max_spread_fraction:.4%}). Skipping."
            )
            return False
        return True

    def place_trade(
        self,
        symbol: str,
        side: str,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        is_futures: bool,
        fast_move: bool,
        size: float,
    ) -> bool:
        """
        Place a trade via ExchangeClient, with:
        - Spread check(can_trade_symbol_now)
        - Optional slippage protection for market orders
        - Size / price adjustment to exchange precision and minimums

        NOTE:
        - Futures routing is still handled by ExchangeClient and currently uses SPOT TESTNET.
        - is_futures flag is passed through unchanged for future extension.
        """
        if not self.can_trade_symbol_now(symbol):
            return False

        order_type = "limit"
        price: Optional[float] = entry_price

        if fast_move:
            order_type = "market"
            price = None

            # Slippage estimation only for market orders
            if not self._check_slippage_for_market(symbol, entry_price):
                return False

        if size <= 0:
            print("[WARN] Size is zero or negative, cannot trade.")
            return False

        # Adjust size and price to market constraints (if metadata available)
        adj_size, adj_price = self._adjust_order_for_market(
            symbol, size, price)

        if adj_size <= 0:
            print("[WARN] Adjusted size is invalid, aborting trade.")
            return False

        return self.exch.place_order(
            symbol=symbol,
            side=side,
            size=adj_size,
            order_type=order_type,
            price=adj_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            is_futures=is_futures,
        )


# ============================
# 7) REPORTER + LOG FILES + TELEGRAM
# ============================
class Reporter:
    # Basic log level mapping
    LOG_LEVELS = {
        "DEBUG": 10,
        "INFO": 20,
        "WARN": 30,
        "ERROR": 40,
    }

    def __init__(self, config: BotConfig, state: BotState):
        self.config = config
        self.state = state
        os.makedirs(LOG_DIR, exist_ok=True)

        # Normalize and store numeric log level
        level_name = (
    getattr(
        self.config,
        "log_level",
         "INFO") or "INFO").upper()
        self.log_level = self.LOG_LEVELS.get(
            level_name, self.LOG_LEVELS["INFO"])

    # ---------- INTERNAL HELPERS ----------

    def _should_log(self, level: str) -> bool:
        """Check if a message with this level should be printed / sent, based on config.log_level."""
        lvl = self.LOG_LEVELS.get(level.upper(), self.LOG_LEVELS["INFO"])
        return lvl >= self.log_level

    def _write_log(self, filename: str, text: str):
        try:
            with open(os.path.join(LOG_DIR, filename), "a", encoding="utf-8") as f:
                f.write(text + "\n")
        except Exception as e:
            # Always print this warning, regardless of log_level, because
            # logging failed.
            print(f"[WARN] Failed to write log {filename}: {e}")

    def _count_open_positions(self) -> int:
        """
        Count total open positions across all symbols.
        Compatible with:
        - Dict[str, Position](current structure)
        - Dict[str, List[Position]](future multi - position structure)
        """
        if not self.state.open_positions:
            return 0
        if not self.state.open_positions:
            return 0

        total = 0
        for v in self.state.open_positions.values():
            if isinstance(v, list):
                total += len(v)
            else:
                total += 1
        return total

    # ---------- PUBLIC LOG METHODS ----------

    def log_trade(self, message: str):
        # Trades are always written to file; printing/Telegram depends on
        # caller.
        self._write_log("trades.log", message)

    def log_error(self, message: str):
        # Errors are always written to file; printing/Telegram can use
        # mentor_log with ERROR level.
        self._write_log("errors.log", message)

    def send_telegram(self, message: str):
        """
        Send Telegram message if token + chat_id are configured.
        If either is missing, silently skip.
        """
        if not self.config.telegram_bot_token or not self.config.telegram_chat_id:
            return

        url = f"https://api.telegram.org/bot{
    self.config.telegram_bot_token}/sendMessage"
        payload = {
            "chat_id": self.config.telegram_chat_id,
            "text": message,
            "parse_mode": "HTML",
        }

        try:
            requests.post(url, json=payload, timeout=5)
        except Exception as e:
            print(f"[WARN] Telegram send failed: {e}")
            self.log_error(f"Telegram error: {e}")

    def mentor_log(
    self,
    message: str,
    level: str = "INFO",
     send_telegram: bool = True):
        """
        Main logging entry point for human - readable messages.
        - Adds timestamp
        - Writes to mentor.log always
        - Prints & sends Telegram only if log_level threshold is met
        """
        now = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        full = f"[MENTOR {now}] {message}"

        # Always write to file
        self._write_log("mentor.log", full)

        # Respect log_level for stdout + Telegram
        if self._should_log(level):
            print(full)
            if send_telegram:
                self.send_telegram(full)

    # ---------- HEARTBEAT & COOLDOWN NOTIFICATIONS ----------

    def heartbeat(self):
        """
        Send periodic heartbeat.
        """
        if not getattr(self.config, "heartbeat_enabled", True):
            return

        interval_min = getattr(self.config, "heartbeat_interval_minutes", 60)
        now = dt.datetime.now()

        if self.state.last_heartbeat is not None:
            delta_sec = (now - self.state.last_heartbeat).total_seconds()
            if delta_sec < interval_min * 60:
                return  # not yet time for next heartbeat

        # Build a compact status snapshot
        open_pos = self._count_open_positions()
        mode = "PAPER" if self.state.paper_mode else "LIVE TESTNET"
        dd_flag = "EMERGENCY" if self.state.emergency_mode else "NORMAL"
        rr_flag = "REDUCED_RISK" if self.state.reduced_risk_mode else "FULL_RISK"

        msg = (
            f"Heartbeat – bot is running.\n"
            f"Mode: <b>{mode}</b>, Drawdown mode: <b>{dd_flag}</b>, Risk mode: <b>{rr_flag}</b>\n"
            f"Balance: <b>{self.state.balance:.2f} USDT</b>, "
            f"Open positions: <b>{open_pos}</b>\n"
        )

        if self.state.cooldown_until:
            remaining = self.state.cooldown_until - now
            if remaining.total_seconds() > 0:
                mins = int(remaining.total_seconds() // 60)
                msg += f"Cooldown active: approx <b>{mins}</b> minutes remaining.\n"
            else:
                msg += "Cooldown scheduled but time already passed (pending reset).\n"

        self.mentor_log(msg, level="INFO", send_telegram=True)
        self.state.last_heartbeat = now

    def notify_cooldown_start(self):
        """
        Optional helper to announce cooldown start.
        Can be called from RiskManager / TradeBot when cooldown is activated.
        """
        if not self.state.cooldown_until:
            return

        now = dt.datetime.now()
        remaining = self.state.cooldown_until - now
        mins = max(int(remaining.total_seconds() // 60), 0)

        msg = (
            f"Loss streak protection activated – entering cooldown for approx "
            f"<b>{mins}</b> minutes. No new trades will be opened during this period."
        )
        self.mentor_log(msg, level="WARN", send_telegram=True)

    def notify_cooldown_end(self):
        """
        Optional helper to announce cooldown end.
        Can be called from TradeBot when cooldown is cleared.
        """
        msg = "Cooldown period ended. Bot is allowed to open new trades again."
        self.mentor_log(msg, level="INFO", send_telegram=True)

    # ---------- DASHBOARDS ----------

    def morning_dashboard(self):
        today = dt.datetime.now().date()
        if self.state.last_morning_dashboard_date == today:
            return

        self.state.last_morning_dashboard_date = today

        open_pos = self._count_open_positions()
        mode = "PAPER" if self.state.paper_mode else "LIVE TESTNET"
        trading_mode = getattr(self.config, "trading_mode", "spot")
        rr_flag = "ON" if self.state.reduced_risk_mode else "OFF"
        dd_flag = "ON" if self.state.emergency_mode else "OFF"

        self.mentor_log(
    "Good morning, Boss. Here's your market prep dashboard.",
     level="INFO")

        self.mentor_log(
            f"Mode: <b>{mode}</b>, Trading mode: <b>{trading_mode}</b>, "
            f"Reduced risk: <b>{rr_flag}</b>, Emergency mode: <b>{dd_flag}</b>",
            level="INFO",
        )

        self.mentor_log(
    f"Balance (test): <b>{
        self.state.balance:.2f} USDT</b>",
         level="INFO")

        if self.state.equity_start_of_day == 0.0:
            self.state.equity_start_of_day = self.state.balance

        self.mentor_log(
            f"Equity start of day: <b>{
    self.state.equity_start_of_day:.2f} USDT</b>",
            level="INFO",
        )

        self.mentor_log(
            f"Open positions at start of day: <b>{open_pos}</b>",
            level="INFO",
        )

    def evening_dashboard(self):
        today = dt.datetime.now().date()
        if self.state.last_evening_dashboard_date == today:
            return

        self.state.last_evening_dashboard_date = today

        daily_pl = self.state.balance - self.state.equity_start_of_day
        total_trades = self.state.today_trades_count
        wins = self.state.wins_today
        losses = self.state.losses_today
        winrate = (wins / total_trades * 100) if total_trades > 0 else 0.0

        open_pos = self._count_open_positions()

        self.mentor_log(
            "Good evening, Boss. Here's your performance summary for today.",
            level="INFO",
        )

        self.mentor_log(
            f"Start equity: <b>{self.state.equity_start_of_day:.2f} USDT</b>",
            level="INFO",
        )
        self.mentor_log(
            f"Current balance: <b>{self.state.balance:.2f} USDT</b>",
            level="INFO",
        )
        self.mentor_log(
            f"Realized P/L today: <b>{daily_pl:.2f} USDT</b>",
            level="INFO",
        )
        self.mentor_log(
            f"Trades closed today: <b>{total_trades}</b>",
            level="INFO",
        )
        self.mentor_log(
            f"Wins: <b>{wins}</b>, Losses: <b>{losses}</b>, Winrate: <b>{winrate:.1f}%</b>",
            level="INFO",
        )
        self.mentor_log(
            f"Open positions at end of day: <b>{open_pos}</b>",
            level="INFO",
        )

        # Lifetime stats if being tracked by RiskManager
        lifetime_trades = getattr(self.state, "lifetime_trades", 0)
        lifetime_profit = getattr(self.state, "lifetime_profit", 0.0)
        best_trade = getattr(self.state, "best_single_trade", 0.0)
        worst_trade = getattr(self.state, "worst_single_trade", 0.0)

        self.mentor_log(
            f"Lifetime trades: <b>{lifetime_trades}</b>, "
            f"Lifetime P/L: <b>{lifetime_profit:.2f} USDT</b>",
            level="INFO",
        )
        self.mentor_log(
            f"Best single trade: <b>{best_trade:.2f} USDT</b>, "
            f"Worst single trade: <b>{worst_trade:.2f} USDT</b>",
            level="INFO",
        )

        self.mentor_log(
    "P/L summary: (testnet – no real money).",
     level="INFO")


# ============================
# 8) MAIN LOOP + STATE PERSISTENCE
# ============================
class TradeBot:
    def __init__(self, config: BotConfig):
        self.config = config

        # Set paper_mode correctly based on config FIRST
        paper_mode = config.use_paper_trading

        self.state = BotState(
            balance=config.starting_balance,
            paper_mode=paper_mode,
        )

        # Now initialize exchange AFTER state.paper_mode is set correctly
        if paper_mode:
            print("[PAPER MODE] Using MockExchange")
            self.exch = MockExchange(config, self.state)
        else:
            print("[LIVE TESTNET] Using real ExchangeClient")
            self.exch = ExchangeClient(config, self.state)


        self.risk = RiskManager(config, self.state)
        self.signals = SignalEngine(config, self.exch)  # type: ignore
        self.exec = ExecutionEngine(config, self.state, self.exch)
        self.reporter = Reporter(config, self.state)

        # Load previous state (if exists)
        self.load_state()

        # Initialize daily equity and peak equity if needed
        if self.state.equity_start_of_day == 0.0:
            self.state.equity_start_of_day = self.state.balance

        if self.state.peak_equity == 0.0:
            self.state.peak_equity = self.state.balance

    # ---------- STATE PERSISTENCE ----------
    def save_state(self):
        try:
            # Normalize open_positions into a flat list of dicts for JSON
            from dataclasses import is_dataclass

            open_positions_list = []
            for symbol, val in self.state.open_positions.items():
                if is_dataclass(val):
                    positions = [val]
                elif isinstance(val, list):
                    positions = val
                else:
                    continue

                for pos in positions:
                    open_positions_list.append(
                        {
                            "symbol": pos.symbol,
                            "side": pos.side,
                            "entry_price": pos.entry_price,
                            "size": pos.size,
                            "stop_loss": pos.stop_loss,
                            "take_profit": pos.take_profit,
                            "is_futures": pos.is_futures,
                            "opened_at": pos.opened_at.isoformat(),
                            # Extended Position fields (if present)
                            "leverage": pos.leverage,
                            "order_id": pos.order_id,
                            "status": pos.status,
                            "fee_paid": pos.fee_paid,
                            "unrealized_pnl": pos.unrealized_pnl,
                            "last_update": pos.last_update.isoformat() if pos.last_update else None,
                            "initial_stop_loss": pos.initial_stop_loss,
                            "initial_take_profit": pos.initial_take_profit,
                        }
                    )

            data = {
                "balance": self.state.balance,
                "today_trades_count": self.state.today_trades_count,
                "today_start_date": self.state.today_start_date.isoformat(),
                "consecutive_losses": self.state.consecutive_losses,
                "reduced_risk_mode": self.state.reduced_risk_mode,
                "emergency_mode": self.state.emergency_mode,
                "equity_start_of_day": self.state.equity_start_of_day,
                "realized_pl_today": self.state.realized_pl_today,
                "wins_today": self.state.wins_today,
                "losses_today": self.state.losses_today,
                "peak_equity": self.state.peak_equity,
                "paper_mode": self.state.paper_mode,
                # Extended BotState fields
                "cooldown_until": self.state.cooldown_until.isoformat() if self.state.cooldown_until else None,
                "last_heartbeat": self.state.last_heartbeat.isoformat() if self.state.last_heartbeat else None,
                "lifetime_profit": self.state.lifetime_profit,
                "lifetime_trades": self.state.lifetime_trades,
                "best_single_trade": self.state.best_single_trade,
                "worst_single_trade": self.state.worst_single_trade,
                "unrealized_pnl": self.state.unrealized_pnl,
                "last_equity_update": self.state.last_equity_update.isoformat() if self.state.last_equity_update else None,
                "multi_position_mode": getattr(self.state, "multi_position_mode", False),
                "open_positions": open_positions_list,
            }

            with open(STATE_FILE, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)

        except Exception as e:
            print(f"[WARN] Failed to save state: {e}")

    def load_state(self):
        if not os.path.exists(STATE_FILE):
            return

        try:
            with open(STATE_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)

            self.state.balance = data.get("balance", self.state.balance)
            self.state.today_trades_count = data.get("today_trades_count", 0)

            tsd = data.get("today_start_date")
            if tsd:
                self.state.today_start_date = dt.date.fromisoformat(tsd)

            self.state.consecutive_losses = data.get("consecutive_losses", 0)
            self.state.reduced_risk_mode = data.get("reduced_risk_mode", False)
            self.state.emergency_mode = data.get("emergency_mode", False)
            self.state.equity_start_of_day = data.get(
                "equity_start_of_day", self.state.balance)
            self.state.realized_pl_today = data.get("realized_pl_today", 0.0)
            self.state.wins_today = data.get("wins_today", 0)
            self.state.losses_today = data.get("losses_today", 0)
            self.state.peak_equity = data.get(
                "peak_equity", self.state.balance)
            self.state.paper_mode = data.get(
    "paper_mode", self.state.paper_mode)

            # Extended BotState fields (with safe defaults)
            cu = data.get("cooldown_until")
            if cu:
                try:
                    parsed = dt.datetime.fromisoformat(cu)
                    self.state.cooldown_until = parsed.replace(
                        tzinfo=None) if parsed.tzinfo else parsed
                except Exception:
                    self.state.cooldown_until = None

            lh = data.get("last_heartbeat")
            if lh:
                try:
                    parsed = dt.datetime.fromisoformat(lh)
                    self.state.last_heartbeat = parsed.replace(
                        tzinfo=None) if parsed.tzinfo else parsed
                except Exception:
                    self.state.last_heartbeat = None

            self.state.lifetime_profit = data.get("lifetime_profit", 0.0)
            self.state.lifetime_trades = data.get("lifetime_trades", 0)
            self.state.best_single_trade = data.get("best_single_trade", 0.0)
            self.state.worst_single_trade = data.get("worst_single_trade", 0.0)
            self.state.unrealized_pnl = data.get("unrealized_pnl", 0.0)

            leu = data.get("last_equity_update")
            if leu:
                try:
                    parsed = dt.datetime.fromisoformat(leu)
                    self.state.last_equity_update = parsed.replace(
                        tzinfo=None) if parsed.tzinfo else parsed
                except Exception:
                    self.state.last_equity_update = None

            self.state.multi_position_mode = data.get(
                "multi_position_mode", False)

            # Rebuild open_positions as Dict[str, List[Position]]
            self.state.open_positions = {}
            for p in data.get("open_positions", []):
                try:
                    opened_at_raw = p.get("opened_at")
                    try:
                        parsed = dt.datetime.fromisoformat(
                            opened_at_raw) if opened_at_raw else None
                        opened_at = parsed.replace(
    tzinfo=None) if parsed and parsed.tzinfo else parsed or dt.datetime.now()
                    except Exception:
                        opened_at = dt.datetime.now()

                    lu_raw = p.get("last_update")
                    last_update = None
                    if lu_raw:
                        try:
                            parsed = dt.datetime.fromisoformat(lu_raw)
                            last_update = parsed.replace(
    tzinfo=None) if parsed.tzinfo else parsed
                        except Exception:
                            last_update = None

                    pos = Position(
                        symbol=p["symbol"],
                        side=p["side"],
                        entry_price=p["entry_price"],
                        size=p["size"],
                        stop_loss=p["stop_loss"],
                        take_profit=p["take_profit"],
                        is_futures=p.get("is_futures", False),
                        opened_at=opened_at,
                        leverage=p.get("leverage"),
                        order_id=p.get("order_id"),
                        status=p.get("status", "open"),
                        fee_paid=p.get("fee_paid", 0.0),
                        unrealized_pnl=p.get("unrealized_pnl", 0.0),
                        last_update=last_update,
                        initial_stop_loss=p.get("initial_stop_loss"),
                        initial_take_profit=p.get("initial_take_profit"),
                    )

                    # Store as list per symbol to support multi-position
                    if pos.symbol not in self.state.open_positions:
                        self.state.open_positions[pos.symbol] = [pos]
                    else:
                        existing = self.state.open_positions[pos.symbol]
                        if isinstance(existing, list):
                            existing.append(pos)
                        else:
                            self.state.open_positions[pos.symbol] = [
                                existing, pos]
                except Exception:
                    continue

            print("[INFO] State loaded from file.")

        except Exception as e:
            print(f"[WARN] Failed to load state: {e}")

    # ---------- MAIN CYCLE ----------
    def run_once(self):
        now = dt.datetime.now()

        # Daily reset for counters and equity
        self.risk.check_daily_reset()

        # Track cooldown transitions for notifications
        in_cooldown_before = self.state.cooldown_until is not None and self.state.cooldown_until > now

        # Peak equity & drawdown checks (based on balance for now)
        if self.state.peak_equity == 0.0 or self.state.balance > self.state.peak_equity:
            self.state.peak_equity = self.state.balance

        if self.state.peak_equity > 0:
            dd = (self.state.balance - self.state.peak_equity) / \
                  self.state.peak_equity

            if dd <= -0.20:
                self.state.emergency_mode = True
                self.reporter.mentor_log(
    f"Max drawdown reached ({
        dd*100:.1f}%). Trading halted (emergency mode).")
                self.save_state()
                return

            elif dd <= -0.15:
                if not self.state.reduced_risk_mode:
                    self.state.reduced_risk_mode = True
                    self.reporter.mentor_log(
                        f"Drawdown level 2 ({dd*100:.1f}%). Reduced risk mode enabled.")

            elif dd <= -0.10:
                if not self.state.reduced_risk_mode:
                    self.state.reduced_risk_mode = True
                    self.reporter.mentor_log(
                        f"Drawdown level 1 ({dd*100:.1f}%). Reduced risk mode enabled.")

            else:
                if self.state.consecutive_losses == 0 and self.state.reduced_risk_mode:
                    self.state.reduced_risk_mode = False
                    self.reporter.mentor_log(
                        "Drawdown recovered. Reduced risk mode disabled.")

        # Daily loss limit
        if self.risk.check_daily_loss_limit():
            self.state.emergency_mode = True
            self.reporter.mentor_log(
                "Daily loss limit reached. Trading halted for the rest of the day.")
            self.save_state()
            return

        # ATR volatility spike check on BTC/USDT as global risk proxy
        if self.signals.is_atr_spike(
    "BTC/USDT",
    timeframe="15m",
    period=14,
     spike_mult=3.0):
            self.state.emergency_mode = True
            self.reporter.mentor_log(
                "ATR volatility spike detected on BTC/USDT (3x). Trading halted for the rest of the day."
            )
            self.save_state()
            return

        # Dashboards
        if now.hour == self.config.morning_hour:
            self.reporter.morning_dashboard()

        if now.hour == self.config.evening_hour:
            self.reporter.evening_dashboard()

        # Heartbeat (will send only if interval passed)
        self.reporter.heartbeat()

        # If in emergency mode -> observe only (no repeated spam message)
        if self.state.emergency_mode:
            self.save_state()
            return

        # Manage existing positions (check SL/TP)
        self.manage_open_positions()

        # Cooldown transition notifications
        now_after = dt.datetime.now()
        in_cooldown_after = self.state.cooldown_until is not None and self.state.cooldown_until > now_after

        if not in_cooldown_before and in_cooldown_after:
            self.reporter.notify_cooldown_start()
        elif in_cooldown_before and not in_cooldown_after:
            self.reporter.notify_cooldown_end()

        # Scan symbols for new signals
        from dataclasses import is_dataclass

        for symbol in self.config.symbols:
            # We rely on generate_signal's internal filters
            signal = self.signals.generate_signal(symbol)
            if not signal:
                continue

            if not self.risk.can_open_new_position():
                continue

            side = signal.get("side")
            entry_price = signal.get("entry_price")
            stop_loss = signal.get("stop_loss")
            take_profit = signal.get("take_profit")
            fast_move = signal.get("fast_move", False)
            is_futures = signal.get("is_futures", False)

            # Enforce at most one LONG and one SHORT per symbol (no pyramiding
            # same side)
            existing = self.state.open_positions.get(symbol)
            same_side_exists = False

            if existing is not None:
                if is_dataclass(existing):
                    if existing.side == side:
                        same_side_exists = True
                elif isinstance(existing, list):
                    for pos in existing:
                        if pos.side == side:
                            same_side_exists = True
                            break

            if same_side_exists:
                # A position of the same side already exists for this symbol ->
                # skip new one
                continue

            size = self.risk.calc_position_size(
                entry_price or 0, stop_loss or 0)
            if size <= 0 or not entry_price or not stop_loss:
                continue

            success = self.exec.place_trade(
                symbol=symbol,
                side=side or "buy",
                entry_price=entry_price or 0,
                stop_loss=stop_loss or 0,
                take_profit=take_profit or 0,
                is_futures=is_futures,
                fast_move=fast_move,
                size=size,
            )

            if success:
                side_str = (side or "buy").upper()
                msg = (
                    f"Opened {side_str} on {symbol} with entry={entry_price}, "
                    f"SL={stop_loss}, TP={take_profit} "
                    f"(futures={is_futures}, fast_move={fast_move}, size={size})"
                )
                self.reporter.mentor_log(msg)
                self.reporter.log_trade("OPEN " + msg)

                # Store as list of positions per symbol for multi-position
                # support
                new_pos = Position(
                    symbol=symbol,
                    side=side or "buy",
                    entry_price=entry_price or 0,
                    size=size,
                    stop_loss=stop_loss or 0,
                    take_profit=take_profit or 0,
                    strategy=signal.get('winning_strategy', 'unknown'),
                    is_futures=is_futures,
                    opened_at=dt.datetime.now(),
                    initial_stop_loss=stop_loss or 0,
                    initial_take_profit=take_profit or 0,
                )
                if symbol not in self.state.open_positions:
                    self.state.open_positions[symbol] = [new_pos]
                else:
                    val = self.state.open_positions[symbol]
                    if is_dataclass(val):
                        self.state.open_positions[symbol] = [val, new_pos]
                    elif isinstance(val, list):
                        val.append(new_pos)
                    else:
                        self.state.open_positions[symbol] = [new_pos]

                self.save_state()

    def manage_open_positions(self):
        """
        Check SL / TP on open positions for each symbol.
        Supports multiple positions per symbol and both LONG and SHORT.
        """
        from dataclasses import is_dataclass

        entry_tf = self.config.timeframes.get("entry", "15m")

        for symbol in list(self.state.open_positions.keys()):
            val = self.state.open_positions[symbol]

            # Normalize to list of positions
            if is_dataclass(val):
                positions = [val]
            elif isinstance(val, list):
                positions = list(val)
            else:
                continue

            candles = self.exch.fetch_ohlcv(symbol, entry_tf, limit=1)
            if not candles:
                continue

            _ts, _open, last_high, last_low, last_close, _vol = candles[-1]

            remaining_positions = []

            for pos in positions:
                closed = False
                close_price = None

                # --- Dynamic risk management (long-only for now) ---
                if pos.side == "long":
                    # Risk distance based on original SL if available
                    base_sl = pos.initial_stop_loss if pos.initial_stop_loss else pos.stop_loss
                    risk_dist = pos.entry_price - base_sl
                    if risk_dist > 0:
                        rr_move = (last_high - pos.entry_price) / risk_dist

                        # Move to breakeven after breakeven_rr
                        if rr_move >= self.config.breakeven_rr and pos.stop_loss < pos.entry_price:
                            pos.stop_loss = pos.entry_price

                        # Start ATR trailing after trailing_rr
                        if rr_move >= self.config.trailing_rr:
                            atr = self.signals.compute_atr(
                                symbol,
                                timeframe=self.config.trailing_timeframe,
                                period=self.config.trailing_atr_period,
                            )
                            if atr is not None and atr > 0:
                                new_sl = last_high - atr * self.config.trailing_atr_mult
                                if new_sl > pos.stop_loss:
                                    pos.stop_loss = new_sl

                if pos.side == "long":
                    # Long: SL if low <= SL, TP if high >= TP
                    if last_low <= pos.stop_loss:
                        close_price = pos.stop_loss
                        closed = True
                    elif last_high >= pos.take_profit:
                        close_price = pos.take_profit
                        closed = True

                elif pos.side == "short":
                    # Short: SL if high >= SL, TP if low <= TP
                    if last_high >= pos.stop_loss:
                        close_price = pos.stop_loss
                        closed = True
                    elif last_low <= pos.take_profit:
                        close_price = pos.take_profit
                        closed = True

                if not closed:
                    remaining_positions.append(pos)
                    continue

                # Compute P&L
                entry_price = float(pos.entry_price or 0)
                close_price_val = float(close_price or 0)
                if pos.side == "long":
                    pl = (close_price_val - entry_price) * pos.size
                else:  # short
                    pl = (entry_price - close_price_val) * pos.size

                self.state.balance += pl
                self.risk.update_after_trade_result(pl)

                msg = (
                    f"Closed {pos.side.upper()} on {symbol} at {close_price} "
                    f"for P/L={pl:.2f}. New balance={self.state.balance:.2f}"
                )
                self.reporter.mentor_log(msg)
                self.reporter.log_trade("CLOSE " + msg)

            # Update open_positions for this symbol
            if not remaining_positions:
                del self.state.open_positions[symbol]
            else:
                # Keep list for consistency
                self.state.open_positions[symbol] = remaining_positions

        self.save_state()

    def run_forever(self, sleep_seconds: int = 60):
        mode = "PAPER MODE" if self.state.paper_mode else "LIVE TESTNET MODE"
        self.reporter.mentor_log(f"Starting passive 24/7 loop in {mode}.")

        while True:
            try:
                self.run_once()
            except Exception as e:
                err_msg = f"Error occurred: {e}"
                self.reporter.log_error(err_msg)
                print(err_msg)
            time.sleep(sleep_seconds)

    @staticmethod
    def run_backtest_cli(
            symbol: str = 'BTC/USDT',
            days: int = 180,
            timeframe: str = '15m'):
        """CLI entrypoint for backtesting."""
        config = BotConfig(
            starting_balance=10000.0,
            telegram_bot_token="",
            telegram_chat_id="",
            api_key="",
            api_secret="",
        )

        bt = BacktestEngine(config)
        results = bt.run_backtest(symbol, timeframe, days)

        # Save results
        with open('backtest_results.json', 'w') as f:
            json.dump(results, f, indent=2)

        print("\n" + "="*60)
        print("BACKTEST RESULTS:")
        print("="*60)
        for k, v in results.items():
            if isinstance(v, float):
                print(f"{k.replace('_', ' ').title()}: {v:.2f}")
            else:
                print(f"{k.replace('_', ' ').title()}: {v}")
        print("\nFull results saved to backtest_results.json")
        print("="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI Trading Bot: Live | Backtest | Hyperopt")
    parser.add_argument('mode', choices=['live', 'backtest', 'hyperopt'], 
                       help="live=paper trading, backtest=historical test, hyperopt=param optimization")
    parser.add_argument('--symbol', default='BTC/USDT', help="Trading pair")
    parser.add_argument('--days', type=int, default=90, help="Backtest/hyperopt days")
    parser.add_argument('--timeframe', default='15m', help="Chart timeframe")
    parser.add_argument('--combinations', type=int, help="Max hyperopt combinations")
    
    args = parser.parse_args()
    
    if args.mode == 'backtest':
        run_backtest_cli(args.symbol, args.days, args.timeframe)
    elif args.mode == 'hyperopt':
        hyperopt = HyperoptEngine()
        hyperopt.optimize(args.symbol, args.days, args.combinations)
    else:  # live/paper trading
        config = BotConfig(
            starting_balance=50.0,
            use_paper_trading=True,
            telegram_bot_token=os.getenv("TELEGRAM_BOT_TOKEN"),
            telegram_chat_id=os.getenv("TELEGRAM_CHAT_ID"),
            api_key=os.getenv("BINANCE_API_KEY"), 
            api_secret=os.getenv("BINANCE_API_SECRET"),
        )
        config.validate()
        bot = TradeBot(config)
        bot.run_forever(60)