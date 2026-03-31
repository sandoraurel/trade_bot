from prometheus_client import start_http_server, Gauge, Counter, Histogram
import time


class MetricsCollector:
    _server_started = False
    _initialized = False

    def __init__(self, port: int = 8000):
        if not MetricsCollector._server_started:
            try:
                start_http_server(port)
            except OSError:
                pass
            MetricsCollector._server_started = True
        if not MetricsCollector._initialized:
            MetricsCollector.balance = Gauge("bot_balance", "Current account balance")
            MetricsCollector.open_positions = Gauge("bot_open_positions", "Number of open positions")
            MetricsCollector.gross_exposure = Gauge("bot_gross_exposure", "Current gross notional exposure")
            MetricsCollector.net_exposure = Gauge("bot_net_exposure", "Current net notional exposure")
            MetricsCollector.reconciliation_ok = Gauge("bot_reconciliation_ok", "1 if reconciliation is healthy, else 0")
            MetricsCollector.risk_halts_total = Counter("bot_risk_halts_total", "Total number of risk halts", ["reason"])
            MetricsCollector.order_fill_ratio = Histogram(
                "bot_order_fill_ratio",
                "Observed simulated order fill ratio",
                buckets=(0.1, 0.25, 0.5, 0.75, 0.9, 1.0),
            )
            MetricsCollector.order_retries_total = Counter("bot_order_retries_total", "Total broker retries attempted")
            MetricsCollector.orders_total = Counter(
                "bot_orders_total",
                "Total number of orders",
                ["status"]
            )
            MetricsCollector.api_latency = Histogram(
                "bot_api_latency_seconds",
                "API latency",
                buckets=(0.05, 0.1, 0.2, 0.5, 1, 2, 5)
            )
            MetricsCollector._initialized = True

        self.balance = MetricsCollector.balance
        self.open_positions = MetricsCollector.open_positions
        self.gross_exposure = MetricsCollector.gross_exposure
        self.net_exposure = MetricsCollector.net_exposure
        self.reconciliation_ok = MetricsCollector.reconciliation_ok
        self.risk_halts_total = MetricsCollector.risk_halts_total
        self.order_fill_ratio = MetricsCollector.order_fill_ratio
        self.order_retries_total = MetricsCollector.order_retries_total
        self.orders_total = MetricsCollector.orders_total
        self.api_latency = MetricsCollector.api_latency

    def set_balance(self, value: float):
        self.balance.set(value)

    def set_open_positions(self, count: int):
        self.open_positions.set(count)

    def set_exposures(self, gross: float, net: float):
        self.gross_exposure.set(gross)
        self.net_exposure.set(net)

    def set_reconciliation(self, ok: bool):
        self.reconciliation_ok.set(1 if ok else 0)

    def inc_order_success(self):
        self.orders_total.labels(status="success").inc()

    def inc_order_failure(self):
        self.orders_total.labels(status="failure").inc()

    def inc_risk_halt(self, reason: str):
        self.risk_halts_total.labels(reason=reason).inc()

    def observe_fill_ratio(self, ratio: float):
        self.order_fill_ratio.observe(max(0.0, min(ratio, 1.0)))

    def inc_order_retry(self):
        self.order_retries_total.inc()

    def measure_api_call(self, func, *args, **kwargs):
        start = time.time()
        try:
            return func(*args, **kwargs)
        finally:
            duration = time.time() - start
            self.api_latency.observe(duration)
