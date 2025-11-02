"""实时行情与交易执行模块"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Deque, Dict, Iterator, List, Optional, Type

import ccxt  # type: ignore
import pandas as pd

from phoenix_quant.backtest.engine import Order, Trade
from phoenix_quant.config import (
    DataSourceConfig,
    EngineConfig,
    ExchangeConfig,
    LiveTradingConfig,
)
from phoenix_quant.strategies.elastic_dip import ElasticDipStrategy

LOGGER = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# 数据源
# ----------------------------------------------------------------------
class LiveDataFeed:
    """基于轮询的实时行情数据源"""

    def __init__(
        self,
        exchange: ccxt.Exchange,
        symbol: str,
        timeframe: str,
        *,
        poll_interval: float = 30.0,
        backfill_limit: int = 1000,
    ) -> None:
        self.exchange = exchange
        self.symbol = symbol
        self.timeframe = timeframe
        self.poll_interval = max(poll_interval, 1.0)
        self.backfill_limit = max(2, backfill_limit)

        self._buffer: Deque[List[float]] = deque(maxlen=self.backfill_limit)
        self._last_ts: Optional[int] = None

    def fetch_history(self) -> List[List[float]]:
        LOGGER.info("Fetching historical candles: %s %s limit=%s", self.symbol, self.timeframe, self.backfill_limit)
        candles = self.exchange.fetch_ohlcv(
            self.symbol,
            timeframe=self.timeframe,
            limit=self.backfill_limit,
        )
        if not candles:
            raise RuntimeError(f"未获取到 {self.symbol} K线数据")
        self._buffer.extend(candles)
        self._last_ts = candles[-1][0]
        return list(self._buffer)

    def stream(self) -> Iterator[List[float]]:
        if self._last_ts is None:
            self.fetch_history()

        assert self._last_ts is not None
        last_ts = self._last_ts

        while True:
            try:
                candles = self.exchange.fetch_ohlcv(self.symbol, timeframe=self.timeframe, limit=2)
            except Exception as exc:  # pragma: no cover
                LOGGER.warning("获取实时K线失败: %s", exc)
                time.sleep(self.poll_interval)
                continue

            new_items: List[List[float]] = []
            for candle in candles:
                if candle[0] > last_ts:
                    new_items.append(candle)
                    last_ts = candle[0]

            if new_items:
                for candle in new_items:
                    self._buffer.append(candle)
                    yield candle
            else:
                LOGGER.debug("暂无新K线，等待 %.2fs", self.poll_interval)

            time.sleep(self.poll_interval)


# ----------------------------------------------------------------------
# 交易所执行器
# ----------------------------------------------------------------------
class ExchangeExecutor:
    """封装 ccxt 交易所接口"""

    def __init__(self, exchange_cfg: ExchangeConfig, data_cfg: DataSourceConfig) -> None:
        if not hasattr(ccxt, exchange_cfg.exchange_id):
            raise ValueError(f"不支持的交易所: {exchange_cfg.exchange_id}")

        exchange_class = getattr(ccxt, exchange_cfg.exchange_id)
        init_kwargs: Dict[str, Any] = {
            "apiKey": exchange_cfg.api_key,
            "secret": exchange_cfg.secret,
            "enableRateLimit": exchange_cfg.enable_rate_limit,
        }
        if exchange_cfg.password:
            init_kwargs["password"] = exchange_cfg.password
        init_kwargs.update(exchange_cfg.params)

        self.exchange: ccxt.Exchange = exchange_class(init_kwargs)
        if exchange_cfg.options:
            self.exchange.options.update(exchange_cfg.options)

        if data_cfg.use_testnet and hasattr(self.exchange, "set_sandbox_mode"):
            self.exchange.set_sandbox_mode(True)

        LOGGER.info("加载交易所市场: %s", exchange_cfg.exchange_id)
        self.exchange.load_markets()

    # 账户同步 ---------------------------------------------------------
    def fetch_balance(self) -> Dict[str, Any]:
        return self.exchange.fetch_balance()

    def fetch_positions(self, symbol: str) -> Optional[Dict[str, Any]]:
        if not hasattr(self.exchange, "fetch_positions"):
            return None
        try:
            positions = self.exchange.fetch_positions([symbol])
        except Exception as exc:  # pragma: no cover
            LOGGER.warning("获取持仓失败: %s", exc)
            return None

        for pos in positions:
            code = pos.get("symbol") or pos.get("info", {}).get("symbol")
            if code == symbol or (code and code.replace("/", "") == symbol.replace("/", "")):
                return pos
        return None

    def set_leverage(self, symbol: str, leverage: float) -> None:
        if leverage <= 0 or not hasattr(self.exchange, "set_leverage"):
            return
        try:
            self.exchange.set_leverage(int(leverage), symbol)
        except Exception as exc:  # pragma: no cover
            LOGGER.warning("设置杠杆失败: %s", exc)

    # 下单 -------------------------------------------------------------
    def create_limit_order(
        self,
        symbol: str,
        side: str,
        amount: float,
        price: float,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        return self.exchange.create_order(symbol, "limit", side, amount, price, params or {})

    def create_market_order(
        self,
        symbol: str,
        side: str,
        amount: float,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        return self.exchange.create_order(symbol, "market", side, amount, None, params or {})

    def cancel_order(self, order_id: str, symbol: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return self.exchange.cancel_order(order_id, symbol, params or {})

    def cancel_all_orders(self, symbol: str, params: Optional[Dict[str, Any]] = None) -> None:
        if hasattr(self.exchange, "cancel_all_orders"):
            self.exchange.cancel_all_orders(symbol, params or {})
            return
        for order in self.fetch_open_orders(symbol):
            try:
                self.cancel_order(order["id"], symbol, params)
            except Exception as exc:  # pragma: no cover
                LOGGER.warning("撤单失败 %s: %s", order["id"], exc)

    def fetch_open_orders(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        return self.exchange.fetch_open_orders(symbol)

    def fetch_order(self, order_id: str, symbol: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return self.exchange.fetch_order(order_id, symbol, params or {})

    def fetch_my_trades(
        self,
        symbol: str,
        since: Optional[int] = None,
        limit: Optional[int] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        return self.exchange.fetch_my_trades(symbol, since, limit, params or {})


# ----------------------------------------------------------------------
# 实盘引擎
# ----------------------------------------------------------------------
@dataclass
class LivePosition:
    symbol: str
    quantity: float = 0.0
    avg_price: float = 0.0
    margin: float = 0.0
    updated_at: float = field(default_factory=time.time)

    @property
    def notional(self) -> float:
        return abs(self.quantity) * self.avg_price


class LiveEngine:
    def __init__(self, executor: ExchangeExecutor, symbol: str, engine_cfg: EngineConfig) -> None:
        self.executor = executor
        self.symbol = symbol
        self.config = engine_cfg

        self.balance: float = engine_cfg.initial_balance
        self.leverage: float = max(engine_cfg.leverage, 1.0)
        self.cash_currency: str = "USDT"

        self.current_prices: Dict[str, float] = {}
        self.current_timestamp: Optional[int] = None

        self.orders: Dict[str, Order] = {}
        self.open_orders: Dict[str, Order] = {}
        self.trades: List[Trade] = []
        self.position = LivePosition(symbol)

        self.trading_enabled = True

    # 同步 -------------------------------------------------------------
    def sync(self) -> None:
        self._sync_balance()
        self._sync_position()
        self._sync_orders()

    def _sync_balance(self) -> None:
        try:
            balance = self.executor.fetch_balance()
        except Exception as exc:  # pragma: no cover
            LOGGER.warning("获取账户余额失败: %s", exc)
            return
        total = balance.get("total") or balance.get("free") or {}
        if isinstance(total, dict):
            cash = total.get(self.cash_currency)
            if isinstance(cash, (int, float)):
                self.balance = float(cash)

    def _sync_position(self) -> None:
        pos = self.executor.fetch_positions(self.symbol)
        if not pos:
            self.position = LivePosition(self.symbol)
            return
        contracts = float(pos.get("contracts") or pos.get("contractSize") or pos.get("size") or 0.0)
        quantity = float(pos.get("positionAmt") or pos.get("amount") or contracts)
        entry_price = float(pos.get("entryPrice") or pos.get("average") or pos.get("avgPrice") or 0.0)
        margin = float(pos.get("margin") or pos.get("initialMargin") or 0.0)
        self.position = LivePosition(self.symbol, quantity=quantity, avg_price=entry_price, margin=margin)

    def _sync_orders(self) -> None:
        try:
            open_orders = self.executor.fetch_open_orders(self.symbol)
        except Exception as exc:  # pragma: no cover
            LOGGER.warning("获取挂单失败: %s", exc)
            return

        open_ids = set()
        for raw in open_orders:
            order_id = str(raw["id"])
            open_ids.add(order_id)
            order = self.orders.get(order_id)
            if not order:
                order = self._build_order(raw)
                self.orders[order_id] = order
            order.status = raw.get("status", "open")
            order.filled_qty = float(raw.get("filled") or order.filled_qty)
            order.quantity = float(raw.get("amount") or order.quantity)
            order.price = float(raw.get("price") or order.price)
            self.open_orders[order_id] = order

        for order_id in list(self.open_orders):
            if order_id in open_ids:
                continue
            order = self.open_orders.pop(order_id, None)
            if not order:
                continue
            try:
                detail = self.executor.fetch_order(order_id, self.symbol)
            except Exception as exc:  # pragma: no cover
                LOGGER.warning("获取订单详情失败: %s", exc)
                continue
            order.status = detail.get("status", "closed")
            order.filled_qty = float(detail.get("filled") or order.filled_qty)
            order.price = float(detail.get("average") or detail.get("price") or order.price)
            self._append_trade(detail, order.tag)

    def _build_order(self, data: Dict[str, Any], tag: str = "") -> Order:
        return Order(
            order_id=str(data.get("id")),
            symbol=self.symbol,
            side=str(data.get("side", "buy")),
            order_type=str(data.get("type", "limit")),
            price=float(data.get("price") or 0.0),
            quantity=float(data.get("amount") or 0.0),
            tag=tag,
            status=str(data.get("status", "open")),
        )

    def _append_trade(self, data: Dict[str, Any], tag: str) -> None:
        timestamp = int(data.get("timestamp") or int(time.time() * 1000))
        price = float(data.get("average") or data.get("price") or 0.0)
        quantity = float(data.get("filled") or data.get("amount") or 0.0)
        fee_info = data.get("fee") or {}
        fee = float(fee_info.get("cost") or 0.0)
        pnl = float(data.get("pnl") or 0.0)
        side = str(data.get("side") or "buy")

        self.trades.append(
            Trade(
                timestamp=timestamp,
                symbol=self.symbol,
                side=side,
                price=price,
                quantity=quantity,
                fee=fee,
                tag=tag,
                pnl=pnl,
            )
        )

    # 策略接口 ---------------------------------------------------------
    def enable_trading(self) -> None:
        self.trading_enabled = True

    def disable_trading(self) -> None:
        self.trading_enabled = False

    def update_market(self, symbol: str, candle: List[float]) -> None:
        self.current_timestamp = int(candle[0])
        self.current_prices[symbol] = float(candle[4])

    def submit_limit_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        *,
        tag: str = "",
        reduce_only: bool = False,
    ) -> Order:
        if not self.trading_enabled:
            LOGGER.debug("交易关闭，跳过限价单 %s %s@%s", side, quantity, price)
            return self._placeholder_order(symbol, side, quantity, price, tag, status="skipped")
        params = {"reduceOnly": reduce_only} if reduce_only else {}
        response = self.executor.create_limit_order(symbol, side, quantity, price, params)
        order = self._build_order(response, tag=tag)
        order.reduce_only = reduce_only
        self.orders[order.order_id] = order
        if order.status == "open":
            self.open_orders[order.order_id] = order
        return order

    def submit_market_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        *,
        tag: str = "",
    ) -> Order:
        if not self.trading_enabled:
            LOGGER.debug("交易关闭，跳过市价单 %s %s", side, quantity)
            return self._placeholder_order(symbol, side, quantity, self.current_prices.get(symbol, 0.0), tag, status="skipped")

        response = self.executor.create_market_order(symbol, side, quantity)
        order = self._build_order(response, tag=tag)
        order.order_type = "market"
        self.orders[order.order_id] = order
        if order.status != "open":
            self._append_trade(response, tag)
        return order

    def cancel_orders(self, *, tag_prefix: Optional[str] = None) -> None:
        for order in list(self.open_orders.values()):
            if tag_prefix and not order.tag.startswith(tag_prefix):
                continue
            try:
                self.executor.cancel_order(order.order_id, self.symbol)
                order.status = "canceled"
                self.open_orders.pop(order.order_id, None)
            except Exception as exc:  # pragma: no cover
                LOGGER.warning("撤单失败 %s: %s", order.order_id, exc)

    def close_position(self, symbol: str, *, portion: float = 1.0, tag: str = "exit") -> Optional[Order]:
        if self.position.quantity == 0:
            return None
        side = "sell" if self.position.quantity > 0 else "buy"
        qty = abs(self.position.quantity) * portion
        return self.submit_market_order(symbol, side, qty, tag=tag)

    def count_open_orders(self, tag_prefix: Optional[str] = None) -> int:
        if not tag_prefix:
            return len(self.open_orders)
        return sum(1 for order in self.open_orders.values() if order.tag.startswith(tag_prefix))

    def get_orders(self, *, status: Optional[str] = None, tag_prefix: Optional[str] = None) -> List[Order]:
        result = []
        for order in self.orders.values():
            if status and order.status != status:
                continue
            if tag_prefix and not order.tag.startswith(tag_prefix):
                continue
            result.append(order)
        return result

    def get_position(self, symbol: str) -> LivePosition:
        if symbol != self.symbol:
            return LivePosition(symbol)
        return self.position

    def get_total_equity(self, symbol: str) -> float:
        if symbol != self.symbol:
            return self.balance
        price = self.current_prices.get(symbol, self.position.avg_price)
        unrealized = (price - self.position.avg_price) * self.position.quantity
        return self.balance + self.position.margin + unrealized

    def get_equity_dataframe(self) -> pd.DataFrame:
        data = [
            {
                "timestamp": self.current_timestamp or int(time.time() * 1000),
                "symbol": self.symbol,
                "equity": self.get_total_equity(self.symbol),
                "balance": self.balance,
                "position_value": self.position.margin,
            }
        ]
        return pd.DataFrame(data)

    def get_trades_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([trade.__dict__ for trade in self.trades])

    def _placeholder_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        tag: str,
        status: str = "pending",
    ) -> Order:
        order = Order(
            order_id=f"PL-{int(time.time() * 1000)}",
            symbol=symbol,
            side=side,
            order_type="limit",
            price=price,
            quantity=quantity,
            tag=tag,
            status=status,
        )
        self.orders[order.order_id] = order
        return order

# ----------------------------------------------------------------------
# 调度器
# ----------------------------------------------------------------------
class LiveTrader:
    """运行策略的调度器"""

    def __init__(self, config: LiveTradingConfig, strategy_cls: Type[ElasticDipStrategy] = ElasticDipStrategy) -> None:
        self.config = config
        self.executor = ExchangeExecutor(config.exchange, config.data)
        self.executor.set_leverage(config.symbol, config.engine.leverage)

        self.engine = LiveEngine(self.executor, config.symbol, config.engine)
        self.strategy = strategy_cls(self.engine, config.symbol, config.strategy)
        self.feed = LiveDataFeed(
            self.executor.exchange,
            config.symbol,
            config.timeframe,
            poll_interval=config.settings.poll_interval,
            backfill_limit=config.settings.backfill_limit,
        )

        if not config.settings.enable_trading or config.settings.dry_run:
            LOGGER.warning("当前为干跑模式（dry-run），不会提交真实订单。")
            self.engine.disable_trading()

    def warmup(self) -> None:
        history = self.feed.fetch_history()
        LOGGER.info("Warmup 历史K线数量: %s", len(history))

        prev_state = self.engine.trading_enabled
        self.engine.disable_trading()
        for candle in history:
            self.engine.update_market(self.config.symbol, candle)
            self.strategy.on_bar(candle)
        if prev_state and self.config.settings.enable_trading and not self.config.settings.dry_run:
            self.engine.enable_trading()

    def run(self) -> None:
        self.warmup()
        self.engine.sync()

        LOGGER.info("开始实盘循环: %s %s", self.config.symbol, self.config.timeframe)
        try:
            for candle in self.feed.stream():
                start = time.time()
                self.engine.update_market(self.config.symbol, candle)
                self.engine.sync()
                self.strategy.on_bar(candle)
                self.engine.sync()
                LOGGER.debug("处理bar耗时 %.3fs", time.time() - start)
        except KeyboardInterrupt:
            LOGGER.info("收到中断信号，准备停止...")
        finally:
            self.shutdown()

    def shutdown(self) -> None:
        LOGGER.info("实盘调度器已停止，刷新账户状态")
        try:
            self.engine.sync()
        except Exception as exc:  # pragma: no cover
            LOGGER.warning("同步失败: %s", exc)


__all__ = ["LiveDataFeed", "ExchangeExecutor", "LiveEngine", "LiveTrader"]
