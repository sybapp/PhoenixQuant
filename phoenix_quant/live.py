"""实时行情与交易执行模块"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
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

        # 根据 market_type 自动调整配置
        market_type = exchange_cfg.market_type.lower()
        if market_type not in ["spot", "future"]:
            raise ValueError(f"不支持的市场类型: {exchange_cfg.market_type}，必须是 'spot' 或 'future'")

        self.market_type = market_type

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

        # 设置市场类型
        if not exchange_cfg.options:
            exchange_cfg.options = {}
        exchange_cfg.options["defaultType"] = market_type
        self.exchange.options.update(exchange_cfg.options)

        LOGGER.info("市场类型: %s | 杠杆: %.1fx", market_type.upper(), exchange_cfg.leverage)

        # 沙盒/测试模式设置
        # 注意：Binance 期货已废弃 testnet，改用 demo 模式（通过 options.demo = true）
        if data_cfg.use_testnet and hasattr(self.exchange, "set_sandbox_mode"):
            # 检查是否是 Binance 期货（已废弃 testnet）
            if exchange_cfg.exchange_id in ["binanceusdm", "binancecoinm"]:
                LOGGER.warning(
                    "Binance 期货已废弃测试网模式！请使用 demo 模式：\n"
                    "  1. 设置 data.use_testnet = false\n"
                    "  2. 设置 exchange.options.demo = true\n"
                    "  3. 使用真实网 API Key（demo 模式会模拟交易）"
                )
                # 不调用 set_sandbox_mode，避免报错
            else:
                # 其他交易所仍然支持 sandbox 模式
                try:
                    self.exchange.set_sandbox_mode(True)
                    LOGGER.info("已启用沙盒模式（testnet）")
                except Exception as exc:
                    LOGGER.warning("启用沙盒模式失败: %s", exc)

        # 检查是否启用了 demo 模式
        if exchange_cfg.options and exchange_cfg.options.get("demo"):
            LOGGER.info("✅ Demo 模式已启用（模拟交易，不会下真实订单）")

        LOGGER.info("加载交易所市场: %s", exchange_cfg.exchange_id)
        self.exchange.load_markets()

    # 账户同步 ---------------------------------------------------------
    def fetch_balance(self) -> Dict[str, Any]:
        return self.exchange.fetch_balance()

    def fetch_positions(self, symbol: str) -> Optional[Dict[str, Any]]:
        if self.market_type == "spot":
            LOGGER.debug("现货模式跳过 fetch_positions")
            return None
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

    def set_leverage(self, symbol: str, leverage: float, market_type: str = "future") -> None:
        """设置杠杆倍数（仅合约支持）"""
        # 现货交易不支持杠杆
        if market_type.lower() == "spot":
            if leverage > 1.0:
                LOGGER.warning("现货交易不支持杠杆，leverage 参数将被忽略")
            return

        # 合约交易杠杆设置
        if leverage <= 1.0:
            LOGGER.info("杠杆设置为 %.1f（无杠杆模式）", leverage)
            return
        if not hasattr(self.exchange, "set_leverage"):
            LOGGER.warning("当前交易所不支持设置杠杆，将使用默认杠杆")
            return
        try:
            self.exchange.set_leverage(int(leverage), symbol)
            LOGGER.info("成功设置杠杆: %dx for %s", int(leverage), symbol)
        except Exception as exc:  # pragma: no cover
            LOGGER.warning("设置杠杆失败（测试网通常不支持杠杆）: %s | 将以无杠杆模式运行", exc)

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
        except ccxt.AuthenticationError as exc:  # type: ignore[attr-defined]  # pragma: no cover
            LOGGER.warning(
                "获取账户余额失败（认证失败）: %s | 请确认 API Key/Secret 是否正确、权限已启用，且环境与 use_testnet 设置匹配",
                exc,
            )
            return
        except Exception as exc:  # pragma: no cover
            LOGGER.warning("获取账户余额失败: %s", exc)
            return
        total = balance.get("total") or balance.get("free") or {}
        if isinstance(total, dict):
            cash = total.get(self.cash_currency)
            if isinstance(cash, (int, float)):
                prev_balance = self.balance
                self.balance = float(cash)
                LOGGER.debug("账户余额同步: %.2f %s (变化: %+.2f)", self.balance, self.cash_currency, self.balance - prev_balance)

    def _sync_position(self) -> None:
        pos = self.executor.fetch_positions(self.symbol)
        if not pos:
            if self.position.quantity != 0:
                LOGGER.info("【持仓】已清空")
            self.position = LivePosition(self.symbol)
            return
        contracts = float(pos.get("contracts") or pos.get("contractSize") or pos.get("size") or 0.0)
        quantity = float(pos.get("positionAmt") or pos.get("amount") or contracts)
        entry_price = float(pos.get("entryPrice") or pos.get("average") or pos.get("avgPrice") or 0.0)
        margin = float(pos.get("margin") or pos.get("initialMargin") or 0.0)

        prev_qty = self.position.quantity
        self.position = LivePosition(self.symbol, quantity=quantity, avg_price=entry_price, margin=margin)

        if quantity != 0 and quantity != prev_qty:
            current_price = self.current_prices.get(self.symbol, entry_price)
            unrealized_pnl = (current_price - entry_price) * quantity if current_price and entry_price else 0
            unrealized_pnl_pct = (unrealized_pnl / (abs(quantity) * entry_price) * 100) if entry_price and quantity else 0
            LOGGER.info(
                "【持仓】数量: %.4f | 均价: %.2f | 当前价: %.2f | 未实现盈亏: %.2f (%.2f%%)",
                quantity, entry_price, current_price, unrealized_pnl, unrealized_pnl_pct
            )

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

    def _simulate_order_matching(self, symbol: str, candle: List[float]) -> None:
        """DRY-RUN模式下的模拟订单撮合

        Args:
            symbol: 交易对
            candle: [timestamp, open, high, low, close, volume]
        """
        if symbol != self.symbol:
            return

        high = float(candle[2])
        low = float(candle[3])

        filled_orders = []

        for order_id, order in list(self.open_orders.items()):
            if order.symbol != symbol or order.status != "pending":
                continue

            # 检查是否成交
            filled = False
            fill_price = order.price

            if order.side == "buy":
                # 买单：当前K线最低价 <= 订单价格时成交
                if low <= order.price:
                    filled = True
                    fill_price = min(order.price, float(candle[1]))  # 使用开盘价或订单价格（取较小）
            else:  # sell
                # 卖单：当前K线最高价 >= 订单价格时成交
                if high >= order.price:
                    filled = True
                    fill_price = max(order.price, float(candle[1]))  # 使用开盘价或订单价格（取较大）

            if filled:
                # 更新订单状态
                order.status = "filled"
                order.filled_qty = order.quantity
                order.price = fill_price  # 使用实际成交价格

                # 从挂单列表移除
                self.open_orders.pop(order_id, None)
                filled_orders.append(order)

                # 更新持仓
                self._update_position_from_fill(order)

                LOGGER.info("【模拟成交】%s | %s %.4f @ %.2f | 标签: %s",
                           symbol, order.side.upper(), order.quantity, fill_price, order.tag)

        # 记录交易
        for order in filled_orders:
            self.trades.append(
                Trade(
                    timestamp=self.current_timestamp or int(time.time() * 1000),
                    symbol=symbol,
                    side=order.side,
                    price=order.price,
                    quantity=order.quantity,
                    fee=0.0,  # DRY-RUN模式不收手续费
                    tag=order.tag,
                    pnl=0.0,
                )
            )

    def _update_position_from_fill(self, order: Order) -> None:
        """根据订单成交更新持仓"""
        if order.side == "buy":
            # 买入增加持仓（做多）
            new_qty = self.position.quantity + order.quantity
            if self.position.quantity == 0:
                self.position.avg_price = order.price
            else:
                # 加仓：重新计算平均价格
                total_value = self.position.quantity * self.position.avg_price + order.quantity * order.price
                self.position.avg_price = total_value / new_qty if new_qty != 0 else 0.0
            self.position.quantity = new_qty
        else:  # sell
            # 卖出减少持仓（如果有多仓）或建立空仓
            new_qty = self.position.quantity - order.quantity
            if self.position.quantity == 0:
                # 开空仓
                self.position.avg_price = order.price
                self.position.quantity = -order.quantity
            elif self.position.quantity > 0:
                # 平多仓
                if new_qty >= 0:
                    self.position.quantity = new_qty
                else:
                    # 平多后反手开空
                    self.position.avg_price = order.price
                    self.position.quantity = new_qty
            else:
                # 加空仓
                total_value = abs(self.position.quantity) * self.position.avg_price + order.quantity * order.price
                self.position.avg_price = total_value / abs(new_qty) if new_qty != 0 else 0.0
                self.position.quantity = new_qty

        self.position.updated_at = time.time()

    def update_market(self, symbol: str, candle: List[float]) -> None:
        self.current_timestamp = int(candle[0])
        self.current_prices[symbol] = float(candle[4])

        # 输出市场数据更新
        ts = datetime.fromtimestamp(int(candle[0]) / 1000).strftime("%Y-%m-%d %H:%M:%S")
        LOGGER.info(
            "市场更新 | 时间: %s | 价格: O=%.2f H=%.2f L=%.2f C=%.2f | 成交量: %.2f",
            ts, candle[1], candle[2], candle[3], candle[4], candle[5]
        )

        # DRY-RUN模式下的模拟撮合
        if not self.trading_enabled:
            self._simulate_order_matching(symbol, candle)

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
            LOGGER.info("【DRY-RUN】限价单 | %s | %s %.4f @ %.2f | 标签: %s", symbol, side.upper(), quantity, price, tag)
            return self._placeholder_order(symbol, side, quantity, price, tag, status="pending")
        params = {"reduceOnly": reduce_only} if reduce_only else {}
        response = self.executor.create_limit_order(symbol, side, quantity, price, params)
        order = self._build_order(response, tag=tag)
        order.reduce_only = reduce_only
        self.orders[order.order_id] = order
        if order.status == "open":
            self.open_orders[order.order_id] = order
        LOGGER.info("【订单创建】限价单 | ID: %s | %s %s %.4f @ %.2f | 标签: %s",
                   order.order_id, symbol, side.upper(), quantity, price, tag)
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
            price = self.current_prices.get(symbol, 0.0)
            LOGGER.info("【DRY-RUN】市价单 | %s | %s %.4f @ ~%.2f | 标签: %s", symbol, side.upper(), quantity, price, tag)
            return self._placeholder_order(symbol, side, quantity, price, tag, status="skipped")

        response = self.executor.create_market_order(symbol, side, quantity)
        order = self._build_order(response, tag=tag)
        order.order_type = "market"
        self.orders[order.order_id] = order
        if order.status != "open":
            self._append_trade(response, tag)
        LOGGER.info("【订单成交】市价单 | ID: %s | %s %s %.4f @ %.2f | 标签: %s",
                   order.order_id, symbol, side.upper(), quantity, order.price, tag)
        return order

    def cancel_orders(self, *, tag_prefix: Optional[str] = None) -> None:
        canceled_count = 0
        for order in list(self.open_orders.values()):
            if tag_prefix and not order.tag.startswith(tag_prefix):
                continue
            try:
                if self.trading_enabled:
                    self.executor.cancel_order(order.order_id, self.symbol)
                order.status = "canceled"
                self.open_orders.pop(order.order_id, None)
                canceled_count += 1
                LOGGER.debug("【撤单】ID: %s | 标签: %s | 价格: %.2f", order.order_id, order.tag, order.price)
            except Exception as exc:  # pragma: no cover
                LOGGER.warning("撤单失败 %s: %s", order.order_id, exc)
        if canceled_count > 0:
            prefix_info = f"标签前缀={tag_prefix}" if tag_prefix else "全部"
            LOGGER.info("【批量撤单】已撤销 %d 个订单 | %s", canceled_count, prefix_info)

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
        # DRY-RUN模式下的挂单也应该添加到 open_orders 中以便跟踪
        if status == "pending":
            self.open_orders[order.order_id] = order
        return order

# ----------------------------------------------------------------------
# 调度器
# ----------------------------------------------------------------------
class LiveTrader:
    """运行策略的调度器"""

    def __init__(self, config: LiveTradingConfig, strategy_cls: Type[ElasticDipStrategy] = ElasticDipStrategy) -> None:
        self.config = config
        self.executor = ExchangeExecutor(config.exchange, config.data)
        self.executor.set_leverage(config.symbol, config.engine.leverage, config.exchange.market_type)

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

        LOGGER.info("=" * 80)
        LOGGER.info("开始实盘循环: %s %s", self.config.symbol, self.config.timeframe)
        LOGGER.info("交易模式: %s", "DRY-RUN（模拟）" if not self.engine.trading_enabled else "实盘交易")
        LOGGER.info("初始余额: %.2f USDT", self.engine.balance)
        LOGGER.info("=" * 80)

        bar_count = 0
        last_heartbeat = time.time()
        heartbeat_interval = self.config.settings.heartbeat_interval if hasattr(self.config.settings, 'heartbeat_interval') else 120

        try:
            for candle in self.feed.stream():
                start = time.time()
                bar_count += 1

                self.engine.update_market(self.config.symbol, candle)
                self.engine.sync()
                self.strategy.on_bar(candle)
                self.engine.sync()

                elapsed = time.time() - start
                LOGGER.debug("处理bar耗时 %.3fs", elapsed)

                # 心跳日志
                if time.time() - last_heartbeat >= heartbeat_interval:
                    self._log_heartbeat(bar_count)
                    last_heartbeat = time.time()

        except KeyboardInterrupt:
            LOGGER.info("收到中断信号，准备停止...")
        finally:
            self.shutdown()

    def _log_heartbeat(self, bar_count: int) -> None:
        """输出心跳日志，显示系统运行状态"""
        equity = self.engine.get_total_equity(self.config.symbol)
        position = self.engine.position
        open_orders = len(self.engine.open_orders)

        LOGGER.info("=" * 80)
        LOGGER.info("【心跳】已处理 %d 根K线", bar_count)
        LOGGER.info("【账户】权益: %.2f USDT | 余额: %.2f USDT | 收益率: %+.2f%%",
                   equity, self.engine.balance, (equity / self.config.engine.initial_balance - 1) * 100)

        if position.quantity != 0:
            current_price = self.engine.current_prices.get(self.config.symbol, position.avg_price)
            unrealized = (current_price - position.avg_price) * position.quantity
            unrealized_pct = (unrealized / (abs(position.quantity) * position.avg_price) * 100) if position.avg_price else 0
            LOGGER.info("【持仓】数量: %.4f | 均价: %.2f | 当前价: %.2f | 未实现: %.2f (%.2f%%)",
                       position.quantity, position.avg_price, current_price, unrealized, unrealized_pct)
        else:
            LOGGER.info("【持仓】空仓")

        LOGGER.info("【订单】挂单数量: %d", open_orders)
        LOGGER.info("【策略】状态: %s", self.strategy.state)

        if self.strategy.cooldown_until:
            remaining = (self.strategy.cooldown_until - datetime.now()).total_seconds() / 60
            if remaining > 0:
                LOGGER.info("【冷却】剩余 %.1f 分钟", remaining)

        LOGGER.info("=" * 80)

    def shutdown(self) -> None:
        LOGGER.info("实盘调度器已停止，刷新账户状态")
        try:
            self.engine.sync()
        except Exception as exc:  # pragma: no cover
            LOGGER.warning("同步失败: %s", exc)


__all__ = ["LiveDataFeed", "ExchangeExecutor", "LiveEngine", "LiveTrader"]
