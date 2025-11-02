"""简化且可扩展的回测引擎实现"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import pandas as pd

from phoenix_quant.config import EngineConfig


@dataclass
class Order:
    """挂单对象"""

    order_id: str
    symbol: str
    side: str  # buy/sell
    order_type: str  # limit/market
    price: float
    quantity: float
    tag: str = ""
    reduce_only: bool = False
    status: str = "open"
    filled_qty: float = 0.0


@dataclass
class Trade:
    """成交记录"""

    timestamp: int
    symbol: str
    side: str
    price: float
    quantity: float
    fee: float
    tag: str


@dataclass
class Position:
    """持仓信息"""

    symbol: str
    quantity: float = 0.0
    avg_price: float = 0.0
    last_update: Optional[int] = None

    def add(self, price: float, qty: float, timestamp: int) -> None:
        total_cost = self.avg_price * self.quantity + price * qty
        self.quantity += qty
        if self.quantity > 0:
            self.avg_price = total_cost / self.quantity
        self.last_update = timestamp

    def reduce(self, price: float, qty: float, timestamp: int) -> float:
        qty = min(qty, self.quantity)
        pnl = (price - self.avg_price) * qty
        self.quantity -= qty
        if self.quantity <= 0:
            self.quantity = 0.0
            self.avg_price = 0.0
        self.last_update = timestamp
        return pnl


class BacktestEngine:
    """用于策略验证的事件驱动式撮合引擎"""

    def __init__(self, config: EngineConfig):
        self.config = config
        self.balance = config.initial_balance
        self._order_seq = 0
        self.orders: Dict[str, Order] = {}
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        self.equity_records: List[Dict] = []
        self.current_prices: Dict[str, float] = {}
        self.current_timestamp: Optional[int] = None

    # ------------------------------------------------------------------
    # 订单管理
    # ------------------------------------------------------------------
    def _next_order_id(self) -> str:
        self._order_seq += 1
        return f"BT-{self._order_seq:05d}"

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
        """提交限价单"""

        order = Order(
            order_id=self._next_order_id(),
            symbol=symbol,
            side=side,
            order_type="limit",
            price=price,
            quantity=quantity,
            tag=tag,
            reduce_only=reduce_only,
        )
        self.orders[order.order_id] = order
        return order

    def submit_market_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        *,
        tag: str = "",
    ) -> Order:
        """提交市价单，按最新价格立即成交"""

        if symbol not in self.current_prices:
            raise ValueError("当前价格未知，无法执行市价单")
        order = Order(
            order_id=self._next_order_id(),
            symbol=symbol,
            side=side,
            order_type="market",
            price=self.current_prices[symbol],
            quantity=quantity,
            tag=tag,
        )
        self._fill_order(order, self.current_prices[symbol], quantity)
        order.status = "filled"
        self.orders[order.order_id] = order
        return order

    def cancel_orders(self, *, tag_prefix: Optional[str] = None) -> None:
        """取消指定标签的订单"""

        for order in self.orders.values():
            if order.status != "open":
                continue
            if tag_prefix and not order.tag.startswith(tag_prefix):
                continue
            order.status = "canceled"

    # ------------------------------------------------------------------
    # 市场驱动
    # ------------------------------------------------------------------
    def update_market(self, symbol: str, candle: List[float]) -> None:
        """更新最新K线并尝试撮合订单"""

        timestamp = int(candle[0])
        high, low, close = candle[2], candle[3], candle[4]
        self.current_prices[symbol] = close
        self.current_timestamp = timestamp

        # 限价撮合
        for order in list(self.orders.values()):
            if order.symbol != symbol or order.status != "open":
                continue
            if order.side == "buy" and low <= order.price:
                self._fill_order(order, order.price, order.quantity)
                order.status = "filled"
            elif order.side == "sell" and high >= order.price:
                self._fill_order(order, order.price, order.quantity)
                order.status = "filled"

        self._record_equity(symbol, timestamp)

    # ------------------------------------------------------------------
    # 权益与持仓
    # ------------------------------------------------------------------
    def _record_equity(self, symbol: str, timestamp: int) -> None:
        position = self.positions.get(symbol)
        position_value = 0.0
        if position and position.quantity > 0:
            position_value = position.quantity * self.current_prices[symbol]
        equity = self.balance + position_value
        self.equity_records.append(
            {
                "timestamp": timestamp,
                "symbol": symbol,
                "equity": equity,
                "balance": self.balance,
                "position_value": position_value,
            }
        )

    def get_equity_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.equity_records)

    def get_trades_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([trade.__dict__ for trade in self.trades])

    def get_total_equity(self, symbol: str) -> float:
        position = self.positions.get(symbol)
        position_value = 0.0
        if position and symbol in self.current_prices:
            position_value = position.quantity * self.current_prices[symbol]
        return self.balance + position_value

    def get_position(self, symbol: str) -> Position:
        if symbol not in self.positions:
            self.positions[symbol] = Position(symbol=symbol)
        return self.positions[symbol]

    # ------------------------------------------------------------------
    # 私有撮合逻辑
    # ------------------------------------------------------------------
    def _fill_order(self, order: Order, price: float, quantity: float) -> None:
        position = self.get_position(order.symbol)
        qty = quantity
        fee_rate = self.config.taker_fee if order.order_type == "market" else self.config.maker_fee
        fee = price * qty * fee_rate

        if order.side == "buy":
            cost = price * qty + fee
            if cost > self.balance:
                qty = self.balance / (price * (1 + fee_rate))
                cost = price * qty + price * qty * fee_rate
                fee = cost - price * qty
            self.balance -= cost
            position.add(price, qty, self.current_timestamp or 0)
        else:
            proceeds = price * qty - fee
            pnl = position.reduce(price, qty, self.current_timestamp or 0)
            self.balance += proceeds
            self.balance += pnl

        order.filled_qty += qty
        trade = Trade(
            timestamp=self.current_timestamp or 0,
            symbol=order.symbol,
            side=order.side,
            price=price,
            quantity=qty,
            fee=fee,
            tag=order.tag,
        )
        self.trades.append(trade)

    # ------------------------------------------------------------------
    # 辅助方法
    # ------------------------------------------------------------------
    def close_position(self, symbol: str, *, portion: float = 1.0, tag: str = "exit") -> Optional[Order]:
        position = self.positions.get(symbol)
        if not position or position.quantity <= 0:
            return None
        qty = position.quantity * portion
        return self.submit_market_order(symbol, "sell", qty, tag=tag)

    def count_open_orders(self, tag_prefix: Optional[str] = None) -> int:
        count = 0
        for order in self.orders.values():
            if order.status != "open":
                continue
            if tag_prefix and not order.tag.startswith(tag_prefix):
                continue
            count += 1
        return count

    def get_orders(self, *, status: Optional[str] = None, tag_prefix: Optional[str] = None) -> List[Order]:
        result = []
        for order in self.orders.values():
            if status and order.status != status:
                continue
            if tag_prefix and not order.tag.startswith(tag_prefix):
                continue
            result.append(order)
        return result
