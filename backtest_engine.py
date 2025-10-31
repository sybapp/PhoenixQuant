# -*- coding: utf-8 -*-
"""
回测引擎 - 基于币安测试网历史数据
支持弹性抄底策略的历史模拟回测
"""

import asyncio
import time
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
import numpy as np
import pandas as pd
import ccxt


class BacktestOrder:
    """回测订单类"""
    def __init__(self, order_id: str, symbol: str, side: str, order_type: str,
                 price: float, quantity: float, timestamp: float):
        self.id = order_id
        self.symbol = symbol
        self.side = side  # 'buy' or 'sell'
        self.type = order_type  # 'limit', 'market', 'stop_market'
        self.price = price
        self.quantity = quantity
        self.timestamp = timestamp
        self.filled_qty = 0.0
        self.filled_price = 0.0
        self.status = 'new'  # 'new', 'filled', 'partially_filled', 'canceled'
        self.stop_price = None

    def __repr__(self):
        return (f"Order({self.id}, {self.side} {self.quantity}@{self.price}, "
                f"status={self.status}, filled={self.filled_qty})")


class BacktestPosition:
    """回测持仓类"""
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.quantity = 0.0
        self.avg_price = 0.0
        self.realized_pnl = 0.0
        self.trades = []

    def add_position(self, price: float, qty: float):
        """增加持仓"""
        if self.quantity == 0:
            self.avg_price = price
            self.quantity = qty
        else:
            total_cost = self.avg_price * self.quantity + price * qty
            self.quantity += qty
            self.avg_price = total_cost / self.quantity if self.quantity > 0 else 0.0

    def reduce_position(self, price: float, qty: float) -> float:
        """减少持仓，返回实现盈亏"""
        if qty > self.quantity:
            qty = self.quantity
        pnl = (price - self.avg_price) * qty
        self.quantity -= qty
        self.realized_pnl += pnl

        if self.quantity <= 0:
            self.quantity = 0.0
            self.avg_price = 0.0

        return pnl

    def get_unrealized_pnl(self, current_price: float) -> float:
        """获取未实现盈亏"""
        if self.quantity <= 0:
            return 0.0
        return (current_price - self.avg_price) * self.quantity


class BacktestEngine:
    """回测引擎"""
    def __init__(self, initial_balance: float = 10000.0,
                 taker_fee: float = 0.0004, maker_fee: float = 0.0002):
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.taker_fee = taker_fee
        self.maker_fee = maker_fee

        # 订单和持仓
        self.orders: List[BacktestOrder] = []
        self.positions: Dict[str, BacktestPosition] = {}
        self.order_id_counter = 0

        # 回测数据
        self.current_time = 0.0
        self.current_prices = {}

        # 统计数据
        self.equity_curve = []
        self.trades_history = []
        self.balance_history = []
        self.timestamps = []

    def create_order(self, symbol: str, side: str, order_type: str,
                    quantity: float, price: Optional[float] = None,
                    stop_price: Optional[float] = None) -> BacktestOrder:
        """创建订单"""
        self.order_id_counter += 1
        order = BacktestOrder(
            order_id=f"BT_{self.order_id_counter}",
            symbol=symbol,
            side=side,
            order_type=order_type,
            price=price or 0.0,
            quantity=quantity,
            timestamp=self.current_time
        )
        order.stop_price = stop_price
        self.orders.append(order)
        return order

    def cancel_order(self, order_id: str):
        """取消订单"""
        for order in self.orders:
            if order.id == order_id and order.status == 'new':
                order.status = 'canceled'

    def _execute_trade(self, order: BacktestOrder, fill_price: float, fill_qty: float):
        """执行交易"""
        symbol = order.symbol

        # 确保持仓对象存在
        if symbol not in self.positions:
            self.positions[symbol] = BacktestPosition(symbol)

        position = self.positions[symbol]

        # 计算手续费
        fee = fill_price * fill_qty * (self.taker_fee if order.type == 'market' else self.maker_fee)

        # 执行交易
        if order.side == 'buy':
            # 买入
            cost = fill_price * fill_qty + fee
            if cost > self.balance:
                # 余额不足，按比例减少成交量
                fill_qty = (self.balance / (fill_price * (1 + self.taker_fee)))
                cost = self.balance
                fee = cost - fill_price * fill_qty

            self.balance -= cost
            position.add_position(fill_price, fill_qty)

        else:  # sell
            # 卖出
            proceeds = fill_price * fill_qty - fee
            self.balance += proceeds
            pnl = position.reduce_position(fill_price, fill_qty)

        # 更新订单状态
        order.filled_qty += fill_qty
        order.filled_price = ((order.filled_price * (order.filled_qty - fill_qty) +
                               fill_price * fill_qty) / order.filled_qty
                              if order.filled_qty > 0 else fill_price)

        if order.filled_qty >= order.quantity:
            order.status = 'filled'
        else:
            order.status = 'partially_filled'

        # 记录交易
        self.trades_history.append({
            'timestamp': self.current_time,
            'symbol': symbol,
            'side': order.side,
            'price': fill_price,
            'quantity': fill_qty,
            'fee': fee,
            'order_id': order.id
        })

    def update_market(self, timestamp: float, ohlcv_data: Dict[str, List]):
        """更新市场数据并尝试撮合订单

        Args:
            timestamp: 当前时间戳
            ohlcv_data: 格式 {symbol: [timestamp, open, high, low, close, volume]}
        """
        self.current_time = timestamp

        # 更新当前价格
        for symbol, candle in ohlcv_data.items():
            self.current_prices[symbol] = candle[4]  # close price

            # 撮合订单
            self._match_orders(symbol, candle)

        # 记录权益曲线
        equity = self.get_total_equity()
        self.equity_curve.append(equity)
        self.balance_history.append(self.balance)
        self.timestamps.append(timestamp)

    def _match_orders(self, symbol: str, candle: List):
        """撮合订单"""
        _, open_price, high, low, close, volume = candle

        for order in self.orders:
            if order.symbol != symbol or order.status not in ('new', 'partially_filled'):
                continue

            remaining_qty = order.quantity - order.filled_qty

            if order.type == 'limit':
                # 限价单撮合
                if order.side == 'buy' and low <= order.price:
                    # 买单：价格跌到或低于限价
                    fill_price = min(order.price, open_price)
                    self._execute_trade(order, fill_price, remaining_qty)

                elif order.side == 'sell' and high >= order.price:
                    # 卖单：价格涨到或高于限价
                    fill_price = max(order.price, open_price)
                    self._execute_trade(order, fill_price, remaining_qty)

            elif order.type == 'market':
                # 市价单立即成交
                fill_price = open_price
                self._execute_trade(order, fill_price, remaining_qty)

            elif order.type == 'stop_market' or order.type == 'STOP_MARKET':
                # 止损单
                if order.stop_price and low <= order.stop_price:
                    # 触发止损
                    fill_price = min(order.stop_price, open_price)
                    self._execute_trade(order, fill_price, remaining_qty)

    def get_position(self, symbol: str) -> BacktestPosition:
        """获取持仓"""
        if symbol not in self.positions:
            self.positions[symbol] = BacktestPosition(symbol)
        return self.positions[symbol]

    def get_total_equity(self) -> float:
        """获取总权益（余额 + 持仓市值）"""
        equity = self.balance
        for symbol, position in self.positions.items():
            if position.quantity > 0 and symbol in self.current_prices:
                equity += position.quantity * self.current_prices[symbol]
        return equity

    def get_statistics(self) -> Dict:
        """获取回测统计数据"""
        if len(self.equity_curve) < 2:
            return {}

        equity_array = np.array(self.equity_curve)
        returns = np.diff(equity_array) / equity_array[:-1]

        # 计算各项指标
        total_return = (equity_array[-1] - self.initial_balance) / self.initial_balance * 100

        # 夏普比率 (假设无风险利率为0，年化252天)
        if len(returns) > 0 and np.std(returns) > 0:
            sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252 * 24 * 60)  # 分钟数据
        else:
            sharpe_ratio = 0.0

        # 最大回撤
        cummax = np.maximum.accumulate(equity_array)
        drawdown = (equity_array - cummax) / cummax * 100
        max_drawdown = np.min(drawdown)

        # 胜率统计
        winning_trades = [t for t in self.trades_history if t['side'] == 'sell']
        if winning_trades:
            # 这里简化处理，实际需要配对买卖单
            win_count = len([t for t in winning_trades if self.positions.get(t['symbol'], None)
                           and self.positions[t['symbol']].realized_pnl > 0])
            win_rate = win_count / len(winning_trades) * 100 if winning_trades else 0
        else:
            win_rate = 0.0

        # 总手续费
        total_fees = sum(t['fee'] for t in self.trades_history)

        stats = {
            'initial_balance': self.initial_balance,
            'final_equity': equity_array[-1],
            'total_return_pct': total_return,
            'max_drawdown_pct': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'total_trades': len(self.trades_history),
            'win_rate_pct': win_rate,
            'total_fees': total_fees,
            'final_balance': self.balance,
        }

        return stats

    def get_equity_dataframe(self) -> pd.DataFrame:
        """获取权益曲线DataFrame"""
        return pd.DataFrame({
            'timestamp': self.timestamps,
            'datetime': [datetime.fromtimestamp(ts/1000) for ts in self.timestamps],
            'equity': self.equity_curve,
            'balance': self.balance_history
        })

    def get_trades_dataframe(self) -> pd.DataFrame:
        """获取交易记录DataFrame"""
        if not self.trades_history:
            return pd.DataFrame()

        df = pd.DataFrame(self.trades_history)
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df


class HistoricalDataFetcher:
    """历史数据获取器"""
    def __init__(self, exchange: ccxt.Exchange, use_testnet: bool = True):
        self.exchange = exchange
        self.use_testnet = use_testnet

        if use_testnet:
            # 币安测试网配置
            self.exchange.set_sandbox_mode(True)

    async def fetch_historical_data(self, symbol: str, timeframe: str = '1m',
                                   start_time: Optional[datetime] = None,
                                   end_time: Optional[datetime] = None,
                                   limit: int = 1000) -> pd.DataFrame:
        """获取历史K线数据"""
        all_candles = []

        if start_time:
            since = int(start_time.timestamp() * 1000)
        else:
            since = int((datetime.now() - timedelta(days=7)).timestamp() * 1000)

        if end_time:
            until = int(end_time.timestamp() * 1000)
        else:
            until = int(datetime.now().timestamp() * 1000)

        print(f"正在获取 {symbol} 历史数据: {start_time} 至 {end_time}")

        current_since = since
        while current_since < until:
            try:
                # 使用ccxt获取历史数据
                ohlcv = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self.exchange.fetch_ohlcv(
                        symbol, timeframe, since=current_since, limit=limit
                    )
                )

                if not ohlcv:
                    break

                all_candles.extend(ohlcv)

                # 更新时间戳
                current_since = ohlcv[-1][0] + 1

                # 避免请求过快
                await asyncio.sleep(0.2)

                print(f"已获取 {len(all_candles)} 条K线数据...")

            except Exception as e:
                print(f"获取数据出错: {e}")
                await asyncio.sleep(1)
                continue

        # 转换为DataFrame
        if all_candles:
            df = pd.DataFrame(
                all_candles,
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')

            # 过滤时间范围
            df = df[(df['timestamp'] >= since) & (df['timestamp'] <= until)]

            print(f"总共获取 {len(df)} 条K线数据")
            return df
        else:
            return pd.DataFrame()

    def save_data(self, df: pd.DataFrame, filename: str):
        """保存数据到文件"""
        df.to_csv(filename, index=False)
        print(f"数据已保存到 {filename}")

    def load_data(self, filename: str) -> pd.DataFrame:
        """从文件加载数据"""
        df = pd.read_csv(filename)
        df['datetime'] = pd.to_datetime(df['datetime'])
        print(f"从 {filename} 加载了 {len(df)} 条数据")
        return df
