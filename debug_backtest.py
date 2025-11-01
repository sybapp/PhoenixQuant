#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
调试版回测 - 添加详细日志和更宽松的条件
"""

import asyncio
from datetime import datetime, timedelta
import ccxt
import numpy as np
import pandas as pd
from typing import Dict, List

from backtest_engine import BacktestEngine, HistoricalDataFetcher
from backtest_strategy import BacktestElasticDipBot
from backtest_analysis import BacktestAnalyzer

# 超宽松策略参数
DEBUG_PARAMS = {
    "timeframe": "5m",
    "poll_sec": 2,

    # 大幅放宽价格条件
    "drop_pct_single": 0.5,      # 单根K线跌幅0.5%
    "drop_pct_window": 1.0,      # 窗口跌幅1.0%
    "window_min": 5,

    # 趋势过滤
    "ema_fast": 10,
    "ema_slow": 30,

    # 动量指标
    "rsi_period": 14,
    "rsi_oversold": 40.0,        # 大幅放宽超卖条件

    # 爆仓和资金费率 - 几乎不要求
    "funding_extreme_neg": -1.0,  # 几乎不限制
    "liq_notional_threshold": 1000,  # 极低阈值

    # 延迟触发
    "delayed_trigger_pct": 0.1,   # 极小反弹
    "delayed_window_sec": 60 * 60,  # 1小时

    # 分层买入
    "layer_pcts": [0.5, 1.0, 1.5],  # 更少层级
    "layer_pos_ratio": [0.33, 0.33, 0.34],

    # 资金管理
    "total_capital": 1000,
    "max_account_ratio": 0.20,

    # 止盈止损
    "take_profit_pct": [2.0],
    "hard_stop_extra": 2.0,
    "sl_time_grace_sec": 120,

    # 成交量条件 - 大幅放宽
    "vol_shrink_ratio": 0.9,
    "vol_recover_ma_short": 3,
    "vol_recover_ma_long": 10,
    "vol_recover_ratio": 1.05,
    "tick_vol_ratio": 1.1,

    # 风险管理
    "max_daily_trades": 10,
    "max_consecutive_losses": 10,
}


class DebugElasticDipBot(BacktestElasticDipBot):
    """调试版策略 - 添加详细日志"""

    def __init__(self, backtest_engine: BacktestEngine, symbol: str, params: Dict):
        super().__init__(backtest_engine, symbol, params)
        self.debug_count = 0

    def step(self, candles: List, current_timestamp: float):
        """策略步骤 - 添加调试信息"""
        if len(candles) < max(self.p["ema_slow"], self.p["rsi_period"]) + 10:
            return

        current_price = candles[-1][4]

        # 每100根K线打印一次调试信息
        self.debug_count += 1
        if self.debug_count % 100 == 0:
            self._print_debug_info(candles, current_timestamp, current_price)

        # 模拟市场条件
        self.simulate_market_conditions(candles)

        # 计算信号强度
        self.signal_strength = self.calculate_signal_strength(candles)

        # 检查风险管理
        if not self.can_trade(current_timestamp):
            return

        # 检查订单成交
        if self.attack_orders:
            self.check_order_fills()

        # 更新移动止损
        if self.position_qty > 0:
            self.update_trailing_stop(current_price)

        # 状态机
        if self.state == 'IDLE':
            self._debug_idle_state(candles, current_timestamp, current_price)
        elif self.state == 'WAIT_FOR_BOUNCE':
            self._wait_bounce_state(candles, current_timestamp, current_price)
        elif self.state == 'WAIT_ORDERS':
            self._wait_orders_state(candles, current_timestamp, current_price)
        elif self.state == 'MANAGE':
            self._manage_state(candles, current_timestamp, current_price)
        elif self.state == 'COOLDOWN':
            if self.cooldown_until and current_timestamp >= self.cooldown_until:
                self.state = 'IDLE'
                self.cooldown_until = None

    def _print_debug_info(self, candles, current_timestamp, current_price):
        """打印调试信息"""
        fast_drop = self.is_fast_drop(candles)
        oversold = self.is_oversold(candles)
        liq_spike = self.is_liquidation_spike()
        funding_extreme = self.is_funding_extreme()

        print(f"\n[调试] 价格: ${current_price:.2f}, 状态: {self.state}")
        print(f"      快速下跌: {fast_drop}, 超卖: {oversold}")
        print(f"      爆仓: {liq_spike}, 资金费率: {funding_extreme}")
        print(f"      信号强度: {self.signal_strength:.1f}%")

    def _debug_idle_state(self, candles, current_timestamp, current_price):
        """调试空闲状态 - 极宽松条件"""
        # 检查基本条件
        fast_drop = self.is_fast_drop(candles)
        oversold = self.is_oversold(candles)

        # 极宽松条件：只要满足快速下跌或超卖
        if fast_drop or oversold:
            self.reference_price = current_price
            self.trigger_time = current_timestamp
            self.state = 'WAIT_FOR_BOUNCE'
            self.daily_trade_count += 1

            print(f"\n[触发] 快速下跌:{fast_drop} 超卖:{oversold}")
            print(f"      价格: ${current_price:.2f}, 信号强度: {self.signal_strength:.1f}%")

    def calculate_signal_strength(self, candles) -> float:
        """极宽松信号强度计算"""
        if len(candles) < 10:
            return 0.0

        # 简单计算：基于价格跌幅
        closes = [c[4] for c in candles]
        current_price = closes[-1]

        # 计算近期高点
        lookback = min(20, len(closes))
        high_price = max(closes[-lookback:])
        drawdown = (high_price - current_price) / high_price * 100

        # 信号强度基于跌幅
        signal = min(100.0, drawdown * 10)

        return signal


async def run_debug_backtest():
    """运行调试版回测"""
    print("="*70)
    print(" " * 20 + "调试版弹性抄底策略回测")
    print("="*70)

    # 使用模拟数据
    df = generate_debug_data()

    print(f"数据准备完成，共 {len(df)} 根K线\n")

    # 初始化回测引擎
    engine = BacktestEngine(initial_balance=10000.0)

    # 初始化调试策略
    strategy = DebugElasticDipBot(engine, "BTC/USDT", DEBUG_PARAMS)

    # 运行回测
    print("开始回测...\n")
    print("-" * 70)

    candle_buffer = []
    buffer_size = 100

    for idx, row in df.iterrows():
        candle = [row['timestamp'], row['open'], row['high'], row['low'], row['close'], row['volume']]
        candle_buffer.append(candle)
        if len(candle_buffer) > buffer_size:
            candle_buffer.pop(0)

        engine.update_market(row['timestamp'], {"BTC/USDT": candle})

        if len(candle_buffer) >= buffer_size:
            strategy.step(candle_buffer, row['timestamp'])

        if idx % 500 == 0:
            equity = engine.get_total_equity()
            print(f"进度: {idx}/{len(df)}, 权益: ${equity:.2f}")

    print("\n" + "-" * 70)
    print("\n回测完成!\n")

    # 分析结果
    equity_df = engine.get_equity_dataframe()
    trades_df = engine.get_trades_dataframe()
    analyzer = BacktestAnalyzer(equity_df, trades_df, 10000.0)
    analyzer.print_report()

    print("回测结束!")


def generate_debug_data():
    """生成包含大幅下跌的调试数据"""
    print("生成调试数据...")

    np.random.seed(42)
    n_candles = 2000
    base_price = 40000

    # 生成时间戳
    start_time = datetime.now() - timedelta(days=7)
    timestamps = [int((start_time + timedelta(minutes=5*i)).timestamp() * 1000)
                  for i in range(n_candles)]

    prices = []
    current_price = base_price

    # 专门设计包含大幅下跌的数据
    for i in range(n_candles):
        # 在特定位置制造大幅下跌
        if i in [300, 500, 800, 1200, 1500, 1800]:
            # 制造3-8%的大幅下跌
            drop_pct = np.random.uniform(0.03, 0.08)
            current_price *= (1 - drop_pct)
            print(f"[数据生成] 在第{i}根K线制造{drop_pct*100:.1f}%下跌，价格: ${current_price:.2f}")
        else:
            # 正常波动
            change = np.random.normal(0, 0.001)
            current_price *= (1 + change)

        # 生成OHLCV数据
        open_price = current_price * (1 + np.random.uniform(-0.001, 0.001))
        high = max(open_price, current_price) * (1 + abs(np.random.uniform(0, 0.003)))
        low = min(open_price, current_price) * (1 - abs(np.random.uniform(0, 0.003)))
        close_price = current_price
        volume = np.random.uniform(100, 1000)

        prices.append([timestamps[i], open_price, high, low, close_price, volume])

    df = pd.DataFrame(prices, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')

    print(f"生成了 {len(df)} 根包含大幅下跌的K线数据")
    return df


if __name__ == "__main__":
    asyncio.run(run_debug_backtest())