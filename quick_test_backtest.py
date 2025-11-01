#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速测试回测 - 使用现有数据验证交易触发
"""

import asyncio
from datetime import datetime
import pandas as pd
from backtest_engine import BacktestEngine, HistoricalDataFetcher
import ccxt

# 导入修复后的策略类
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

# 直接在这里定义修复版的策略类
from backtest_strategy import BacktestElasticDipBot
from typing import Dict
import numpy as np


class FixedElasticDipBot(BacktestElasticDipBot):
    """修复版弹性抄底策略 - 移除ATR过滤"""

    def is_fast_drop(self, candles):
        """重写快速下跌判断 - 移除过于严格的波动率过滤"""
        w = self.p["window_min"]
        o, h, l, c = candles[-1][1:5]

        # 单根K线跌幅
        single_drop = (c < o) and ((o - c) / o * 100 >= self.p["drop_pct_single"])

        # 窗口内跌幅
        sub = candles[-w:]
        hi = max(x[2] for x in sub)
        window_drop = (hi - sub[-1][4]) / hi * 100 >= self.p["drop_pct_window"]

        # 移除ATR波动率过滤，直接返回价格条件
        return single_drop or window_drop

    def _wait_bounce_state(self, candles, current_timestamp, current_price):
        """重写等待反弹状态 - 极度宽松的条件便于触发"""
        # 检查超时
        if current_timestamp - self.trigger_time > self.p["delayed_window_sec"] * 1000:
            print(f"[回测超时] 延迟窗口过期，重置")
            self.reset()
            return

        # 极度放宽的反弹确认 - 只要价格不再继续下跌即可
        price_stable = current_price >= self.reference_price * 0.998

        # 如果价格稳定，立即进场
        if price_stable:
            plan = self.compute_attack_plan(current_price)
            self.attack_orders = plan

            for order_plan in plan:
                order = self.engine.create_order(
                    self.symbol, 'buy', 'limit',
                    order_plan["qty"], order_plan["price"]
                )
                order_plan["id"] = order.id

            self.state = 'WAIT_ORDERS'
            total_investment = sum(p['qty'] * p['price'] for p in plan)
            print(f"[下单触发] 参考价:${self.reference_price:.4f}, 当前价:${current_price:.4f}, "
                  f"{len(plan)}档订单，总投资: ${total_investment:.2f}")


# 测试配置
SYMBOL = "DOGE/USDT"
TIMEFRAME = "1m"
INITIAL_BALANCE = 100.0

# 极度宽松的测试参数 - 确保能触发交易
TEST_PARAMS = {
    "timeframe": TIMEFRAME,
    "poll_sec": 2,

    # 极度宽松的触发条件
    "drop_pct_single": 0.2,      # 超低阈值：0.2%的下跌就触发
    "drop_pct_window": 0.5,      # 超低阈值：0.5%的窗口下跌
    "window_min": 5,

    # 趋势过滤
    "ema_fast": 10,
    "ema_slow": 30,

    # 极度宽松的动量指标
    "rsi_period": 14,
    "rsi_oversold": 45.0,        # 超宽松：RSI<45就算超卖

    # 爆仓和资金费率
    "funding_extreme_neg": -0.03,
    "liq_notional_threshold": 2_000_000,

    # 极度宽松的延迟触发
    "delayed_trigger_pct": 0.05,  # 只需0.05%的反弹
    "delayed_window_sec": 60 * 60 * 12,  # 延长到12小时

    # 分层买入
    "layer_pcts": [0.3, 0.6, 0.9, 1.2],
    "layer_pos_ratio": [0.40, 0.30, 0.20, 0.10],

    # 资金管理
    "total_capital": 2000,
    "max_account_ratio": 0.20,

    # 止盈止损
    "take_profit_pct": [0.5, 1.0, 2.0],
    "hard_stop_extra": 1.0,
    "sl_time_grace_sec": 300,
    "trailing_stop_pct": 0.5,

    # 成交量条件 - 极度宽松
    "vol_shrink_ratio": 0.5,
    "vol_recover_ma_short": 2,
    "vol_recover_ma_long": 15,
    "vol_recover_ratio": 1.05,   # 只需5%的放量
    "tick_vol_ratio": 1.1,

    # 风险管理 - 宽松
    "max_daily_trades": 10,
    "max_consecutive_losses": 10,
}


async def quick_test():
    """快速测试回测"""
    print("="*70)
    print(" " * 20 + "快速测试 - 验证交易触发")
    print("="*70)
    print(f"\n交易对: {SYMBOL}")
    print(f"时间框架: {TIMEFRAME}")
    print("\n使用极度宽松的参数进行测试...\n")

    # 初始化交易所
    exchange = ccxt.binance({
        "apiKey": "kflCxmrjxzyNuaM60yvhFTCvFZBMRzCX2hniLLfEMycGJ2j2e6OMrsOE8Gzd5H7P",
        "secret": "Z9GOv6MoF2WQfi7iE21zkFliHzMJ1ENRtLixnvkp51lX4JA9jxsKnZ9ONak573An",
        "enableRateLimit": True,
        "options": {"defaultType": "future"},
    })
    exchange.set_sandbox_mode(True)

    # 查找现有数据文件
    import os
    import glob

    # 查找匹配的数据文件
    pattern = f"*testnet_data_{SYMBOL.replace('/', '_')}_{TIMEFRAME}_*.csv"
    files = glob.glob(pattern)

    if not files:
        pattern2 = f"improved_testnet_data_{SYMBOL.replace('/', '_')}_{TIMEFRAME}_*.csv"
        files = glob.glob(pattern2)

    if not files:
        print(f"错误: 未找到数据文件 {pattern}")
        print("请先运行 real_market_backtest.py 或 improved_real_market_backtest.py 获取数据")
        return

    data_file = files[0]
    print(f"使用数据文件: {data_file}\n")

    # 加载数据
    data_fetcher = HistoricalDataFetcher(exchange, use_testnet=True)
    df = data_fetcher.load_data(data_file)

    if df.empty:
        print("错误: 数据文件为空")
        return

    print(f"数据加载完成，共 {len(df)} 根K线\n")

    # 初始化回测引擎
    engine = BacktestEngine(
        initial_balance=INITIAL_BALANCE,
        taker_fee=0.0004,
        maker_fee=0.0002
    )

    # 初始化策略 - 使用修复版策略类
    strategy = FixedElasticDipBot(engine, SYMBOL, TEST_PARAMS)

    # 运行回测
    print("开始测试回测...\n")
    print("-" * 70)

    candle_buffer = []
    buffer_size = max(TEST_PARAMS["ema_slow"], TEST_PARAMS["rsi_period"]) + 50

    trade_triggers = 0

    for idx, row in df.iterrows():
        candle = [
            row['timestamp'],
            row['open'],
            row['high'],
            row['low'],
            row['close'],
            row['volume']
        ]

        candle_buffer.append(candle)
        if len(candle_buffer) > buffer_size:
            candle_buffer.pop(0)

        # 更新市场数据
        engine.update_market(row['timestamp'], {SYMBOL: candle})

        # 执行策略
        if len(candle_buffer) >= buffer_size:
            old_state = strategy.state
            strategy.step(candle_buffer, row['timestamp'])

            # 检测状态变化
            if old_state == 'IDLE' and strategy.state != 'IDLE':
                trade_triggers += 1
                print(f"[触发 #{trade_triggers}] 时间: {row['datetime']}, 价格: ${row['close']:.4f}")

        # 每1000根K线打印进度
        if idx % 1000 == 0 and idx > 0:
            equity = engine.get_total_equity()
            returns = (equity - INITIAL_BALANCE) / INITIAL_BALANCE * 100
            print(f"[进度 {idx}/{len(df)}] 权益: ${equity:.2f}, 收益: {returns:.2f}%, 触发次数: {trade_triggers}")

    print("\n" + "-" * 70)
    print(f"\n测试完成!")
    print(f"总触发次数: {trade_triggers}")

    # 获取交易记录
    trades_df = engine.get_trades_dataframe()

    if not trades_df.empty:
        print(f"总成交笔数: {len(trades_df)}")
        print(f"\n前5笔交易:")
        print(trades_df.head())
    else:
        print("\n⚠️  警告: 没有产生任何交易!")
        print("\n可能的原因:")
        print("1. 数据中确实没有符合条件的下跌")
        print("2. 策略逻辑有问题")
        print("3. 参数仍然过于严格")
        print("\n建议: 检查数据文件中的价格波动范围")

        # 分析数据
        print(f"\n数据分析:")
        print(f"价格范围: ${df['close'].min():.4f} - ${df['close'].max():.4f}")
        print(f"平均价格: ${df['close'].mean():.4f}")

        # 计算最大单根K线跌幅
        df['pct_change'] = df['close'].pct_change() * 100
        max_drop = df['pct_change'].min()
        print(f"最大单根K线跌幅: {max_drop:.2f}%")

        # 统计下跌超过0.2%的K线数量
        drops = df[df['pct_change'] < -0.2]
        print(f"下跌超过0.2%的K线数量: {len(drops)}")

    # 最终权益
    final_equity = engine.get_total_equity()
    final_returns = (final_equity - INITIAL_BALANCE) / INITIAL_BALANCE * 100
    print(f"\n最终权益: ${final_equity:.2f}")
    print(f"最终收益: {final_returns:.2f}%")


if __name__ == "__main__":
    asyncio.run(quick_test())
