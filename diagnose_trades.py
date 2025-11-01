#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
诊断交易统计问题
"""

import pandas as pd
from backtest_engine import BacktestEngine, HistoricalDataFetcher
import ccxt
import asyncio
from datetime import datetime

async def diagnose():
    # 加载数据
    exchange = ccxt.binance({
        "apiKey": "kflCxmrjxzyNuaM60yvhFTCvFZBMRzCX2hniLLfEMycGJ2j2e6OMrsOE8Gzd5H7P",
        "secret": "Z9GOv6MoF2WQfi7iE21zkFliHzMJ1ENRtLixnvkp51lX4JA9jxsKnZ9ONak573An",
        "enableRateLimit": True,
        "options": {"defaultType": "future"},
    })
    exchange.set_sandbox_mode(True)

    data_fetcher = HistoricalDataFetcher(exchange, use_testnet=True)

    # 找数据文件
    import glob
    files = glob.glob("testnet_data_DOGE_USDT_1m_*.csv")
    if not files:
        files = glob.glob("improved_testnet_data_DOGE_USDT_1m_*.csv")

    if not files:
        print("未找到数据文件")
        return

    df = data_fetcher.load_data(files[0])
    print(f"数据文件: {files[0]}")
    print(f"K线数量: {len(df)}\n")

    # 运行回测（使用改进的策略）
    from backtest_strategy import BacktestElasticDipBot
    import numpy as np

    class FixedElasticDipBot(BacktestElasticDipBot):
        def is_fast_drop(self, candles):
            w = self.p["window_min"]
            o, h, l, c = candles[-1][1:5]
            single_drop = (c < o) and ((o - c) / o * 100 >= self.p["drop_pct_single"])
            sub = candles[-w:]
            hi = max(x[2] for x in sub)
            window_drop = (hi - sub[-1][4]) / hi * 100 >= self.p["drop_pct_window"]
            return single_drop or window_drop

        def _wait_bounce_state(self, candles, current_timestamp, current_price):
            if current_timestamp - self.trigger_time > self.p["delayed_window_sec"] * 1000:
                self.reset()
                return

            price_stable = current_price >= self.reference_price * 0.998

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

    TEST_PARAMS = {
        "timeframe": "1m",
        "poll_sec": 2,
        "drop_pct_single": 0.2,
        "drop_pct_window": 0.5,
        "window_min": 5,
        "ema_fast": 10,
        "ema_slow": 30,
        "rsi_period": 14,
        "rsi_oversold": 45.0,
        "funding_extreme_neg": -0.03,
        "liq_notional_threshold": 2_000_000,
        "delayed_trigger_pct": 0.05,
        "delayed_window_sec": 60 * 60 * 12,
        "layer_pcts": [0.3, 0.6, 0.9, 1.2],
        "layer_pos_ratio": [0.40, 0.30, 0.20, 0.10],
        "total_capital": 2000,
        "max_account_ratio": 0.20,
        "take_profit_pct": [0.5, 1.0, 2.0],
        "hard_stop_extra": 1.0,
        "sl_time_grace_sec": 300,
        "trailing_stop_pct": 0.5,
        "vol_shrink_ratio": 0.5,
        "vol_recover_ma_short": 2,
        "vol_recover_ma_long": 15,
        "vol_recover_ratio": 1.05,
        "tick_vol_ratio": 1.1,
        "max_daily_trades": 10,
        "max_consecutive_losses": 10,
    }

    engine = BacktestEngine(initial_balance=100.0, taker_fee=0.0004, maker_fee=0.0002)
    strategy = FixedElasticDipBot(engine, "DOGE/USDT", TEST_PARAMS)

    candle_buffer = []
    buffer_size = 100

    for idx, row in df.iterrows():
        candle = [row['timestamp'], row['open'], row['high'], row['low'], row['close'], row['volume']]
        candle_buffer.append(candle)
        if len(candle_buffer) > buffer_size:
            candle_buffer.pop(0)

        engine.update_market(row['timestamp'], {"DOGE/USDT": candle})

        if len(candle_buffer) >= buffer_size:
            strategy.step(candle_buffer, row['timestamp'])

    print("="*70)
    print("交易记录分析")
    print("="*70)

    trades_df = engine.get_trades_dataframe()
    print(f"\n总交易记录数: {len(trades_df)}")

    if not trades_df.empty:
        buy_trades = trades_df[trades_df['side'] == 'buy']
        sell_trades = trades_df[trades_df['side'] == 'sell']

        print(f"买入次数: {len(buy_trades)}")
        print(f"卖出次数: {len(sell_trades)}")
        print(f"\n买入总数量: {buy_trades['quantity'].sum():.4f}")
        print(f"卖出总数量: {sell_trades['quantity'].sum():.4f}")
        print(f"未平仓数量: {buy_trades['quantity'].sum() - sell_trades['quantity'].sum():.4f}")

        print(f"\n前10笔交易:")
        print(trades_df[['datetime', 'side', 'price', 'quantity', 'fee']].head(10))

        if len(trades_df) > 10:
            print(f"\n后10笔交易:")
            print(trades_df[['datetime', 'side', 'price', 'quantity', 'fee']].tail(10))

    # 检查持仓
    print(f"\n="*70)
    print("持仓分析")
    print("="*70)
    positions = engine.positions
    if positions:
        for symbol, pos in positions.items():
            print(f"\n{symbol}:")
            print(f"  数量: {pos.quantity:.4f}")
            print(f"  成本价: ${pos.avg_price:.4f}")
            print(f"  当前市值: ${pos.market_value:.2f}")
            print(f"  未实现盈亏: ${pos.unrealized_pnl:.2f}")
    else:
        print("无持仓")

    # 权益分析
    print(f"\n="*70)
    print("权益分析")
    print("="*70)
    print(f"现金余额: ${engine.balance:.2f}")
    print(f"持仓市值: ${sum(p.market_value for p in engine.positions.values()):.2f}")
    print(f"总权益: ${engine.get_total_equity():.2f}")

if __name__ == "__main__":
    asyncio.run(diagnose())
