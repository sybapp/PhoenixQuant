# -*- coding: utf-8 -*-
"""
弹性抄底策略回测主程序
使用币安测试网数据运行历史回测
"""

import asyncio
from datetime import datetime, timedelta
import ccxt
import sys

from backtest_engine import BacktestEngine, HistoricalDataFetcher
from backtest_strategy import BacktestElasticDipBot
from backtest_analysis import BacktestAnalyzer

# ========= 回测配置 =========
# 币安测试网API（如果需要实时获取数据）
API_KEY = "kflCxmrjxzyNuaM60yvhFTCvFZBMRzCX2hniLLfEMycGJ2j2e6OMrsOE8Gzd5H7P"
API_SECRET = "Z9GOv6MoF2WQfi7iE21zkFliHzMJ1ENRtLixnvkp51lX4JA9jxsKnZ9ONak573An"

# 回测参数
SYMBOL = "BTC/USDT"
INITIAL_BALANCE = 10000.0  # 初始资金
TIMEFRAME = "1m"

# 回测时间范围
BACKTEST_START = datetime(2025, 10, 1)  # 开始日期
BACKTEST_END = datetime(2025, 10, 7)    # 结束日期
# 也可以设置为最近N天
# BACKTEST_START = datetime.now() - timedelta(days=7)
# BACKTEST_END = datetime.now()

# 策略参数（使用BTC预设）
STRATEGY_PARAMS = {
    "timeframe": "1m",
    "poll_sec": 2,
    "drop_pct_single": 1.0,
    "drop_pct_window": 3.0,
    "window_min": 5,
    "ema_fast": 20,
    "ema_slow": 60,
    "vol_shrink_ratio": 0.6,
    "rsi_period": 14,
    "rsi_oversold": 25.0,
    "funding_extreme_neg": -0.05,
    "liq_window_sec": 60,
    "liq_notional_threshold": 8_000_000,
    "delayed_trigger_pct": 1.0,
    "delayed_window_sec": 60 * 60 * 12,  # 12小时
    "layer_pcts": [0.8, 1.4, 2.0, 2.6, 3.3],
    "layer_pos_ratio": [0.10, 0.15, 0.20, 0.25, 0.30],
    "total_capital": 1000,
    "max_account_ratio": 0.30,
    "take_profit_pct": 1.0,
    "hard_stop_extra": 0.8,
    "sl_time_grace_sec": 30,
    "vol_recover_ma_short": 5,
    "vol_recover_ma_long": 20,
    "vol_recover_ratio": 1.15,
    "tick_vol_ratio": 1.30,
}


async def run_backtest():
    """运行回测"""
    print("="*70)
    print(" " * 20 + "弹性抄底策略回测")
    print("="*70)
    print(f"\n交易对: {SYMBOL}")
    print(f"初始资金: ${INITIAL_BALANCE:,.2f}")
    print(f"回测时间: {BACKTEST_START} 至 {BACKTEST_END}")
    print(f"K线周期: {TIMEFRAME}")
    print("\n正在准备数据...\n")

    # 1. 初始化交易所
    exchange = ccxt.binance({
        "apiKey": API_KEY,
        "secret": API_SECRET,
        "enableRateLimit": True,
        "options": {"defaultType": "future"},
    })
    exchange.set_sandbox_mode(True)  # 使用测试网

    # 2. 获取历史数据
    data_fetcher = HistoricalDataFetcher(exchange, use_testnet=True)

    try:
        # 尝试从文件加载数据（可选）
        import os
        data_file = f"backtest_data_{SYMBOL.replace('/', '_')}_{BACKTEST_START.strftime('%Y%m%d')}_{BACKTEST_END.strftime('%Y%m%d')}.csv"

        if os.path.exists(data_file):
            print(f"从缓存文件加载数据: {data_file}")
            df = data_fetcher.load_data(data_file)
        else:
            print("从币安测试网获取历史数据...")
            df = await data_fetcher.fetch_historical_data(
                SYMBOL,
                timeframe=TIMEFRAME,
                start_time=BACKTEST_START,
                end_time=BACKTEST_END,
                limit=1000
            )

            if df.empty:
                print("错误: 无法获取历史数据")
                return

            # 保存数据供下次使用
            data_fetcher.save_data(df, data_file)

    except Exception as e:
        print(f"获取数据时出错: {e}")
        print("\n提示: 币安测试网的历史数据可能有限，建议使用主网API获取数据")
        print("或者将 use_testnet 设置为 False 使用主网数据（仅读取，不下单）")
        return

    if len(df) < 100:
        print(f"警告: 数据量较少（仅{len(df)}条），回测结果可能不准确")

    print(f"\n数据准备完成，共 {len(df)} 根K线\n")

    # 3. 初始化回测引擎
    engine = BacktestEngine(
        initial_balance=INITIAL_BALANCE,
        taker_fee=0.0004,  # 0.04% taker fee
        maker_fee=0.0002   # 0.02% maker fee
    )

    # 4. 初始化策略
    strategy = BacktestElasticDipBot(engine, SYMBOL, STRATEGY_PARAMS)

    # 5. 运行回测
    print("开始回测...\n")
    print("-" * 70)

    candle_buffer = []
    buffer_size = max(STRATEGY_PARAMS["ema_slow"], STRATEGY_PARAMS["rsi_period"]) + 50

    for idx, row in df.iterrows():
        # 构建K线数据 [timestamp, open, high, low, close, volume]
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

        # 更新市场数据到引擎
        engine.update_market(row['timestamp'], {SYMBOL: candle})

        # 执行策略步骤
        if len(candle_buffer) >= buffer_size:
            strategy.step(candle_buffer, row['timestamp'])

        # 每1000根K线打印进度
        if idx % 1000 == 0:
            equity = engine.get_total_equity()
            returns = (equity - INITIAL_BALANCE) / INITIAL_BALANCE * 100
            print(f"[{row['datetime']}] 进度: {idx}/{len(df)}, "
                  f"权益: ${equity:.2f}, 收益: {returns:.2f}%")

    print("\n" + "-" * 70)
    print("\n回测完成!\n")

    # 6. 生成分析报告
    equity_df = engine.get_equity_dataframe()
    trades_df = engine.get_trades_dataframe()

    analyzer = BacktestAnalyzer(equity_df, trades_df, INITIAL_BALANCE)

    # 打印报告
    analyzer.print_report()

    # 绘制图表
    print("正在生成回测图表...")
    try:
        analyzer.plot_results(save_path=f"backtest_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    except Exception as e:
        print(f"绘图时出错: {e}")
        print("提示: 如果在无图形界面环境运行，可能无法显示图表")

    # 导出Excel报告
    try:
        excel_file = f"backtest_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        analyzer.export_to_excel(excel_file)
    except Exception as e:
        print(f"导出Excel时出错: {e}")

    print("\n回测结束!")


async def quick_backtest_example():
    """快速回测示例（使用模拟数据）"""
    print("快速回测示例 - 使用模拟数据\n")

    # 创建模拟价格数据
    import numpy as np
    import pandas as pd

    np.random.seed(42)
    n_candles = 10000
    base_price = 40000
    timestamps = [int((datetime.now() - timedelta(minutes=n_candles - i)).timestamp() * 1000)
                  for i in range(n_candles)]

    prices = []
    current_price = base_price
    for i in range(n_candles):
        # 模拟价格波动，偶尔有大幅下跌
        if np.random.random() < 0.01:  # 1%概率大跌
            change = -np.random.uniform(0.02, 0.04)  # -2%到-4%
        else:
            change = np.random.normal(0, 0.002)  # 正常波动

        current_price *= (1 + change)
        high = current_price * (1 + abs(np.random.uniform(0, 0.005)))
        low = current_price * (1 - abs(np.random.uniform(0, 0.005)))
        open_price = current_price * (1 + np.random.uniform(-0.003, 0.003))
        volume = np.random.uniform(100, 1000)

        prices.append([timestamps[i], open_price, high, low, current_price, volume])

    df = pd.DataFrame(prices, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')

    print(f"生成了 {len(df)} 根模拟K线数据\n")

    # 运行回测
    engine = BacktestEngine(initial_balance=10000.0)
    strategy = BacktestElasticDipBot(engine, "BTC/USDT", STRATEGY_PARAMS)

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

    # 分析结果
    equity_df = engine.get_equity_dataframe()
    trades_df = engine.get_trades_dataframe()
    analyzer = BacktestAnalyzer(equity_df, trades_df, 10000.0)
    analyzer.print_report()


if __name__ == "__main__":
    # 选择运行模式
    if len(sys.argv) > 1 and sys.argv[1] == "--demo":
        # 快速演示模式（使用模拟数据）
        asyncio.run(quick_backtest_example())
    else:
        # 正式回测模式（使用币安测试网数据）
        asyncio.run(run_backtest())
