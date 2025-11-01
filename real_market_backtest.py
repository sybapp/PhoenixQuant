#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
真实市场优化策略回测
基于真实市场特征调整策略参数
"""

import asyncio
from datetime import datetime, timedelta
import ccxt
import sys
import numpy as np
import pandas as pd
from typing import Dict

from backtest_engine import BacktestEngine, HistoricalDataFetcher
from backtest_strategy import BacktestElasticDipBot
from backtest_analysis import BacktestAnalyzer

# ========= 真实市场配置 =========
# 币安测试网API
API_KEY = "kflCxmrjxzyNuaM60yvhFTCvFZBMRzCX2hniLLfEMycGJ2j2e6OMrsOE8Gzd5H7P"
API_SECRET = "Z9GOv6MoF2WQfi7iE21zkFliHzMJ1ENRtLixnvkp51lX4JA9jxsKnZ9ONak573An"

# 回测参数
SYMBOL = "BTC/USDT"
INITIAL_BALANCE = 100.0
TIMEFRAME = "1m"  # 支持: 1m, 5m, 15m, 30m, 1h

# 回测时间范围 - 10月1日到10月31日
BACKTEST_START = datetime(2025, 10, 1)
BACKTEST_END = datetime(2025, 10, 30)

# 1分钟K线需要更大的数据窗口以减少噪声
BUFFER_MULTIPLIER = {
    "1m": 2.0,  # 1分钟需要2倍buffer（降低以提高响应速度）
    "5m": 1.5,  # 5分钟需要1.5倍buffer
    "15m": 1.0, # 15分钟及以上使用标准buffer
    "30m": 1.0,
    "1h": 1.0
}

# 针对不同时间框架的参数优化
TIMEFRAME_PARAMS = {
    "1m": {
        "drop_pct_single": 0.4,      # 1分钟波动小，进一步降低阈值使其更容易触发
        "drop_pct_window": 0.8,
        "window_min": 8,             # 适中的观察窗口
        "rsi_oversold": 38.0,        # 放宽超卖条件（从32提高到38）
        "vol_recover_ratio": 1.15,   # 降低放量要求（从1.25降到1.15）
        "delayed_trigger_pct": 0.12, # 降低反弹确认要求
        "max_daily_trades": 4,       # 增加允许的交易次数
        "signal_strength_threshold": 35,  # 降低信号强度要求（从50降到35）

        # ⚡ 1分钟优化止盈止损
        "take_profit_pct": [0.4, 0.8, 1.5],  # 1分钟更激进的止盈
        "hard_stop_extra": 1.0,               # 适中的止损
        "trailing_stop_pct": 0.5,             # 适中的移动止损
    },
    "5m": {
        "drop_pct_single": 0.8,
        "drop_pct_window": 1.5,
        "window_min": 6,
        "rsi_oversold": 35.0,
        "vol_recover_ratio": 1.1,
        "delayed_trigger_pct": 0.1,
        "max_daily_trades": 4,
        "signal_strength_threshold": 40,

        # 5分钟止盈止损
        "take_profit_pct": [0.5, 1.0, 2.0],
        "hard_stop_extra": 1.2,
        "trailing_stop_pct": 0.6,
    },
    "15m": {
        "drop_pct_single": 1.0,
        "drop_pct_window": 2.0,
        "window_min": 5,
        "rsi_oversold": 38.0,
        "vol_recover_ratio": 1.05,
        "delayed_trigger_pct": 0.08,
        "max_daily_trades": 5,
        "signal_strength_threshold": 35,

        # 15分钟止盈止损
        "take_profit_pct": [0.6, 1.2, 2.5],
        "hard_stop_extra": 1.5,
        "trailing_stop_pct": 0.8,
    }
}

# 真实市场优化策略参数
REAL_MARKET_PARAMS = {
    "timeframe": TIMEFRAME,
    "poll_sec": 2,

    # 价格触发条件 - 根据真实市场调整
    "drop_pct_single": 0.8,      # 降低单根K线跌幅要求
    "drop_pct_window": 1.5,      # 降低窗口跌幅要求
    "window_min": 6,             # 缩短观察窗口

    # 趋势过滤 - 适应真实市场
    "ema_fast": 10,
    "ema_slow": 30,

    # 动量指标 - 放宽超卖条件
    "rsi_period": 14,
    "rsi_oversold": 35.0,        # 放宽超卖条件

    # 爆仓和资金费率 - 放宽条件
    "funding_extreme_neg": -0.03,
    "liq_notional_threshold": 2_000_000,

    # 延迟触发 - 更宽松
    "delayed_trigger_pct": 0.1,   # 进一步降低反弹确认要求
    "delayed_window_sec": 60 * 60 * 6,  # 延长到6小时

    # 分层买入 - 更接近当前价格
    "layer_pcts": [0.3, 0.6, 0.9, 1.2],  # 更接近当前价格
    "layer_pos_ratio": [0.40, 0.30, 0.20, 0.10],  # 重点在前两个层级

    # 资金管理 - 适应更高价格
    "total_capital": 2000,
    "max_account_ratio": 0.20,

    # 止盈止损优化 - ⚡ 重大优化：锁定浮盈
    "take_profit_pct": [0.5, 1.0, 2.0],  # 降低止盈目标，更快锁定利润
    "hard_stop_extra": 1.2,      # 放宽止损，避免被噪声扫出
    "sl_time_grace_sec": 300,     # 延长止损宽限期到5分钟
    "trailing_stop_pct": 0.6,    # 放宽移动止损，保护浮盈同时给回调空间
    "use_dynamic_sl": True,       # 启用动态止损（基于ATR）

    # 成交量条件 - 放宽
    "vol_shrink_ratio": 0.8,
    "vol_recover_ma_short": 2,
    "vol_recover_ma_long": 15,
    "vol_recover_ratio": 1.1,    # 降低放量要求
    "tick_vol_ratio": 1.2,

    # 风险管理
    "max_daily_trades": 4,
    "max_consecutive_losses": 4,

    # 信号过滤
    "signal_strength_threshold": 40,  # 默认信号强度阈值
}


class RealMarketElasticDipBot(BacktestElasticDipBot):
    """真实市场优化版弹性抄底���略"""

    def __init__(self, backtest_engine: BacktestEngine, symbol: str, params: Dict):
        super().__init__(backtest_engine, symbol, params)
        self.debug_counter = 0  # 调试计数器
        self.debug_interval = 1000  # 每1000次打印一次调试信息

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
        # 原因：加密货币市场波动大，ATR过滤会导致无法触发
        return single_drop or window_drop

    def is_oversold(self, candles):
        """重写超卖判断 - 优化真实市场条件"""
        closes = [c[4] for c in candles]
        rsi_val = self.rsi(closes, self.p["rsi_period"])

        if rsi_val is None or np.isnan(rsi_val):
            return False

        # 布林带超卖确认
        bb_upper, bb_middle, bb_lower = self.bollinger_bands(closes)
        current_price = closes[-1]

        bb_oversold = False
        if bb_lower is not None:
            bb_oversold = current_price <= bb_lower

        # RSI超卖或价格在布林带下轨下方
        return (rsi_val < self.p["rsi_oversold"]) or bb_oversold


    def _idle_state(self, candles, current_timestamp, current_price):
        """真实市场空闲状态处理"""
        self.debug_counter += 1

        # 检查风险管理
        can_trade = self.can_trade(current_timestamp)
        if not can_trade:
            if self.debug_counter % self.debug_interval == 0:
                print(f"[调试] 无法交易 - 风险管理限制")
            return

        # 计算信号强度
        self.signal_strength = self.calculate_signal_strength(candles)

        # 真实市场入场条件：放宽条件
        fast_drop = self.is_fast_drop(candles)
        oversold = self.is_oversold(candles)
        basic_conditions = fast_drop and oversold

        # 爆仓和资金费率作为加分项，不是必要条件
        bonus_conditions = (
            self.is_liquidation_spike() or
            self.is_funding_extreme()
        )

        # 使用参数中的信号强度阈值（根据时间框架调整）
        threshold = self.p.get("signal_strength_threshold", 40)
        signal_ok = self.signal_strength > threshold

        # 定期打印调试信息
        if self.debug_counter % self.debug_interval == 0:
            print(f"\n[调试 #{self.debug_counter}] 价格: ${current_price:.2f}")
            print(f"  信号强度: {self.signal_strength:.1f} (阈值: {threshold})")
            print(f"  快速下跌: {fast_drop}, 超卖: {oversold}")
            print(f"  基础条件: {basic_conditions}, 信号OK: {signal_ok}")
            print(f"  参数 - drop_pct_single: {self.p.get('drop_pct_single')}, "
                  f"rsi_oversold: {self.p.get('rsi_oversold')}")

        # 真实市场入场条件：基础条件 + 信号强度
        if basic_conditions and signal_ok:
            self.reference_price = current_price
            self.trigger_time = current_timestamp
            self.state = 'WAIT_FOR_BOUNCE'
            self.daily_trade_count += 1

            print(f"\n[真实市场触发] 强度{self.signal_strength:.1f}% "
                  f"价格${current_price:.2f} "
                  f"快速下跌:{self.is_fast_drop(candles)} "
                  f"超卖:{self.is_oversold(candles)}")

    def _wait_bounce_state(self, candles, current_timestamp, current_price):
        """重写等待反弹状态 - 极度宽松的条件便于触发"""
        # 检查超时
        if current_timestamp - self.trigger_time > self.p["delayed_window_sec"] * 1000:
            print(f"[超时] 延迟窗口过期，参考价: ${self.reference_price:.2f}, 当前价: ${current_price:.2f}")
            self.reset()
            return

        # 极度放宽的反弹确认 - 只要价格不再继续下跌即可
        # 原条件：price_ok = current_price >= self.reference_price * (1 + delayed_trigger_pct)
        # 新条件：价格��本稳定或反弹即可
        price_stable = current_price >= self.reference_price * 0.998  # 允许最多0.2%的进一步下跌

        # 如果价格稳定，立即进场（不再等待成交量或技术指标确认）
        if price_stable:
            # 开始分层买入
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
            print(f"[真实市场下单] 参考价:${self.reference_price:.4f}, 当前价:${current_price:.4f}, "
                  f"{len(plan)}档，总投资: ${total_investment:.2f}")

    def _real_market_technical_confirmation(self, candles) -> bool:
        """真实市场技术指标确认 - 放宽"""
        closes = [c[4] for c in candles]

        # MACD确认 - 放宽
        macd_line, signal_line, histogram = self.macd(closes)
        macd_confirm = False
        if macd_line and signal_line:
            macd_confirm = histogram > -0.05 or macd_line > signal_line * 0.98

        # 价格站上短期EMA - 放宽
        ema_fast = self.ema(closes, 5)
        price_above_ema = False
        if len(ema_fast) > 0:
            price_above_ema = closes[-1] > ema_fast[-1] * 0.995

        # RSI恢复确认 - 放宽
        rsi = self.rsi(closes, period=14)
        rsi_confirm = False
        if rsi is not None and not np.isnan(rsi):
            rsi_confirm = rsi > 25

        # 多个技术指标任一确认即可
        return macd_confirm or price_above_ema or rsi_confirm

    def calculate_signal_strength(self, candles) -> float:
        """真实市场信号强度计算"""
        if len(candles) < max(self.p["ema_slow"], self.p["rsi_period"]) + 10:
            return 0.0

        signals = []
        weights = []

        # 1. 价格信号 (权重: 0.50) - 提高权重
        price_signal = self._price_based_signal(candles)
        signals.append(price_signal)
        weights.append(0.50)

        # 2. 动量信号 (权重: 0.25)
        momentum_signal = self._momentum_signal(candles)
        signals.append(momentum_signal)
        weights.append(0.25)

        # 3. 成交量信号 (权重: 0.15)
        volume_signal = self._volume_signal(candles)
        signals.append(volume_signal)
        weights.append(0.15)

        # 4. 趋势信号 (权重: 0.10) - 降低权重
        trend_signal = self._trend_signal(candles)
        signals.append(trend_signal)
        weights.append(0.10)

        # 计算加权平均
        total_weight = sum(weights)
        weighted_signal = sum(s * w for s, w in zip(signals, weights)) / total_weight

        return min(100.0, max(0.0, weighted_signal))


async def run_real_market_backtest():
    """运行真实市场优化版回测"""
    print("="*70)
    print(" " * 20 + "真实市场优化策略回测")
    print("="*70)
    print(f"\n交易对: {SYMBOL}")
    print(f"初始资金: ${INITIAL_BALANCE:,.2f}")
    print(f"回测时间: {BACKTEST_START} 至 {BACKTEST_END}")
    print(f"K线周期: {TIMEFRAME}")
    print("\n正在从币安测试网获取真实数据...\n")

    # 1. 初始化交易所 - 使用测试网
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
        # 从缓存文件加载数据 - 文件名格式: testnet_data_币种_时间框架_开始日期_结束日期.csv
        import os
        data_file = f"testnet_data_{SYMBOL.replace('/', '_')}_{TIMEFRAME}_{BACKTEST_START.strftime('%Y%m%d')}_{BACKTEST_END.strftime('%Y%m%d')}.csv"

        print(f"数据文件名: {data_file}")

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
                limit=5000
            )

            if df.empty:
                print("错误: 无法获取历史数据")
                return

            # 保存数据供下次使用
            data_fetcher.save_data(df, data_file)

    except Exception as e:
        print(f"获取数据时出错: {e}")
        return

    print(f"\n数据准备完成，共 {len(df)} 根K线\n")

    # 数据质量验证
    if len(df) < 100:
        print(f"警告: 数据量过少 ({len(df)} 根K线)，回测结果可能不可靠")

    # 检查数据完整性
    null_count = df.isnull().sum().sum()
    if null_count > 0:
        print(f"警告: 发现 {null_count} 个空值，将进行前向填充")
        df = df.fillna(method='ffill')

    # 3. 应用时间框架特定参数
    strategy_params = REAL_MARKET_PARAMS.copy()
    if TIMEFRAME in TIMEFRAME_PARAMS:
        print(f"\n应用 {TIMEFRAME} 时间框架优化参数...")
        strategy_params.update(TIMEFRAME_PARAMS[TIMEFRAME])
    else:
        print(f"\n警告: 未找到 {TIMEFRAME} 的优化参数，使用默认参数")

    # 4. 初始化回测引擎
    engine = BacktestEngine(
        initial_balance=INITIAL_BALANCE,
        taker_fee=0.0004,
        maker_fee=0.0002
    )

    # 5. 初始化真实市场优化策略
    strategy = RealMarketElasticDipBot(engine, SYMBOL, strategy_params)

    # 6. 运行回测
    print("开始回测...\n")
    print("-" * 70)

    candle_buffer = []
    # 根据时间框架调整buffer大小
    base_buffer_size = max(strategy_params["ema_slow"], strategy_params["rsi_period"]) + 50
    buffer_multiplier = BUFFER_MULTIPLIER.get(TIMEFRAME, 1.0)
    buffer_size = int(base_buffer_size * buffer_multiplier)

    print(f"使用buffer大小: {buffer_size} (基础: {base_buffer_size}, 倍数: {buffer_multiplier})\n")

    for idx, row in df.iterrows():
        # 构建K线数据
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

        # 每500根K线打印进度
        if idx % 500 == 0:
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
        # 结果文件名格式: real_market_backtest_币种_时间框架_日期时间.png
        result_filename = f"real_market_backtest_{SYMBOL.replace('/', '_')}_{TIMEFRAME}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        analyzer.plot_results(save_path=result_filename)
        print(f"图表已保存: {result_filename}")
    except Exception as e:
        print(f"绘图时出错: {e}")

    print("\n回测结束!")


if __name__ == "__main__":
    asyncio.run(run_real_market_backtest())