#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
改进的真实市场优化策略回测
解决1分钟K线回测不稳定问题，支持多资产参数优化
"""

import asyncio
from datetime import datetime, timedelta
import ccxt
import sys
import numpy as np
import pandas as pd
from typing import Dict, List

from backtest_engine import BacktestEngine, HistoricalDataFetcher
from backtest_strategy import BacktestElasticDipBot
from backtest_analysis import BacktestAnalyzer

# ========= 真实市场配置 =========
# 币安测试网API
API_KEY = "kflCxmrjxzyNuaM60yvhFTCvFZBMRzCX2hniLLfEMycGJ2j2e6OMrsOE8Gzd5H7P"
API_SECRET = "Z9GOv6MoF2WQfi7iE21zkFliHzMJ1ENRtLixnvkp51lX4JA9jxsKnZ9ONak573An"

# 回测参数
SYMBOLS = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "DOGE/USDT", "XRP/USDT"]
INITIAL_BALANCE = 100.0
TIMEFRAME = "1m"

# 回测时间范围 - 10月1日到10月31日
BACKTEST_START = datetime(2025, 10, 1)
BACKTEST_END = datetime(2025, 10, 30)

# 资产特定参数配置
ASSET_SPECIFIC_PARAMS = {
    "BTC/USDT": {
        # 比特币 - 低波动，需要更敏感的触发条件
        "drop_pct_single": 0.6,
        "drop_pct_window": 1.2,
        "rsi_oversold": 32.0,
        "layer_pcts": [0.2, 0.4, 0.6, 0.8],
        "take_profit_pct": [0.6, 1.2, 2.0],
        "hard_stop_extra": 0.8,
        "vol_recover_ratio": 1.05,
    },
    "ETH/USDT": {
        # 以太坊 - 中等波动
        "drop_pct_single": 0.7,
        "drop_pct_window": 1.4,
        "rsi_oversold": 33.0,
        "layer_pcts": [0.25, 0.5, 0.75, 1.0],
        "take_profit_pct": [0.7, 1.4, 2.2],
        "hard_stop_extra": 0.7,
        "vol_recover_ratio": 1.08,
    },
    "SOL/USDT": {
        # Solana - 高波动
        "drop_pct_single": 1.0,
        "drop_pct_window": 2.0,
        "rsi_oversold": 38.0,
        "layer_pcts": [0.4, 0.8, 1.2, 1.6],
        "take_profit_pct": [1.0, 2.0, 3.0],
        "hard_stop_extra": 0.5,
        "vol_recover_ratio": 1.15,
    },
    "BNB/USDT": {
        # BNB - 中等波动
        "drop_pct_single": 0.8,
        "drop_pct_window": 1.6,
        "rsi_oversold": 35.0,
        "layer_pcts": [0.3, 0.6, 0.9, 1.2],
        "take_profit_pct": [0.8, 1.6, 2.4],
        "hard_stop_extra": 0.6,
        "vol_recover_ratio": 1.10,
    },
    "DOGE/USDT": {
        # Dogecoin - 高波动
        "drop_pct_single": 1.2,
        "drop_pct_window": 2.5,
        "rsi_oversold": 40.0,
        "layer_pcts": [0.5, 1.0, 1.5, 2.0],
        "take_profit_pct": [1.2, 2.4, 3.6],
        "hard_stop_extra": 0.4,
        "vol_recover_ratio": 1.20,
    },
    "XRP/USDT": {
        # XRP - 高波动
        "drop_pct_single": 1.1,
        "drop_pct_window": 2.2,
        "rsi_oversold": 38.0,
        "layer_pcts": [0.45, 0.9, 1.35, 1.8],
        "take_profit_pct": [1.1, 2.2, 3.3],
        "hard_stop_extra": 0.45,
        "vol_recover_ratio": 1.18,
    }
}

# 基础策略参数
BASE_PARAMS = {
    "timeframe": TIMEFRAME,
    "poll_sec": 2,
    "window_min": 6,
    "ema_fast": 10,
    "ema_slow": 30,
    "rsi_period": 14,
    "funding_extreme_neg": -0.03,
    "liq_notional_threshold": 2_000_000,
    "delayed_trigger_pct": 0.1,
    "delayed_window_sec": 60 * 60 * 6,
    "layer_pos_ratio": [0.40, 0.30, 0.20, 0.10],
    "total_capital": 2000,
    "max_account_ratio": 0.20,
    "sl_time_grace_sec": 240,
    "trailing_stop_pct": 0.4,
    "vol_shrink_ratio": 0.8,
    "vol_recover_ma_short": 2,
    "vol_recover_ma_long": 15,
    "tick_vol_ratio": 1.2,
    "max_daily_trades": 4,
    "max_consecutive_losses": 4,
}


class ImprovedRealMarketElasticDipBot(BacktestElasticDipBot):
    """改进的真实市场优化版弹性抄底策略"""

    def __init__(self, backtest_engine: BacktestEngine, symbol: str, params: Dict):
        super().__init__(backtest_engine, symbol, params)
        self.symbol = symbol
        self.volatility_profile = self._calculate_volatility_profile()

    def _calculate_volatility_profile(self) -> str:
        """根据交易对计算波动率特征"""
        high_vol_assets = ["SOL", "DOGE", "XRP"]
        medium_vol_assets = ["ETH", "BNB"]

        base_asset = self.symbol.split('/')[0]

        if base_asset in high_vol_assets:
            return "high"
        elif base_asset in medium_vol_assets:
            return "medium"
        else:
            return "low"

    def _idle_state(self, candles, current_timestamp, current_price):
        """改进的真实市场空闲状态处理"""
        # 检查风险管理
        if not self.can_trade(current_timestamp):
            return

        # 计算自适应信号强度
        self.signal_strength = self._calculate_adaptive_signal_strength(candles)

        # 自适应入场条件
        basic_conditions = (
            self._adaptive_fast_drop(candles) and
            self._adaptive_oversold(candles)
        )

        # 根据波动率调整信号强度要求
        volatility_threshold = {
            "low": 35,
            "medium": 40,
            "high": 45
        }
        signal_ok = self.signal_strength > volatility_threshold[self.volatility_profile]

        # 改进的入场条件
        if basic_conditions and signal_ok:
            self.reference_price = current_price
            self.trigger_time = current_timestamp
            self.state = 'WAIT_FOR_BOUNCE'
            self.daily_trade_count += 1

            print(f"\n[改进策略触发] {self.symbol} 强度{self.signal_strength:.1f}% "
                  f"价格${current_price:.2f} 波动率:{self.volatility_profile}")

    def _calculate_adaptive_signal_strength(self, candles) -> float:
        """自适应信号强度计算"""
        if len(candles) < max(self.p["ema_slow"], self.p["rsi_period"]) + 10:
            return 0.0

        signals = []
        weights = []

        # 根据波动率调整权重
        if self.volatility_profile == "low":
            # 低波动资产：更注重价格和趋势信号
            weights = [0.60, 0.20, 0.10, 0.10]  # 价格, 动量, 成交量, 趋势
        elif self.volatility_profile == "medium":
            weights = [0.50, 0.25, 0.15, 0.10]
        else:  # high volatility
            # 高波动资产：更注重成交量和动量信号
            weights = [0.40, 0.30, 0.20, 0.10]

        # 1. 价格信号
        price_signal = self._price_based_signal(candles)
        signals.append(price_signal)

        # 2. 动量信号
        momentum_signal = self._momentum_signal(candles)
        signals.append(momentum_signal)

        # 3. 成交量信号
        volume_signal = self._volume_signal(candles)
        signals.append(volume_signal)

        # 4. 趋势信号
        trend_signal = self._trend_signal(candles)
        signals.append(trend_signal)

        # 计算加权平均
        total_weight = sum(weights)
        weighted_signal = sum(s * w for s, w in zip(signals, weights)) / total_weight

        return min(100.0, max(0.0, weighted_signal))

    def _adaptive_fast_drop(self, candles) -> bool:
        """自适应快速下跌检测"""
        # 根据波动率调整下跌阈值
        volatility_multiplier = {
            "low": 0.8,
            "medium": 1.0,
            "high": 1.2
        }

        multiplier = volatility_multiplier[self.volatility_profile]

        # 临时保存原始参数
        original_drop_pct_single = self.p["drop_pct_single"]
        original_drop_pct_window = self.p["drop_pct_window"]

        # 临时设置调整后的参数
        self.p["drop_pct_single"] = original_drop_pct_single * multiplier
        self.p["drop_pct_window"] = original_drop_pct_window * multiplier

        # 使用调整后的阈值进行检测
        result = self.is_fast_drop(candles)

        # 恢复原始参数
        self.p["drop_pct_single"] = original_drop_pct_single
        self.p["drop_pct_window"] = original_drop_pct_window

        return result

    def _adaptive_oversold(self, candles) -> bool:
        """自适应超卖检测"""
        # 根据波动率调整超卖阈值
        volatility_adjustment = {
            "low": -3.0,   # 更敏感
            "medium": 0.0,  # 标准
            "high": +3.0    # 更宽松
        }

        adjustment = volatility_adjustment[self.volatility_profile]

        # 临时保存原始参数
        original_rsi_oversold = self.p["rsi_oversold"]

        # 临时设置调整后的参数
        self.p["rsi_oversold"] = original_rsi_oversold + adjustment

        # 使用调整后的阈值进行检测
        result = self.is_oversold(candles)

        # 恢复原始参数
        self.p["rsi_oversold"] = original_rsi_oversold

        return result

    def _wait_bounce_state(self, candles, current_timestamp, current_price):
        """改进的等待反弹状态"""
        # 检查超时
        if current_timestamp - self.trigger_time > self.p["delayed_window_sec"] * 1000:
            print(f"[超时] {self.symbol} 延迟窗口过期，参考价: ${self.reference_price:.2f}, 当前价: ${current_price:.2f}")
            self.reset()
            return

        # 自适应反弹确认条件
        price_ok = current_price >= self.reference_price * (1 + self.p["delayed_trigger_pct"] / 100.0)

        # 成交量恢复作为可选条件
        vol_ok = self.volume_recovered(
            candles,
            ma_short=self.p["vol_recover_ma_short"],
            ma_long=self.p["vol_recover_ma_long"],
            ratio=self.p["vol_recover_ratio"],
            tick_ratio=self.p["tick_vol_ratio"]
        )

        # 自适应技术指标确认
        tech_confirm = self._adaptive_technical_confirmation(candles)

        # 改进的条件：价格确认 + (成交量或技术指标)
        if price_ok and (vol_ok or tech_confirm):
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
            print(f"[改进策略下单] {self.symbol} {len(plan)}档，总投资: ${total_investment:.2f}")

    def _adaptive_technical_confirmation(self, candles) -> bool:
        """自适应技术指标确认"""
        closes = [c[4] for c in candles]

        # 根据波动率调整确认阈值
        if self.volatility_profile == "low":
            # 低波动：更严格的确认
            macd_threshold = -0.02
            ema_threshold = 0.998
            rsi_threshold = 28
        elif self.volatility_profile == "medium":
            macd_threshold = -0.05
            ema_threshold = 0.995
            rsi_threshold = 25
        else:  # high
            # 高波动：更宽松的确认
            macd_threshold = -0.08
            ema_threshold = 0.990
            rsi_threshold = 22

        # MACD确认
        macd_line, signal_line, histogram = self.macd(closes)
        macd_confirm = False
        if macd_line and signal_line:
            macd_confirm = histogram > macd_threshold or macd_line > signal_line * 0.98

        # 价格站上短期EMA
        ema_fast = self.ema(closes, 5)
        price_above_ema = False
        if len(ema_fast) > 0:
            price_above_ema = closes[-1] > ema_fast[-1] * ema_threshold

        # RSI恢复确认
        rsi = self.rsi(closes, period=14)
        rsi_confirm = False
        if rsi is not None and not np.isnan(rsi):
            rsi_confirm = rsi > rsi_threshold

        # 多个技术指标任一确认即可
        return macd_confirm or price_above_ema or rsi_confirm


async def run_improved_backtest_for_symbol(symbol: str):
    """为单个交易对运行改进的回测"""
    print(f"\n{'='*70}")
    print(f" " * 20 + f"改进策略回测 - {symbol}")
    print(f"{'='*70}")
    print(f"\n交易对: {symbol}")
    print(f"初始资金: ${INITIAL_BALANCE:,.2f}")
    print(f"回测时间: {BACKTEST_START} 至 {BACKTEST_END}")
    print(f"K线周期: {TIMEFRAME}")
    print(f"\n正在从币安测试网获取真实数据...\n")

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
        # 从缓存文件加载数据
        import os
        data_file = f"improved_testnet_data_{symbol.replace('/', '_')}_{TIMEFRAME}_{BACKTEST_START.strftime('%Y%m%d')}_{BACKTEST_END.strftime('%Y%m%d')}.csv"

        if os.path.exists(data_file):
            print(f"从缓存文件加载数据: {data_file}")
            df = data_fetcher.load_data(data_file)
        else:
            print("从币安测试网获取历史数据...")
            df = await data_fetcher.fetch_historical_data(
                symbol,
                timeframe=TIMEFRAME,
                start_time=BACKTEST_START,
                end_time=BACKTEST_END,
                limit=5000
            )

            if df.empty:
                print(f"错误: 无法获取 {symbol} 的历史数据")
                return None

            # 保存数据供下次使用
            data_fetcher.save_data(df, data_file)

    except Exception as e:
        print(f"获取 {symbol} 数据时出错: {e}")
        return None

    print(f"\n数据准备完成，共 {len(df)} 根K线\n")

    # 3. 构建策略参数
    strategy_params = BASE_PARAMS.copy()
    if symbol in ASSET_SPECIFIC_PARAMS:
        strategy_params.update(ASSET_SPECIFIC_PARAMS[symbol])

    # 4. 初始化回测引擎
    engine = BacktestEngine(
        initial_balance=INITIAL_BALANCE,
        taker_fee=0.0004,
        maker_fee=0.0002
    )

    # 5. 初始化改进的真实市场优化策略
    strategy = ImprovedRealMarketElasticDipBot(engine, symbol, strategy_params)

    # 6. 运行回测
    print("开始回测...\n")
    print("-" * 70)

    candle_buffer = []
    buffer_size = max(strategy_params["ema_slow"], strategy_params["rsi_period"]) + 50

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
        engine.update_market(row['timestamp'], {symbol: candle})

        # 执行策略步骤
        if len(candle_buffer) >= buffer_size:
            strategy.step(candle_buffer, row['timestamp'])

        # 每500根K线打印进度
        if idx % 500 == 0:
            equity = engine.get_total_equity()
            returns = (equity - INITIAL_BALANCE) / INITIAL_BALANCE * 100
            print(f"[{row['datetime']}] {symbol} 进度: {idx}/{len(df)}, "
                  f"权益: ${equity:.2f}, 收益: {returns:.2f}%")

    print("\n" + "-" * 70)
    print("\n回测完成!\n")

    # 7. 生成分析报告
    equity_df = engine.get_equity_dataframe()
    trades_df = engine.get_trades_dataframe()

    analyzer = BacktestAnalyzer(equity_df, trades_df, INITIAL_BALANCE)

    # 打印报告
    analyzer.print_report()

    # 绘制图表
    print("正在生成回测图表...")
    try:
        timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
        analyzer.plot_results(save_path=f"improved_backtest_result_{symbol.replace('/', '_')}_{timestamp_str}.png")
    except Exception as e:
        print(f"绘图时出错: {e}")

    # 导出到Excel
    excel_filename = f"improved_backtest_report_{symbol.replace('/', '_')}_{timestamp_str}.xlsx"
    analyzer.export_to_excel(excel_filename)

    return analyzer.calculate_metrics()


async def run_comparative_backtest():
    """运行多资产比较回测"""
    print("="*70)
    print(" " * 15 + "多资产改进策略回测比较")
    print("="*70)

    results = {}

    for symbol in SYMBOLS:
        metrics = await run_improved_backtest_for_symbol(symbol)
        if metrics:
            results[symbol] = metrics

    # 打印比较结果
    if results:
        print("\n" + "="*70)
        print(" " * 20 + "多资产回测比较结果")
        print("="*70)

        print(f"\n{'交易对':<12} {'总收益率':<10} {'最大回撤':<10} {'夏普比率':<10} {'胜率':<8} {'交易次数':<8}")
        print("-" * 70)

        for symbol, metrics in results.items():
            print(f"{symbol:<12} {metrics['total_return_pct']:>8.2f}% {metrics['max_drawdown_pct']:>9.2f}% "
                  f"{metrics['sharpe_ratio']:>9.3f} {metrics['win_rate_pct']:>7.1f}% {metrics['total_trades']:>8.0f}")


if __name__ == "__main__":
    # 运行多资产比较回测
    asyncio.run(run_comparative_backtest())