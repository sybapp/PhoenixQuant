#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速测试收益率分布图修复
使用模拟数据验证图表显示
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from backtest_analysis import BacktestAnalyzer

# 创建模拟数据
print("创建模拟权益曲线...")

# 模拟41,761根K线（与真实回测相同）
num_bars = 1000  # 使用较小数量快速测试
initial_balance = 100.0

# 生成时间序列
start_time = datetime(2025, 10, 1)
timestamps = [int((start_time + timedelta(minutes=i)).timestamp() * 1000) for i in range(num_bars)]
datetimes = [start_time + timedelta(minutes=i) for i in range(num_bars)]

# 模拟权益曲线：
# - 大部分时间权益不变（模拟无交易期）
# - 偶尔有大幅跳跃（模拟交易发生）
equity = np.ones(num_bars) * initial_balance
balance = np.ones(num_bars) * initial_balance

# 在几个点上添加大幅变化（模拟交易）
trade_indices = [100, 200, 300, 400, 500, 600, 700, 800]
for idx in trade_indices:
    jump = np.random.uniform(5, 50)  # 5-50美元的跳跃
    equity[idx:] += jump
    balance[idx:] += jump * 0.3  # 部分是现金，部分是持仓

# 添加少量持仓价格波动（模拟未实现盈亏变化）
for i in range(num_bars):
    if i > 0 and i not in trade_indices:
        # 添加微小波动（0-0.02%）
        noise = np.random.uniform(-0.0002, 0.0002)
        equity[i] = equity[i-1] * (1 + noise)

equity_df = pd.DataFrame({
    'timestamp': timestamps,
    'datetime': datetimes,
    'equity': equity,
    'balance': balance
})

# 创建空的交易记录（只测试图表）
trades_df = pd.DataFrame()

print(f"模拟数据创建完成：{len(equity_df)} 根K线")

# 创建分析器并生成图表
analyzer = BacktestAnalyzer(equity_df, trades_df, initial_balance)

print("\n生成测试图表...")
analyzer.plot_results(save_path="test_chart_fix_output.png")

print("\n✅ 测试完成！请查看 test_chart_fix_output.png")
print("   收益率分布图应该显示：")
print("   - 只有明显的收益率变化（>0.05%）")
print("   - 大约8个明显的峰值（对应8次交易）")
print("   - 不应该有接近0的巨大峰值")
