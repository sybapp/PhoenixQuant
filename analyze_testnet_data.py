#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析测试网数据特征
"""

import pandas as pd
import numpy as np

def analyze_testnet_data():
    """分析测试网数据特征"""
    df = pd.read_csv('testnet_data_BTC_USDT_20251001_20251031.csv')

    print("="*60)
    print("测试网数据特征分析")
    print("="*60)

    print(f'\n数据范围: {df["datetime"].min()} 到 {df["datetime"].max()}')
    print(f'总K线数: {len(df)}')

    print(f'\n价格统计:')
    print(f'  最低价: ${df["close"].min():.2f}')
    print(f'  最高价: ${df["close"].max():.2f}')
    print(f'  平均价: ${df["close"].mean():.2f}')
    print(f'  价格标准差: ${df["close"].std():.2f}')

    # 计算整体跌幅
    max_drawdown = (df["close"].min() / df["close"].max() - 1) * 100
    print(f'  最大整体跌幅: {max_drawdown:.2f}%')

    # 计算5分钟级别的价格变化
    returns = df['close'].pct_change()

    print(f'\n5分钟收益率统计:')
    print(f'  平均收益率: {returns.mean() * 100:.4f}%')
    print(f'  收益率标准差: {returns.std() * 100:.4f}%')
    print(f'  最大单根K线涨幅: {returns.max() * 100:.4f}%')
    print(f'  最大单根K线跌幅: {returns.min() * 100:.4f}%')

    print(f'\n跌幅统计:')
    for threshold in [0.005, 0.01, 0.015, 0.02, 0.025, 0.03]:
        count = (returns < -threshold).sum()
        percentage = count / len(returns) * 100
        print(f'  跌幅超过{threshold*100:.1f}%的次数: {count} ({percentage:.2f}%)')

    # 分析连续下跌
    print(f'\n连续下跌分析:')
    consecutive_drops = 0
    max_consecutive = 0
    current_streak = 0

    for ret in returns:
        if ret < 0:
            current_streak += 1
            max_consecutive = max(max_consecutive, current_streak)
        else:
            current_streak = 0

    print(f'  最长连续下跌K线数: {max_consecutive}')

    # 分析成交量
    print(f'\n成交量统计:')
    print(f'  平均成交量: {df["volume"].mean():.2f}')
    print(f'  成交量标准差: {df["volume"].std():.2f}')
    print(f'  最大成交量: {df["volume"].max():.2f}')

    # 分析策略触发条件
    print(f'\n策略触发条件分析:')

    # 检查是否有符合我们策略条件的下跌
    window_size = 8  # 8根K线窗口
    for i in range(window_size, len(df)):
        window_high = df['close'].iloc[i-window_size:i].max()
        current_price = df['close'].iloc[i]
        window_drop = (current_price - window_high) / window_high * 100

        if window_drop < -2.0:  # 窗口跌幅超过2%
            single_drop = (df['close'].iloc[i] - df['close'].iloc[i-1]) / df['close'].iloc[i-1] * 100
            if single_drop < -1.2:  # 单根K线跌幅超过1.2%
                print(f'  发现潜在触发点: 时间 {df["datetime"].iloc[i]}, '
                      f'窗口跌幅 {window_drop:.2f}%, 单根跌幅 {single_drop:.2f}%')
                break
    else:
        print(f'  未发现符合策略条件的下跌模式')

    print("\n" + "="*60)

if __name__ == "__main__":
    analyze_testnet_data()