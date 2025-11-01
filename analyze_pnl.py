#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析交易统计问题 - 简化版
"""

import pandas as pd
import numpy as np

# 模拟修复后的PnL计算逻辑
def analyze_pnl_calculation():
    # 创建模拟交易数据
    trades_data = [
        # 第1组交易 - 4买1卖，亏损
        {'side': 'buy', 'price': 0.2200, 'quantity': 35.7962, 'fee': 0.001575},
        {'side': 'buy', 'price': 0.2200, 'quantity': 26.9282, 'fee': 0.001185},
        {'side': 'buy', 'price': 0.2200, 'quantity': 18.0065, 'fee': 0.000792},
        {'side': 'buy', 'price': 0.2200, 'quantity': 9.0306, 'fee': 0.000397},
        {'side': 'sell', 'price': 0.21769, 'quantity': 89.7615, 'fee': 0.003908},

        # 第2组交易 - 假设4买但没卖（未平仓）
        {'side': 'buy', 'price': 0.2100, 'quantity': 30.0, 'fee': 0.001},
        {'side': 'buy', 'price': 0.2100, 'quantity': 25.0, 'fee': 0.001},
        {'side': 'buy', 'price': 0.2100, 'quantity': 20.0, 'fee': 0.001},
        {'side': 'buy', 'price': 0.2100, 'quantity': 15.0, 'fee': 0.001},

        # 第3组交易 - 假设又平仓了
        {'side': 'sell', 'price': 0.2500, 'quantity': 90.0, 'fee': 0.004},
    ]

    trades_df = pd.DataFrame(trades_data)

    print("="*70)
    print("交易记录")
    print("="*70)
    print(trades_df)
    print(f"\n买入次数: {len(trades_df[trades_df['side']=='buy'])}")
    print(f"卖出次数: {len(trades_df[trades_df['side']=='sell'])}")

    # 使用修复后的PnL计算逻辑
    print("\n" + "="*70)
    print("FIFO配对PnL计算")
    print("="*70)

    profit_trades = 0
    loss_trades = 0
    total_profit = 0
    total_loss = 0

    buy_queue = []

    for idx, trade in trades_df.iterrows():
        if trade['side'] == 'buy':
            buy_queue.append({
                'price': trade['price'],
                'quantity': trade['quantity'],
                'fee': trade['fee']
            })
            print(f"[买入] 价格: ${trade['price']:.4f}, 数量: {trade['quantity']:.4f}")
        elif trade['side'] == 'sell':
            sell_qty = trade['quantity']
            sell_price = trade['price']
            sell_fee = trade['fee']

            print(f"\n[卖出] 价格: ${sell_price:.4f}, 数量: {sell_qty:.4f}")

            while sell_qty > 0 and buy_queue:
                buy = buy_queue[0]

                matched_qty = min(sell_qty, buy['quantity'])

                buy_cost = buy['price'] * matched_qty + buy['fee'] * (matched_qty / buy['quantity'])
                sell_revenue = sell_price * matched_qty - sell_fee * (matched_qty / trade['quantity'])
                pnl = sell_revenue - buy_cost

                print(f"  配对 {matched_qty:.4f} @ ${buy['price']:.4f} -> ${sell_price:.4f}")
                print(f"    买入成本: ${buy_cost:.4f}")
                print(f"    卖出收入: ${sell_revenue:.4f}")
                print(f"    PnL: ${pnl:.4f}")

                if pnl > 0:
                    profit_trades += 1
                    total_profit += pnl
                else:
                    loss_trades += 1
                    total_loss += abs(pnl)

                sell_qty -= matched_qty
                buy['quantity'] -= matched_qty

                if buy['quantity'] <= 1e-8:
                    buy_queue.pop(0)

    print(f"\n" + "="*70)
    print("统计结果")
    print("="*70)
    print(f"盈利交易数: {profit_trades}")
    print(f"亏损交易数: {loss_trades}")
    print(f"总盈利: ${total_profit:.4f}")
    print(f"总亏损: ${total_loss:.4f}")
    print(f"胜率: {profit_trades/(profit_trades+loss_trades)*100 if (profit_trades+loss_trades)>0 else 0:.2f}%")
    print(f"盈亏比: {total_profit/total_loss if total_loss>0 else 0:.2f}")
    print(f"平均盈利: ${total_profit/profit_trades if profit_trades>0 else 0:.4f}")
    print(f"平均亏损: ${total_loss/loss_trades if loss_trades>0 else 0:.4f}")

    if buy_queue:
        print(f"\n未平仓数量: {sum(b['quantity'] for b in buy_queue):.4f}")
        print("未平仓详情:")
        for b in buy_queue:
            print(f"  ${b['price']:.4f} x {b['quantity']:.4f}")

if __name__ == "__main__":
    analyze_pnl_calculation()
