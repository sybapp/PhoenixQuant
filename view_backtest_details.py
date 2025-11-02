"""查看详细回测结果"""

from phoenix_quant import load_backtest_config, run_backtest

def main():
    config = load_backtest_config("configs/elastic_dip.yaml")
    analyzer = run_backtest(config)

    # 打印基本报告
    analyzer.print_report()

    # 显示权益曲线样本数据
    print("\n=== 权益曲线 (前10行和后10行) ===")
    equity_df = analyzer.equity
    print("\n前10行:")
    print(equity_df.head(10).to_string())
    print(f"\n...(共{len(equity_df)}行)...\n")
    print("后10行:")
    print(equity_df.tail(10).to_string())

    # 显示交易记录
    print("\n\n=== 交易记录 (前20笔和后20笔) ===")
    trades_df = analyzer.trades
    if not trades_df.empty:
        print("\n前20笔:")
        print(trades_df.head(20).to_string())
        print(f"\n...(共{len(trades_df)}笔)...\n")
        print("后20笔:")
        print(trades_df.tail(20).to_string())

        # 按买卖方向统计
        print("\n\n=== 交易统计 ===")
        buy_trades = trades_df[trades_df['side'] == 'buy']
        sell_trades = trades_df[trades_df['side'] == 'sell']
        print(f"买入笔数: {len(buy_trades)}")
        print(f"卖出笔数: {len(sell_trades)}")
        print(f"总交易量 (买入): {buy_trades['quantity'].sum():.4f}")
        print(f"总交易量 (卖出): {sell_trades['quantity'].sum():.4f}")
        print(f"平均买入价: {buy_trades['price'].mean():.2f}")
        print(f"平均卖出价: {sell_trades['price'].mean():.2f}")
    else:
        print("无交易记录")

    # 导出数据
    print("\n\n=== 导出数据文件 ===")
    equity_df.to_csv("backtest_equity.csv", index=False)
    print("权益曲线已保存: backtest_equity.csv")

    if not trades_df.empty:
        trades_df.to_csv("backtest_trades.csv", index=False)
        print("交易记录已保存: backtest_trades.csv")

if __name__ == "__main__":
    main()
