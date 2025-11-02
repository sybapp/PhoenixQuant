"""回测结果可视化分析"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from phoenix_quant import load_backtest_config, run_backtest

# 设置中文字体支持
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial Unicode MS", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False


def _load_price_from_config(config) -> pd.DataFrame | None:
    data_cfg = config.data
    if not data_cfg.cache:
        return None
    cache = Path(data_cfg.cache)
    if not cache.exists():
        return None
    df = pd.read_csv(cache)
    if "datetime" not in df.columns:
        df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")
    else:
        df["datetime"] = pd.to_datetime(df["datetime"])
    return df[["datetime", "close"]].rename(columns={"close": "price"})


def load_data(config_path: str | None, output_dir: Path | None = None):
    """加载或生成回测结果数据"""
    price_df = None
    config = None

    if config_path:
        config = load_backtest_config(config_path)
        analyzer = run_backtest(config)
        equity_df = analyzer.equity.copy()
        trades_df = analyzer.trades.copy()
        price_df = _load_price_from_config(config)

        equity_df["datetime"] = pd.to_datetime(equity_df["timestamp"], unit="ms")
        trades_df["datetime"] = pd.to_datetime(trades_df["timestamp"], unit="ms")

        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
            equity_df.to_csv(output_dir / f"{Path(config_path).stem}_equity.csv", index=False)
            trades_df.to_csv(output_dir / f"{Path(config_path).stem}_trades.csv", index=False)
    else:
        equity_df = pd.read_csv("backtest_equity.csv")
        trades_df = pd.read_csv("backtest_trades.csv")
        equity_df["datetime"] = pd.to_datetime(equity_df["timestamp"], unit="ms")
        trades_df["datetime"] = pd.to_datetime(trades_df["timestamp"], unit="ms")

    return equity_df, trades_df, price_df, config


def calculate_metrics(equity_df: pd.DataFrame) -> dict:
    """计算关键指标"""
    equity = equity_df["equity"].values

    cummax = pd.Series(equity).cummax()
    drawdown = (equity - cummax) / cummax * 100
    returns = pd.Series(equity).pct_change().fillna(0)

    sharpe = returns.mean() / returns.std() * (365 * 24 * 60) ** 0.5 if returns.std() > 0 else 0.0

    return {
        "drawdown": drawdown,
        "returns": returns * 100,
        "cummax": cummax,
        "sharpe_ratio": sharpe,
        "total_return": (equity[-1] / equity[0] - 1) * 100 if len(equity) > 1 else 0.0,
    }

def plot_equity_curve(equity_df, metrics, ax):
    """绘制权益曲线"""
    ax.plot(
        equity_df["datetime"],
        equity_df["equity"],
        label="Equity",
        color="#2E86AB",
        linewidth=1.5,
    )
    ax.plot(
        equity_df["datetime"],
        metrics["cummax"],
        label="Historical High",
        color="#A23B72",
        linestyle="--",
        alpha=0.6,
        linewidth=1,
    )

    ax.set_title(
        f"Equity Curve (Return: {metrics['total_return']:.2f}%, Sharpe: {metrics['sharpe_ratio']:.2f})",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_xlabel("Date")
    ax.set_ylabel("Equity (USDT)")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    ax.tick_params(axis="x", rotation=45)

def plot_drawdown(equity_df, metrics, ax):
    """绘制回撤曲线"""
    drawdown = metrics['drawdown']
    max_dd = drawdown.min()

    ax.fill_between(equity_df['datetime'], 0, drawdown,
                     color='#F18F01', alpha=0.6, label='Drawdown')
    ax.axhline(y=max_dd, color='red', linestyle='--',
               linewidth=1, label=f'Max DD: {max_dd:.2f}%')

    ax.set_title('Drawdown Analysis', fontsize=12, fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Drawdown (%)')
    ax.legend(loc='lower left')
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    ax.tick_params(axis='x', rotation=45)

def plot_position_distribution(equity_df, ax):
    """绘制仓位分布"""
    equity_df['position_ratio'] = equity_df['position_value'] / equity_df['equity'] * 100

    ax.fill_between(equity_df['datetime'], 0, equity_df['position_ratio'],
                     color='#6A4C93', alpha=0.5)
    ax.axhline(y=50, color='orange', linestyle='--',
               linewidth=1, alpha=0.6, label='50% Line')

    avg_position = equity_df['position_ratio'].mean()
    ax.axhline(y=avg_position, color='green', linestyle='-',
               linewidth=1.5, label=f'Avg: {avg_position:.1f}%')

    ax.set_title('Position Ratio Over Time', fontsize=12, fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Position Ratio (%)')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 100])
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    ax.tick_params(axis='x', rotation=45)

def plot_trade_analysis(trades_df, ax):
    """绘制交易分析"""
    buys = trades_df[trades_df['side'] == 'buy']
    sells = trades_df[trades_df['side'] == 'sell']

    # 按层级统计买入
    layer_counts = buys['tag'].value_counts().sort_index()

    colors = ['#2E86AB', '#A23B72', '#F18F01', '#6A4C93', '#1B998B']
    bars = ax.bar(range(len(layer_counts)), layer_counts.values, color=colors[:len(layer_counts)])

    ax.set_title(f'Buy Orders by Layer (Total: {len(buys)} buys, {len(sells)} sells)',
                 fontsize=12, fontweight='bold')
    ax.set_xlabel('Layer')
    ax.set_ylabel('Order Count')
    ax.set_xticks(range(len(layer_counts)))
    ax.set_xticklabels(layer_counts.index)
    ax.grid(True, alpha=0.3, axis='y')

    # 在柱子上标注数值
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontsize=9)

def plot_price_vs_trades(trades_df, ax, price_df=None):
    """绘制价格与交易点位"""
    buys = trades_df[trades_df['side'] == 'buy']
    sells = trades_df[trades_df['side'] == 'sell']

    if price_df is not None and not price_df.empty:
        price_series = price_df.set_index("datetime")["price"].resample("1h").mean().interpolate()
    else:
        all_prices = pd.concat([buys[['datetime', 'price']], sells[['datetime', 'price']]]).sort_values('datetime')
        all_prices.set_index('datetime', inplace=True)
        price_series = all_prices['price'].resample('1h').mean().dropna()

    if not price_series.empty:
        ax.plot(
            price_series.index,
            price_series.values,
            color="gray",
            alpha=0.5,
            linewidth=1,
            label="Price (1h MA)",
        )

    ax.scatter(buys['datetime'], buys['price'],
               c='green', marker='^', s=20, alpha=0.6, label='Buy')
    ax.scatter(sells['datetime'], sells['price'],
               c='red', marker='v', s=50, alpha=0.8, label='Sell')

    ax.set_title('Price Chart with Trade Markers', fontsize=12, fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('BTC Price (USDT)')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    ax.tick_params(axis='x', rotation=45)

def plot_daily_returns(metrics, ax):
    """绘制每日收益分布"""
    returns = metrics["returns"]

    ax.hist(returns, bins=50, color="#2E86AB", alpha=0.7, edgecolor="black")
    ax.axvline(x=0, color="red", linestyle="--", linewidth=1.5)
    ax.axvline(
        x=returns.mean(),
        color="green",
        linestyle="--",
        linewidth=1.5,
        label=f"Mean: {returns.mean():.4f}%",
    )

    ax.set_title(
        f"Return Distribution (Positive: {(returns > 0).sum()}, Negative: {(returns < 0).sum()})",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_xlabel("Return (%)")
    ax.set_ylabel("Frequency")
    ax.legend()
    ax.grid(True, alpha=0.3)

def analyze_trade_performance(trades_df):
    """分析交易表现"""
    buys = trades_df[trades_df['side'] == 'buy'].copy()
    sells = trades_df[trades_df['side'] == 'sell'].copy()

    print("\n" + "="*60)
    print("交易表现深度分析")
    print("="*60)

    # 买入分析
    print(f"\n【买入分析】")
    print(f"总买入次数: {len(buys)}")
    print(f"总买入量: {buys['quantity'].sum():.4f} BTC")
    print(f"总买入成本: {(buys['price'] * buys['quantity']).sum():.2f} USDT")
    print(f"平均买入价: {buys['price'].mean():.2f} USDT")
    print(f"买入价格范围: {buys['price'].min():.2f} - {buys['price'].max():.2f} USDT")

    # 按层级分析
    print(f"\n【分层买入分析】")
    for layer in sorted(buys['tag'].unique()):
        layer_buys = buys[buys['tag'] == layer]
        print(f"{layer}: {len(layer_buys)}笔, "
              f"总量: {layer_buys['quantity'].sum():.4f} BTC, "
              f"均价: {layer_buys['price'].mean():.2f} USDT")

    # 卖出分析
    print(f"\n【卖出分析】")
    print(f"总卖出次数: {len(sells)}")
    print(f"总卖出量: {sells['quantity'].sum():.4f} BTC")
    print(f"总卖出收入: {(sells['price'] * sells['quantity']).sum():.2f} USDT")
    print(f"平均卖出价: {sells['price'].mean():.2f} USDT")
    print(f"卖出价格范围: {sells['price'].min():.2f} - {sells['price'].max():.2f} USDT")

    # 卖出触发分析
    sell_tags = sells['tag'].value_counts()
    print(f"\n【卖出触发原因】")
    for tag, count in sell_tags.items():
        print(f"{tag}: {count}次")

    # 简单盈亏分析（粗略估算）
    total_buy_cost = (buys['price'] * buys['quantity']).sum()
    total_sell_revenue = (sells['price'] * sells['quantity']).sum()
    total_fees = buys['fee'].sum() + sells['fee'].sum()

    print(f"\n【粗略盈亏分析】")
    print(f"总买入成本: {total_buy_cost:.2f} USDT")
    print(f"总卖出收入: {total_sell_revenue:.2f} USDT")
    print(f"已实现盈亏: {total_sell_revenue - (total_buy_cost * sells['quantity'].sum() / buys['quantity'].sum()):.2f} USDT")
    print(f"总手续费: {total_fees:.2f} USDT")

def main():
    parser = argparse.ArgumentParser(description="回测可视化")
    parser.add_argument("--config", help="配置文件路径，提供后将自动运行回测并生成图表")
    parser.add_argument("--output-dir", help="输出目录，用于保存CSV与图表")
    parser.add_argument("--no-show", action="store_true", help="生成图表但不显示窗口")
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else None

    print("加载回测数据...")
    equity_df, trades_df, price_df, config = load_data(args.config, output_dir)

    print("计算性能指标...")
    metrics = calculate_metrics(equity_df)

    print("\n【关键指标】")
    print(f"夏普比率 (年化): {metrics['sharpe_ratio']:.2f}")
    print(f"最大回撤: {metrics['drawdown'].min():.2f}%")
    print(f"累计收益: {metrics['total_return']:.2f}%")
    print(f"回测期间: {equity_df['datetime'].iloc[0]} 至 {equity_df['datetime'].iloc[-1]}")
    print(f"数据点数: {len(equity_df)} 根K线")

    # 创建可视化
    print("\n生成可视化图表...")
    fig = plt.figure(figsize=(16, 12))

    # 2x3布局
    ax1 = plt.subplot(3, 2, 1)
    plot_equity_curve(equity_df, metrics, ax1)

    ax2 = plt.subplot(3, 2, 2)
    plot_drawdown(equity_df, metrics, ax2)

    ax3 = plt.subplot(3, 2, 3)
    plot_position_distribution(equity_df, ax3)

    ax4 = plt.subplot(3, 2, 4)
    plot_trade_analysis(trades_df, ax4)

    ax5 = plt.subplot(3, 2, 5)
    plot_price_vs_trades(trades_df, ax5, price_df=price_df)

    ax6 = plt.subplot(3, 2, 6)
    plot_daily_returns(metrics, ax6)

    plt.tight_layout()

    # 保存图表
    output_name = f"{Path(args.config).stem if args.config else 'backtest'}_analysis.png"
    if output_dir:
        output_file = output_dir / output_name
    else:
        output_file = Path(output_name)
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    print(f"\n可视化图表已保存: {output_file}")

    if not args.no_show:
        plt.show()
    else:
        plt.close()

    # 交易表现分析
    analyze_trade_performance(trades_df)

    print("\n" + "="*60)
    print("分析完成！")
    print("="*60)

if __name__ == "__main__":
    main()
