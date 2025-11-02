"""批量测试多个配置并生成对比报告"""

import pandas as pd
from pathlib import Path
from phoenix_quant import load_backtest_config, run_backtest

def run_single_backtest(config_path: str):
    """运行单个配置的回测"""
    print(f"\n{'='*60}")
    print(f"测试配置: {config_path}")
    print('='*60)

    config = load_backtest_config(config_path)
    analyzer = run_backtest(config)

    summary = analyzer.build_summary()
    trades = analyzer.trades

    # 计算额外指标
    buys = trades[trades['side'] == 'buy']
    sells = trades[trades['side'] == 'sell']

    # 卖出原因统计
    sell_reasons = {}
    if not sells.empty:
        sell_counts = sells['tag'].value_counts()
        sell_reasons = sell_counts.to_dict()

    # 分层统计
    layer_stats = {}
    if not buys.empty:
        layer_counts = buys['tag'].value_counts()
        layer_stats = layer_counts.to_dict()

    result = {
        'config': Path(config_path).stem,
        'initial_balance': summary.initial_balance,
        'final_equity': summary.final_equity,
        'return_pct': summary.total_return_pct,
        'max_drawdown_pct': summary.max_drawdown_pct,
        'win_rate_pct': summary.win_rate_pct,
        'trade_count': summary.trade_count,
        'buy_count': len(buys),
        'sell_count': len(sells),
        'take_profit_count': sell_reasons.get('take-profit', 0),
        'timeout_count': sell_reasons.get('timeout', 0),
        'stop_count': sell_reasons.get('stop', 0),
        'layer1_count': layer_stats.get('layer-1', 0),
        'layer2_count': layer_stats.get('layer-2', 0),
        'layer3_count': layer_stats.get('layer-3', 0),
    }

    # 计算夏普比率
    if not analyzer.equity.empty:
        returns = analyzer.equity['equity'].pct_change().fillna(0)
        sharpe = returns.mean() / returns.std() * (365 * 24 * 60) ** 0.5 if returns.std() > 0 else 0
        result['sharpe_ratio'] = sharpe
    else:
        result['sharpe_ratio'] = 0

    # 打印简报
    print(f"\n收益率: {result['return_pct']:.2f}%")
    print(f"最大回撤: {result['max_drawdown_pct']:.2f}%")
    print(f"夏普比率: {result['sharpe_ratio']:.2f}")
    print(f"交易次数: {result['trade_count']} (买:{result['buy_count']}, 卖:{result['sell_count']})")
    print(f"胜率: {result['win_rate_pct']:.2f}%")
    print(f"卖出原因: TP={result['take_profit_count']}, Timeout={result['timeout_count']}, SL={result['stop_count']}")

    return result

def main():
    """批量测试所有配置"""
    configs = [
        "configs/elastic_dip.yaml",  # 基准配置
        "configs/opt_signal_quality.yaml",
        "configs/opt_layer_balance.yaml",
        "configs/opt_risk_control.yaml",
        "configs/opt_comprehensive.yaml",
    ]

    results = []

    for config_path in configs:
        try:
            result = run_single_backtest(config_path)
            results.append(result)
        except Exception as e:
            print(f"错误: {config_path} - {e}")
            import traceback
            traceback.print_exc()
            continue

    # 生成对比表格
    df = pd.DataFrame(results)

    print("\n\n" + "="*80)
    print("对比分析结果")
    print("="*80)

    # 关键指标对比
    print("\n【关键指标对比】")
    key_metrics = df[['config', 'return_pct', 'max_drawdown_pct', 'sharpe_ratio', 'win_rate_pct']]
    print(key_metrics.to_string(index=False))

    # 交易统计对比
    print("\n【交易统计对比】")
    trade_stats = df[['config', 'trade_count', 'buy_count', 'sell_count',
                      'take_profit_count', 'timeout_count', 'stop_count']]
    print(trade_stats.to_string(index=False))

    # 分层统计对比
    print("\n【分层统计对比】")
    layer_stats = df[['config', 'layer1_count', 'layer2_count', 'layer3_count']]
    print(layer_stats.to_string(index=False))

    # 找出最佳配置
    print("\n【最佳配置排名】")
    print(f"最高收益: {df.loc[df['return_pct'].idxmax(), 'config']} ({df['return_pct'].max():.2f}%)")
    print(f"最小回撤: {df.loc[df['max_drawdown_pct'].idxmax(), 'config']} ({df['max_drawdown_pct'].max():.2f}%)")
    print(f"最高夏普: {df.loc[df['sharpe_ratio'].idxmax(), 'config']} ({df['sharpe_ratio'].max():.2f})")
    print(f"最高胜率: {df.loc[df['win_rate_pct'].idxmax(), 'config']} ({df['win_rate_pct'].max():.2f}%)")

    # 综合评分（简单加权）
    df['综合得分'] = (
        df['return_pct'] / 10 +  # 收益权重
        (df['max_drawdown_pct'] + 20) / 5 +  # 回撤权重（加20是因为负值）
        df['sharpe_ratio'] * 5 +  # 夏普权重
        df['win_rate_pct'] / 10  # 胜率权重
    )

    best_idx = df['综合得分'].idxmax()
    print(f"\n综合最佳: {df.loc[best_idx, 'config']} (得分: {df.loc[best_idx, '综合得分']:.2f})")

    # 保存完整结果
    df.to_csv('backtest_comparison.csv', index=False)
    print(f"\n完整对比结果已保存: backtest_comparison.csv")

    print("\n" + "="*80)

if __name__ == "__main__":
    main()
