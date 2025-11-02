"""生成优化配置对比可视化"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 读取对比数据
df = pd.read_csv('backtest_comparison.csv')

# 简化配置名称
df['config'] = df['config'].str.replace('opt_', '').str.replace('_', ' ').str.title()
df.loc[df['config'] == 'Elastic Dip', 'config'] = 'Baseline'

fig = plt.figure(figsize=(16, 10))

# 1. 收益率对比
ax1 = plt.subplot(2, 3, 1)
colors = ['#808080' if c == 'Baseline' else '#2E86AB' for c in df['config']]
bars = ax1.bar(range(len(df)), df['return_pct'], color=colors, alpha=0.8)
ax1.set_title('Return Comparison', fontsize=13, fontweight='bold')
ax1.set_ylabel('Return (%)')
ax1.set_xticks(range(len(df)))
ax1.set_xticklabels(df['config'], rotation=45, ha='right')
ax1.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5)
ax1.grid(True, alpha=0.3, axis='y')
for i, (bar, val) in enumerate(zip(bars, df['return_pct'])):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
             f'{val:.1f}%', ha='center', va='bottom' if val > 0 else 'top', fontsize=9)

# 2. 最大回撤对比
ax2 = plt.subplot(2, 3, 2)
colors = ['#808080' if c == 'Baseline' else '#F18F01' for c in df['config']]
bars = ax2.bar(range(len(df)), df['max_drawdown_pct'], color=colors, alpha=0.8)
ax2.set_title('Max Drawdown Comparison', fontsize=13, fontweight='bold')
ax2.set_ylabel('Drawdown (%)')
ax2.set_xticks(range(len(df)))
ax2.set_xticklabels(df['config'], rotation=45, ha='right')
ax2.axhline(y=-15, color='red', linestyle='--', linewidth=1, alpha=0.5, label='-15% Warning')
ax2.grid(True, alpha=0.3, axis='y')
ax2.legend(fontsize=8)
for bar, val in zip(bars, df['max_drawdown_pct']):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
             f'{val:.1f}%', ha='center', va='top', fontsize=9)

# 3. 夏普比率对比
ax3 = plt.subplot(2, 3, 3)
colors = ['#808080' if c == 'Baseline' else '#6A4C93' for c in df['config']]
bars = ax3.bar(range(len(df)), df['sharpe_ratio'], color=colors, alpha=0.8)
ax3.set_title('Sharpe Ratio Comparison', fontsize=13, fontweight='bold')
ax3.set_ylabel('Sharpe Ratio')
ax3.set_xticks(range(len(df)))
ax3.set_xticklabels(df['config'], rotation=45, ha='right')
ax3.axhline(y=2.0, color='green', linestyle='--', linewidth=1, alpha=0.5, label='Target: 2.0')
ax3.grid(True, alpha=0.3, axis='y')
ax3.legend(fontsize=8)
for bar, val in zip(bars, df['sharpe_ratio']):
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height,
             f'{val:.2f}', ha='center', va='bottom' if val > 0 else 'top', fontsize=9)

# 4. 胜率对比
ax4 = plt.subplot(2, 3, 4)
colors = ['#808080' if c == 'Baseline' else '#1B998B' for c in df['config']]
bars = ax4.bar(range(len(df)), df['win_rate_pct'], color=colors, alpha=0.8)
ax4.set_title('Win Rate Comparison', fontsize=13, fontweight='bold')
ax4.set_ylabel('Win Rate (%)')
ax4.set_xticks(range(len(df)))
ax4.set_xticklabels(df['config'], rotation=45, ha='right')
ax4.axhline(y=30, color='orange', linestyle='--', linewidth=1, alpha=0.5, label='Target: 30%')
ax4.grid(True, alpha=0.3, axis='y')
ax4.legend(fontsize=8)
for bar, val in zip(bars, df['win_rate_pct']):
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height,
             f'{val:.1f}%', ha='center', va='bottom', fontsize=9)

# 5. 交易次数对比
ax5 = plt.subplot(2, 3, 5)
colors = ['#808080' if c == 'Baseline' else '#A23B72' for c in df['config']]
bars = ax5.bar(range(len(df)), df['trade_count'], color=colors, alpha=0.8)
ax5.set_title('Trade Count Comparison', fontsize=13, fontweight='bold')
ax5.set_ylabel('Number of Trades')
ax5.set_xticks(range(len(df)))
ax5.set_xticklabels(df['config'], rotation=45, ha='right')
ax5.grid(True, alpha=0.3, axis='y')
for bar, val in zip(bars, df['trade_count']):
    height = bar.get_height()
    ax5.text(bar.get_x() + bar.get_width()/2., height,
             f'{int(val)}', ha='center', va='bottom', fontsize=9)

# 6. 综合得分雷达图
ax6 = plt.subplot(2, 3, 6, projection='polar')

# 标准化指标
metrics = ['Return', 'Low DD', 'Sharpe', 'Win Rate', 'Low Trades']
baseline_idx = df[df['config'] == 'Baseline'].index[0]
best_config_idx = df['综合得分'].idxmax()

# 标准化到0-100分
def normalize(series, higher_better=True):
    if higher_better:
        return (series - series.min()) / (series.max() - series.min()) * 100
    else:
        return (series.max() - series) / (series.max() - series.min()) * 100

scores_baseline = [
    normalize(df['return_pct'])[baseline_idx],
    normalize(df['max_drawdown_pct'], False)[baseline_idx],
    normalize(df['sharpe_ratio'])[baseline_idx],
    normalize(df['win_rate_pct'])[baseline_idx],
    normalize(df['trade_count'], False)[baseline_idx]
]

scores_best = [
    normalize(df['return_pct'])[best_config_idx],
    normalize(df['max_drawdown_pct'], False)[best_config_idx],
    normalize(df['sharpe_ratio'])[best_config_idx],
    normalize(df['win_rate_pct'])[best_config_idx],
    normalize(df['trade_count'], False)[best_config_idx]
]

angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
scores_baseline += scores_baseline[:1]
scores_best += scores_best[:1]
angles += angles[:1]

ax6.plot(angles, scores_baseline, 'o-', linewidth=2, label='Baseline', color='gray')
ax6.fill(angles, scores_baseline, alpha=0.15, color='gray')
ax6.plot(angles, scores_best, 'o-', linewidth=2, label=f'Best: {df.loc[best_config_idx, "config"]}', color='#2E86AB')
ax6.fill(angles, scores_best, alpha=0.25, color='#2E86AB')
ax6.set_xticks(angles[:-1])
ax6.set_xticklabels(metrics, fontsize=9)
ax6.set_ylim(0, 100)
ax6.set_title('Performance Radar', fontsize=13, fontweight='bold', pad=20)
ax6.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=9)
ax6.grid(True)

plt.tight_layout()
plt.savefig('optimization_comparison.png', dpi=150, bbox_inches='tight')
print("优化对比图表已保存: optimization_comparison.png")

# 打印总结
print("\n" + "="*80)
print("优化方案总结")
print("="*80)

for idx, row in df.iterrows():
    print(f"\n【{row['config']}】")
    print(f"  收益率: {row['return_pct']:.2f}% (基准: {df.loc[baseline_idx, 'return_pct']:.2f}%)")
    print(f"  最大回撤: {row['max_drawdown_pct']:.2f}% (基准: {df.loc[baseline_idx, 'max_drawdown_pct']:.2f}%)")
    print(f"  夏普比率: {row['sharpe_ratio']:.2f} (基准: {df.loc[baseline_idx, 'sharpe_ratio']:.2f})")
    print(f"  胜率: {row['win_rate_pct']:.2f}% (基准: {df.loc[baseline_idx, 'win_rate_pct']:.2f}%)")
    print(f"  交易次数: {int(row['trade_count'])} (基准: {int(df.loc[baseline_idx, 'trade_count'])})")
    print(f"  综合得分: {row['综合得分']:.2f}")

print("\n" + "="*80)
print("推荐配置")
print("="*80)
print(f"\n✅ 激进型（最高收益）: {df.loc[df['return_pct'].idxmax(), 'config']}")
print(f"   - 收益率: {df['return_pct'].max():.2f}%")
print(f"   - 适合风险承受能力强的交易者")

print(f"\n✅ 稳健型（最优综合）: {df.loc[best_config_idx, 'config']}")
print(f"   - 综合得分: {df.loc[best_config_idx, '综合得分']:.2f}")
print(f"   - 平衡收益与风险，适合大多数交易者")

print(f"\n✅ 保守型（最低回撤）: {df.loc[df['max_drawdown_pct'].idxmax(), 'config']}")
print(f"   - 最大回撤: {df['max_drawdown_pct'].max():.2f}%")
print(f"   - 适合风险厌恶型交易者")

print("\n" + "="*80)
