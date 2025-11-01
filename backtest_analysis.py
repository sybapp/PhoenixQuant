# -*- coding: utf-8 -*-
"""
回测分析与可视化模块
提供回测结果的统计分析和图表展示
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from typing import Dict, List
import matplotlib.font_manager as fm
import os


class BacktestAnalyzer:
    """回测分析器"""

    def __init__(self, equity_df: pd.DataFrame, trades_df: pd.DataFrame,
                 initial_balance: float):
        self.equity_df = equity_df
        self.trades_df = trades_df
        self.initial_balance = initial_balance

    def _setup_chinese_fonts(self):
        """设置中文字体支持"""
        # 定义中文字体优先级列表
        chinese_fonts = [
            'WenQuanYi Micro Hei',      # 文泉驿微米黑 (Linux)
            'Source Han Sans CN',       # 思源黑体 (Linux/Mac)
            'Noto Sans CJK SC',         # Noto字体 (Linux)
            'SimHei',                   # 黑体 (Windows)
            'Microsoft YaHei',          # 微软雅黑 (Windows)
            'PingFang SC',              # 苹方 (Mac)
            'Hiragino Sans GB',         # 冬青黑体 (Mac)
            'STHeiti',                  # 华文黑体 (Mac)
            'DejaVu Sans',              # 回退字体
            'Arial Unicode MS'          # 回退字体
        ]

        # 检查系统中实际可用的字体
        available_fonts = set(f.name for f in fm.fontManager.ttflist)

        # 按优先级选择可用的字体
        selected_fonts = []
        for font in chinese_fonts:
            if font in available_fonts:
                selected_fonts.append(font)
                print(f"✓ 找到字体: {font}")

        # 如果没有找到任何中文字体，使用默认字体
        if not selected_fonts:
            print("⚠ 未找到中文字体，使用默认字体")
            selected_fonts = ['DejaVu Sans']
        else:
            print(f"✅ 使用字体: {selected_fonts[0]}")

        # 设置matplotlib字体
        plt.rcParams['font.sans-serif'] = selected_fonts
        plt.rcParams['axes.unicode_minus'] = False
        plt.rcParams['font.size'] = 10

        # 对于保存图片的情况，确保字体嵌入
        plt.rcParams['pdf.fonttype'] = 42
        plt.rcParams['ps.fonttype'] = 42

        return selected_fonts[0]

    def calculate_metrics(self) -> Dict:
        """计算详细的回测指标"""
        if len(self.equity_df) < 2:
            return {}

        equity = self.equity_df['equity'].values
        timestamps = self.equity_df['timestamp'].values

        # 基础收益指标
        final_equity = equity[-1]
        total_return = (final_equity - self.initial_balance) / self.initial_balance * 100

        # 计算收益率序列
        returns = np.diff(equity) / equity[:-1]

        # 最大回撤
        cummax = np.maximum.accumulate(equity)
        drawdown = (equity - cummax) / cummax * 100
        max_drawdown = np.min(drawdown)
        max_drawdown_idx = np.argmin(drawdown)

        # 回撤持续时间
        underwater = drawdown < 0
        drawdown_periods = []
        start_idx = None
        for i, is_underwater in enumerate(underwater):
            if is_underwater and start_idx is None:
                start_idx = i
            elif not is_underwater and start_idx is not None:
                drawdown_periods.append(i - start_idx)
                start_idx = None

        avg_drawdown_duration = np.mean(drawdown_periods) if drawdown_periods else 0
        max_drawdown_duration = np.max(drawdown_periods) if drawdown_periods else 0

        # 夏普比率（假设年化，1分钟数据）
        if len(returns) > 0 and np.std(returns) > 0:
            periods_per_year = 252 * 24 * 60  # 1分钟K线
            sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(periods_per_year)
        else:
            sharpe_ratio = 0.0

        # Sortino比率（只考虑下行波动）
        downside_returns = returns[returns < 0]
        if len(downside_returns) > 0 and np.std(downside_returns) > 0:
            sortino_ratio = np.mean(returns) / np.std(downside_returns) * np.sqrt(periods_per_year)
        else:
            sortino_ratio = 0.0

        # Calmar比率（年化收益/最大回撤）
        days = (timestamps[-1] - timestamps[0]) / (1000 * 86400)
        annualized_return = total_return * (365 / days) if days > 0 else 0
        calmar_ratio = abs(annualized_return / max_drawdown) if max_drawdown < 0 else 0

        # 交易统计
        if not self.trades_df.empty:
            buy_trades = self.trades_df[self.trades_df['side'] == 'buy']
            sell_trades = self.trades_df[self.trades_df['side'] == 'sell']

            total_trades = len(buy_trades)
            total_fees = self.trades_df['fee'].sum()
            avg_trade_size = buy_trades['quantity'].mean() if len(buy_trades) > 0 else 0

            # 盈利交易统计（FIFO配对买卖计算实际PnL）
            profit_trades = 0
            loss_trades = 0
            total_profit = 0
            total_loss = 0

            # 使用FIFO方法配对买卖计算PnL
            buy_queue = []  # 买入队列

            for _, trade in self.trades_df.iterrows():
                if trade['side'] == 'buy':
                    # 加入买入队列
                    buy_queue.append({
                        'price': trade['price'],
                        'quantity': trade['quantity'],
                        'fee': trade['fee']
                    })
                elif trade['side'] == 'sell':
                    # 卖出时与买入配对计算PnL
                    sell_qty = trade['quantity']
                    sell_price = trade['price']
                    sell_fee = trade['fee']

                    while sell_qty > 0 and buy_queue:
                        buy = buy_queue[0]

                        # 计算本次配对的数量
                        matched_qty = min(sell_qty, buy['quantity'])

                        # 计算PnL: (卖出价 - 买入价) * 数量 - 手续费
                        buy_cost = buy['price'] * matched_qty + buy['fee'] * (matched_qty / buy['quantity'])
                        sell_revenue = sell_price * matched_qty - sell_fee * (matched_qty / trade['quantity'])
                        pnl = sell_revenue - buy_cost

                        if pnl > 0:
                            profit_trades += 1
                            total_profit += pnl
                        else:
                            loss_trades += 1
                            total_loss += abs(pnl)

                        # 更新队列
                        sell_qty -= matched_qty
                        buy['quantity'] -= matched_qty

                        if buy['quantity'] <= 1e-8:  # 防止浮点误差
                            buy_queue.pop(0)

            win_rate = (profit_trades / (profit_trades + loss_trades) * 100) if (profit_trades + loss_trades) > 0 else 0
            avg_win = total_profit / profit_trades if profit_trades > 0 else 0
            avg_loss = total_loss / loss_trades if loss_trades > 0 else 0
            profit_factor = total_profit / total_loss if total_loss > 0 else 0

        else:
            total_trades = 0
            total_fees = 0
            avg_trade_size = 0
            win_rate = 0
            avg_win = 0
            avg_loss = 0
            profit_factor = 0

        metrics = {
            # 收益指标
            'initial_balance': self.initial_balance,
            'final_equity': final_equity,
            'total_return_pct': total_return,
            'annualized_return_pct': annualized_return,

            # 风险指标
            'max_drawdown_pct': max_drawdown,
            'avg_drawdown_duration_bars': avg_drawdown_duration,
            'max_drawdown_duration_bars': max_drawdown_duration,
            'volatility': np.std(returns) * 100 if len(returns) > 0 else 0,

            # 风险调整收益
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,

            # 交易统计
            'total_trades': total_trades,
            'win_rate_pct': win_rate,
            'profit_factor': profit_factor,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'total_fees': total_fees,
            'avg_trade_size': avg_trade_size,

            # 时间统计
            'backtest_days': days,
            'total_bars': len(equity),
        }

        return metrics

    def print_report(self):
        """打印回测报告"""
        metrics = self.calculate_metrics()

        if not metrics:
            print("没有足够的数据生成报告")
            return

        print("\n" + "="*60)
        print(" " * 20 + "回测报告")
        print("="*60)

        print("\n【收益指标】")
        print(f"  初始资金:          ${metrics['initial_balance']:,.2f}")
        print(f"  最终权益:          ${metrics['final_equity']:,.2f}")
        print(f"  总收益率:          {metrics['total_return_pct']:.2f}%")
        print(f"  年化收益率:        {metrics['annualized_return_pct']:.2f}%")

        print("\n【风险指标】")
        print(f"  最大回撤:          {metrics['max_drawdown_pct']:.2f}%")
        print(f"  波动率:            {metrics['volatility']:.2f}%")
        print(f"  平均回撤持续:      {metrics['avg_drawdown_duration_bars']:.0f} 根K线")
        print(f"  最长回撤持续:      {metrics['max_drawdown_duration_bars']:.0f} 根K线")

        print("\n【风险调整收益】")
        print(f"  夏普比率:          {metrics['sharpe_ratio']:.3f}")
        print(f"  Sortino比率:       {metrics['sortino_ratio']:.3f}")
        print(f"  Calmar比率:        {metrics['calmar_ratio']:.3f}")

        print("\n【交易统计】")
        print(f"  总交易次数:        {metrics['total_trades']:.0f}")
        print(f"  胜率:              {metrics['win_rate_pct']:.2f}%")
        print(f"  盈亏比:            {metrics['profit_factor']:.2f}")
        print(f"  平均盈利:          ${metrics['avg_win']:.2f}")
        print(f"  平均亏损:          ${metrics['avg_loss']:.2f}")
        print(f"  总手续费:          ${metrics['total_fees']:.2f}")
        print(f"  平均交易规模:      {metrics['avg_trade_size']:.4f}")

        # 提示：如果盈亏比很低但权益增长很多，说明有大量未平仓浮盈
        if metrics['total_return_pct'] > 10 and metrics['profit_factor'] < 0.5:
            print(f"\n  ⚠️  注意: 权益增长主要来自未平仓持仓浮盈")
            print(f"           建议优化止盈策略以锁定利润")

        print("\n【时间统计】")
        print(f"  回测天数:          {metrics['backtest_days']:.1f} 天")
        print(f"  总K线数:           {metrics['total_bars']:.0f}")

        print("\n" + "="*60 + "\n")

    def plot_results(self, save_path: str = None):
        """绘制回测结果图表"""
        fig = plt.figure(figsize=(16, 12))

        # 设置中文字体
        current_font = self._setup_chinese_fonts()
        print(f"图表将使用字体: {current_font}")

        # 1. 权益曲线
        ax1 = plt.subplot(3, 2, 1)
        ax1.plot(self.equity_df['datetime'], self.equity_df['equity'],
                label='总权益', linewidth=2, color='#2E86AB')
        ax1.plot(self.equity_df['datetime'], self.equity_df['balance'],
                label='现金余额', linewidth=1, alpha=0.7, color='#A23B72')
        ax1.axhline(y=self.initial_balance, color='gray', linestyle='--',
                   alpha=0.5, label='初始资金')
        ax1.set_title('权益曲线', fontsize=12, fontweight='bold')
        ax1.set_xlabel('时间')
        ax1.set_ylabel('权益 ($)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. 回撤曲线
        ax2 = plt.subplot(3, 2, 2)
        equity = self.equity_df['equity'].values
        cummax = np.maximum.accumulate(equity)
        drawdown = (equity - cummax) / cummax * 100
        ax2.fill_between(self.equity_df['datetime'], drawdown, 0,
                        color='#F18F01', alpha=0.6)
        ax2.set_title('回撤曲线', fontsize=12, fontweight='bold')
        ax2.set_xlabel('时间')
        ax2.set_ylabel('回撤 (%)')
        ax2.grid(True, alpha=0.3)

        # 3. 收益率分布（改进版：显示有意义的收益率分布）
        ax3 = plt.subplot(3, 2, 3)
        returns = np.diff(equity) / equity[:-1] * 100

        # 过滤掉微小的收益率变化（仅显示>0.05%的变化）
        non_zero_returns = returns[np.abs(returns) > 0.05]

        if len(non_zero_returns) > 0:
            # 移除极端异常值（超过3个标准差）
            mean_ret = np.mean(non_zero_returns)
            std_ret = np.std(non_zero_returns)
            filtered_returns = non_zero_returns[
                np.abs(non_zero_returns - mean_ret) <= 3 * std_ret
            ]

            if len(filtered_returns) > 0:
                ax3.hist(filtered_returns, bins=30, color='#06A77D',
                        alpha=0.7, edgecolor='black')
                ax3.axvline(x=0, color='red', linestyle='--',
                           alpha=0.5, label='零收益')
                ax3.axvline(x=mean_ret, color='blue', linestyle='--',
                           alpha=0.5, label=f'均值: {mean_ret:.4f}%')

                # 添加统计信息
                stats_text = f'样本数: {len(filtered_returns)}\n'
                stats_text += f'均值: {mean_ret:.4f}%\n'
                stats_text += f'标准差: {std_ret:.4f}%\n'
                stats_text += f'过滤阈值: >0.05%'
                ax3.text(0.02, 0.98, stats_text,
                        transform=ax3.transAxes,
                        verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                        fontsize=8)

                ax3.legend(fontsize=8)
            else:
                ax3.text(0.5, 0.5, '数据不足',
                        ha='center', va='center', transform=ax3.transAxes)
        else:
            ax3.text(0.5, 0.5, '无有意义的收益率变化',
                    ha='center', va='center', transform=ax3.transAxes)

        ax3.set_title('收益率分布（|变化|>0.05%）', fontsize=12, fontweight='bold')
        ax3.set_xlabel('收益率 (%)')
        ax3.set_ylabel('频数')
        ax3.grid(True, alpha=0.3)

        # 4. 累计收益率
        ax4 = plt.subplot(3, 2, 4)
        cumulative_returns = (equity / self.initial_balance - 1) * 100
        ax4.plot(self.equity_df['datetime'], cumulative_returns,
                linewidth=2, color='#D81159')
        ax4.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax4.set_title('累计收益率', fontsize=12, fontweight='bold')
        ax4.set_xlabel('时间')
        ax4.set_ylabel('累计收益率 (%)')
        ax4.grid(True, alpha=0.3)

        # 5. 交易分布（如果有交易数据）
        if not self.trades_df.empty:
            ax5 = plt.subplot(3, 2, 5)
            buy_trades = self.trades_df[self.trades_df['side'] == 'buy']
            sell_trades = self.trades_df[self.trades_df['side'] == 'sell']

            ax5.scatter(buy_trades['datetime'], buy_trades['price'],
                       marker='^', color='green', s=50, alpha=0.6, label='买入')
            ax5.scatter(sell_trades['datetime'], sell_trades['price'],
                       marker='v', color='red', s=50, alpha=0.6, label='卖出')
            ax5.set_title('交易分布', fontsize=12, fontweight='bold')
            ax5.set_xlabel('时间')
            ax5.set_ylabel('价格 ($)')
            ax5.legend()
            ax5.grid(True, alpha=0.3)

        # 6. 月度收益热力图
        ax6 = plt.subplot(3, 2, 6)
        equity_df_copy = self.equity_df.copy()
        equity_df_copy['year'] = equity_df_copy['datetime'].dt.year
        equity_df_copy['month'] = equity_df_copy['datetime'].dt.month

        # 计算月度收益
        monthly_returns = equity_df_copy.groupby(['year', 'month'])['equity'].apply(
            lambda x: (x.iloc[-1] - x.iloc[0]) / x.iloc[0] * 100 if len(x) > 0 else 0
        ).unstack(fill_value=0)

        if not monthly_returns.empty:
            sns.heatmap(monthly_returns, annot=True, fmt='.2f', cmap='RdYlGn',
                       center=0, cbar_kws={'label': '收益率 (%)'}, ax=ax6)
            ax6.set_title('月度收益热力图', fontsize=12, fontweight='bold')
            ax6.set_xlabel('月份')
            ax6.set_ylabel('年份')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"图表已保存到: {save_path}")

        plt.show()

    def export_to_excel(self, filename: str):
        """导出回测结果到Excel"""
        metrics = self.calculate_metrics()

        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            # 指标摘要
            metrics_df = pd.DataFrame([metrics]).T
            metrics_df.columns = ['数值']
            metrics_df.to_excel(writer, sheet_name='指标摘要')

            # 权益曲线
            self.equity_df.to_excel(writer, sheet_name='权益曲线', index=False)

            # 交易记录
            if not self.trades_df.empty:
                self.trades_df.to_excel(writer, sheet_name='交易记录', index=False)

        print(f"回测结果已导出到: {filename}")


class PerformanceMonitor:
    """实时性能监控器（用于回测过程中）"""

    def __init__(self):
        self.checkpoints = []

    def checkpoint(self, name: str, equity: float, timestamp: float):
        """记录检查点"""
        self.checkpoints.append({
            'name': name,
            'equity': equity,
            'timestamp': timestamp,
            'datetime': datetime.fromtimestamp(timestamp / 1000)
        })

    def print_progress(self):
        """打印进度"""
        if len(self.checkpoints) < 2:
            return

        latest = self.checkpoints[-1]
        first = self.checkpoints[0]

        total_return = (latest['equity'] - first['equity']) / first['equity'] * 100

        print(f"[{latest['datetime']}] 权益: ${latest['equity']:.2f}, "
              f"收益率: {total_return:.2f}%")
