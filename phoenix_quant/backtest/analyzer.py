"""回测结果分析工具"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class Summary:
    """回测摘要数据"""

    initial_balance: float
    final_equity: float
    total_return_pct: float
    max_drawdown_pct: float
    win_rate_pct: float
    trade_count: int


class BacktestAnalyzer:
    """生成简洁的回测报告"""

    def __init__(self, equity: pd.DataFrame, trades: pd.DataFrame, initial_balance: float):
        self.equity = equity
        self.trades = trades
        self.initial_balance = initial_balance

    def _calculate_drawdown(self, series: pd.Series) -> float:
        cummax = series.cummax()
        drawdown = (series - cummax) / cummax * 100
        return drawdown.min() if not drawdown.empty else 0.0

    def _calculate_win_rate(self) -> float:
        if self.trades.empty:
            return 0.0
        buys = self.trades[self.trades["side"] == "buy"]
        sells = self.trades[self.trades["side"] == "sell"]
        if buys.empty or sells.empty:
            return 0.0

        wins = 0
        losses = 0
        queue = []
        for _, trade in self.trades.iterrows():
            if trade["side"] == "buy":
                queue.append(trade.copy())
            else:
                qty = trade["quantity"]
                while qty > 0 and queue:
                    buy_trade = queue[0]
                    matched = min(qty, buy_trade["quantity"])
                    pnl_piece = (trade["price"] - buy_trade["price"]) * matched
                    if pnl_piece > 0:
                        wins += 1
                    else:
                        losses += 1
                    qty -= matched
                    buy_trade["quantity"] -= matched
                    if buy_trade["quantity"] <= 1e-8:
                        queue.pop(0)
        total = wins + losses
        return wins / total * 100 if total else 0.0

    def build_summary(self) -> Summary:
        final_equity = float(self.equity["equity"].iloc[-1]) if not self.equity.empty else self.initial_balance
        total_return = (final_equity - self.initial_balance) / self.initial_balance * 100
        max_drawdown = self._calculate_drawdown(self.equity["equity"]) if not self.equity.empty else 0.0
        win_rate = self._calculate_win_rate()
        trade_count = len(self.trades)
        return Summary(
            initial_balance=self.initial_balance,
            final_equity=final_equity,
            total_return_pct=total_return,
            max_drawdown_pct=max_drawdown,
            win_rate_pct=win_rate,
            trade_count=trade_count,
        )

    def to_dataframe(self) -> pd.DataFrame:
        summary = self.build_summary()
        return pd.DataFrame([summary.__dict__])

    def print_report(self) -> None:
        summary = self.build_summary()
        print("\n=== 回测结果概览 ===")
        print(f"初始资金: {summary.initial_balance:,.2f}")
        print(f"最终权益: {summary.final_equity:,.2f}")
        print(f"总收益率: {summary.total_return_pct:.2f}%")
        print(f"最大回撤: {summary.max_drawdown_pct:.2f}%")
        print(f"胜率: {summary.win_rate_pct:.2f}%")
        print(f"成交笔数: {summary.trade_count}")
