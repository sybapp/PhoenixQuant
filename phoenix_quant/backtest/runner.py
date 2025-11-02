"""回测执行入口"""

from __future__ import annotations

from pathlib import Path

from phoenix_quant.backtest.analyzer import BacktestAnalyzer
from phoenix_quant.backtest.data import HistoricalDataLoader
from phoenix_quant.backtest.engine import BacktestEngine
from phoenix_quant.config import BacktestConfig, load_config
from phoenix_quant.strategies.elastic_dip import ElasticDipStrategy


def load_backtest_config(path: str | Path) -> BacktestConfig:
    """加载配置文件"""

    return load_config(path)


def run_backtest(config: BacktestConfig) -> BacktestAnalyzer:
    """根据配置执行回测"""

    data_loader = HistoricalDataLoader(config)
    df = data_loader.load()

    engine = BacktestEngine(config.engine)
    strategy = ElasticDipStrategy(engine, config.symbol, config.strategy)

    for _, row in df.iterrows():
        candle = [
            row["timestamp"],
            row["open"],
            row["high"],
            row["low"],
            row["close"],
            row["volume"],
        ]
        engine.update_market(config.symbol, candle)
        strategy.on_bar(candle)

    analyzer = BacktestAnalyzer(
        engine.get_equity_dataframe(),
        engine.get_trades_dataframe(),
        config.engine.initial_balance,
    )
    return analyzer
