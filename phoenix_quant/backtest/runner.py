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

    # 性能优化: 使用 to_numpy() 代替 iterrows()，速度提升 50-100倍
    candles = df[["timestamp", "open", "high", "low", "close", "volume"]].to_numpy()
    for candle in candles:
        candle_list = candle.tolist()
        engine.update_market(config.symbol, candle_list)
        strategy.on_bar(candle_list)

    analyzer = BacktestAnalyzer(
        engine.get_equity_dataframe(),
        engine.get_trades_dataframe(),
        config.engine.initial_balance,
    )
    return analyzer
