"""历史数据加载模块"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import ccxt
import pandas as pd

from phoenix_quant.config import BacktestConfig


COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]


@dataclass
class HistoricalDataLoader:
    """根据配置加载历史K线数据"""

    config: BacktestConfig

    def load(self) -> pd.DataFrame:
        data_cfg = self.config.data
        cache = data_cfg.cache
        if cache and cache.exists():
            return self._load_from_cache(cache)
        df = self._fetch_from_exchange()
        if cache:
            cache.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(cache, index=False)
        return df

    def _load_from_cache(self, path: Path) -> pd.DataFrame:
        df = pd.read_csv(path)
        if "datetime" not in df.columns:
            df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")
        return df

    def _fetch_from_exchange(self) -> pd.DataFrame:
        exchange_id = self.config.data.source.lower()
        if not hasattr(ccxt, exchange_id):
            raise ValueError(f"不支持的交易所: {exchange_id}")
        exchange_class = getattr(ccxt, exchange_id)
        exchange = exchange_class({"enableRateLimit": True})
        if exchange_id == "binance":
            exchange.options = {"defaultType": "future"}
            exchange.set_sandbox_mode(self.config.data.use_testnet)

        timeframe = self.config.timeframe
        since = int(self.config.window.start.timestamp() * 1000) if self.config.window.start else None
        end_ts = int(self.config.window.end.timestamp() * 1000) if self.config.window.end else None
        limit = self.config.data.limit

        all_candles = []
        symbol = self.config.symbol
        while True:
            candles = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=limit)
            if not candles:
                break
            all_candles.extend(candles)
            last_ts = candles[-1][0]
            if end_ts and last_ts >= end_ts:
                break
            since = last_ts + 1
            if len(candles) < limit:
                break

        if not all_candles:
            raise RuntimeError("未获取到任何历史数据")

        df = pd.DataFrame(all_candles, columns=COLUMNS)
        df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")
        if end_ts:
            df = df[df["timestamp"] <= end_ts]
        return df
