"""弹性抄底策略重构版"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set

import pandas as pd

from phoenix_quant.backtest.engine import BacktestEngine
from phoenix_quant.config import StrategyConfig


@dataclass
class LayerState:
    """跟踪每一层挂单的状态"""

    order_id: str
    price: float
    quantity: float
    filled: bool = False


class ElasticDipStrategy:
    """更易扩展的弹性抄底策略"""

    def __init__(self, engine: BacktestEngine, symbol: str, config: StrategyConfig):
        self.engine = engine
        self.symbol = symbol
        self.config = config

        self.history: List[List[float]] = []
        self.layers: Dict[str, LayerState] = {}
        self.processed_fills: Set[str] = set()

        self.state = "IDLE"  # IDLE -> ARMING -> MANAGE
        self.cooldown_until: Optional[datetime] = None
        self.entry_time: Optional[datetime] = None
        self.trailing_stop: Optional[float] = None
        self.trigger_price: Optional[float] = None

    # ------------------------------------------------------------------
    # 指标计算
    # ------------------------------------------------------------------
    def _history_df(self) -> pd.DataFrame:
        return pd.DataFrame(self.history, columns=["timestamp", "open", "high", "low", "close", "volume"])

    @staticmethod
    def _ema(series: pd.Series, span: int) -> pd.Series:
        return series.ewm(span=span, adjust=False).mean()

    @staticmethod
    def _rsi(series: pd.Series, period: int) -> pd.Series:
        delta = series.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
        rs = avg_gain / (avg_loss + 1e-12)
        return 100 - 100 / (1 + rs)

    def _volume_recovered(self, volumes: pd.Series) -> bool:
        if len(volumes) < max(self.config.volume_short, self.config.volume_long):
            return False
        short = volumes.rolling(self.config.volume_short).mean().iloc[-1]
        long = volumes.rolling(self.config.volume_long).mean().iloc[-1]
        last = volumes.iloc[-1]
        return short > long * self.config.volume_recover_ratio or last > long * self.config.volume_tick_ratio

    def _calculate_signal(self, df: pd.DataFrame) -> float:
        if len(df) < max(self.config.ema_slow, self.config.rsi_period) + 10:
            return 0.0

        closes = df["close"]
        highs = df["high"]
        lows = df["low"]
        volumes = df["volume"]

        ema_fast = self._ema(closes, self.config.ema_fast)
        ema_slow = self._ema(closes, self.config.ema_slow)
        rsi_series = self._rsi(closes, self.config.rsi_period)

        latest_close = closes.iloc[-1]
        latest_low = lows.iloc[-1]
        prev_close = closes.iloc[-2]
        rsi_latest = rsi_series.iloc[-1]

        pct_change = closes.pct_change() * 100
        recent_vol = pct_change.rolling(self.config.volatility_window).std().iloc[-1]
        long_vol = pct_change.rolling(self.config.volatility_window * 4).std().iloc[-1]
        recent_vol = float(recent_vol) if pd.notna(recent_vol) else 0.0
        long_vol = float(long_vol) if pd.notna(long_vol) else 0.0
        volatility_ref = max(recent_vol, long_vol, 1e-6)

        window_high = highs.rolling(self.config.drop_window).max().iloc[-1]
        if pd.isna(window_high) or window_high <= 0:
            window_high = latest_close

        single_drop = (prev_close - latest_close) / prev_close * 100
        window_drop = (window_high - latest_close) / window_high * 100 if window_high else 0.0

        single_z = single_drop / volatility_ref if volatility_ref else 0.0
        window_z = window_drop / max(recent_vol, 1e-6)

        signal = 0.0

        if single_z > self.config.drop_single_pct:
            signal += 18
            signal += min((single_z - self.config.drop_single_pct) * 10, 22)
        if window_z > self.config.drop_window_pct:
            signal += 22
            signal += min((window_z - self.config.drop_window_pct) * 8, 25)

        trend_bias = (ema_slow.iloc[-1] - latest_close) / max(ema_slow.iloc[-1], 1e-6) * 100
        if trend_bias > 0:
            signal += min(trend_bias * 6, 15)
        if ema_fast.iloc[-1] < ema_slow.iloc[-1]:
            signal += 8

        if rsi_latest < self.config.rsi_oversold:
            signal += min((self.config.rsi_oversold - rsi_latest) * 0.8, 18)
        elif rsi_latest < self.config.rsi_oversold + 5:
            signal += (self.config.rsi_oversold + 5 - rsi_latest) * 0.3

        if self._volume_recovered(volumes):
            signal += 10

        intraday_drawdown = (latest_close - latest_low) / max(latest_close, 1e-6) * 100
        if intraday_drawdown >= self.config.delayed_trigger_pct:
            signal += min(intraday_drawdown * 1.5, 12)

        return min(signal, 100.0)

    # ------------------------------------------------------------------
    # 核心执行逻辑
    # ------------------------------------------------------------------
    def on_bar(self, candle: List[float]) -> None:
        self.history.append(candle)
        if len(self.history) > 5000:
            self.history = self.history[-5000:]

        df = self._history_df()
        ts = datetime.fromtimestamp(candle[0] / 1000)

        if self.cooldown_until and ts < self.cooldown_until:
            return

        signal_strength = self._calculate_signal(df)

        if self.state == "IDLE" and signal_strength >= self.config.min_signal:
            self._deploy_layers(df.iloc[-1], ts)
            return

        self._refresh_fills()
        self._manage_position(df.iloc[-1], ts)
        self._cleanup_orders(ts)

    def _deploy_layers(self, latest: pd.Series, ts: datetime) -> None:
        if not self.config.layers:
            return

        equity = self.engine.get_total_equity(self.symbol)
        capital = equity * self.config.risk.max_account_ratio * self.config.scale_multiplier
        base_price = latest["close"]

        for idx, layer in enumerate(self.config.layers, start=1):
            target_price = base_price * (1 - layer.offset_pct / 100)
            layer_capital = capital * layer.size_ratio
            quantity = max(layer_capital / target_price, 0.0)
            if quantity <= 0:
                continue
            order = self.engine.submit_limit_order(
                self.symbol,
                "buy",
                quantity,
                target_price,
                tag=f"layer-{idx}",
            )
            self.layers[order.order_id] = LayerState(order.order_id, target_price, quantity)

        self.state = "ARMING"
        self.trigger_price = base_price
        self.entry_time = ts
        self.trailing_stop = None

    def _refresh_fills(self) -> None:
        for order in self.engine.get_orders(tag_prefix="layer-"):
            if order.status == "filled" and order.order_id not in self.processed_fills:
                layer_state = self.layers.get(order.order_id)
                if not layer_state:
                    continue
                layer_state.filled = True
                self.processed_fills.add(order.order_id)
                fill_ts = datetime.fromtimestamp((self.engine.current_timestamp or 0) / 1000)
                self.entry_time = fill_ts
                self.state = "MANAGE"

    def _manage_position(self, latest: pd.Series, ts: datetime) -> None:
        position = self.engine.get_position(self.symbol)
        if position.quantity <= 0:
            if self.state != "IDLE":
                self._reset_state()
            return

        profit_pct = (latest["close"] - position.avg_price) / position.avg_price * 100
        drawdown_pct = (position.avg_price - latest["low"]) / position.avg_price * 100

        # 硬止损
        if drawdown_pct >= self.config.risk.hard_stop_pct:
            self.engine.close_position(self.symbol, tag="stop")
            self._start_cooldown(ts)
            return

        # 止盈
        if profit_pct >= self.config.risk.take_profit_pct:
            self.engine.close_position(self.symbol, tag="take-profit")
            self._start_cooldown(ts)
            return

        # 移动止损激活
        if profit_pct >= self.config.risk.trailing_activation_pct:
            target = latest["close"] * (1 - self.config.risk.trailing_pct / 100)
            if not self.trailing_stop or target > self.trailing_stop:
                self.trailing_stop = target

        if self.trailing_stop and latest["close"] <= self.trailing_stop:
            self.engine.close_position(self.symbol, tag="trailing-stop")
            self._start_cooldown(ts)
            return

        # 最长持仓时间限制
        if self.entry_time and ts - self.entry_time >= timedelta(minutes=self.config.risk.max_hold_minutes):
            self.engine.close_position(self.symbol, tag="timeout")
            self._start_cooldown(ts)

    def _cleanup_orders(self, ts: datetime) -> None:
        if self.state == "MANAGE":
            self.engine.cancel_orders(tag_prefix="layer-")
            return

        if not self.entry_time:
            return

        window = timedelta(minutes=self.config.delayed_window_minutes)
        if ts - self.entry_time > window:
            self.engine.cancel_orders(tag_prefix="layer-")
            if self.engine.get_position(self.symbol).quantity <= 0:
                self._reset_state()

    def _start_cooldown(self, ts: datetime) -> None:
        self.cooldown_until = ts + timedelta(minutes=self.config.risk.cooldown_minutes)
        self._reset_state()

    def _reset_state(self) -> None:
        self.state = "IDLE"
        self.layers.clear()
        self.processed_fills.clear()
        self.entry_time = None
        self.trailing_stop = None
        self.trigger_price = None
