"""弹性抄底策略重构版"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from math import sqrt
from typing import Deque, Dict, List, Optional

from phoenix_quant.backtest.engine import BacktestEngine
from phoenix_quant.config import StrategyConfig


@dataclass
class LayerState:
    """跟踪每一层挂单的状态"""

    order_id: str
    price: float
    quantity: float
    filled: bool = False


@dataclass
class Candle:
    """基础K线结构"""

    timestamp: int
    open: float
    high: float
    low: float
    close: float
    volume: float


class RollingMean:
    """维护固定窗口的简单移动平均"""

    def __init__(self, size: int):
        self.size = max(1, size)
        self.values: Deque[float] = deque()
        self.total = 0.0

    def append(self, value: float) -> None:
        self.values.append(value)
        self.total += value
        if len(self.values) > self.size:
            self.total -= self.values.popleft()

    def is_ready(self) -> bool:
        return len(self.values) >= self.size

    def mean(self) -> float:
        if not self.values:
            return 0.0
        return self.total / len(self.values)


class RollingMax:
    """维护固定窗口的最大值"""

    def __init__(self, size: int):
        self.size = max(1, size)
        self.data: Deque[tuple[float, int]] = deque()

    def append(self, value: float, index: int) -> None:
        while self.data and self.data[-1][0] <= value:
            self.data.pop()
        self.data.append((value, index))

        boundary = index - self.size + 1
        while self.data and self.data[0][1] < boundary:
            self.data.popleft()

    def current(self) -> Optional[float]:
        return self.data[0][0] if self.data else None


class ElasticDipStrategy:
    """更易扩展的弹性抄底策略"""

    def __init__(self, engine: BacktestEngine, symbol: str, config: StrategyConfig):
        self.engine = engine
        self.symbol = symbol
        self.config = config

        self.max_buffer = 5000
        self.candles: Deque[Candle] = deque(maxlen=self.max_buffer)
        self.layers: Dict[str, LayerState] = {}

        self.state = "IDLE"  # IDLE -> ARMING -> MANAGE
        self.cooldown_until: Optional[datetime] = None
        self.entry_time: Optional[datetime] = None
        self.trailing_stop: Optional[float] = None
        self.trigger_price: Optional[float] = None

        self.prev_close_value: Optional[float] = None
        self.last_close: Optional[float] = None
        self.last_low: Optional[float] = None
        self.last_volume: float = 0.0

        self.ema_fast_value: Optional[float] = None
        self.ema_slow_value: Optional[float] = None

        self.rsi_period = config.rsi_period
        self.rsi_avg_gain: Optional[float] = None
        self.rsi_avg_loss: Optional[float] = None
        self.rsi_gains: Deque[float] = deque(maxlen=self.rsi_period)
        self.rsi_losses: Deque[float] = deque(maxlen=self.rsi_period)
        self.rsi_value: Optional[float] = None

        self.volatility_window = config.volatility_window
        self.returns: Deque[float] = deque(maxlen=max(1, self.volatility_window * 4))

        self.volume_short_ma = RollingMean(config.volume_short)
        self.volume_long_ma = RollingMean(config.volume_long)
        self.window_high_tracker = RollingMax(config.drop_window)

        self.bar_index = 0
        self.min_signal_bars = max(config.ema_slow, config.rsi_period) + 10

    # ------------------------------------------------------------------
    # 指标计算
    # ------------------------------------------------------------------

    def _volume_recovered(self) -> bool:
        if not self.volume_long_ma.is_ready():
            return False
        short_avg = self.volume_short_ma.mean()
        long_avg = self.volume_long_ma.mean()
        last = self.last_volume
        return short_avg > long_avg * self.config.volume_recover_ratio or last > long_avg * self.config.volume_tick_ratio

    @staticmethod
    def _std(values: List[float]) -> float:
        length = len(values)
        if length <= 1:
            return 0.0
        mu = sum(values) / length
        variance = sum((v - mu) ** 2 for v in values) / (length - 1)
        return sqrt(variance)

    def _volatility(self, window: int) -> float:
        if window <= 1 or len(self.returns) < window:
            return 0.0
        window_values = list(self.returns)[-window:]
        return self._std(window_values)

    def _update_state(self, candle: Candle) -> None:
        self.bar_index += 1
        prev_close = self.last_close

        if prev_close is not None:
            self.prev_close_value = prev_close

        self.candles.append(candle)
        self.last_close = candle.close
        self.last_low = candle.low
        self.last_volume = candle.volume

        self.volume_short_ma.append(candle.volume)
        self.volume_long_ma.append(candle.volume)
        self.window_high_tracker.append(candle.high, self.bar_index)

        if prev_close is not None and prev_close > 0:
            ret_pct = (candle.close - prev_close) / prev_close * 100
            self.returns.append(ret_pct)

        self._update_ema(candle.close)
        self._update_rsi(prev_close, candle.close)

    def _update_ema(self, close: float) -> None:
        alpha_fast = 2 / (self.config.ema_fast + 1)
        alpha_slow = 2 / (self.config.ema_slow + 1)

        if self.ema_fast_value is None:
            self.ema_fast_value = close
        else:
            self.ema_fast_value += alpha_fast * (close - self.ema_fast_value)

        if self.ema_slow_value is None:
            self.ema_slow_value = close
        else:
            self.ema_slow_value += alpha_slow * (close - self.ema_slow_value)

    def _update_rsi(self, prev_close: Optional[float], close: float) -> None:
        if prev_close is None:
            return

        delta = close - prev_close
        gain = max(delta, 0.0)
        loss = max(-delta, 0.0)

        self.rsi_gains.append(gain)
        self.rsi_losses.append(loss)

        if self.rsi_avg_gain is None or self.rsi_avg_loss is None:
            if len(self.rsi_gains) == self.rsi_period:
                self.rsi_avg_gain = sum(self.rsi_gains) / self.rsi_period
                self.rsi_avg_loss = sum(self.rsi_losses) / self.rsi_period
            else:
                return
        else:
            self.rsi_avg_gain = (self.rsi_avg_gain * (self.rsi_period - 1) + gain) / self.rsi_period
            self.rsi_avg_loss = (self.rsi_avg_loss * (self.rsi_period - 1) + loss) / self.rsi_period

        rs = self.rsi_avg_gain / (self.rsi_avg_loss + 1e-12)
        self.rsi_value = 100 - 100 / (1 + rs)

    def _calculate_signal(self) -> float:
        if len(self.candles) < self.min_signal_bars:
            return 0.0
        if self.prev_close_value is None or self.last_close is None or self.last_low is None:
            return 0.0
        if self.ema_fast_value is None or self.ema_slow_value is None or self.rsi_value is None:
            return 0.0

        latest_close = self.last_close
        latest_low = self.last_low
        prev_close = self.prev_close_value
        rsi_latest = self.rsi_value

        recent_vol = self._volatility(self.config.volatility_window)
        long_vol = self._volatility(self.config.volatility_window * 4)
        volatility_ref = max(recent_vol, long_vol, 1e-6)

        window_high = self.window_high_tracker.current()
        if window_high is None or window_high <= 0:
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

        trend_bias = (self.ema_slow_value - latest_close) / max(self.ema_slow_value, 1e-6) * 100
        if trend_bias > 0:
            signal += min(trend_bias * 6, 15)
        if self.ema_fast_value < self.ema_slow_value:
            signal += 8

        if rsi_latest < self.config.rsi_oversold:
            signal += min((self.config.rsi_oversold - rsi_latest) * 0.8, 18)
        elif rsi_latest < self.config.rsi_oversold + 5:
            signal += (self.config.rsi_oversold + 5 - rsi_latest) * 0.3

        if self._volume_recovered():
            signal += 10

        intraday_drawdown = (latest_close - latest_low) / max(latest_close, 1e-6) * 100
        if intraday_drawdown >= self.config.delayed_trigger_pct:
            signal += min(intraday_drawdown * 1.5, 12)

        return min(signal, 100.0)

    # ------------------------------------------------------------------
    # 核心执行逻辑
    # ------------------------------------------------------------------
    def on_bar(self, candle: List[float]) -> None:
        current = Candle(
            timestamp=int(candle[0]),
            open=float(candle[1]),
            high=float(candle[2]),
            low=float(candle[3]),
            close=float(candle[4]),
            volume=float(candle[5]),
        )
        self._update_state(current)

        ts = datetime.fromtimestamp(current.timestamp / 1000)

        if self.cooldown_until and ts < self.cooldown_until:
            return

        # 性能优化: 只在IDLE状态下计算信号，避免不必要的计算
        if self.state == "IDLE":
            signal_strength = self._calculate_signal()
            if signal_strength >= self.config.min_signal:
                self._deploy_layers(current, ts)
                return

        # 对于ARMING/MANAGE状态，只需要管理仓位和订单
        self._refresh_fills()
        self._manage_position(current, ts)
        self._cleanup_orders(ts)

    def _deploy_layers(self, latest: Candle, ts: datetime) -> None:
        if not self.config.layers:
            return

        equity = self.engine.get_total_equity(self.symbol)
        capital = equity * self.config.risk.max_account_ratio * self.config.scale_multiplier
        base_price = latest.close

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
        for order_id, layer_state in self.layers.items():
            if layer_state.filled:
                continue
            order = self.engine.orders.get(order_id)
            if not order or order.status != "filled":
                continue
            layer_state.filled = True
            fill_ts = datetime.fromtimestamp((self.engine.current_timestamp or 0) / 1000)
            self.entry_time = fill_ts
            self.state = "MANAGE"

    def _manage_position(self, latest: Candle, ts: datetime) -> None:
        position = self.engine.get_position(self.symbol)
        if position.quantity <= 0:
            if self.state != "IDLE":
                self._reset_state()
            return

        profit_pct = (latest.close - position.avg_price) / position.avg_price * 100
        drawdown_pct = (position.avg_price - latest.low) / position.avg_price * 100

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
            target = latest.close * (1 - self.config.risk.trailing_pct / 100)
            if not self.trailing_stop or target > self.trailing_stop:
                self.trailing_stop = target

        if self.trailing_stop and latest.close <= self.trailing_stop:
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
        self.entry_time = None
        self.trailing_stop = None
        self.trigger_price = None
