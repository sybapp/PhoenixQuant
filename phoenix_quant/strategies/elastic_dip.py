"""弹性抄底策略重构版"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from math import sqrt
from typing import Deque, Dict, List, Optional

from phoenix_quant.backtest.engine import BacktestEngine
from phoenix_quant.config import StrategyConfig

LOGGER = logging.getLogger(__name__)


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
        self.active_direction: Optional[str] = None  # 当前持仓/挂单方向: "long" 或 "short"

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

    def _calculate_signal(self) -> tuple[str, float]:
        """根据配置的方向计算信号

        Returns:
            (direction, signal_strength): 方向("long"或"short")和信号强度(0-100)
        """
        if self.config.direction == "short":
            return ("short", self._calculate_short_signal())
        elif self.config.direction == "both":
            # 双向模式：同时计算做多和做空信号，选择更强的
            long_signal = self._calculate_long_signal()
            short_signal = self._calculate_short_signal()

            # 如果两个信号都不够强，返回空信号
            if long_signal < self.config.min_signal and short_signal < self.config.min_signal:
                return ("long", 0.0)  # 默认返回做多方向，但信号为0

            # 返回更强的信号
            if long_signal >= short_signal:
                LOGGER.info("【双向模式】做多信号(%.1f) >= 做空信号(%.1f)，选择做多", long_signal, short_signal)
                return ("long", long_signal)
            else:
                LOGGER.info("【双向模式】做空信号(%.1f) > 做多信号(%.1f)，选择做空", short_signal, long_signal)
                return ("short", short_signal)
        else:
            # 默认做多
            return ("long", self._calculate_long_signal())

    def _calculate_long_signal(self) -> float:
        """计算做多信号（原逻辑）"""
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
        signal_components = {}

        # 单K线跌幅
        if single_z > self.config.drop_single_pct:
            component = 18 + min((single_z - self.config.drop_single_pct) * 10, 22)
            signal += component
            signal_components["单K跌幅"] = component

        # 窗口跌幅
        if window_z > self.config.drop_window_pct:
            component = 22 + min((window_z - self.config.drop_window_pct) * 8, 25)
            signal += component
            signal_components["窗口跌幅"] = component

        # 趋势偏离
        trend_bias = (self.ema_slow_value - latest_close) / max(self.ema_slow_value, 1e-6) * 100
        if trend_bias > 0:
            component = min(trend_bias * 6, 15)
            signal += component
            signal_components["趋势偏离"] = component

        # EMA死叉
        if self.ema_fast_value < self.ema_slow_value:
            signal += 8
            signal_components["EMA死叉"] = 8

        # RSI超卖
        if rsi_latest < self.config.rsi_oversold:
            component = min((self.config.rsi_oversold - rsi_latest) * 0.8, 18)
            signal += component
            signal_components["RSI超卖"] = component
        elif rsi_latest < self.config.rsi_oversold + 5:
            component = (self.config.rsi_oversold + 5 - rsi_latest) * 0.3
            signal += component
            signal_components["RSI接近超卖"] = component

        # 成交量恢复
        if self._volume_recovered():
            signal += 10
            signal_components["成交量恢复"] = 10

        # 日内回撤
        intraday_drawdown = (latest_close - latest_low) / max(latest_close, 1e-6) * 100
        if intraday_drawdown >= self.config.delayed_trigger_pct:
            component = min(intraday_drawdown * 1.5, 12)
            signal += component
            signal_components["日内回撤"] = component

        final_signal = min(signal, 100.0)

        # 输出详细信号信息
        if final_signal >= self.config.min_signal * 0.7:
            LOGGER.info(
                "【做多信号】总分: %.1f (阈值: %.1f) | 价格: %.2f | RSI: %.1f | EMA快: %.2f | EMA慢: %.2f",
                final_signal, self.config.min_signal, latest_close, rsi_latest,
                self.ema_fast_value, self.ema_slow_value
            )
            if signal_components:
                components_str = " | ".join([f"{k}={v:.1f}" for k, v in signal_components.items()])
                LOGGER.info("【信号分量】%s", components_str)

        return final_signal

    def _calculate_short_signal(self) -> float:
        """计算做空信号（做多的反向逻辑）"""
        if len(self.candles) < self.min_signal_bars:
            return 0.0
        if self.prev_close_value is None or self.last_close is None or self.last_low is None:
            return 0.0
        if self.ema_fast_value is None or self.ema_slow_value is None or self.rsi_value is None:
            return 0.0

        latest_close = self.last_close
        latest_high = self.candles[-1].high if self.candles else latest_close
        prev_close = self.prev_close_value
        rsi_latest = self.rsi_value

        recent_vol = self._volatility(self.config.volatility_window)
        long_vol = self._volatility(self.config.volatility_window * 4)
        volatility_ref = max(recent_vol, long_vol, 1e-6)

        # 追踪窗口最低价（做空时用）
        window_low = min([c.low for c in list(self.candles)[-self.config.drop_window:]] or [latest_close])

        single_rise = (latest_close - prev_close) / prev_close * 100
        window_rise = (latest_close - window_low) / window_low * 100 if window_low else 0.0

        single_z = single_rise / volatility_ref if volatility_ref else 0.0
        window_z = window_rise / max(recent_vol, 1e-6)

        signal = 0.0
        signal_components = {}

        # 单K线涨幅
        if single_z > self.config.drop_single_pct:
            component = 18 + min((single_z - self.config.drop_single_pct) * 10, 22)
            signal += component
            signal_components["单K涨幅"] = component

        # 窗口涨幅
        if window_z > self.config.drop_window_pct:
            component = 22 + min((window_z - self.config.drop_window_pct) * 8, 25)
            signal += component
            signal_components["窗口涨幅"] = component

        # 趋势偏离（价格高于EMA）
        trend_bias = (latest_close - self.ema_slow_value) / max(self.ema_slow_value, 1e-6) * 100
        if trend_bias > 0:
            component = min(trend_bias * 6, 15)
            signal += component
            signal_components["趋势偏离"] = component

        # EMA金叉（快线>慢线，看空信号）
        if self.ema_fast_value > self.ema_slow_value:
            signal += 8
            signal_components["EMA金叉"] = 8

        # RSI超买
        if rsi_latest > self.config.rsi_overbought:
            component = min((rsi_latest - self.config.rsi_overbought) * 0.8, 18)
            signal += component
            signal_components["RSI超买"] = component
        elif rsi_latest > self.config.rsi_overbought - 5:
            component = (rsi_latest - (self.config.rsi_overbought - 5)) * 0.3
            signal += component
            signal_components["RSI接近超买"] = component

        # 成交量恢复
        if self._volume_recovered():
            signal += 10
            signal_components["成交量恢复"] = 10

        # 日内拉升
        intraday_pump = (latest_high - latest_close) / max(latest_close, 1e-6) * 100
        if intraday_pump >= self.config.delayed_trigger_pct:
            component = min(intraday_pump * 1.5, 12)
            signal += component
            signal_components["日内拉升"] = component

        final_signal = min(signal, 100.0)

        # 输出详细信号信息
        if final_signal >= self.config.min_signal * 0.7:
            LOGGER.info(
                "【做空信号】总分: %.1f (阈值: %.1f) | 价格: %.2f | RSI: %.1f | EMA快: %.2f | EMA慢: %.2f",
                final_signal, self.config.min_signal, latest_close, rsi_latest,
                self.ema_fast_value, self.ema_slow_value
            )
            if signal_components:
                components_str = " | ".join([f"{k}={v:.1f}" for k, v in signal_components.items()])
                LOGGER.info("【信号分量】%s", components_str)

        return final_signal

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
            signal_direction, signal_strength = self._calculate_signal()
            if signal_strength >= self.config.min_signal:
                LOGGER.info("【触发信号】%s | 信号强度 %.1f >= 阈值 %.1f，准备部署订单层",
                           "做多" if signal_direction == "long" else "做空",
                           signal_strength, self.config.min_signal)
                self._deploy_layers(current, ts, signal_direction)
                return

        # 对于ARMING/MANAGE状态，只需要管理仓位和订单
        self._refresh_fills()
        self._manage_position(current, ts)
        self._cleanup_orders(ts)

    def _deploy_layers(self, latest: Candle, ts: datetime, direction: str) -> None:
        """部署订单层

        Args:
            latest: 最新K线
            ts: 时间戳
            direction: 交易方向 "long" 或 "short"
        """
        if not self.config.layers:
            return

        equity = self.engine.get_total_equity(self.symbol)
        capital = equity * self.config.risk.max_account_ratio * self.config.scale_multiplier
        base_price = latest.close

        # 确定交易方向
        is_long = direction == "long"
        side = "buy" if is_long else "sell"
        direction_label = "做多" if is_long else "做空"

        LOGGER.info("【部署订单层】%s | 触发价格: %.2f | 总资金: %.2f | 分配资金: %.2f (%.1f%%)",
                   direction_label, base_price, equity, capital,
                   self.config.risk.max_account_ratio * self.config.scale_multiplier * 100)

        for idx, layer in enumerate(self.config.layers, start=1):
            # 做多：价格在下方；做空：价格在上方
            if is_long:
                target_price = base_price * (1 - layer.offset_pct / 100)
            else:
                target_price = base_price * (1 + layer.offset_pct / 100)

            layer_capital = capital * layer.size_ratio
            quantity = max(layer_capital / target_price, 0.0)
            if quantity <= 0:
                continue
            order = self.engine.submit_limit_order(
                self.symbol,
                side,
                quantity,
                target_price,
                tag=f"layer-{idx}",
            )
            self.layers[order.order_id] = LayerState(order.order_id, target_price, quantity)
            offset_sign = "-" if is_long else "+"
            LOGGER.debug("  层级-%d | 价格: %.2f (%s%.1f%%) | 数量: %.4f | 资金: %.2f",
                        idx, target_price, offset_sign, layer.offset_pct, quantity, layer_capital)

        self.state = "ARMING"
        self.trigger_price = base_price
        self.entry_time = ts
        self.trailing_stop = None
        self.active_direction = direction  # 记录当前方向
        LOGGER.info("【状态切换】IDLE -> ARMING | 已部署 %d 个%s订单层", len(self.layers), direction_label)

    def _refresh_fills(self) -> None:
        newly_filled = []
        for order_id, layer_state in self.layers.items():
            if layer_state.filled:
                continue
            order = self.engine.orders.get(order_id)
            if not order or order.status != "filled":
                continue
            layer_state.filled = True
            fill_ts = datetime.fromtimestamp((self.engine.current_timestamp or 0) / 1000)
            self.entry_time = fill_ts
            newly_filled.append((order.tag, order.price, order.quantity))
            if self.state != "MANAGE":
                self.state = "MANAGE"
                LOGGER.info("【状态切换】ARMING -> MANAGE | 订单开始成交")

        for tag, price, qty in newly_filled:
            LOGGER.info("【订单成交】%s | 价格: %.2f | 数量: %.4f", tag, price, qty)

    def _manage_position(self, latest: Candle, ts: datetime) -> None:
        position = self.engine.get_position(self.symbol)
        # 检查是否有持仓（做多quantity>0，做空quantity<0）
        if position.quantity == 0:
            if self.state != "IDLE":
                self._reset_state()
            return

        # 判断持仓方向
        is_long = position.quantity > 0

        # 计算盈亏（做多和做空逻辑相反）
        if is_long:
            profit_pct = (latest.close - position.avg_price) / position.avg_price * 100
            drawdown_pct = (position.avg_price - latest.low) / position.avg_price * 100
        else:
            # 做空：价格下跌赚钱
            profit_pct = (position.avg_price - latest.close) / position.avg_price * 100
            drawdown_pct = (latest.high - position.avg_price) / position.avg_price * 100

        # 定期输出持仓状态（每10根K线输出一次）
        if self.bar_index % 10 == 0:
            hold_time = (ts - self.entry_time).total_seconds() / 60 if self.entry_time else 0
            direction_label = "做多" if is_long else "做空"
            LOGGER.info(
                "【仓位监控】%s | 持仓: %.4f | 均价: %.2f | 当前: %.2f | 盈亏: %+.2f%% | 最大回撤: %.2f%% | 持仓时间: %.0f分钟",
                direction_label, position.quantity, position.avg_price, latest.close, profit_pct, drawdown_pct, hold_time
            )
            if self.trailing_stop:
                LOGGER.info("  移动止损已激活 | 止损价: %.2f", self.trailing_stop)

        # 硬止损
        if drawdown_pct >= self.config.risk.hard_stop_pct:
            LOGGER.warning("【触发止损】回撤 %.2f%% >= 硬止损 %.2f%%，平仓",
                          drawdown_pct, self.config.risk.hard_stop_pct)
            self.engine.close_position(self.symbol, tag="stop")
            self._start_cooldown(ts)
            return

        # 止盈
        if profit_pct >= self.config.risk.take_profit_pct:
            LOGGER.info("【触发止盈】盈利 %.2f%% >= 止盈线 %.2f%%，平仓",
                       profit_pct, self.config.risk.take_profit_pct)
            self.engine.close_position(self.symbol, tag="take-profit")
            self._start_cooldown(ts)
            return

        # 移动止损激活
        if profit_pct >= self.config.risk.trailing_activation_pct:
            # 做多：止损价上移；做空：止损价下移
            if is_long:
                target = latest.close * (1 - self.config.risk.trailing_pct / 100)
                should_update = not self.trailing_stop or target > self.trailing_stop
                check_hit = latest.close <= self.trailing_stop if self.trailing_stop else False
            else:
                target = latest.close * (1 + self.config.risk.trailing_pct / 100)
                should_update = not self.trailing_stop or target < self.trailing_stop
                check_hit = latest.close >= self.trailing_stop if self.trailing_stop else False

            if should_update:
                if not self.trailing_stop:
                    self.trailing_stop = target
                    LOGGER.info("【移动止损激活】当前盈利 %.2f%% | 止损价: %.2f", profit_pct, target)
                else:
                    old_stop = self.trailing_stop
                    self.trailing_stop = target
                    LOGGER.debug("【移动止损更新】%.2f -> %.2f", old_stop, target)

            if check_hit:
                LOGGER.info("【触发移动止损】当前价 %.2f %s 止损价 %.2f，平仓",
                           latest.close, "<=" if is_long else ">=", self.trailing_stop)
                self.engine.close_position(self.symbol, tag="trailing-stop")
                self._start_cooldown(ts)
                return

        # 最长持仓时间限制
        if self.entry_time and ts - self.entry_time >= timedelta(minutes=self.config.risk.max_hold_minutes):
            hold_minutes = (ts - self.entry_time).total_seconds() / 60
            LOGGER.info("【触发超时平仓】持仓时间 %.0f 分钟 >= 最大持仓 %d 分钟",
                       hold_minutes, self.config.risk.max_hold_minutes)
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
            if self.engine.get_position(self.symbol).quantity == 0:
                self._reset_state()

    def _start_cooldown(self, ts: datetime) -> None:
        self.cooldown_until = ts + timedelta(minutes=self.config.risk.cooldown_minutes)
        LOGGER.info("【进入冷却】将冷却 %d 分钟，冷却结束时间: %s",
                   self.config.risk.cooldown_minutes,
                   self.cooldown_until.strftime("%Y-%m-%d %H:%M:%S"))
        self._reset_state()

    def _reset_state(self) -> None:
        if self.state != "IDLE":
            LOGGER.info("【状态重置】%s -> IDLE", self.state)
        self.state = "IDLE"
        self.layers.clear()
        self.entry_time = None
        self.trailing_stop = None
        self.trigger_price = None
        self.active_direction = None  # 清空方向
