# -*- coding: utf-8 -*-
"""
优化版弹性抄底策略回测适配器
增强信号过滤、风险管理和退出机制
"""

import asyncio
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import ccxt

from backtest_engine import BacktestEngine, BacktestOrder


class BacktestElasticDipBot:
    """优化版弹性抄底策略"""

    def __init__(self, backtest_engine: BacktestEngine, symbol: str, params: Dict):
        self.engine = backtest_engine
        self.symbol = symbol
        self.p = params

        # 策略状态
        self.state = 'IDLE'  # IDLE, WAIT_FOR_BOUNCE, WAIT_ORDERS, MANAGE, COOLDOWN
        self.reference_price = None
        self.trigger_time = None
        self.cooldown_until = None

        # 订单和持仓
        self.attack_orders = []
        self.filled_orders = []
        self.break_time = None

        # 持仓信息
        self.position_qty = 0.0
        self.avg_entry = 0.0
        self.lowest_fill = None
        self.highest_price = None

        # 止损止盈订单
        self.tp_orders = []  # 改为多个止盈订单
        self.sl_order_id = None
        self.trailing_stop_price = None

        # 市场数据缓存
        self.price_history = []
        self.volume_history = []
        self.signal_strength = 0.0

        # 绩效跟踪
        self.trade_count = 0
        self.win_count = 0
        self.consecutive_losses = 0
        self.daily_trade_count = 0
        self.last_trade_date = None

        # 模拟的市场数据
        self.simulated_liq_notional = 0.0
        self.simulated_funding_rate = 0.0

    def _round_price(self, price: float) -> float:
        """价格精度处理"""
        return round(price, 2)

    def _round_amount(self, amount: float) -> float:
        """数量精度处理"""
        return round(amount, 4)

    # ========= 增强指标计算 =========
    def ema(self, arr, period):
        """计算EMA"""
        arr = np.asarray(arr, dtype=float)
        if len(arr) < period:
            return np.array([])
        k = 2 / (period + 1)
        e = np.zeros_like(arr)
        e[0] = arr[0]
        for i in range(1, len(arr)):
            e[i] = arr[i] * k + e[i - 1] * (1 - k)
        return e

    def sma(self, arr, period):
        """计算SMA"""
        if len(arr) < period:
            return np.nan
        return np.mean(arr[-period:])

    def rsi(self, arr, period=14):
        """计算RSI"""
        arr = np.asarray(arr, dtype=float)
        if len(arr) < period + 1:
            return np.nan
        deltas = np.diff(arr)
        gains = np.clip(deltas, 0, None)
        losses = -np.clip(deltas, None, 0)
        avg_gain = np.mean(gains[:period])
        avg_loss = np.mean(losses[:period])
        rsis, rs = [], (avg_gain / (avg_loss + 1e-12)) if avg_loss > 0 else np.inf
        rsis.append(100 - 100 / (1 + rs))
        for i in range(period, len(deltas)):
            avg_gain = (avg_gain * (period - 1) + gains[i]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i]) / period
            rs = (avg_gain / (avg_loss + 1e-12)) if avg_loss > 0 else np.inf
            rsis.append(100 - 100 / (1 + rs))
        return rsis[-1] if rsis else np.nan

    def macd(self, prices, fast=12, slow=26, signal=9):
        """计算MACD"""
        if len(prices) < slow:
            return None, None, None
        
        ema_fast = self.ema(prices, fast)
        ema_slow = self.ema(prices, slow)
        
        if len(ema_fast) < slow or len(ema_slow) < slow:
            return None, None, None
            
        macd_line = ema_fast[-1] - ema_slow[-1]
        
        # 计算信号线 (MACD的EMA)
        macd_values = ema_fast - ema_slow
        signal_line = self.ema(macd_values, signal)[-1] if len(macd_values) >= signal else macd_line
        
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram

    def bollinger_bands(self, prices, period=20, std_dev=2):
        """计算布林带"""
        if len(prices) < period:
            return None, None, None
            
        sma = np.mean(prices[-period:])
        std = np.std(prices[-period:])
        
        upper = sma + std_dev * std
        lower = sma - std_dev * std
        
        return upper, sma, lower

    def atr(self, candles, period=14):
        """计算平均真实波幅(ATR)"""
        if len(candles) < period + 1:
            return np.nan
            
        true_ranges = []
        for i in range(1, len(candles)):
            high, low = candles[i][2], candles[i][3]
            prev_close = candles[i-1][4]
            
            tr1 = high - low
            tr2 = abs(high - prev_close)
            tr3 = abs(low - prev_close)
            
            true_ranges.append(max(tr1, tr2, tr3))
            
        return np.mean(true_ranges[-period:])

    def volume_recovered(self, candles, ma_short=5, ma_long=20, ratio=1.15, tick_ratio=None):
        """量能恢复判断"""
        vols = [c[5] for c in candles]
        if len(vols) < max(ma_short, ma_long) + 1:
            return False
        ma_s = float(np.mean(vols[-ma_short:]))
        ma_l = float(np.mean(vols[-ma_long:]))
        cond_a = (ma_s > ma_l * ratio)
        if tick_ratio is None:
            return cond_a
        cond_b = (vols[-1] > ma_l * tick_ratio)
        return cond_a or cond_b

    # ========= 增强信号判断 =========
    def calculate_signal_strength(self, candles) -> float:
        """计算综合信号强度 (0-100)"""
        if len(candles) < max(self.p["ema_slow"], self.p["rsi_period"]) + 10:
            return 0.0

        signals = []
        weights = []

        # 1. 价格信号 (权重: 0.35)
        price_signal = self._price_based_signal(candles)
        signals.append(price_signal)
        weights.append(0.35)

        # 2. 动量信号 (权重: 0.25)
        momentum_signal = self._momentum_signal(candles)
        signals.append(momentum_signal)
        weights.append(0.25)

        # 3. 成交量信号 (权重: 0.20)
        volume_signal = self._volume_signal(candles)
        signals.append(volume_signal)
        weights.append(0.20)

        # 4. 趋势信号 (权重: 0.20)
        trend_signal = self._trend_signal(candles)
        signals.append(trend_signal)
        weights.append(0.20)

        # 计算加权平均
        total_weight = sum(weights)
        weighted_signal = sum(s * w for s, w in zip(signals, weights)) / total_weight

        return min(100.0, max(0.0, weighted_signal))

    def _price_based_signal(self, candles) -> float:
        """基于价格的信号 (0-100)"""
        closes = [c[4] for c in candles]
        current_price = closes[-1]

        # 计算近期跌幅
        lookback = min(20, len(closes))
        high_price = max(closes[-lookback:])
        drawdown = (high_price - current_price) / high_price * 100

        # 跌幅越大信号越强，但超过15%可能趋势已坏
        if drawdown > 15:
            return 0
        return min(100, drawdown * 6)  # 调整系数

    def _momentum_signal(self, candles) -> float:
        """动量信号 (0-100)"""
        closes = [c[4] for c in candles]
        
        # RSI信号
        rsi_val = self.rsi(closes, self.p["rsi_period"])
        rsi_signal = 0.0
        if not np.isnan(rsi_val):
            # RSI超卖程度
            rsi_signal = max(0, (self.p["rsi_oversold"] - rsi_val) / self.p["rsi_oversold"] * 100)

        # MACD信号
        macd_signal = 50.0  # 默认中性
        macd_line, signal_line, histogram = self.macd(closes)
        if macd_line is not None:
            if macd_line > signal_line and histogram > 0:  # 金叉且向上
                macd_signal = 80.0
            elif macd_line < signal_line and histogram < 0:  # 死叉且向下
                macd_signal = 20.0
            elif macd_line > signal_line:  # 金叉但动量减弱
                macd_signal = 60.0
            else:  # 死叉但可能见底
                macd_signal = 40.0

        return (rsi_signal * 0.6 + macd_signal * 0.4)

    def _volume_signal(self, candles) -> float:
        """成交量信号 (0-100)"""
        volumes = [c[5] for c in candles]
        if len(volumes) < 10:
            return 50.0

        current_vol = volumes[-1]
        avg_vol_short = np.mean(volumes[-5:])
        avg_vol_long = np.mean(volumes[-20:])

        # 缩量企稳信号
        if current_vol < avg_vol_long * self.p["vol_shrink_ratio"]:
            return 80.0
        # 放量反弹信号
        elif current_vol > avg_vol_long * self.p.get("vol_recover_ratio", 1.15):
            return 70.0
        # 温和放量
        elif current_vol > avg_vol_short * 1.1:
            return 60.0
        else:
            return 30.0

    def _trend_signal(self, candles) -> float:
        """趋势信号 (0-100)"""
        closes = [c[4] for c in candles]
        
        # EMA趋势
        ema_fast = self.ema(closes, self.p["ema_fast"])
        ema_slow = self.ema(closes, self.p["ema_slow"])
        
        if len(ema_fast) == 0 or len(ema_slow) == 0:
            return 50.0

        current_price = closes[-1]
        ema_fast_val = ema_fast[-1]
        ema_slow_val = ema_slow[-1]

        # 布林带位置
        bb_upper, bb_middle, bb_lower = self.bollinger_bands(closes)
        
        trend_score = 50.0
        
        # 价格在EMA之上为积极信号
        if current_price > ema_slow_val:
            trend_score += 20
        elif current_price > ema_fast_val:
            trend_score += 10
            
        # 价格在布林带下轨附近为超卖信号
        if bb_lower is not None and current_price <= bb_lower:
            trend_score += 15
        elif bb_middle is not None and current_price <= bb_middle:
            trend_score += 5

        return min(100.0, max(0.0, trend_score))

    def is_fast_drop(self, candles):
        """快速下跌判断 - 增强版"""
        w = self.p["window_min"]
        o, h, l, c = candles[-1][1:5]
        
        # 单根K线跌幅
        single_drop = (c < o) and ((o - c) / o * 100 >= self.p["drop_pct_single"])
        
        # 窗口内跌幅
        sub = candles[-w:]
        hi = max(x[2] for x in sub)
        window_drop = (hi - sub[-1][4]) / hi * 100 >= self.p["drop_pct_window"]
        
        # 新增：ATR过滤，避免在波动率过高时入场
        atr_val = self.atr(candles)
        current_range = (h - l) / l * 100
        low_volatility = current_range < atr_val * 2 if atr_val > 0 else True
        
        return (single_drop or window_drop) and low_volatility

    def is_trend_down(self, candles):
        """下降趋势判断 - 增强版"""
        closes = [c[4] for c in candles]
        ef = self.ema(closes, self.p["ema_fast"])
        es = self.ema(closes, self.p["ema_slow"])
        if len(ef) == 0 or len(es) == 0:
            return False
            
        ef_last, ef_prev = ef[-1], ef[-5] if len(ef) >= 5 else ef[0]
        es_last, es_prev = es[-1], es[-5] if len(es) >= 5 else es[0]
        
        # 增强趋势判断：价格在EMA下方且EMA向下
        current_price = closes[-1]
        trend_down = (ef_last < es_last) and (ef_last - ef_prev < 0) and (es_last - es_prev < 0)
        price_below_ema = current_price < es_last
        
        return trend_down and price_below_ema

    def is_volume_dry(self, candles):
        """缩量判断 - 增强版"""
        vols = [c[5] for c in candles]
        if len(vols) < 20:
            return False
            
        current_vol = vols[-1]
        avg_vol_10 = float(np.mean(vols[-10:]))
        avg_vol_20 = float(np.mean(vols[-20:]))
        
        # 双重缩量确认
        return (current_vol < avg_vol_10 * self.p["vol_shrink_ratio"] and 
                current_vol < avg_vol_20 * self.p["vol_shrink_ratio"])

    def is_oversold(self, candles):
        """超卖判断 - 增强版"""
        closes = [c[4] for c in candles]
        rsi_val = self.rsi(closes, self.p["rsi_period"])
        
        if np.isnan(rsi_val):
            return False
            
        # 布林带超卖确认
        bb_upper, bb_middle, bb_lower = self.bollinger_bands(closes)
        current_price = closes[-1]
        
        bb_oversold = False
        if bb_lower is not None:
            bb_oversold = current_price <= bb_lower
            
        # RSI超卖或价格在布林带下轨下方
        return (rsi_val < self.p["rsi_oversold"]) or bb_oversold

    def is_liquidation_spike(self):
        """爆仓事件判断"""
        return self.simulated_liq_notional >= self.p["liq_notional_threshold"]

    def is_funding_extreme(self):
        """极端资金费率判断"""
        return self.simulated_funding_rate <= self.p["funding_extreme_neg"]

    # ========= 风险管理增强 =========
    def can_trade(self, current_timestamp: float) -> bool:
        """检查是否可以交易"""
        current_date = datetime.fromtimestamp(current_timestamp/1000).date()
        
        # 日期切换重置计数
        if self.last_trade_date != current_date:
            self.daily_trade_count = 0
            self.last_trade_date = current_date
            
        # 检查冷却时间
        if self.cooldown_until and current_timestamp < self.cooldown_until:
            return False
            
        # 检查每日交易限额
        max_daily = self.p.get("max_daily_trades", 5)
        if self.daily_trade_count >= max_daily:
            return False
            
        # 检查连续亏损限制
        max_consecutive_losses = self.p.get("max_consecutive_losses", 3)
        if self.consecutive_losses >= max_consecutive_losses:
            return False
            
        return True

    def adjust_position_size(self, base_capital: float) -> float:
        """根据市场条件调整仓位大小"""
        # 波动率调整
        volatility_factor = 1.0
        if len(self.price_history) >= 20:
            returns = np.diff(np.log(self.price_history[-20:]))
            volatility = np.std(returns)
            # 高波动率时减仓
            if volatility > 0.02:  # 2%日波动率
                volatility_factor = 0.7
            elif volatility > 0.015:
                volatility_factor = 0.85
                
        # 连续亏损调整
        loss_factor = 1.0
        if self.consecutive_losses > 0:
            loss_factor = max(0.5, 1.0 - self.consecutive_losses * 0.1)
            
        return base_capital * volatility_factor * loss_factor

    # ========= 增强订单管理 =========
    def compute_attack_plan(self, current_price: float) -> List[Dict]:
        """计算分层买入计划 - 增强版"""
        base_capital = min(self.p["total_capital"],
                          self.engine.balance * self.p["max_account_ratio"])
        
        # 动态调整仓位
        adjusted_capital = self.adjust_position_size(base_capital)
        
        plan = []
        layer_pcts = self.p["layer_pcts"]
        layer_ratios = self.p["layer_pos_ratio"]
        
        # 根据信号强度调整层数
        signal_strength = self.signal_strength
        if signal_strength < 40:  # 弱信号，减少层数
            layer_pcts = layer_pcts[:3]
            layer_ratios = [r * 1.2 for r in layer_ratios[:3]]  # 集中仓位
            layer_ratios = [r / sum(layer_ratios) for r in layer_ratios]  # 重新归一化
        
        for pct, ratio in zip(layer_pcts, layer_ratios):
            price = current_price * (1 - pct / 100.0)
            capital = adjusted_capital * ratio
            qty = capital / price if price > 0 else 0.0
            
            plan.append({
                "price": self._round_price(price),
                "qty": self._round_amount(qty),
                "id": None,
                "filled": False,
                "layer_pct": pct
            })
            
        return plan

    def _recalc_position(self):
        """重新计算持仓信息"""
        if not self.filled_orders:
            self.position_qty = 0.0
            self.avg_entry = 0.0
            self.lowest_fill = None
            self.highest_price = None
            return

        total_qty = sum(o["qty"] for o in self.filled_orders)
        total_cost = sum(o["qty"] * o["price"] for o in self.filled_orders)
        self.position_qty = total_qty
        self.avg_entry = total_cost / total_qty if total_qty > 0 else 0.0
        self.lowest_fill = min(o["price"] for o in self.filled_orders)
        
        # 初始化最高价
        if self.highest_price is None:
            self.highest_price = self.avg_entry

    def check_order_fills(self):
        """检查订单成交情况"""
        position_changed = False

        for plan_order in self.attack_orders:
            if plan_order["filled"]:
                continue

            # 查找对应的引擎订单
            engine_order = None
            for order in self.engine.orders:
                if order.id == plan_order["id"]:
                    engine_order = order
                    break

            if engine_order and engine_order.status == 'filled':
                plan_order["filled"] = True
                plan_order["price"] = engine_order.filled_price
                plan_order["qty"] = engine_order.filled_qty
                self.filled_orders.append(plan_order)
                position_changed = True

                print(f"[回测成交] 层级{plan_order['layer_pct']}%: "
                      f"买入 {plan_order['qty']:.4f} @ ${plan_order['price']:.2f}")

        if position_changed:
            self._recalc_position()
            self._update_tp_sl()

    def _update_tp_sl(self):
        """更新止盈止损订单 - 增强版"""
        if self.position_qty <= 0:
            return

        # 取消旧的止盈订单
        for tp_order in self.tp_orders:
            self.engine.cancel_order(tp_order.id)
        self.tp_orders.clear()

        # 分层止盈
        take_profit_pcts = self.p.get("take_profit_pct", [1.0])
        if isinstance(take_profit_pcts, list):
            # 多个止盈目标
            tp_levels = take_profit_pcts
        else:
            # 单个止盈目标，分为3个级别
            base_tp = take_profit_pcts
            tp_levels = [base_tp * 0.6, base_tp * 1.0, base_tp * 1.5]

        # 计算每个止盈级别的数量和价格
        total_qty = self.position_qty
        level_ratios = [0.3, 0.4, 0.3]  # 各层级卖出比例
        level_ratios = [r / sum(level_ratios) for r in level_ratios]  # 归一化

        for i, (tp_pct, ratio) in enumerate(zip(tp_levels, level_ratios)):
            if i >= len(level_ratios):  # 如果级别数不匹配，跳出
                break
                
            tp_price = self.avg_entry * (1 + tp_pct / 100.0)
            tp_qty = total_qty * ratio
            
            if tp_qty > 0:
                tp_order = self.engine.create_order(
                    self.symbol, 'sell', 'limit', tp_qty, tp_price
                )
                self.tp_orders.append(tp_order)
                print(f"[回测TP{i+1}] 止盈挂单 {tp_qty:.4f} @ ${tp_price:.2f} (+{tp_pct:.1f}%)")

        # 动态止损 - 基于ATR或固定比例
        atr_val = self.atr(self._get_recent_candles(), period=14)
        if atr_val > 0:
            # 使用2倍ATR作为止损距离
            sl_distance = atr_val * 2
            sl_price = self.lowest_fill - sl_distance
        else:
            # 回退到固定比例止损
            sl_price = self.lowest_fill * (1 - self.p["hard_stop_extra"] / 100.0)

        # 更新或创建止损订单
        if self.sl_order_id:
            # 取消旧止损单
            old_sl = None
            for order in self.engine.orders:
                if order.id == self.sl_order_id:
                    old_sl = order
                    break
            if old_sl:
                self.engine.cancel_order(old_sl.id)

        sl_order = self.engine.create_order(
            self.symbol, 'sell', 'STOP_MARKET', self.position_qty,
            stop_price=self._round_price(sl_price)
        )
        self.sl_order_id = sl_order.id
        print(f"[回测SL] 动态止损 @ ${sl_price:.2f}")

        # 初始化移动止损
        if self.trailing_stop_price is None:
            self.trailing_stop_price = sl_price

    def update_trailing_stop(self, current_price: float):
        """更新移动止损"""
        if self.position_qty <= 0 or self.trailing_stop_price is None:
            return

        # 更新最高价
        if current_price > self.highest_price:
            self.highest_price = current_price
            
            # 计算新的移动止损价
            trail_pct = self.p.get("trailing_stop_pct", 0.8)
            new_stop = self.highest_price * (1 - trail_pct / 100.0)
            
            # 移动止损只能向上移动
            if new_stop > self.trailing_stop_price:
                self.trailing_stop_price = new_stop
                
                # 更新止损订单
                if self.sl_order_id:
                    for order in self.engine.orders:
                        if order.id == self.sl_order_id:
                            self.engine.cancel_order(order.id)
                            break
                    
                    new_sl_order = self.engine.create_order(
                        self.symbol, 'sell', 'STOP_MARKET', self.position_qty,
                        stop_price=self._round_price(self.trailing_stop_price)
                    )
                    self.sl_order_id = new_sl_order.id
                    print(f"[回测移动止损] 更新 @ ${self.trailing_stop_price:.2f}")

    # ========= 市场数据模拟 =========
    def simulate_market_conditions(self, candles):
        """模拟市场条件 - 增强版"""
        if len(candles) < 10:
            return

        # 更新价格和成交量历史
        self.price_history.append(candles[-1][4])
        self.volume_history.append(candles[-1][5])
        
        # 保持历史数据长度
        if len(self.price_history) > 100:
            self.price_history.pop(0)
        if len(self.volume_history) > 100:
            self.volume_history.pop(0)

        # 模拟爆仓事件
        vols = [c[5] for c in candles]
        avg_vol = np.mean(vols[-20:]) if len(vols) >= 20 else np.mean(vols)
        current_vol = vols[-1]

        # 基于成交量和价格变动模拟爆仓
        price_change_pct = (candles[-1][4] - candles[-2][4]) / candles[-2][4] * 100 if len(candles) >= 2 else 0
        
        if current_vol > avg_vol * 3 and price_change_pct < -2:
            # 大幅下跌伴随放量，模拟大额爆仓
            self.simulated_liq_notional = 15_000_000
        elif current_vol > avg_vol * 2 and price_change_pct < -1:
            self.simulated_liq_notional = 8_000_000
        else:
            self.simulated_liq_notional = 0

        # 模拟资金费率
        if price_change_pct < -3:
            self.simulated_funding_rate = -0.15  # -15 bps
        elif price_change_pct < -1:
            self.simulated_funding_rate = -0.08  # -8 bps
        else:
            self.simulated_funding_rate = 0.02   # +2 bps

    def _get_recent_candles(self, lookback=50):
        """获取最近的K线数据（简化实现）"""
        # 在实际使用中，这里应该返回最近lookback根K线
        # 这里返回空列表，实际使用时需要根据具体实现调整
        return []

    # ========= 增强策略主逻辑 =========
    def step(self, candles: List, current_timestamp: float):
        """策略步骤 - 增强版"""
        if len(candles) < max(self.p["ema_slow"], self.p["rsi_period"]) + 10:
            return

        current_price = candles[-1][4]

        # 模拟市场条件
        self.simulate_market_conditions(candles)

        # 计算信号强度
        self.signal_strength = self.calculate_signal_strength(candles)

        # 检查风险管理
        if not self.can_trade(current_timestamp):
            if self.state != 'COOLDOWN' and self.state != 'IDLE':
                self.state = 'COOLDOWN'
                print("[风控] 进入冷却状态")
            return

        # 检查订单成交
        if self.attack_orders:
            self.check_order_fills()

        # 更新移动止损
        if self.position_qty > 0:
            self.update_trailing_stop(current_price)

        # 状态机
        if self.state == 'IDLE':
            self._idle_state(candles, current_timestamp, current_price)
            
        elif self.state == 'WAIT_FOR_BOUNCE':
            self._wait_bounce_state(candles, current_timestamp, current_price)
            
        elif self.state == 'WAIT_ORDERS':
            self._wait_orders_state(candles, current_timestamp, current_price)
            
        elif self.state == 'MANAGE':
            self._manage_state(candles, current_timestamp, current_price)
            
        elif self.state == 'COOLDOWN':
            # 检查冷却结束
            if self.cooldown_until and current_timestamp >= self.cooldown_until:
                self.state = 'IDLE'
                self.cooldown_until = None
                print("[风控] 冷却结束，恢复正常交易")

    def _idle_state(self, candles, current_timestamp, current_price):
        """空闲状态处理"""
        # 避免在明显下跌趋势中入场
        if self.is_trend_down(candles) and self.is_volume_dry(candles):
            return

        # 增强入场条件：需要强信号
        strong_signal = self.signal_strength > 50
        basic_conditions = (self.is_fast_drop(candles) and 
                          self.is_oversold(candles) and 
                          self.is_liquidation_spike() and 
                          self.is_funding_extreme())

        if basic_conditions and strong_signal:
            self.reference_price = current_price
            self.trigger_time = current_timestamp
            self.state = 'WAIT_FOR_BOUNCE'
            self.daily_trade_count += 1
            
            print(f"\n[回测触发] 信号强度{self.signal_strength:.1f}% "
                  f"进入等待反弹 @ ${current_price:.2f}")

    def _wait_bounce_state(self, candles, current_timestamp, current_price):
        """等待反弹状态处理"""
        # 检查超时
        if current_timestamp - self.trigger_time > self.p["delayed_window_sec"] * 1000:
            print("[回测超时] 延迟窗口过期，重置")
            self.reset()
            return

        # 增强反弹确认条件
        price_ok = current_price >= self.reference_price * (1 + self.p["delayed_trigger_pct"] / 100.0)
        vol_ok = self.volume_recovered(
            candles,
            ma_short=self.p["vol_recover_ma_short"],
            ma_long=self.p["vol_recover_ma_long"],
            ratio=self.p["vol_recover_ratio"],
            tick_ratio=self.p["tick_vol_ratio"]
        )
        
        # 新增：技术指标确认
        tech_confirm = self._technical_confirmation(candles)

        if price_ok and vol_ok and tech_confirm:
            # 开始分层买入
            plan = self.compute_attack_plan(current_price)
            self.attack_orders = plan

            for order_plan in plan:
                order = self.engine.create_order(
                    self.symbol, 'buy', 'limit',
                    order_plan["qty"], order_plan["price"]
                )
                order_plan["id"] = order.id

            self.state = 'WAIT_ORDERS'
            print(f"[回测下单] 信号强度{self.signal_strength:.1f}% "
                  f"分层买入 {len(plan)} 档")

    def _wait_orders_state(self, candles, current_timestamp, current_price):
        """等待订单状态处理"""
        # 检查止损触发
        if self.lowest_fill:
            # 使用移动止损或固定止损
            stop_price = self.trailing_stop_price if self.trailing_stop_price else \
                        self.lowest_fill * (1 - self.p["hard_stop_extra"] / 100.0)
            
            if current_price <= stop_price:
                if self.break_time is None:
                    self.break_time = current_timestamp
                elif current_timestamp - self.break_time >= self.p["sl_time_grace_sec"] * 1000:
                    print(f"[回测止损] 价格跌破止损线 ${stop_price:.2f}，强制平仓")
                    self._close_position(current_price, "止损")
                    return
            else:
                self.break_time = None

        # 如果有持仓，进入管理状态
        if self.position_qty > 0:
            self.state = 'MANAGE'

    def _manage_state(self, candles, current_timestamp, current_price):
        """持仓管理状态处理"""
        # 检查止盈止损是否成交
        tp_filled = any(order.status == 'filled' for order in self.tp_orders)
        sl_filled = False
        
        for order in self.engine.orders:
            if order.id == self.sl_order_id and order.status == 'filled':
                sl_filled = True
                print(f"[回测SL成交] 止损 @ ${order.filled_price:.2f}")

        # 检查部分止盈
        remaining_tp_orders = [order for order in self.tp_orders if order.status != 'filled']
        if len(remaining_tp_orders) < len(self.tp_orders) and not tp_filled:
            tp_filled = True
            print("[回测部分止盈] 部分仓位已止盈")

        if tp_filled or sl_filled or self.position_qty <= 0:
            profit = 0
            if self.filled_orders:
                # 计算实际收益
                entry_value = sum(o["qty"] * o["price"] for o in self.filled_orders)
                exit_price = current_price
                if sl_filled:
                    # 找到止损订单的成交价
                    for order in self.engine.orders:
                        if order.id == self.sl_order_id and order.status == 'filled':
                            exit_price = order.filled_price
                            break
                
                exit_value = self.position_qty * exit_price
                profit = exit_value - entry_value
                
                # 更新绩效统计
                self.trade_count += 1
                if profit > 0:
                    self.win_count += 1
                    self.consecutive_losses = 0
                else:
                    self.consecutive_losses += 1
                    
                # 根据亏损情况设置冷却时间
                if profit < 0:
                    loss_pct = abs(profit) / entry_value * 100
                    if loss_pct > 5:  # 亏损超过5%
                        cooldown_hours = min(24, int(loss_pct * 2))  # 最多冷却24小时
                        self.cooldown_until = current_timestamp + cooldown_hours * 3600 * 1000
                        print(f"[风控] 大额亏损{loss_pct:.1f}%，冷却{cooldown_hours}小时")

            print(f"[回测完成] 本次交易{'盈利' if profit >= 0 else '亏损'}: ${profit:.2f}")
            self.reset()

    def _technical_confirmation(self, candles) -> bool:
        """技术指标确认"""
        closes = [c[4] for c in candles]
        
        # MACD确认
        macd_line, signal_line, histogram = self.macd(closes)
        if macd_line and signal_line:
            macd_bullish = histogram > 0 or macd_line > signal_line
        else:
            macd_bullish = True
            
        # 价格站上短期EMA
        ema_fast = self.ema(closes, 5)
        if len(ema_fast) > 0:
            price_above_fast_ema = closes[-1] > ema_fast[-1]
        else:
            price_above_fast_ema = True
            
        return macd_bullish and price_above_fast_ema

    def _close_position(self, current_price: float, reason: str):
        """平仓处理"""
        if self.position_qty > 0:
            # 市价平仓
            close_order = self.engine.create_order(
                self.symbol, 'sell', 'market', self.position_qty
            )
            print(f"[回测{reason}] 市价平仓 {self.position_qty:.4f} @ ${current_price:.2f}")
        
        self.reset()

    def reset(self):
        """重置策略状态"""
        self.state = 'IDLE'
        self.reference_price = None
        self.trigger_time = None
        self.attack_orders.clear()
        self.filled_orders.clear()
        self.position_qty = 0.0
        self.avg_entry = 0.0
        self.lowest_fill = None
        self.highest_price = None
        self.break_time = None
        self.tp_orders.clear()
        self.sl_order_id = None
        self.trailing_stop_price = None