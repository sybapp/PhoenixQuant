# -*- coding: utf-8 -*-
"""
Elastic Dip Bot (Async, Binance Futures) + USER DATA STREAM + PRESETS + Volume-Recovery Filter
- Trend/volume filters to avoid slow-drip downtrends
- Realtime liquidation & funding websocket feeds
- Immediate & delayed elastic rebound execution
- Laddered limit buys, TP, hard SL (time + price)
- USER DATA STREAM (listenKey): real fills/positions drive state
- Multi-symbol parameter presets (BTC/ETH/SOL/BNB/DOGE/XRP/ALTS_MAJOR)
- Volume recovery confirmation for delayed-bounce trigger

Dependencies: ccxt, websockets, numpy
  pip install ccxt websockets numpy
"""

import asyncio
import json
import time
from collections import deque
from enum import Enum
import numpy as np
import ccxt
import websockets

# ========= API / SYMBOL =========
API_KEY = "kflCxmrjxzyNuaM60yvhFTCvFZBMRzCX2hniLLfEMycGJ2j2e6OMrsOE8Gzd5H7P"
API_SECRET = "Z9GOv6MoF2WQfi7iE21zkFliHzMJ1ENRtLixnvkp51lX4JA9jxsKnZ9ONak573An"

SYMBOL = "BTC/USDT"           # ccxt symbol, e.g., "BTC/USDT"
STREAM_SYMBOL = "btcusdt"     # ws stream symbol (lowercase, no slash), e.g., "btcusdt"

# Dry-Run 模式（True 不下真实单，只打印；False 实盘下单）
DRY_RUN = True

# ===== 交易品种参数预设 =====
PARAM_PRESETS = {
    "BTCUSDT": {
        "timeframe": "1m", "poll_sec": 2,
        "drop_pct_single": 1.0, "drop_pct_window": 3.0, "window_min": 5,
        "ema_fast": 20, "ema_slow": 60, "vol_shrink_ratio": 0.6,
        "rsi_period": 14, "rsi_oversold": 25.0,
        "funding_extreme_neg": -0.05,        # -5 bps
        "liq_window_sec": 60, "liq_notional_threshold": 8_000_000,
        "delayed_trigger_pct": 1.0, "delayed_window_sec": 60*60*12,
        "layer_pcts": [0.8, 1.4, 2.0, 2.6, 3.3],
        "layer_pos_ratio": [0.10, 0.15, 0.20, 0.25, 0.30],
        "total_capital": 1000, "max_account_ratio": 0.30,
        "take_profit_pct": 1.0, "hard_stop_extra": 0.8, "sl_time_grace_sec": 30,
        "vol_recover_ma_short": 5, "vol_recover_ma_long": 20, "vol_recover_ratio": 1.15,
        "tick_vol_ratio": 1.30,
    },
    "ETHUSDT": {
        "timeframe": "1m", "poll_sec": 2,
        "drop_pct_single": 1.0, "drop_pct_window": 3.0, "window_min": 5,
        "ema_fast": 20, "ema_slow": 60, "vol_shrink_ratio": 0.6,
        "rsi_period": 14, "rsi_oversold": 25.0,
        "funding_extreme_neg": -0.05,
        "liq_window_sec": 60, "liq_notional_threshold": 4_000_000,
        "delayed_trigger_pct": 1.0, "delayed_window_sec": 60*60*12,
        "layer_pcts": [0.9, 1.6, 2.3, 3.0, 3.8],
        "layer_pos_ratio": [0.10, 0.15, 0.20, 0.25, 0.30],
        "total_capital": 800, "max_account_ratio": 0.30,
        "take_profit_pct": 1.0, "hard_stop_extra": 0.9, "sl_time_grace_sec": 30,
        "vol_recover_ma_short": 5, "vol_recover_ma_long": 20, "vol_recover_ratio": 1.15,
        "tick_vol_ratio": 1.30,
    },
    "SOLUSDT": {
        "timeframe": "1m", "poll_sec": 2,
        "drop_pct_single": 1.2, "drop_pct_window": 3.8, "window_min": 5,
        "ema_fast": 20, "ema_slow": 60, "vol_shrink_ratio": 0.65,
        "rsi_period": 14, "rsi_oversold": 24.0,
        "funding_extreme_neg": -0.08,
        "liq_window_sec": 60, "liq_notional_threshold": 2_000_000,
        "delayed_trigger_pct": 1.2, "delayed_window_sec": 60*60*10,
        "layer_pcts": [1.0, 1.8, 2.6, 3.5, 4.5],
        "layer_pos_ratio": [0.08, 0.15, 0.22, 0.25, 0.30],
        "total_capital": 600, "max_account_ratio": 0.25,
        "take_profit_pct": 1.2, "hard_stop_extra": 1.1, "sl_time_grace_sec": 25,
        "vol_recover_ma_short": 5, "vol_recover_ma_long": 20, "vol_recover_ratio": 1.20,
        "tick_vol_ratio": 1.40,
    },
    "BNBUSDT": {
        "timeframe": "1m", "poll_sec": 2,
        "drop_pct_single": 1.0, "drop_pct_window": 3.2, "window_min": 5,
        "ema_fast": 20, "ema_slow": 60, "vol_shrink_ratio": 0.6,
        "rsi_period": 14, "rsi_oversold": 25.0,
        "funding_extreme_neg": -0.06,
        "liq_window_sec": 60, "liq_notional_threshold": 2_500_000,
        "delayed_trigger_pct": 1.0, "delayed_window_sec": 60*60*10,
        "layer_pcts": [0.9, 1.6, 2.3, 3.0, 3.8],
        "layer_pos_ratio": [0.10, 0.15, 0.20, 0.25, 0.30],
        "total_capital": 600, "max_account_ratio": 0.25,
        "take_profit_pct": 1.0, "hard_stop_extra": 0.9, "sl_time_grace_sec": 25,
        "vol_recover_ma_short": 5, "vol_recover_ma_long": 20, "vol_recover_ratio": 1.18,
        "tick_vol_ratio": 1.35,
    },
    "DOGEUSDT": {
        "timeframe": "1m", "poll_sec": 2,
        "drop_pct_single": 1.4, "drop_pct_window": 4.2, "window_min": 5,
        "ema_fast": 20, "ema_slow": 60, "vol_shrink_ratio": 0.7,
        "rsi_period": 14, "rsi_oversold": 23.0,
        "funding_extreme_neg": -0.10,
        "liq_window_sec": 60, "liq_notional_threshold": 1_200_000,
        "delayed_trigger_pct": 1.3, "delayed_window_sec": 60*60*8,
        "layer_pcts": [1.2, 2.0, 3.0, 4.2, 5.5],
        "layer_pos_ratio": [0.08, 0.12, 0.20, 0.25, 0.35],
        "total_capital": 400, "max_account_ratio": 0.20,
        "take_profit_pct": 1.5, "hard_stop_extra": 1.3, "sl_time_grace_sec": 20,
        "vol_recover_ma_short": 5, "vol_recover_ma_long": 20, "vol_recover_ratio": 1.25,
        "tick_vol_ratio": 1.50,
    },
    "XRPUSDT": {
        "timeframe": "1m", "poll_sec": 2,
        "drop_pct_single": 1.2, "drop_pct_window": 3.6, "window_min": 5,
        "ema_fast": 20, "ema_slow": 60, "vol_shrink_ratio": 0.65,
        "rsi_period": 14, "rsi_oversold": 24.0,
        "funding_extreme_neg": -0.08,
        "liq_window_sec": 60, "liq_notional_threshold": 1_500_000,
        "delayed_trigger_pct": 1.2, "delayed_window_sec": 60*60*10,
        "layer_pcts": [1.0, 1.8, 2.6, 3.5, 4.5],
        "layer_pos_ratio": [0.08, 0.15, 0.22, 0.25, 0.30],
        "total_capital": 500, "max_account_ratio": 0.25,
        "take_profit_pct": 1.2, "hard_stop_extra": 1.0, "sl_time_grace_sec": 25,
        "vol_recover_ma_short": 5, "vol_recover_ma_long": 20, "vol_recover_ratio": 1.20,
        "tick_vol_ratio": 1.40,
    },
    "ALTS_MAJOR": {
        "timeframe": "1m", "poll_sec": 2,
        "drop_pct_single": 1.2, "drop_pct_window": 3.8, "window_min": 5,
        "ema_fast": 20, "ema_slow": 60, "vol_shrink_ratio": 0.65,
        "rsi_period": 14, "rsi_oversold": 24.0,
        "funding_extreme_neg": -0.08,
        "liq_window_sec": 60, "liq_notional_threshold": 1_000_000,
        "delayed_trigger_pct": 1.2, "delayed_window_sec": 60*60*10,
        "layer_pcts": [1.0, 1.8, 2.6, 3.5, 4.5],
        "layer_pos_ratio": [0.08, 0.15, 0.22, 0.25, 0.30],
        "total_capital": 400, "max_account_ratio": 0.25,
        "take_profit_pct": 1.2, "hard_stop_extra": 1.1, "sl_time_grace_sec": 25,
        "vol_recover_ma_short": 5, "vol_recover_ma_long": 20, "vol_recover_ratio": 1.20,
        "tick_vol_ratio": 1.45,
    },
}

# ========= 指标工具 =========
def ema(arr, period):
    arr = np.asarray(arr, dtype=float)
    if len(arr) < period:
        return np.array([])
    k = 2 / (period + 1)
    e = np.zeros_like(arr)
    e[0] = arr[0]
    for i in range(1, len(arr)):
        e[i] = arr[i]*k + e[i-1]*(1-k)
    return e

def rsi(arr, period=14):
    arr = np.asarray(arr, dtype=float)
    if len(arr) < period + 1:
        return np.nan
    deltas = np.diff(arr)
    gains = np.clip(deltas, 0, None)
    losses = -np.clip(deltas, None, 0)
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])
    rsis, rs = [], (avg_gain / (avg_loss + 1e-12)) if avg_loss > 0 else np.inf
    rsis.append(100 - 100/(1+rs))
    for i in range(period, len(deltas)):
        avg_gain = (avg_gain*(period-1) + gains[i]) / period
        avg_loss = (avg_loss*(period-1) + losses[i]) / period
        rs = (avg_gain / (avg_loss + 1e-12)) if avg_loss > 0 else np.inf
        rsis.append(100 - 100/(1+rs))
    return rsis[-1] if rsis else np.nan

def volume_recovered(candles, ma_short=5, ma_long=20, ratio=1.15, tick_ratio=None):
    """
    量能恢复条件（任一满足即可）：
      A) MA_short(成交量) > MA_long(成交量) * ratio
      B) (可选)当前K线量 > MA_long * tick_ratio
    """
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

# ========= 实时公共Feed：清算 & Funding =========
class RealtimeFeed:
    def __init__(self, stream_symbol: str, liq_window_sec: int):
        self.stream_symbol = stream_symbol
        self.liq_window_sec = liq_window_sec
        self.liq_events = deque()  # (ts, notional)
        self.funding_rate = 0.0
        self._tasks = []

    async def _liquidation_worker(self):
        url = f"wss://stream.binancefuture.com/ws/{self.stream_symbol}@forceOrder"
        while True:
            try:
                async with websockets.connect(url, ping_interval=20) as ws:
                    async for msg in ws:
                        data = json.loads(msg)
                        o = data.get("o", {})
                        ap = float(o.get("ap", 0.0))
                        q  = float(o.get("q", 0.0))
                        notional = ap * q
                        ts = int(o.get("T", time.time()*1000))/1000.0
                        self.liq_events.append((ts, notional))
                        cutoff = time.time() - self.liq_window_sec
                        while self.liq_events and self.liq_events[0][0] < cutoff:
                            self.liq_events.popleft()
            except Exception as e:
                print("[LIQ WS] reconnect:", e)
                await asyncio.sleep(1)

    async def _funding_worker(self):
        url = f"wss://stream.binancefuture.com/ws/{self.stream_symbol}@fundingRate"
        while True:
            try:
                async with websockets.connect(url, ping_interval=20) as ws:
                    async for msg in ws:
                        data = json.loads(msg)
                        self.funding_rate = float(data.get("p", 0.0))  # 0.0001 = 0.01%
            except Exception as e:
                print("[FUND WS] reconnect:", e)
                await asyncio.sleep(1)

    def get_liq_notional_sum(self):
        cutoff = time.time() - self.liq_window_sec
        return sum(n for (ts, n) in self.liq_events if ts >= cutoff)

    async def start(self):
        self._tasks = [
            asyncio.create_task(self._liquidation_worker()),
            asyncio.create_task(self._funding_worker()),
        ]

    async def stop(self):
        for t in self._tasks:
            t.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)

# ========= 用户数据流：listenKey =========
class UserStream:
    def __init__(self, ex: ccxt.binance, on_event):
        self.ex = ex
        self.on_event = on_event
        self.listen_key = None
        self._task_ws = None
        self._task_keepalive = None
        self._running = False

    async def _ccxt_async(self, func, *args, **kwargs):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: func(*args, **kwargs))

    async def start(self):
        if self.listen_key is None:
            res = await self._ccxt_async(self.ex.fapiPrivatePostListenKey, {})
            self.listen_key = res.get("listenKey")
            print("[USER] listenKey:", self.listen_key)

        self._running = True
        self._task_ws = asyncio.create_task(self._ws_worker())
        self._task_keepalive = asyncio.create_task(self._keepalive_worker())

    async def stop(self):
        self._running = False
        if self._task_ws: self._task_ws.cancel()
        if self._task_keepalive: self._task_keepalive.cancel()
        await asyncio.gather(*(t for t in [self._task_ws, self._task_keepalive] if t), return_exceptions=True)
        try:
            await self._ccxt_async(self.ex.fapiPrivateDeleteListenKey, {"listenKey": self.listen_key})
        except Exception as e:
            print("[USER] delete listenKey err:", e)

    async def _keepalive_worker(self):
        while self._running:
            try:
                await asyncio.sleep(25*60)
                await self._ccxt_async(self.ex.fapiPrivatePutListenKey, {"listenKey": self.listen_key})
                print("[USER] keepalive ok")
            except Exception as e:
                print("[USER] keepalive err:", e)

    async def _ws_worker(self):
        url = f"wss://stream.binancefuture.com/ws/{self.listen_key}"
        while self._running:
            try:
                async with websockets.connect(url, ping_interval=20) as ws:
                    async for msg in ws:
                        data = json.loads(msg)
                        await self.on_event(data)
            except Exception as e:
                print("[USER WS] reconnect:", e)
                await asyncio.sleep(1)

# ========= 状态 =========
class State(Enum):
    IDLE = 0
    WAIT_FOR_BOUNCE = 1
    WAIT_ORDERS = 2
    MANAGE = 3

# ========= 主策略 =========
class ElasticDipBot:
    def __init__(self, exchange, symbol, params, public_feed: RealtimeFeed, user_stream: UserStream|None):
        self.ex = exchange
        self.symbol = symbol
        self.p = params
        self.feed = public_feed
        self.usr = user_stream

        self.state = State.IDLE
        self.reference_price = None
        self.trigger_time = None

        self.market = None
        self.attack_orders = []  # [{'id','price','qty','filled'}]
        self.filled_orders = []  # subset
        self.break_time = None   # SL 时间条件

        # 运行时缓存
        self.position_qty = 0.0
        self.avg_entry = 0.0
        self.lowest_fill = None

    async def init_market(self):
        await self._ccxt_async(self.ex.load_markets)
        self.market = self.ex.market(self.symbol)

    async def _ccxt_async(self, func, *args, **kwargs):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: func(*args, **kwargs))

    async def fetch_candles(self, limit=240):
        return await self._ccxt_async(self.ex.fetch_ohlcv, self.symbol, timeframe=self.p["timeframe"], limit=limit)

    async def fetch_price(self):
        t = await self._ccxt_async(self.ex.fetch_ticker, self.symbol)
        return t["bid"]

    # ---- 过滤与信号 ----
    def is_fast_drop(self, candles):
        w = self.p["window_min"]
        o,h,l,c = candles[-1][1:5]
        single = (c < o) and ((o - c)/o*100 >= self.p["drop_pct_single"])
        sub = candles[-w:]
        hi = max(x[2] for x in sub)
        window_drop = (hi - sub[-1][4])/hi*100 >= self.p["drop_pct_window"]
        return single or window_drop

    def is_trend_down(self, candles):
        closes = [c[4] for c in candles]
        ef = ema(closes, self.p["ema_fast"])
        es = ema(closes, self.p["ema_slow"])
        if len(ef)==0 or len(es)==0: return False
        ef_last, ef_prev = ef[-1], ef[-5] if len(ef)>=5 else ef[-1]
        es_last, es_prev = es[-1], es[-5] if len(es)>=5 else es[-1]
        return (ef_last < es_last) and (ef_last - ef_prev < 0) and (es_last - es_prev < 0)

    def is_volume_dry(self, candles):
        vols = [c[5] for c in candles]
        if len(vols) < 20: return False
        return vols[-1] < float(np.mean(vols[-10:])) * self.p["vol_shrink_ratio"]

    def is_oversold(self, candles):
        closes = [c[4] for c in candles]
        v = rsi(closes, self.p["rsi_period"])
        return (not np.isnan(v)) and (v < self.p["rsi_oversold"])

    def is_liquidation_spike(self):
        return self.feed.get_liq_notional_sum() >= self.p["liq_notional_threshold"]

    def is_funding_extreme(self):
        return self.feed.funding_rate <= self.p["funding_extreme_neg"]

    # ---- 精度/下单 ----
    def _round_price(self, price):
        return float(self.ex.price_to_precision(self.symbol, price))
    def _round_amount(self, amount):
        return float(self.ex.amount_to_precision(self.symbol, amount))

    async def _place_limit_buy(self, price, qty):
        price = self._round_price(price)
        qty   = self._round_amount(qty)
        if DRY_RUN:
            oid = f"DRY_BUY_{price}"
            print(f"[ORDER] (Dry) LIMIT BUY {qty} @ {price}")
            return {"id": oid}
        order = await self._ccxt_async(
            self.ex.create_order, self.symbol, "limit", "buy", qty, price,
            {"timeInForce":"GTC","reduceOnly":False,"positionSide":"BOTH"}
        )
        print("[ORDER] LIMIT BUY", order.get("id"), qty, "@", price)
        return order

    async def _place_limit_sell(self, price, qty):
        price = self._round_price(price)
        qty   = self._round_amount(qty)
        if DRY_RUN:
            oid = f"DRY_SELL_{price}"
            print(f"[ORDER] (Dry) LIMIT SELL {qty} @ {price}")
            return {"id": oid}
        order = await self._ccxt_async(
            self.ex.create_order, self.symbol, "limit", "sell", qty, price,
            {"timeInForce":"GTC","reduceOnly":True,"positionSide":"BOTH"}
        )
        print("[ORDER] LIMIT SELL", order.get("id"), qty, "@", price)
        return order

    async def _place_stop_market_close(self, stop_price):
        stop_price = self._round_price(stop_price)
        if DRY_RUN:
            print(f"[ORDER] (Dry) STOP-MARKET close @ {stop_price}")
            return {"id": f"DRY_SL_{stop_price}"}
        order = await self._ccxt_async(
            self.ex.create_order, self.symbol, "STOP_MARKET", "sell", None, None,
            {"stopPrice": stop_price, "closePosition": True, "positionSide":"BOTH", "workingType":"MARK_PRICE"}
        )
        print("[ORDER] STOP-MARKET closePosition", order.get("id"), "@", stop_price)
        return order

    async def compute_attack_plan(self, current_price):
        bal = await self._ccxt_async(self.ex.fetch_balance)
        usdt = bal["USDT"]["free"]
        max_capital = min(self.p["total_capital"], usdt * self.p["max_account_ratio"])
        plan = []
        for pct, ratio in zip(self.p["layer_pcts"], self.p["layer_pos_ratio"]):
            price = current_price * (1 - pct/100.0)
            capital = max_capital * ratio
            qty = capital / price if price > 0 else 0.0
            plan.append({"price": price, "qty": qty, "id": None, "filled": False})
        return plan

    def _recalc_position(self):
        if not self.filled_orders:
            self.position_qty = 0.0
            self.avg_entry = 0.0
            self.lowest_fill = None
            return
        total_qty = sum(o["qty"] for o in self.filled_orders)
        total_cost= sum(o["qty"]*o["price"] for o in self.filled_orders)
        self.position_qty = total_qty
        self.avg_entry = total_cost / total_qty if total_qty > 0 else 0.0
        self.lowest_fill = min(o["price"] for o in self.filled_orders) if self.filled_orders else None

    # ---- USER DATA 回调 ----
    async def on_user_event(self, data: dict):
        e = data.get("e")
        if e == "ORDER_TRADE_UPDATE":
            o = data.get("o", {})
            s = o.get("s")
            if not self.market or s != self.market.get("id"):
                return
            x = o.get("X")     # FILLED, PARTIALLY_FILLED, NEW, CANCELED...
            side = o.get("S")  # BUY/SELL
            p = float(o.get("p", "0") or 0.0)   # 报价
            ap= float(o.get("ap","0") or 0.0)   # 成交均价
            q = float(o.get("q", "0") or 0.0)   # 委托数量
            z = float(o.get("z", "0") or 0.0)   # 成交数量
            oid = o.get("i")

            if x in ("FILLED","PARTIALLY_FILLED") and side=="BUY":
                m = min(self.attack_orders, key=lambda od: abs(od["price"] - p)) if self.attack_orders else None
                filled_price = ap if ap>0 else p
                filled_qty   = z if z>0 else q
                if m:
                    m["filled"] = True
                    m["id"] = oid
                    m["price"] = filled_price
                    m["qty"] = filled_qty
                    if m not in self.filled_orders:
                        self.filled_orders.append(m)
                else:
                    self.filled_orders.append({"id": oid,"price":filled_price,"qty":filled_qty,"filled":True})
                self._recalc_position()

                if self.state in (State.WAIT_ORDERS, State.MANAGE):
                    tp = self.avg_entry * (1 + self.p["take_profit_pct"]/100.0)
                    sl = self.lowest_fill * (1 - self.p["hard_stop_extra"]/100.0) if self.lowest_fill else None
                    if sl:
                        await self._place_limit_sell(tp, self.position_qty * 0.5)
                        await self._place_stop_market_close(sl)
                        self.state = State.MANAGE
                        print(f"[MANAGE] avg={self.avg_entry:.2f} tp={tp:.2f} sl={sl:.2f}")

            # TODO: 处理 SELL 成交确认TP/SL并 reset()

        elif e == "ACCOUNT_UPDATE":
            # 可扩展：解析持仓/余额变化，校验 position_qty 一致性
            pass

    # ---- 状态机 ----
    async def step(self):
        candles = await self.fetch_candles()

        if self.state == State.IDLE:
            # 避免连续阴跌/缩量下行
            if self.is_trend_down(candles) and self.is_volume_dry(candles):
                return
            # 爆仓+超卖+急跌+极端资金费率 同时满足 → 进入延迟监控
            if self.is_fast_drop(candles) and self.is_oversold(candles) \
               and self.is_liquidation_spike() and self.is_funding_extreme():
                self.reference_price = await self.fetch_price()
                self.trigger_time = time.time()
                self.state = State.WAIT_FOR_BOUNCE
                print(f"[TRIGGER] -> WAIT_FOR_BOUNCE ref={self.reference_price:.2f}")
                return

        elif self.state == State.WAIT_FOR_BOUNCE:
            # 超时
            if time.time() - self.trigger_time > self.p["delayed_window_sec"]:
                print("[TIMEOUT] delayed window expired")
                await self.reset()
                return
            price = await self.fetch_price()
            price_ok = price >= self.reference_price * (1 + self.p["delayed_trigger_pct"]/100.0)
            # 量能恢复判定
            vol_ok = volume_recovered(
                candles,  # NOTE: 可换成 await self.fetch_candles() 以取最新一批
                ma_short=self.p["vol_recover_ma_short"],
                ma_long=self.p["vol_recover_ma_long"],
                ratio=self.p["vol_recover_ratio"],
                tick_ratio=self.p["tick_vol_ratio"],
            )
            if price_ok and vol_ok:
                plan = await self.compute_attack_plan(price)
                self.attack_orders = plan
                for o in plan:
                    res = await self._place_limit_buy(o["price"], o["qty"])
                    o["id"] = res.get("id", None)
                self.state = State.WAIT_ORDERS
                self.break_time = None
                print("[ORDERS] ladder placed (delayed+volume confirm):", len(plan))
                return

        elif self.state == State.WAIT_ORDERS:
            # 成交由用户流驱动；这里做 SL 保护（价格+时间双条件）
            if self.lowest_fill:
                price = await self.fetch_price()
                sl = self.lowest_fill * (1 - self.p["hard_stop_extra"]/100.0)
                if price <= sl:
                    if self.break_time is None:
                        self.break_time = time.time()
                    elif time.time() - self.break_time >= self.p["sl_time_grace_sec"]:
                        print("[SL GUARD] price below SL too long, reset (STOP_MARKET should close)")
                        await self.reset()
                else:
                    self.break_time = None

        elif self.state == State.MANAGE:
            # 如需：根据用户流SELL事件确认TP/SL成交后 reset()
            if self.position_qty <= 0:
                await self.reset()

    async def reset(self):
        self.state = State.IDLE
        self.reference_price = None
        self.trigger_time = None
        self.attack_orders.clear()
        self.filled_orders.clear()
        self.position_qty = 0.0
        self.avg_entry = 0.0
        self.lowest_fill = None
        self.break_time = None
        print("[RESET] -> IDLE")

# ========= 主入口 =========
async def main():
    # 选参数：按 SYMBOL 基础名（无斜杠）选择预设
    base = SYMBOL.replace("/", "").upper()
    params = PARAM_PRESETS.get(base, PARAM_PRESETS["ALTS_MAJOR"])

    exchange = ccxt.binance({
        "apiKey": API_KEY,
        "secret": API_SECRET,
        "enableRateLimit": True,
        "options": {"defaultType": "future"},
    })

    exchange.set_sandbox_mode(True)

    public_feed = RealtimeFeed(STREAM_SYMBOL, params["liq_window_sec"])
    await public_feed.start()

    user_stream = None
    bot = None

    if DRY_RUN:
        bot = ElasticDipBot(exchange, SYMBOL, params, public_feed, None)
        await bot.init_market()
    else:
        bot = ElasticDipBot(exchange, SYMBOL, params, public_feed, None)
        await bot.init_market()
        user_stream = UserStream(exchange, bot.on_user_event)
        await user_stream.start()
        bot.usr = user_stream

    try:
        while True:
            await bot.step()
            await asyncio.sleep(params["poll_sec"])
    finally:
        if user_stream:
            await user_stream.stop()
        await public_feed.stop()

if __name__ == "__main__":
    asyncio.run(main())
