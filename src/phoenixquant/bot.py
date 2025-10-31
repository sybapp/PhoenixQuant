"""Trading bot implementation for PhoenixQuant."""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

import ccxt

from .config import BotParameters
from .feeds import RealtimeFeed, UserStreamProtocol
from .indicators import ema, rsi, volume_recovered


@dataclass(slots=True)
class OrderLayer:
    """Representation of a single ladder order."""

    price: float
    qty: float
    id: Optional[str] = None
    filled: bool = False


@dataclass(slots=True)
class PositionSnapshot:
    """Aggregate view of the currently filled layers."""

    quantity: float = 0.0
    avg_entry: float = 0.0
    lowest_fill: Optional[float] = None

    def reset(self) -> None:
        self.quantity = 0.0
        self.avg_entry = 0.0
        self.lowest_fill = None


class State(Enum):
    IDLE = 0
    WAIT_FOR_BOUNCE = 1
    WAIT_ORDERS = 2
    MANAGE = 3


class ElasticDipBot:
    """Async trading bot that reacts to liquidation spikes and rebounds."""

    def __init__(
        self,
        exchange: ccxt.Exchange,
        symbol: str,
        params: BotParameters,
        public_feed: RealtimeFeed,
        user_stream: Optional[UserStreamProtocol],
        *,
        dry_run: bool,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.exchange = exchange
        self.symbol = symbol
        self.params = params
        self.feed = public_feed
        self.user_stream = user_stream
        self.dry_run = dry_run
        self.logger = logger or logging.getLogger(__name__)

        self.state = State.IDLE
        self.reference_price: Optional[float] = None
        self.trigger_time: Optional[float] = None

        self.market: Optional[Dict[str, Any]] = None
        self.attack_orders: List[OrderLayer] = []
        self.filled_orders: List[OrderLayer] = []
        self.break_time: Optional[float] = None

        self.position = PositionSnapshot()

    async def init_market(self) -> None:
        self.logger.info("Loading exchange markets", extra={"symbol": self.symbol})
        await self._ccxt_async(self.exchange.load_markets)
        self.market = self.exchange.market(self.symbol)
        self.logger.info("Loaded market metadata", extra={"market": self.market})

    async def _ccxt_async(self, func, *args, **kwargs):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: func(*args, **kwargs))

    async def fetch_candles(self, limit: int = 240):
        self.logger.debug("Fetching candles", extra={"limit": limit, "timeframe": self.params.timeframe})
        return await self._ccxt_async(
            self.exchange.fetch_ohlcv,
            self.symbol,
            timeframe=self.params.timeframe,
            limit=limit,
        )

    async def fetch_price(self) -> float:
        ticker = await self._ccxt_async(self.exchange.fetch_ticker, self.symbol)
        price = ticker["bid"]
        self.logger.debug("Fetched bid price", extra={"price": price})
        return price

    # ---- signals ----
    def is_fast_drop(self, candles) -> bool:
        window = self.params.window_min
        open_price, high_price, low_price, close_price = candles[-1][1:5]
        single_drop = (close_price < open_price) and (
            (open_price - close_price) / open_price * 100 >= self.params.drop_pct_single
        )
        sub_window = candles[-window:]
        highest = max(entry[2] for entry in sub_window)
        window_drop = (highest - sub_window[-1][4]) / highest * 100 >= self.params.drop_pct_window
        result = single_drop or window_drop
        self.logger.debug(
            "Fast drop evaluation",
            extra={
                "single_drop": single_drop,
                "window_drop": window_drop,
                "window": window,
                "result": result,
            },
        )
        return result

    def is_trend_down(self, candles) -> bool:
        closes = [candle[4] for candle in candles]
        ema_fast_values = ema(closes, self.params.ema_fast)
        ema_slow_values = ema(closes, self.params.ema_slow)
        if len(ema_fast_values) == 0 or len(ema_slow_values) == 0:
            return False
        ema_fast_last = ema_fast_values[-1]
        ema_fast_prev = ema_fast_values[-5] if len(ema_fast_values) >= 5 else ema_fast_last
        ema_slow_last = ema_slow_values[-1]
        ema_slow_prev = ema_slow_values[-5] if len(ema_slow_values) >= 5 else ema_slow_last
        result = (
            ema_fast_last < ema_slow_last
            and ema_fast_last - ema_fast_prev < 0
            and ema_slow_last - ema_slow_prev < 0
        )
        self.logger.debug(
            "Trend down evaluation",
            extra={
                "ema_fast_last": float(ema_fast_last),
                "ema_fast_prev": float(ema_fast_prev),
                "ema_slow_last": float(ema_slow_last),
                "ema_slow_prev": float(ema_slow_prev),
                "result": result,
            },
        )
        return result

    def is_volume_dry(self, candles) -> bool:
        volumes = [candle[5] for candle in candles]
        if len(volumes) < 20:
            return False
        average_volume = float(sum(volumes[-10:]) / 10.0)
        result = volumes[-1] < average_volume * self.params.vol_shrink_ratio
        self.logger.debug(
            "Volume dry evaluation",
            extra={
                "last_volume": volumes[-1],
                "average_volume": average_volume,
                "ratio": self.params.vol_shrink_ratio,
                "result": result,
            },
        )
        return result

    def is_oversold(self, candles) -> bool:
        closes = [candle[4] for candle in candles]
        value = rsi(closes, self.params.rsi_period)
        result = not (value != value) and value < self.params.rsi_oversold
        self.logger.debug(
            "RSI evaluation",
            extra={
                "rsi_value": float(value),
                "threshold": self.params.rsi_oversold,
                "result": result,
            },
        )
        return result

    def is_liquidation_spike(self) -> bool:
        notional_sum = self.feed.get_liq_notional_sum()
        result = notional_sum >= self.params.liq_notional_threshold
        self.logger.debug(
            "Liquidation spike evaluation",
            extra={
                "notional_sum": notional_sum,
                "threshold": self.params.liq_notional_threshold,
                "result": result,
            },
        )
        return result

    def is_funding_extreme(self) -> bool:
        result = self.feed.funding_rate <= self.params.funding_extreme_neg
        self.logger.debug(
            "Funding rate evaluation",
            extra={
                "funding_rate": self.feed.funding_rate,
                "threshold": self.params.funding_extreme_neg,
                "result": result,
            },
        )
        return result

    # ---- precision helpers ----
    def _round_price(self, price: float) -> float:
        return float(self.exchange.price_to_precision(self.symbol, price))

    def _round_amount(self, amount: float) -> float:
        return float(self.exchange.amount_to_precision(self.symbol, amount))

    async def _place_limit_buy(self, price: float, qty: float):
        price = self._round_price(price)
        qty = self._round_amount(qty)
        if self.dry_run:
            self.logger.info("(Dry) placing limit buy", extra={"qty": qty, "price": price})
            return {"id": f"DRY_BUY_{price}"}
        order = await self._ccxt_async(
            self.exchange.create_order,
            self.symbol,
            "limit",
            "buy",
            qty,
            price,
            {"timeInForce": "GTC", "reduceOnly": False, "positionSide": "BOTH"},
        )
        self.logger.info("Placed limit buy", extra={"order": order})
        return order

    async def _place_limit_sell(self, price: float, qty: float):
        price = self._round_price(price)
        qty = self._round_amount(qty)
        if self.dry_run:
            self.logger.info("(Dry) placing limit sell", extra={"qty": qty, "price": price})
            return {"id": f"DRY_SELL_{price}"}
        order = await self._ccxt_async(
            self.exchange.create_order,
            self.symbol,
            "limit",
            "sell",
            qty,
            price,
            {"timeInForce": "GTC", "reduceOnly": True, "positionSide": "BOTH"},
        )
        self.logger.info("Placed limit sell", extra={"order": order})
        return order

    async def _place_stop_market_close(self, stop_price: float):
        stop_price = self._round_price(stop_price)
        if self.dry_run:
            self.logger.info("(Dry) placing stop market", extra={"stop_price": stop_price})
            return {"id": f"DRY_SL_{stop_price}"}
        order = await self._ccxt_async(
            self.exchange.create_order,
            self.symbol,
            "STOP_MARKET",
            "sell",
            None,
            None,
            {
                "stopPrice": stop_price,
                "closePosition": True,
                "positionSide": "BOTH",
                "workingType": "MARK_PRICE",
            },
        )
        self.logger.info("Placed stop market", extra={"order": order})
        return order

    async def compute_attack_plan(self, current_price: float) -> List[OrderLayer]:
        balance = await self._ccxt_async(self.exchange.fetch_balance)
        usdt_free = 0.0
        if isinstance(balance, dict):
            if "USDT" in balance and isinstance(balance["USDT"], dict):
                wallet = balance["USDT"]
                usdt_free = float(wallet.get("free") or wallet.get("total") or 0.0)
            elif "total" in balance and isinstance(balance["total"], dict):
                usdt_free = float(balance["total"].get("USDT", 0.0))
        if usdt_free <= 0:
            usdt_free = float(self.params.total_capital)
        max_capital = min(self.params.total_capital, usdt_free * self.params.max_account_ratio)
        self.logger.info(
            "Computed capital allocation",
            extra={
                "free_usdt": usdt_free,
                "max_capital": max_capital,
                "total_capital": self.params.total_capital,
                "max_account_ratio": self.params.max_account_ratio,
            },
        )
        plan: List[OrderLayer] = []
        for drop_pct, ratio in zip(self.params.layer_pcts, self.params.layer_pos_ratio):
            price = current_price * (1 - drop_pct / 100.0)
            capital = max_capital * ratio
            qty = capital / price if price > 0 else 0.0
            plan.append(OrderLayer(price=price, qty=qty))
            self.logger.debug(
                "Layer prepared",
                extra={
                    "drop_pct": drop_pct,
                    "ratio": ratio,
                    "price": price,
                    "capital": capital,
                    "qty": qty,
                },
            )
        return plan

    def _recalc_position(self) -> None:
        if not self.filled_orders:
            self.position.reset()
            return
        total_qty = sum(order.qty for order in self.filled_orders)
        total_cost = sum(order.qty * order.price for order in self.filled_orders)
        self.position.quantity = total_qty
        self.position.avg_entry = total_cost / total_qty if total_qty > 0 else 0.0
        self.position.lowest_fill = min(order.price for order in self.filled_orders)
        self.logger.info(
            "Recalculated position",
            extra={
                "position_qty": self.position.quantity,
                "avg_entry": self.position.avg_entry,
                "lowest_fill": self.position.lowest_fill,
            },
        )

    async def on_user_event(self, data: Dict[str, Any]) -> None:
        event_type = data.get("e")
        if event_type == "ORDER_TRADE_UPDATE":
            order_data = data.get("o", {})
            symbol = order_data.get("s")
            if not self.market or symbol != self.market.get("id"):
                return
            status = order_data.get("X")
            side = order_data.get("S")
            price = float(order_data.get("p", "0") or 0.0)
            avg_price = float(order_data.get("ap", "0") or 0.0)
            quantity = float(order_data.get("q", "0") or 0.0)
            filled_qty = float(order_data.get("z", "0") or 0.0)
            order_id = order_data.get("i")

            self.logger.info(
                "Order trade update received",
                extra={
                    "status": status,
                    "side": side,
                    "price": price,
                    "avg_price": avg_price,
                    "quantity": quantity,
                    "filled_qty": filled_qty,
                    "order_id": order_id,
                },
            )

            if status in ("FILLED", "PARTIALLY_FILLED") and side == "BUY":
                match: Optional[OrderLayer] = None
                candidates = [order for order in self.attack_orders if not order.filled]
                if candidates:
                    match = min(candidates, key=lambda order: abs(order.price - price))
                filled_price = avg_price if avg_price > 0 else price
                filled_quantity = filled_qty if filled_qty > 0 else quantity
                if match:
                    match.filled = True
                    match.id = order_id
                    match.price = filled_price
                    match.qty = filled_quantity
                    if match not in self.filled_orders:
                        self.filled_orders.append(match)
                else:
                    self.filled_orders.append(
                        OrderLayer(
                            id=order_id,
                            price=filled_price,
                            qty=filled_quantity,
                            filled=True,
                        )
                    )
                self._recalc_position()

                if self.state in (State.WAIT_ORDERS, State.MANAGE):
                    tp = self.position.avg_entry * (1 + self.params.take_profit_pct / 100.0)
                    sl = (
                        self.position.lowest_fill * (1 - self.params.hard_stop_extra / 100.0)
                        if self.position.lowest_fill
                        else None
                    )
                    if sl:
                        await self._place_limit_sell(tp, self.position.quantity * 0.5)
                        await self._place_stop_market_close(sl)
                        self.state = State.MANAGE
                        self.logger.info(
                            "Manage state setup",
                            extra={
                                "avg_entry": self.position.avg_entry,
                                "take_profit": tp,
                                "stop_loss": sl,
                            },
                        )
        elif event_type == "ACCOUNT_UPDATE":
            self.logger.debug("Account update event", extra={"data": data})

    async def step(self) -> None:
        candles = await self.fetch_candles()

        if self.state == State.IDLE:
            trend_down = self.is_trend_down(candles)
            volume_dry = self.is_volume_dry(candles)
            if trend_down and volume_dry:
                self.logger.info("Skipping due to downtrend with dry volume")
                return
            fast_drop = self.is_fast_drop(candles)
            oversold = self.is_oversold(candles)
            liq_spike = self.is_liquidation_spike()
            funding_extreme = self.is_funding_extreme()
            self.logger.debug(
                "Signal snapshot",
                extra={
                    "trend_down": trend_down,
                    "volume_dry": volume_dry,
                    "fast_drop": fast_drop,
                    "oversold": oversold,
                    "liq_spike": liq_spike,
                    "funding_extreme": funding_extreme,
                },
            )
            if fast_drop and oversold and liq_spike and funding_extreme:
                self.reference_price = await self.fetch_price()
                self.trigger_time = time.time()
                self.state = State.WAIT_FOR_BOUNCE
                self.logger.info(
                    "Trigger detected; waiting for bounce",
                    extra={"reference_price": self.reference_price},
                )
                return

        elif self.state == State.WAIT_FOR_BOUNCE:
            if time.time() - (self.trigger_time or 0) > self.params.delayed_window_sec:
                self.logger.info("Delayed window expired, resetting")
                await self.reset()
                return
            price = await self.fetch_price()
            price_ok = price >= (self.reference_price or price) * (
                1 + self.params.delayed_trigger_pct / 100.0
            )
            volume_ok = volume_recovered(
                candles,
                ma_short=self.params.vol_recover_ma_short,
                ma_long=self.params.vol_recover_ma_long,
                ratio=self.params.vol_recover_ratio,
                tick_ratio=self.params.tick_vol_ratio,
            )
            self.logger.debug(
                "Bounce evaluation",
                extra={"price_ok": price_ok, "volume_ok": volume_ok, "price": price},
            )
            if price_ok and volume_ok:
                plan = await self.compute_attack_plan(price)
                self.attack_orders = plan
                for order in plan:
                    response = await self._place_limit_buy(order.price, order.qty)
                    order.id = response.get("id")
                self.state = State.WAIT_ORDERS
                self.break_time = None
                self.logger.info("Placed ladder orders", extra={"count": len(plan)})
                return

        elif self.state == State.WAIT_ORDERS:
            if self.position.lowest_fill:
                price = await self.fetch_price()
                stop_loss = self.position.lowest_fill * (1 - self.params.hard_stop_extra / 100.0)
                if price <= stop_loss:
                    if self.break_time is None:
                        self.break_time = time.time()
                    elif time.time() - self.break_time >= self.params.sl_time_grace_sec:
                        self.logger.warning("Stop loss guard triggered; resetting")
                        await self.reset()
                else:
                    self.break_time = None

        elif self.state == State.MANAGE:
            if self.position.quantity <= 0:
                await self.reset()

    async def reset(self) -> None:
        self.state = State.IDLE
        self.reference_price = None
        self.trigger_time = None
        self.attack_orders.clear()
        self.filled_orders.clear()
        self.position.reset()
        self.break_time = None
        self.logger.info("Bot state reset", extra={"state": self.state.name})

