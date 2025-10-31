"""Realtime websocket feeds for PhoenixQuant."""
from __future__ import annotations

import asyncio
import json
import time
from collections import deque
from typing import Awaitable, Callable, Deque, Dict, Optional, Tuple

import websockets


class RealtimeFeed:
    """Track liquidation and funding rate streams."""

    def __init__(self, stream_symbol: str, liq_window_sec: int, *, logger):
        self.stream_symbol = stream_symbol
        self.liq_window_sec = liq_window_sec
        self.logger = logger
        self.liq_events: Deque[Tuple[float, float]] = deque()
        self.funding_rate: float = 0.0
        self._tasks: list[asyncio.Task[None]] = []

    async def _liquidation_worker(self) -> None:
        url = f"wss://fstream.binance.com/ws/{self.stream_symbol}@forceOrder"
        while True:
            try:
                async with websockets.connect(url, ping_interval=20) as ws:
                    async for msg in ws:
                        data = json.loads(msg)
                        order = data.get("o", {})
                        average_price = float(order.get("ap", 0.0))
                        quantity = float(order.get("q", 0.0))
                        notional = average_price * quantity
                        timestamp = int(order.get("T", time.time() * 1000)) / 1000.0
                        self.liq_events.append((timestamp, notional))
                        cutoff = time.time() - self.liq_window_sec
                        while self.liq_events and self.liq_events[0][0] < cutoff:
                            self.liq_events.popleft()
            except Exception as exc:  # pragma: no cover - network failure path
                self.logger.warning("[LIQ WS] reconnect", exc_info=exc)
                await asyncio.sleep(1)

    async def _funding_worker(self) -> None:
        url = f"wss://fstream.binance.com/ws/{self.stream_symbol}@fundingRate"
        while True:
            try:
                async with websockets.connect(url, ping_interval=20) as ws:
                    async for msg in ws:
                        data = json.loads(msg)
                        self.funding_rate = float(data.get("p", 0.0))
            except Exception as exc:  # pragma: no cover - network failure path
                self.logger.warning("[FUND WS] reconnect", exc_info=exc)
                await asyncio.sleep(1)

    def get_liq_notional_sum(self) -> float:
        """Return the current liquidation notional sum for the window."""

        cutoff = time.time() - self.liq_window_sec
        return sum(notional for timestamp, notional in self.liq_events if timestamp >= cutoff)

    async def start(self) -> None:
        """Start both websocket workers."""

        self.logger.info("Starting realtime feeds", extra={"stream_symbol": self.stream_symbol})
        self._tasks = [
            asyncio.create_task(self._liquidation_worker()),
            asyncio.create_task(self._funding_worker()),
        ]

    async def stop(self) -> None:
        """Stop websocket workers."""

        self.logger.info("Stopping realtime feeds", extra={"stream_symbol": self.stream_symbol})
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)


class UserStream:
    """Manage Binance Futures user data stream."""

    def __init__(self, exchange, on_event: Callable[[Dict[str, object]], Awaitable[None]], *, logger):
        self.exchange = exchange
        self.on_event = on_event
        self.logger = logger
        self.listen_key: Optional[str] = None
        self._task_ws: Optional[asyncio.Task[None]] = None
        self._task_keepalive: Optional[asyncio.Task[None]] = None
        self._running = False

    async def _ccxt_async(self, func, *args, **kwargs):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: func(*args, **kwargs))

    async def start(self) -> None:
        if self.listen_key is None:
            response = await self._ccxt_async(self.exchange.fapiPrivatePostListenKey, {})
            self.listen_key = response.get("listenKey")
            self.logger.info("Obtained listen key", extra={"listen_key": self.listen_key})

        self._running = True
        self._task_ws = asyncio.create_task(self._ws_worker())
        self._task_keepalive = asyncio.create_task(self._keepalive_worker())

    async def stop(self) -> None:
        self._running = False
        if self._task_ws:
            self._task_ws.cancel()
        if self._task_keepalive:
            self._task_keepalive.cancel()
        await asyncio.gather(
            *(task for task in (self._task_ws, self._task_keepalive) if task),
            return_exceptions=True,
        )
        if self.listen_key:
            try:
                await self._ccxt_async(self.exchange.fapiPrivateDeleteListenKey, {"listenKey": self.listen_key})
                self.logger.info("Deleted listen key", extra={"listen_key": self.listen_key})
            except Exception as exc:  # pragma: no cover - network failure path
                self.logger.warning("[USER] delete listenKey err", exc_info=exc)

    async def _keepalive_worker(self) -> None:
        while self._running:
            try:
                await asyncio.sleep(25 * 60)
                await self._ccxt_async(self.exchange.fapiPrivatePutListenKey, {"listenKey": self.listen_key})
                self.logger.debug("Refreshed listen key", extra={"listen_key": self.listen_key})
            except Exception as exc:  # pragma: no cover - network failure path
                self.logger.warning("[USER] keepalive err", exc_info=exc)

    async def _ws_worker(self) -> None:
        url = f"wss://fstream.binance.com/ws/{self.listen_key}"
        while self._running:
            try:
                async with websockets.connect(url, ping_interval=20) as ws:
                    async for msg in ws:
                        data = json.loads(msg)
                        await self.on_event(data)
            except Exception as exc:  # pragma: no cover - network failure path
                self.logger.warning("[USER WS] reconnect", exc_info=exc)
                await asyncio.sleep(1)

