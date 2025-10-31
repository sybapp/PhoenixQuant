"""Realtime websocket feeds for PhoenixQuant."""
from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import json
import time
from collections import deque
from dataclasses import dataclass
from typing import Awaitable, Callable, Deque, Dict, Optional, Protocol, Tuple

import websockets


@dataclass(slots=True)
class ExchangeCredentials:
    """Credentials required for establishing private websocket streams."""

    api_key: str
    api_secret: str
    passphrase: Optional[str] = None


class UserStreamProtocol(Protocol):
    """Protocol describing the behaviour of user stream helpers."""

    async def start(self) -> None:  # pragma: no cover - network I/O
        ...

    async def stop(self) -> None:  # pragma: no cover - network I/O
        ...


class RealtimeFeed:
    """Track liquidation and funding rate streams for multiple exchanges."""

    def __init__(
        self,
        exchange_id: str,
        stream_symbol: str,
        liq_window_sec: int,
        *,
        logger,
        stream_type: Optional[str] = None,
    ) -> None:
        self.exchange_id = exchange_id
        self.stream_symbol = stream_symbol
        self.stream_type = stream_type
        self.liq_window_sec = liq_window_sec
        self.logger = logger
        self.liq_events: Deque[Tuple[float, float]] = deque()
        self.funding_rate: float = 0.0
        self._tasks: list[asyncio.Task[None]] = []

    async def _liquidation_worker(self) -> None:  # pragma: no cover - network I/O
        if self.exchange_id == "binance":
            url = f"wss://fstream.binance.com/ws/{self.stream_symbol}@forceOrder"
            subscribe_message = None
        elif self.exchange_id == "okx":
            url = "wss://ws.okx.com:8443/ws/v5/public"
            subscribe_message = json.dumps(
                {
                    "op": "subscribe",
                    "args": [
                        {
                            "channel": "liquidation-orders",
                            "instType": self.stream_type or "SWAP",
                            "instId": self.stream_symbol,
                        }
                    ],
                }
            )
        elif self.exchange_id == "bitget":
            url = "wss://ws.bitget.com/mix/v1/stream"
            subscribe_message = json.dumps(
                {
                    "op": "subscribe",
                    "args": [
                        {
                            "channel": "forceOrders",
                            "instType": (self.stream_type or "UMCBL").upper(),
                            "instId": self.stream_symbol,
                        }
                    ],
                }
            )
        else:
            raise ValueError(f"Unsupported exchange for realtime feed: {self.exchange_id}")

        while True:
            try:
                async with websockets.connect(url, ping_interval=20) as websocket:
                    if subscribe_message:
                        await websocket.send(subscribe_message)
                    async for message in websocket:
                        data = json.loads(message)
                        self._handle_liquidation_message(data)
            except Exception as exc:  # pragma: no cover - network failure path
                self.logger.warning("[LIQ WS] reconnect", exc_info=exc)
                await asyncio.sleep(1)

    def _handle_liquidation_message(self, data: Dict[str, object]) -> None:
        """Normalise liquidation payloads across exchanges."""

        if self.exchange_id == "binance":
            order = data.get("o", {})
            average_price = float(order.get("ap", 0.0))
            quantity = float(order.get("q", 0.0))
            timestamp = int(order.get("T", time.time() * 1000)) / 1000.0
            notional = average_price * quantity
        elif self.exchange_id == "okx":
            payload = next(iter(data.get("data", []) or []), None)
            if not payload:
                return
            price = float(payload.get("fillPx") or payload.get("px") or 0.0)
            size = float(payload.get("sz", 0.0))
            timestamp = int(payload.get("ts", time.time() * 1000)) / 1000.0
            notional = price * size
        elif self.exchange_id == "bitget":
            payload = next(iter(data.get("data", []) or []), None)
            if not payload:
                return
            price = float(payload.get("price", 0.0) or payload.get("fillPrice", 0.0))
            size = float(payload.get("size", 0.0) or payload.get("filledQty", 0.0))
            timestamp = int(payload.get("ts", time.time() * 1000)) / 1000.0
            notional = price * size
        else:  # pragma: no cover - guard for future extensions
            return

        self.liq_events.append((timestamp, notional))
        cutoff = time.time() - self.liq_window_sec
        while self.liq_events and self.liq_events[0][0] < cutoff:
            self.liq_events.popleft()

    async def _funding_worker(self) -> None:  # pragma: no cover - network I/O
        if self.exchange_id == "binance":
            url = f"wss://fstream.binance.com/ws/{self.stream_symbol}@fundingRate"
            subscribe_message = None
        elif self.exchange_id == "okx":
            url = "wss://ws.okx.com:8443/ws/v5/public"
            subscribe_message = json.dumps(
                {
                    "op": "subscribe",
                    "args": [
                        {
                            "channel": "funding-rate",
                            "instId": self.stream_symbol,
                            "instType": self.stream_type or "SWAP",
                        }
                    ],
                }
            )
        elif self.exchange_id == "bitget":
            url = "wss://ws.bitget.com/mix/v1/stream"
            subscribe_message = json.dumps(
                {
                    "op": "subscribe",
                    "args": [
                        {
                            "channel": "fundingRate",
                            "instType": (self.stream_type or "UMCBL").upper(),
                            "instId": self.stream_symbol,
                        }
                    ],
                }
            )
        else:
            raise ValueError(f"Unsupported exchange for realtime feed: {self.exchange_id}")

        while True:
            try:
                async with websockets.connect(url, ping_interval=20) as websocket:
                    if subscribe_message:
                        await websocket.send(subscribe_message)
                    async for message in websocket:
                        data = json.loads(message)
                        self._handle_funding_message(data)
            except Exception as exc:  # pragma: no cover - network failure path
                self.logger.warning("[FUND WS] reconnect", exc_info=exc)
                await asyncio.sleep(1)

    def _handle_funding_message(self, data: Dict[str, object]) -> None:
        if self.exchange_id == "binance":
            self.funding_rate = float(data.get("p", 0.0))
        elif self.exchange_id == "okx":
            payload = next(iter(data.get("data", []) or []), None)
            if payload:
                self.funding_rate = float(payload.get("fundingRate", 0.0))
        elif self.exchange_id == "bitget":
            payload = next(iter(data.get("data", []) or []), None)
            if payload:
                self.funding_rate = float(payload.get("fundingRate", 0.0))
        else:  # pragma: no cover - guard for future extensions
            return

    def get_liq_notional_sum(self) -> float:
        """Return the current liquidation notional sum for the window."""

        cutoff = time.time() - self.liq_window_sec
        return sum(notional for timestamp, notional in self.liq_events if timestamp >= cutoff)

    async def start(self) -> None:
        """Start both websocket workers."""

        self.logger.info(
            "Starting realtime feeds",
            extra={
                "exchange": self.exchange_id,
                "stream_symbol": self.stream_symbol,
                "stream_type": self.stream_type,
            },
        )
        self._tasks = [
            asyncio.create_task(self._liquidation_worker()),
            asyncio.create_task(self._funding_worker()),
        ]

    async def stop(self) -> None:
        """Stop websocket workers."""

        self.logger.info(
            "Stopping realtime feeds",
            extra={
                "exchange": self.exchange_id,
                "stream_symbol": self.stream_symbol,
                "stream_type": self.stream_type,
            },
        )
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)


class BinanceUserStream(UserStreamProtocol):
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

    async def start(self) -> None:  # pragma: no cover - network I/O
        if self.listen_key is None:
            response = await self._ccxt_async(self.exchange.fapiPrivatePostListenKey, {})
            self.listen_key = response.get("listenKey")
            self.logger.info("Obtained listen key", extra={"listen_key": self.listen_key})

        self._running = True
        self._task_ws = asyncio.create_task(self._ws_worker())
        self._task_keepalive = asyncio.create_task(self._keepalive_worker())

    async def stop(self) -> None:  # pragma: no cover - network I/O
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

    async def _keepalive_worker(self) -> None:  # pragma: no cover - network I/O
        while self._running:
            try:
                await asyncio.sleep(25 * 60)
                await self._ccxt_async(self.exchange.fapiPrivatePutListenKey, {"listenKey": self.listen_key})
                self.logger.debug("Refreshed listen key", extra={"listen_key": self.listen_key})
            except Exception as exc:  # pragma: no cover - network failure path
                self.logger.warning("[USER] keepalive err", exc_info=exc)

    async def _ws_worker(self) -> None:  # pragma: no cover - network I/O
        url = f"wss://fstream.binance.com/ws/{self.listen_key}"
        while self._running:
            try:
                async with websockets.connect(url, ping_interval=20) as websocket:
                    async for msg in websocket:
                        data = json.loads(msg)
                        await self.on_event(data)
            except Exception as exc:  # pragma: no cover - network failure path
                self.logger.warning("[USER WS] reconnect", exc_info=exc)
                await asyncio.sleep(1)


class OkxUserStream(UserStreamProtocol):
    """Private order stream for OKX perpetual instruments."""

    def __init__(
        self,
        on_event: Callable[[Dict[str, object]], Awaitable[None]],
        *,
        logger,
        credentials: ExchangeCredentials,
        stream_type: Optional[str] = None,
    ) -> None:
        self.on_event = on_event
        self.logger = logger
        self.credentials = credentials
        self.stream_type = stream_type or "SWAP"
        self._running = False
        self._task_ws: Optional[asyncio.Task[None]] = None

    async def start(self) -> None:  # pragma: no cover - network I/O
        self._running = True
        self._task_ws = asyncio.create_task(self._ws_worker())

    async def stop(self) -> None:  # pragma: no cover - network I/O
        self._running = False
        if self._task_ws:
            self._task_ws.cancel()
            await asyncio.gather(self._task_ws, return_exceptions=True)
            self._task_ws = None

    async def _ws_worker(self) -> None:  # pragma: no cover - network I/O
        url = "wss://ws.okx.com:8443/ws/v5/private"
        while self._running:
            try:
                async with websockets.connect(url, ping_interval=20) as websocket:
                    await self._login(websocket)
                    await self._subscribe_orders(websocket)
                    async for message in websocket:
                        data = json.loads(message)
                        if data.get("event") in {"login", "subscribe"}:
                            self.logger.debug("OKX user stream event", extra={"event": data.get("event")})
                            continue
                        await self._handle_message(data)
            except Exception as exc:  # pragma: no cover - network failure path
                self.logger.warning("[OKX USER] reconnect", exc_info=exc)
                await asyncio.sleep(1)

    async def _login(self, websocket) -> None:
        timestamp = str(time.time())
        message = f"{timestamp}GET/users/self/verify"
        signature = base64.b64encode(
            hmac.new(self.credentials.api_secret.encode(), message.encode(), hashlib.sha256).digest()
        ).decode()
        payload = {
            "op": "login",
            "args": [
                {
                    "apiKey": self.credentials.api_key,
                    "passphrase": self.credentials.passphrase or "",
                    "timestamp": timestamp,
                    "sign": signature,
                }
            ],
        }
        await websocket.send(json.dumps(payload))

    async def _subscribe_orders(self, websocket) -> None:
        payload = {
            "op": "subscribe",
            "args": [
                {
                    "channel": "orders",
                    "instType": self.stream_type,
                }
            ],
        }
        await websocket.send(json.dumps(payload))

    async def _handle_message(self, data: Dict[str, object]) -> None:
        if data.get("arg", {}).get("channel") != "orders":
            return
        for entry in data.get("data", []) or []:
            event = {
                "e": "ORDER_TRADE_UPDATE",
                "o": {
                    "s": entry.get("instId"),
                    "X": str(entry.get("state", "")).upper(),
                    "S": str(entry.get("side", "")).upper(),
                    "p": entry.get("px", "0"),
                    "ap": entry.get("avgPx", "0"),
                    "q": entry.get("sz", "0"),
                    "z": entry.get("accFillSz", "0"),
                    "i": entry.get("ordId"),
                },
            }
            await self.on_event(event)


class BitgetUserStream(UserStreamProtocol):
    """Private order stream for Bitget perpetual instruments."""

    def __init__(
        self,
        on_event: Callable[[Dict[str, object]], Awaitable[None]],
        *,
        logger,
        credentials: ExchangeCredentials,
        stream_type: Optional[str] = None,
    ) -> None:
        self.on_event = on_event
        self.logger = logger
        self.credentials = credentials
        self.stream_type = (stream_type or "UMCBL").upper()
        self._running = False
        self._task_ws: Optional[asyncio.Task[None]] = None

    async def start(self) -> None:  # pragma: no cover - network I/O
        self._running = True
        self._task_ws = asyncio.create_task(self._ws_worker())

    async def stop(self) -> None:  # pragma: no cover - network I/O
        self._running = False
        if self._task_ws:
            self._task_ws.cancel()
            await asyncio.gather(self._task_ws, return_exceptions=True)
            self._task_ws = None

    async def _ws_worker(self) -> None:  # pragma: no cover - network I/O
        url = "wss://ws.bitget.com/mix/v1/stream"
        while self._running:
            try:
                async with websockets.connect(url, ping_interval=20) as websocket:
                    await self._login(websocket)
                    await self._subscribe_orders(websocket)
                    async for message in websocket:
                        data = json.loads(message)
                        if data.get("event") in {"login", "subscribe"}:
                            self.logger.debug("Bitget user stream event", extra={"event": data.get("event")})
                            continue
                        await self._handle_message(data)
            except Exception as exc:  # pragma: no cover - network failure path
                self.logger.warning("[BITGET USER] reconnect", exc_info=exc)
                await asyncio.sleep(1)

    async def _login(self, websocket) -> None:
        timestamp = str(int(time.time() * 1000))
        message = f"{timestamp}GET/user/verify"
        signature = base64.b64encode(
            hmac.new(self.credentials.api_secret.encode(), message.encode(), hashlib.sha256).digest()
        ).decode()
        payload = {
            "op": "login",
            "args": [
                {
                    "apiKey": self.credentials.api_key,
                    "passphrase": self.credentials.passphrase or "",
                    "timestamp": timestamp,
                    "sign": signature,
                }
            ],
        }
        await websocket.send(json.dumps(payload))

    async def _subscribe_orders(self, websocket) -> None:
        payload = {
            "op": "subscribe",
            "args": [
                {
                    "channel": "orders",
                    "instType": self.stream_type,
                }
            ],
        }
        await websocket.send(json.dumps(payload))

    async def _handle_message(self, data: Dict[str, object]) -> None:
        if data.get("arg", {}).get("channel") != "orders":
            return
        for entry in data.get("data", []) or []:
            event = {
                "e": "ORDER_TRADE_UPDATE",
                "o": {
                    "s": entry.get("instId") or entry.get("symbol"),
                    "X": str(entry.get("state", "")).upper(),
                    "S": str(entry.get("side", "")).upper(),
                    "p": entry.get("price", "0"),
                    "ap": entry.get("fillPrice", "0"),
                    "q": entry.get("size", "0"),
                    "z": entry.get("accFillSize", entry.get("filledQty", "0")),
                    "i": entry.get("orderId") or entry.get("order_id"),
                },
            }
            await self.on_event(event)


__all__ = [
    "ExchangeCredentials",
    "RealtimeFeed",
    "UserStreamProtocol",
    "BinanceUserStream",
    "OkxUserStream",
    "BitgetUserStream",
]


