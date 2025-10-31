"""Application orchestration helpers for PhoenixQuant."""
from __future__ import annotations

import asyncio
import logging
from contextlib import AbstractAsyncContextManager
from typing import Optional

import ccxt

from .bot import ElasticDipBot
from .config import BotParameters
from .feeds import (
    BinanceUserStream,
    BitgetUserStream,
    ExchangeCredentials,
    OkxUserStream,
    RealtimeFeed,
    UserStreamProtocol,
)


class PhoenixQuantApp(AbstractAsyncContextManager["PhoenixQuantApp"]):
    """Manage lifecycle of the bot, realtime feed and user stream."""

    def __init__(
        self,
        *,
        exchange: ccxt.Exchange,
        exchange_id: str,
        symbol: str,
        stream_symbol: str,
        stream_type: Optional[str],
        params: BotParameters,
        dry_run: bool,
        poll_logger: logging.Logger,
        feed_logger: logging.Logger,
        user_logger: logging.Logger,
        credentials: ExchangeCredentials,
    ) -> None:
        self.exchange = exchange
        self.exchange_id = exchange_id
        self.symbol = symbol
        self.stream_symbol = stream_symbol
        self.stream_type = stream_type
        self.params = params
        self.dry_run = dry_run
        self.poll_logger = poll_logger
        self.feed_logger = feed_logger
        self.user_logger = user_logger
        self.credentials = credentials

        self.feed = RealtimeFeed(
            exchange_id,
            stream_symbol,
            params.liq_window_sec,
            logger=feed_logger,
            stream_type=stream_type,
        )
        self.bot = ElasticDipBot(
            exchange,
            symbol,
            params,
            self.feed,
            None,
            dry_run=dry_run,
            logger=poll_logger,
        )
        self.user_stream: Optional[UserStreamProtocol] = None

    async def __aenter__(self) -> "PhoenixQuantApp":
        await self.feed.start()
        await self.bot.init_market()
        if not self.dry_run:
            self.user_stream = self._build_user_stream()
            if self.user_stream:
                await self.user_stream.start()
                self.bot.user_stream = self.user_stream
            else:
                self.user_logger.warning(
                    "User stream unavailable for exchange", extra={"exchange": self.exchange_id}
                )
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:  # noqa: D401 - standard context method signature
        await self.stop()

    async def run_forever(self) -> None:
        """Poll the bot until cancelled."""

        self.poll_logger.info("Starting bot loop", extra={"poll_sec": self.params.poll_sec})
        try:
            while True:
                await self.bot.step()
                await asyncio.sleep(self.params.poll_sec)
        finally:
            self.poll_logger.info("Bot loop stopping")
            await self.stop()

    async def stop(self) -> None:
        """Stop all running services."""

        if self.user_stream:
            await self.user_stream.stop()
            self.bot.user_stream = None
            self.user_stream = None
        await self.feed.stop()

    def _build_user_stream(self) -> Optional[UserStreamProtocol]:
        if self.exchange_id == "binance":
            return BinanceUserStream(self.exchange, self.bot.on_user_event, logger=self.user_logger)
        if self.exchange_id == "okx":
            if not (self.credentials.api_key and self.credentials.api_secret and self.credentials.passphrase):
                self.user_logger.error(
                    "OKX user stream requires api key, secret and passphrase",
                    extra={"credentials_provided": bool(self.credentials.api_key and self.credentials.api_secret)},
                )
                return None
            return OkxUserStream(
                self.bot.on_user_event,
                logger=self.user_logger,
                credentials=self.credentials,
                stream_type=self.stream_type,
            )
        if self.exchange_id == "bitget":
            if not (self.credentials.api_key and self.credentials.api_secret and self.credentials.passphrase):
                self.user_logger.error(
                    "Bitget user stream requires api key, secret and passphrase",
                    extra={"credentials_provided": bool(self.credentials.api_key and self.credentials.api_secret)},
                )
                return None
            return BitgetUserStream(
                self.bot.on_user_event,
                logger=self.user_logger,
                credentials=self.credentials,
                stream_type=self.stream_type,
            )
        self.user_logger.error("Unsupported exchange for user stream", extra={"exchange": self.exchange_id})
        return None

