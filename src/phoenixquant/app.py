"""Application orchestration helpers for PhoenixQuant."""
from __future__ import annotations

import asyncio
import logging
from contextlib import AbstractAsyncContextManager
from typing import Optional

import ccxt

from .bot import ElasticDipBot
from .config import BotParameters
from .feeds import RealtimeFeed, UserStream


class PhoenixQuantApp(AbstractAsyncContextManager["PhoenixQuantApp"]):
    """Manage lifecycle of the bot, realtime feed and user stream."""

    def __init__(
        self,
        *,
        exchange: ccxt.binance,
        symbol: str,
        stream_symbol: str,
        params: BotParameters,
        dry_run: bool,
        poll_logger: logging.Logger,
        feed_logger: logging.Logger,
        user_logger: logging.Logger,
    ) -> None:
        self.exchange = exchange
        self.symbol = symbol
        self.stream_symbol = stream_symbol
        self.params = params
        self.dry_run = dry_run
        self.poll_logger = poll_logger
        self.feed_logger = feed_logger
        self.user_logger = user_logger

        self.feed = RealtimeFeed(stream_symbol, params.liq_window_sec, logger=feed_logger)
        self.bot = ElasticDipBot(
            exchange,
            symbol,
            params,
            self.feed,
            None,
            dry_run=dry_run,
            logger=poll_logger,
        )
        self.user_stream: Optional[UserStream] = None

    async def __aenter__(self) -> "PhoenixQuantApp":
        await self.feed.start()
        await self.bot.init_market()
        if not self.dry_run:
            self.user_stream = UserStream(self.exchange, self.bot.on_user_event, logger=self.user_logger)
            await self.user_stream.start()
            self.bot.user_stream = self.user_stream
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

