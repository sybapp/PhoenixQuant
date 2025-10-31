"""CLI entry point for PhoenixQuant."""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
from typing import Optional

import ccxt

from .bot import ElasticDipBot
from .config import PARAM_PRESETS, BotParameters
from .feeds import RealtimeFeed, UserStream
from .logging_config import setup_logging


def parse_args() -> argparse.Namespace:
    presets = sorted(PARAM_PRESETS.keys())
    parser = argparse.ArgumentParser(description="Run the PhoenixQuant elastic dip bot")
    parser.add_argument("symbol", help="Trading symbol, e.g., BTC/USDT")
    parser.add_argument("stream_symbol", help="Stream symbol, e.g., btcusdt")
    parser.add_argument(
        "--preset",
        choices=presets,
        default="ALTS_MAJOR",
        help="Parameter preset to apply",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Enable dry run mode (no real orders).",
    )
    parser.add_argument(
        "--log-level",
        default=os.environ.get("PHOENIXQUANT_LOG_LEVEL", "INFO"),
        help="Logging level (DEBUG, INFO, WARNING, ...)",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("BINANCE_API_KEY", ""),
        help="Binance API key",
    )
    parser.add_argument(
        "--api-secret",
        default=os.environ.get("BINANCE_API_SECRET", ""),
        help="Binance API secret",
    )
    return parser.parse_args()


async def _run_bot(args: argparse.Namespace) -> None:
    setup_logging(level=getattr(logging, args.log_level.upper(), logging.INFO))
    params: BotParameters = PARAM_PRESETS[args.preset]
    params.log_parameters(logging.getLogger("phoenixquant.parameters"))

    exchange = ccxt.binance(
        {
            "apiKey": args.api_key,
            "secret": args.api_secret,
            "enableRateLimit": True,
            "options": {"defaultType": "future"},
        }
    )

    feed_logger = logging.getLogger("phoenixquant.feed")
    public_feed = RealtimeFeed(args.stream_symbol, params.liq_window_sec, logger=feed_logger)
    await public_feed.start()

    bot = ElasticDipBot(
        exchange,
        args.symbol,
        params,
        public_feed,
        None,
        dry_run=args.dry_run,
    )
    await bot.init_market()

    user_stream: Optional[UserStream] = None
    if not args.dry_run:
        user_logger = logging.getLogger("phoenixquant.user_stream")
        user_stream = UserStream(exchange, bot.on_user_event, logger=user_logger)
        await user_stream.start()
        bot.user_stream = user_stream

    try:
        while True:
            await bot.step()
            await asyncio.sleep(params.poll_sec)
    finally:
        if user_stream:
            await user_stream.stop()
        await public_feed.stop()


def main() -> None:
    args = parse_args()
    asyncio.run(_run_bot(args))


if __name__ == "__main__":
    main()

