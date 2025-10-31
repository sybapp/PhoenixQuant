"""CLI entry point for PhoenixQuant."""
from __future__ import annotations

import argparse
import asyncio
import logging
import os

import ccxt

from .app import PhoenixQuantApp
from .config import PARAM_PRESETS, BotParameters
from .feeds import ExchangeCredentials
from .logging_config import setup_logging


def parse_args() -> argparse.Namespace:
    presets = sorted(PARAM_PRESETS.keys())
    parser = argparse.ArgumentParser(description="Run the PhoenixQuant elastic dip bot")
    parser.add_argument("symbol", help="Trading symbol, e.g., BTC/USDT")
    parser.add_argument("stream_symbol", help="Stream identifier (e.g., btcusdt, BTC-USDT-SWAP)")
    parser.add_argument(
        "--preset",
        choices=presets,
        default="ALTS_MAJOR",
        help="Parameter preset to apply",
    )
    parser.add_argument(
        "--exchange",
        choices=["binance", "okx", "bitget"],
        default="binance",
        help="Target derivatives exchange",
    )
    parser.add_argument(
        "--inst-type",
        default=None,
        help="Stream instrument type (e.g., SWAP, UMCBL) for OKX/Bitget feeds",
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
    parser.add_argument("--api-key", default="", help="Exchange API key")
    parser.add_argument("--api-secret", default="", help="Exchange API secret")
    parser.add_argument("--api-passphrase", default="", help="Exchange API passphrase (if required)")
    return parser.parse_args()


async def _run_bot(args: argparse.Namespace) -> None:
    setup_logging(level=getattr(logging, args.log_level.upper(), logging.INFO))
    params: BotParameters = PARAM_PRESETS[args.preset]
    params.log_parameters(logging.getLogger("phoenixquant.parameters"))

    exchange_id = args.exchange
    env_prefix = exchange_id.upper()
    api_key = args.api_key or os.environ.get(f"{env_prefix}_API_KEY", "")
    api_secret = args.api_secret or os.environ.get(f"{env_prefix}_API_SECRET", "")
    api_passphrase = args.api_passphrase or os.environ.get(f"{env_prefix}_API_PASSPHRASE", "")

    if exchange_id == "binance":
        exchange = ccxt.binance(
            {
                "apiKey": api_key,
                "secret": api_secret,
                "enableRateLimit": True,
                "options": {"defaultType": "future"},
            }
        )
    elif exchange_id == "okx":
        exchange = ccxt.okx(
            {
                "apiKey": api_key,
                "secret": api_secret,
                "password": api_passphrase,
                "enableRateLimit": True,
                "options": {"defaultType": "swap"},
            }
        )
    elif exchange_id == "bitget":
        exchange = ccxt.bitget(
            {
                "apiKey": api_key,
                "secret": api_secret,
                "password": api_passphrase,
                "enableRateLimit": True,
                "options": {"defaultType": "swap"},
            }
        )
    else:  # pragma: no cover - guarded by argparse choices
        raise ValueError(f"Unsupported exchange: {exchange_id}")

    credentials = ExchangeCredentials(api_key=api_key, api_secret=api_secret, passphrase=api_passphrase or None)

    poll_logger = logging.getLogger("phoenixquant.bot")
    feed_logger = logging.getLogger("phoenixquant.feed")
    user_logger = logging.getLogger("phoenixquant.user_stream")

    async with PhoenixQuantApp(
        exchange=exchange,
        exchange_id=exchange_id,
        symbol=args.symbol,
        stream_symbol=args.stream_symbol,
        stream_type=args.inst_type,
        params=params,
        dry_run=args.dry_run,
        poll_logger=poll_logger,
        feed_logger=feed_logger,
        user_logger=user_logger,
        credentials=credentials,
    ) as app:
        await app.run_forever()


def main() -> None:
    args = parse_args()
    asyncio.run(_run_bot(args))


if __name__ == "__main__":
    main()

