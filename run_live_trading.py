"""实时执行弹性抄底策略"""

from __future__ import annotations

import argparse
import logging
import sys

from phoenix_quant import load_live_config
from phoenix_quant.live import LiveTrader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行 PhoenixQuant 实盘策略")
    parser.add_argument("--config", required=True, help="实盘配置文件路径")
    parser.add_argument("--dry-run", action="store_true", help="强制使用干跑模式，不下真实订单")
    parser.add_argument("--log", default="INFO", help="日志级别 (DEBUG/INFO/WARNING/ERROR)")
    return parser.parse_args()


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s - %(levelname)s - %(name)s: %(message)s",
    )


def main() -> None:
    args = parse_args()
    setup_logging(args.log)

    config = load_live_config(args.config)
    if args.dry_run:
        config.settings.enable_trading = False
        config.settings.dry_run = True

    trader = LiveTrader(config)
    trader.run()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pragma: no cover
        logging.exception("实盘运行失败: %s", exc)
        sys.exit(1)
