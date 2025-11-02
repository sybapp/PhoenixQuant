"""基于配置文件的回测启动脚本"""

from __future__ import annotations

import argparse

from phoenix_quant import load_backtest_config, run_backtest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行弹性抄底策略回测")
    parser.add_argument(
        "--config",
        default="configs/elastic_dip.yaml",
        help="配置文件路径",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_backtest_config(args.config)
    analyzer = run_backtest(config)
    analyzer.print_report()


if __name__ == "__main__":
    main()
