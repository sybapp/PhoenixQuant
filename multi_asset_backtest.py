"""多资产长周期回测批处理"""

from __future__ import annotations

from pathlib import Path
from typing import List

import pandas as pd

from phoenix_quant import load_backtest_config, run_backtest


def collect_configs() -> List[Path]:
    """收集 configs/ 下所有 long_term_* 配置"""
    return sorted(Path("configs").glob("long_term_*.yaml"))


def run_backtest_safe(config_path: Path):
    """对单个配置执行回测，缺数据时跳过"""
    config = load_backtest_config(config_path)
    cache_path = config.data.cache
    if cache_path and not cache_path.exists():
        raise FileNotFoundError(
            f"缺少缓存文件: {cache_path}. 请先准备对应交易对的CSV数据。"
        )

    analyzer = run_backtest(config)
    summary = analyzer.build_summary()

    result = {
        "config": config_path.stem,
        "symbol": config.symbol,
        "initial_balance": summary.initial_balance,
        "final_equity": summary.final_equity,
        "return_pct": summary.total_return_pct,
        "max_drawdown_pct": summary.max_drawdown_pct,
        "win_rate_pct": summary.win_rate_pct,
        "trade_count": summary.trade_count,
    }
    return result


def main() -> None:
    configs = collect_configs()
    if not configs:
        print("未找到 long_term_* 配置文件，可在 configs/ 中新增。")
        return

    results = []
    skipped: List[str] = []

    for cfg in configs:
        print(f"\n{'=' * 70}\n运行配置: {cfg}\n{'=' * 70}")
        try:
            result = run_backtest_safe(cfg)
        except FileNotFoundError as exc:
            skipped.append(f"{cfg.name}: {exc}")
            print(f"跳过: {exc}")
            continue
        except RuntimeError as exc:
            skipped.append(f"{cfg.name}: {exc}")
            print(f"跳过: {exc}")
            continue
        results.append(result)
        print(
            f"收益率: {result['return_pct']:.2f}%, "
            f"最大回撤: {result['max_drawdown_pct']:.2f}%, "
            f"胜率: {result['win_rate_pct']:.2f}%, "
            f"交易数: {result['trade_count']}"
        )

    if results:
        df = pd.DataFrame(results).sort_values("return_pct", ascending=False)
        output = Path("backtest_comparison_long_term.csv")
        df.to_csv(output, index=False)

        print("\n\n====== 长周期多资产对比 ======")
        print(df.to_string(index=False, justify="center"))
        print(f"\n结果已保存至: {output}")
    else:
        print("\n未产生任何回测结果。")

    if skipped:
        print("\n以下配置被跳过，请补充数据或检查错误:")
        for item in skipped:
            print(f"- {item}")


if __name__ == "__main__":
    main()
