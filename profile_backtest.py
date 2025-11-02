"""性能分析脚本"""
import cProfile
import pstats
from pathlib import Path
from phoenix_quant.backtest.runner import run_backtest, load_backtest_config

def main():
    config_path = Path("configs/elastic_dip.yaml")
    config = load_backtest_config(config_path)

    profiler = cProfile.Profile()
    profiler.enable()

    analyzer = run_backtest(config)

    profiler.disable()

    # 输出性能统计
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    print("\n=== 按累计时间排序的前30个函数 ===")
    stats.print_stats(30)

    print("\n=== 回测结果 ===")
    print(f"总收益率: {analyzer.total_return_pct:.2f}%")
    print(f"最大回撤: {analyzer.max_drawdown_pct:.2f}%")

if __name__ == "__main__":
    main()
