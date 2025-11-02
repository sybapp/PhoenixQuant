#!/bin/bash
# PhoenixQuant 优化策略快速启动脚本

echo "=========================================="
echo "PhoenixQuant 策略回测快速启动"
echo "=========================================="
echo ""

# 检查虚拟环境
if [ ! -d ".venv" ]; then
    echo "❌ 未检测到虚拟环境，请先运行: pip install -e ."
    exit 1
fi

echo "选择回测配置："
echo "1) Baseline (原始配置)"
echo "2) Signal Quality (提高信号质量)"
echo "3) Risk Control (最高收益 ⭐)"
echo "4) Comprehensive (最均衡，推荐)"
echo "5) 批量对比所有配置"
echo ""
read -p "请输入选项 (1-5): " choice

case $choice in
    1)
        CONFIG="configs/elastic_dip.yaml"
        ;;
    2)
        CONFIG="configs/opt_signal_quality.yaml"
        ;;
    3)
        CONFIG="configs/opt_risk_control.yaml"
        ;;
    4)
        CONFIG="configs/opt_comprehensive.yaml"
        ;;
    5)
        echo ""
        echo "🚀 开始批量回测..."
        python batch_backtest.py
        echo ""
        echo "📊 生成对比图表..."
        python visualize_comparison.py
        echo ""
        echo "✅ 完成！查看以下文件："
        echo "  - backtest_comparison.csv (详细数据)"
        echo "  - optimization_comparison.png (可视化对比)"
        echo "  - OPTIMIZATION_SUMMARY.md (完整报告)"
        exit 0
        ;;
    *)
        echo "❌ 无效选项"
        exit 1
        ;;
esac

echo ""
echo "🚀 运行配置: $CONFIG"
python run_backtest.py --config $CONFIG

echo ""
read -p "是否查看详细分析？(y/n): " detail

if [ "$detail" = "y" ] || [ "$detail" = "Y" ]; then
    echo ""
    echo "📊 生成详细分析..."
    python visualize_backtest.py
    echo ""
    echo "✅ 完成！查看以下文件："
    echo "  - backtest_equity.csv (权益曲线)"
    echo "  - backtest_trades.csv (交易记录)"
    echo "  - backtest_analysis.png (可视化图表)"
fi

echo ""
echo "=========================================="
echo "回测完成！"
echo "=========================================="
