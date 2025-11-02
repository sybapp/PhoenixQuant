# PhoenixQuant 回测框架

PhoenixQuant 是一个专注于数字货币弹性抄底策略的轻量化回测框架。本次重构对核心模块进行了拆分，使引擎、策略与配置解耦，更便于拓展与维护。

## 目录结构

```
phoenix_quant/
├── backtest/
│   ├── analyzer.py        # 回测结果分析
│   ├── data.py            # 历史数据加载
│   ├── engine.py          # 撮合与资金管理
│   └── runner.py          # 回测调度
├── strategies/
│   └── elastic_dip.py     # 弹性抄底策略
└── __init__.py
configs/
└── elastic_dip.yaml        # 策略与引擎配置
run_backtest.py              # 命令行入口
```

## 安装依赖

```bash
pip install -e .
```

## 配置参数

所有参数均放置在 `configs/elastic_dip.yaml` 中，可根据需要复制多份配置文件。配置主要分为以下几部分：

- `engine`：初始资金与手续费设置。
- `data`：历史数据来源、缓存路径及抓取限制。
- `window`：回测时间范围（ISO 时间）。
- `strategy`：策略细节，包括触发条件、分层挂单及风险控制。

示例（节选）：

```yaml
strategy:
  drop_single_pct: 0.8        # 单根K线跌幅阈值
  drop_window_pct: 2.5        # 窗口跌幅阈值
  rsi_oversold: 28            # RSI 超卖线
  volatility_window: 30       # 自适应波动率窗口
  layers:                     # 分层挂单
    - offset_pct: 0.8
      size_ratio: 0.15
  risk:
    take_profit_pct: 1.2      # 止盈百分比
    hard_stop_pct: 2.4        # 强制止损百分比
    cooldown_minutes: 45      # 冷静期
```

## 运行回测

```bash
python run_backtest.py --config configs/elastic_dip.yaml
```

脚本会自动：

1. 加载配置并抓取/读取历史数据（仓库已附带 `data/btcusdt_1m.csv`，离线环境可直接使用缓存）。
2. 使用重构后的撮合引擎执行弹性抄底策略。
3. 输出核心统计指标（收益率、最大回撤、胜率等）。

## 策略亮点

- **信号评分体系**：结合跌幅、EMA 位置、RSI、量能恢复，并以滚动波动率自适应阈值，为每根K线计算综合信号，减少噪声交易。
- **分层进场**：通过配置多层挂单，在行情急跌中逐步建仓，有效平滑成本。
- **动态风控**：支持硬止损、止盈与移动止损，同时限定最大持仓时间与冷静期。
- **完全参数化**：策略核心参数全部拆分为配置文件，便于快速迭代或批量调优。

## 拓展建议

- 为不同品种编写独立的 YAML 配置，以适配波动特性。
- 结合第三方数据源或本地 CSV，扩展 `HistoricalDataLoader`；如无法访问外部交易所，可改写为读取自有缓存。
- 在 `ElasticDipStrategy` 中新增自定义信号模块（如链上指标或资金费率）。

希望 PhoenixQuant 能帮助你更高效地验证交易想法！
