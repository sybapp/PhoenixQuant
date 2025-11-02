# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PhoenixQuant is a lightweight cryptocurrency backtesting and live trading framework focused on "elastic dip buying" strategies. The codebase is in Chinese with English code structure. The framework implements an event-driven backtest engine that executes layered limit orders during market dips, with support for both backtesting and live trading.

## Installation & Setup

```bash
# Install project and dependencies
pip install -e .
```

## Commands

### Backtesting
```bash
# Run backtest with default config
python run_backtest.py

# Run with custom config
python run_backtest.py --config configs/elastic_dip.yaml

# Batch backtest multiple configs
python batch_backtest.py

# Multi-asset backtest
python multi_asset_backtest.py
```

### Live Trading
```bash
# Run live trading (requires API keys in config)
python run_live_trading.py --config configs/live_template.yaml

# Dry-run mode (no real orders)
python run_live_trading.py --config configs/live_template.yaml --dry-run

# With custom log level
python run_live_trading.py --config configs/live_template.yaml --log DEBUG
```

### Visualization & Analysis
```bash
# Visualize backtest results
python visualize_backtest.py --config configs/elastic_dip.yaml

# View detailed trade breakdown
python view_backtest_details.py --config configs/elastic_dip.yaml

# Compare multiple backtests
python visualize_comparison.py

# Generate optimization report
python generate_optimization_report.py

# Profile backtest performance
python profile_backtest.py
```

## Architecture

### Core Modules

**phoenix_quant/backtest/**
- `engine.py` - Event-driven matching engine with order management, position tracking, and equity recording
- `runner.py` - Orchestrates data loading, engine initialization, and strategy execution
- `data.py` - Fetches OHLCV data from exchanges via CCXT; auto-downloads from Binance testnet and caches to `data/binance_testnet/`
- `analyzer.py` - Calculates backtest metrics (returns, drawdown, win rate)

**phoenix_quant/strategies/**
- `elastic_dip.py` - Implements the core trading strategy with signal scoring system

**phoenix_quant/**
- `config.py` - Configuration dataclasses and YAML parsing (includes fallback parser when PyYAML unavailable)
- `live.py` - Live trading implementation with real-time data feed and order execution

### Strategy Logic Flow

1. **Signal Calculation** (`ElasticDipStrategy._calculate_signal`): Scores each candle 0-100 based on:
   - Single-candle and windowed price drops (volatility-adjusted Z-scores)
   - EMA positioning (21/55 periods)
   - RSI oversold conditions (14-period)
   - Volume recovery patterns
   - Intraday drawdown triggers

2. **State Machine** (IDLE → ARMING → MANAGE):
   - **IDLE**: Waiting for signal >= `min_signal` threshold
   - **ARMING**: Layered limit orders placed below current price
   - **MANAGE**: Position active; monitors TP/SL/trailing stops

3. **Layered Entry** (`_deploy_layers`): Deploys multiple limit orders at configured offsets with size ratios

4. **Risk Management** (`_manage_position`):
   - Hard stop loss at `hard_stop_pct`
   - Take profit at `take_profit_pct`
   - Trailing stop activated above `trailing_activation_pct`
   - Max hold time and cooldown periods

### Configuration Structure

All parameters in YAML config files under `configs/`:

**Backtest configs:**
- `elastic_dip.yaml` - Default backtest configuration
- `opt_*.yaml` - Optimized parameter sets (balanced, comprehensive, risk_control, etc.)
- `long_term_*.yaml` - Long-term backtest configs for different assets (BTC, ETH, BNB, XRP, DOGE)

**Live trading configs:**
- `live_template.yaml` - Template for live trading (copy and add API keys)

**Configuration sections:**
- `engine`: Initial balance, fees, leverage
- `data`: Exchange source (Binance testnet), caching, data limits
- `window`: Backtest time range (ISO format) - only for backtesting
- `exchange`: API keys and exchange settings - only for live trading
- `live`: Polling interval, warmup bars, dry-run mode - only for live trading
- `strategy`: Signal thresholds, EMA/RSI periods, layers, risk controls

Layers defined as:
```yaml
layers:
  - offset_pct: 0.8      # Price offset from trigger
    size_ratio: 0.15     # Fraction of allocated capital
```

## Key Implementation Details

**Order Matching** (`BacktestEngine.update_market`):
- Limit orders fill when candle high/low crosses order price
- Market orders execute immediately at current price
- Tracks equity, balance, and position value each candle

**Data Loading**:
- Defaults to Binance testnet futures with sandbox mode
- Auto-caches to `data/binance_testnet/{symbol}_{timeframe}.csv`
- Fetches iteratively with pagination until `window.end` reached

**Position Tracking** (`Position` class in `engine.py`):
- Maintains weighted average entry price
- Calculates realized PnL on position reduction
- Updates timestamp on each trade

**Live Trading Architecture** (`phoenix_quant/live.py`):
- `LiveDataFeed`: Poll-based real-time OHLCV data with backfill support
- `ExchangeExecutor`: Wraps CCXT for order placement and management
- `LiveTrader`: Main orchestrator that runs the strategy loop with real-time data
- Supports dry-run mode for testing without placing real orders
- Heartbeat logging for monitoring system health

**Dual-Mode Design**:
The same `ElasticDipStrategy` class works in both backtest and live modes:
- Backtest: Uses `BacktestEngine` for simulated order matching
- Live: Uses `ExchangeExecutor` via adapter pattern with same order interface
- Strategy logic remains unchanged between modes

## Development Notes

- This project uses `uv` for dependency management (evidenced by `uv.lock`)
- No test suite currently exists
- The project is installed as an editable package (`pip install -e .`)
- Comments and print output are in Chinese; variable/function names are English
