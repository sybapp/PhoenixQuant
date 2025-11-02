# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PhoenixQuant is a lightweight cryptocurrency backtesting framework focused on "elastic dip buying" strategies. The codebase is in Chinese with English code structure. The framework implements an event-driven backtest engine that executes layered limit orders during market dips.

## Installation & Setup

```bash
# Install project and dependencies
pip install -e .
```

## Running Backtests

```bash
# Run backtest with default config
python run_backtest.py

# Run with custom config
python run_backtest.py --config configs/elastic_dip.yaml
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

All parameters in `configs/elastic_dip.yaml`:
- `engine`: Initial balance, fees
- `data`: Exchange source (Binance testnet), caching, data limits
- `window`: Backtest time range (ISO format)
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

## Development Notes

- This project uses `uv` for dependency management (evidenced by `uv.lock`)
- No test suite currently exists
- The project is installed as an editable package (`pip install -e .`)
- Comments and print output are in Chinese; variable/function names are English
