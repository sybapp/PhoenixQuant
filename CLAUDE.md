# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PhoenixQuant is a Python-based quantitative trading backtesting system focused on cryptocurrency trading strategies. The system implements an "elastic dip bot" strategy that aims to buy during market dips using sophisticated technical analysis and risk management.

## Development Commands

### Running Backtests
```bash
# Run main backtest with Binance testnet data
python run_backtest.py

# Run demo with simulated data
python run_backtest.py --demo

# Run optimized backtest with enhanced parameters
python run_backtest_ds.py
```

### Package Management
This project uses uv for Python package management:
```bash
# Install dependencies
uv sync

# Add new dependency
uv add package_name

# Run with uv
uv run python run_backtest.py
```

## Architecture

### Core Components

1. **Backtest Engine** (`backtest_engine.py`)
   - `BacktestEngine`: Main simulation engine with order management, position tracking, and market data processing
   - `BacktestOrder`: Order representation with status tracking
   - `BacktestPosition`: Position management with P&L calculation
   - `HistoricalDataFetcher`: Data acquisition from Binance API (testnet and mainnet)

2. **Strategy Implementation** (`backtest_strategy.py`)
   - `BacktestElasticDipBot`: Core strategy implementing elastic dip methodology
   - Multi-layer entry system with dynamic position sizing
   - Advanced technical indicators (EMA, RSI, MACD, Bollinger Bands, ATR)
   - Comprehensive risk management with stop-loss and take-profit

3. **Analysis & Visualization** (`backtest_analysis.py`)
   - `BacktestAnalyzer`: Performance metrics calculation and reporting
   - Statistical analysis (Sharpe, Sortino, Calmar ratios, max drawdown)
   - Visualization with matplotlib/seaborn
   - Excel export functionality

### Key Design Patterns

- **State Machine Pattern**: Strategy uses states (IDLE, WAIT_FOR_BOUNCE, WAIT_ORDERS, MANAGE, COOLDOWN)
- **Event-Driven Simulation**: Market data updates trigger order matching and strategy decisions
- **Layered Architecture**: Clear separation between engine, strategy, and analysis components

## Strategy Logic

### Elastic Dip Strategy Flow
1. **Signal Detection**: Identifies rapid price drops with oversold conditions
2. **Confirmation**: Waits for volume recovery and technical confirmation
3. **Layered Entry**: Executes multiple limit orders at progressively lower prices
4. **Risk Management**: Dynamic stop-loss and multiple take-profit levels
5. **Performance Tracking**: Real-time P&L and position management

### Technical Indicators Used
- EMA (Exponential Moving Average) for trend analysis
- RSI (Relative Strength Index) for momentum
- MACD for trend changes
- Bollinger Bands for volatility and mean reversion
- ATR (Average True Range) for volatility-adjusted position sizing

## Configuration

### Strategy Parameters
Key parameters in `run_backtest.py`:
- `drop_pct_single`: Single candle drop percentage threshold
- `drop_pct_window`: Window-based drop percentage threshold
- `layer_pcts`: Price levels for layered entry orders
- `layer_pos_ratio`: Position sizing for each layer
- Risk management parameters for stop-loss and take-profit

### Exchange Configuration
- Uses Binance testnet by default (`set_sandbox_mode(True)`)
- Can be configured for mainnet data (read-only)
- API credentials stored in code (should be externalized for production)

## Data Management

### Historical Data Sources
- Binance API (testnet and mainnet)
- Local CSV caching for repeated backtests
- Simulated data generation for testing

### Data Format
K-line data structure: `[timestamp, open, high, low, close, volume]`

## Development Notes

### Code Style
- Chinese comments throughout the codebase
- Comprehensive docstrings with parameter descriptions
- Type hints used extensively
- Async/await pattern for API calls

### Testing
- Simulated data generation for strategy testing
- Progress tracking during long backtests
- Comprehensive error handling for API failures

### Performance Considerations
- Efficient pandas/numpy operations for large datasets
- Memory management for long backtest periods
- Rate limiting for API calls

## Common Tasks

### Adding New Strategies
1. Extend `BacktestElasticDipBot` class
2. Implement new signal detection logic
3. Add custom risk management rules
4. Update analysis metrics as needed

### Parameter Optimization
1. Modify parameters in `STRATEGY_PARAMS`
2. Run comparative backtests
3. Use analysis metrics to evaluate performance

### Data Source Integration
1. Extend `HistoricalDataFetcher` for new exchanges
2. Implement data format conversion
3. Add caching mechanisms for efficiency