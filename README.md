# PhoenixQuant

PhoenixQuant is an async elastic dip trading bot for major perpetual futures venues
(Binance Futures, OKX Swap, Bitget Mix). The project follows a standard Python
package layout with a CLI entry point and structured JSON logging for easier
observability.

## Installation

```bash
pip install -e .
```

## Usage

```bash
phoenixquant BTC/USDT btcusdt --preset BTCUSDT --exchange binance --dry-run
```

To run against OKX or Bitget you can switch the `--exchange` flag and provide the
appropriate websocket identifiers, for example:

```bash
phoenixquant BTC/USDT BTC-USDT-SWAP --exchange okx --inst-type SWAP
phoenixquant BTC/USDT BTCUSDT --exchange bitget --inst-type UMCBL
```

Environment variables can be used to supply credentials and logging configuration
for each venue:

- `BINANCE_API_KEY`, `BINANCE_API_SECRET`
- `OKX_API_KEY`, `OKX_API_SECRET`, `OKX_API_PASSPHRASE`
- `BITGET_API_KEY`, `BITGET_API_SECRET`, `BITGET_API_PASSPHRASE`
- `PHOENIXQUANT_LOG_LEVEL`

Run `phoenixquant --help` to see the full list of options.

## Project Structure

```
src/phoenixquant/
├── __init__.py
├── bot.py
├── app.py
├── config.py
├── feeds.py
├── indicators.py
├── logging_config.py
└── main.py
```

- `config.py` holds dataclasses and parameter presets.
- `app.py` coordinates the exchange, realtime feeds, and bot lifecycle using an async
  context manager.
- `bot.py` contains the core state machine and trading logic with detailed logging of
  trigger evaluations and order management.
- `feeds.py` manages realtime public and private websocket streams.
- `indicators.py` provides the EMA, RSI and volume recovery helpers.
- `logging_config.py` configures JSON formatted logging output.
- `main.py` exposes the CLI entry point and wiring for the bot.

## Additional Quantitative Strategies

While PhoenixQuant focuses on elastic rebound behaviour on perpetual futures, the
quantitative trading landscape is broad. Teams commonly mix and match tactics
such as:

- **Trend following & momentum** – trade in the direction of medium/long-term
  price trends using moving averages, breakout detection, or cross-asset
  relative strength overlays.
- **Mean reversion** – exploit short-term price dislocations around statistical
  baselines (z-score bands, Bollinger, Ornstein–Uhlenbeck models) across single
  assets or correlated pairs.
- **Statistical arbitrage** – build market-neutral portfolios via factor models
  or cointegration relationships, rebalancing when spreads deviate from their
  learned equilibria.
- **Market making & liquidity provision** – quote two-sided limit orders with
  dynamic inventory and spread control to capture bid/ask edge while managing
  adverse selection risk.
- **Event-driven** – react to scheduled announcements (macro releases, funding
  rolls, earnings) or blockchain-specific signals (on-chain flows, liquidation
  cascades) with tailored execution playbooks.
- **Machine learning & reinforcement learning** – ingest high-dimensional
  features (order book states, alt data) to forecast returns or optimise policy
  decisions through supervised or RL frameworks.
- **Options & volatility strategies** – price and hedge derivatives via implied
  volatility surfaces, gamma scalping, or dispersion trades tied to
  cryptocurrency options markets.

Each category can be adapted to the exchange/instrument mix you operate on and
augmented with risk overlays (volatility targeting, Kelly sizing, drawdown
limits) to fit your capital profile.

