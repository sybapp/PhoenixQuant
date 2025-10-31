# PhoenixQuant

PhoenixQuant is an async elastic dip trading bot for Binance Futures. The project now
follows a standard Python package layout with a CLI entry point and structured JSON
logging for easier observability.

## Installation

```bash
pip install -e .
```

## Usage

```bash
phoenixquant BTC/USDT btcusdt --preset BTCUSDT --dry-run
```

Environment variables can be used to supply credentials and logging configuration:

- `BINANCE_API_KEY`
- `BINANCE_API_SECRET`
- `PHOENIXQUANT_LOG_LEVEL`

Run `phoenixquant --help` to see the full list of options.

## Project Structure

```
src/phoenixquant/
├── __init__.py
├── bot.py
├── config.py
├── feeds.py
├── indicators.py
├── logging_config.py
└── main.py
```

- `config.py` holds dataclasses and parameter presets.
- `bot.py` contains the core state machine and trading logic with detailed logging of
  trigger evaluations and order management.
- `feeds.py` manages realtime public and private websocket streams.
- `indicators.py` provides the EMA, RSI and volume recovery helpers.
- `logging_config.py` configures JSON formatted logging output.
- `main.py` exposes the CLI entry point and wiring for the bot.

