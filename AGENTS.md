# Repository Guidelines

## Project Structure & Module Organization
- Core library: `phoenix_quant/backtest/` (engine, data, runner, analyzer) plus `phoenix_quant/strategies/` (current baseline `elastic_dip.py`).
- CLI helpers (`run_backtest.py`, `batch_backtest.py`, `visualize_backtest.py`, `profile_backtest.py`) drive single runs, batch comparisons, charting, and profiling.
- Configuration YAMLs live in `configs/`; cached market data and generated CSV/PNG reports stay under `data/` or the repo root.

## Build, Test, and Development Commands
- `pip install -e .` — install dependencies against Python ≥3.12.
- `python run_backtest.py --config configs/elastic_dip.yaml` — run the baseline config and print key metrics.
- `python batch_backtest.py` — batch all curated configs and write `backtest_comparison.csv`.
- `python visualize_backtest.py --config <path>` — render equity curves and trade heatmaps for review.
- `bash quick_start.sh` — optional bootstrap that chains install + sample run; update it if CLI flags change.

## Coding Style & Naming Conventions
- PEP 8, 4-space indentation, and type hints; docstrings may stay bilingual but should be concise.
- Prefer `dataclass` models for config/state objects, mirroring `config.py` and `backtest/engine.py`.
- Functions and modules use snake_case; classes use PascalCase (`BacktestEngine`, `ElasticDipStrategy`).
- Expose CLIs through `argparse` with explicit long-form flags.

## Testing Guidelines
- Treat scripted backtests as regression tests: run `python run_backtest.py --config configs/elastic_dip.yaml` and review console stats plus new CSV/PNG artifacts.
- After batch-related changes, confirm `python batch_backtest.py` succeeds and `backtest_comparison.csv` includes expected metrics.
- New pure-python utilities should ship with `pytest` coverage in `tests/`, mirroring the package structure (`tests/backtest/test_engine.py`).
- Log deterministic seeds or extra data requirements inside the corresponding YAML so reviewers can replay offline.

## Commit & Pull Request Guidelines
- Prefer Conventional Commit prefixes (`feat:`, `fix:`, `refactor:`) as in `9ad2d8c feat: adapt elastic dip signal to volatility`; keep subjects ≤72 characters.
- Squash WIP commits before opening a PR; reserve merge commits for long-running branches.
- PR descriptions should list touched modules, include relevant command output or screenshots, and point to updated configs/data.
- Reference issues when available and note any external data or environment steps reviewers must perform.

## Configuration & Data Notes
- Version reusable YAMLs in `configs/` instead of mutating shared baselines.
- Keep cached market data in `data/`; never commit credentials. Document required env vars and prefer cached CSV fallbacks.

## Answer Style
Answer by Chinese.