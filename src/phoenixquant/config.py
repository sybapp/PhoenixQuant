"""Configuration structures and presets for PhoenixQuant."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass(slots=True)
class BotParameters:
    """Parameter set that controls the behaviour of :class:`ElasticDipBot`."""

    timeframe: str
    poll_sec: float
    drop_pct_single: float
    drop_pct_window: float
    window_min: int
    ema_fast: int
    ema_slow: int
    vol_shrink_ratio: float
    rsi_period: int
    rsi_oversold: float
    funding_extreme_neg: float
    liq_window_sec: int
    liq_notional_threshold: float
    delayed_trigger_pct: float
    delayed_window_sec: float
    layer_pcts: List[float]
    layer_pos_ratio: List[float]
    total_capital: float
    max_account_ratio: float
    take_profit_pct: float
    hard_stop_extra: float
    sl_time_grace_sec: float
    vol_recover_ma_short: int
    vol_recover_ma_long: int
    vol_recover_ratio: float
    tick_vol_ratio: float

    def log_parameters(self, logger) -> None:
        """Emit all parameters in a single structured log statement."""

        logger.info(
            "Loaded bot parameters",
            extra={
                "params": {
                    "timeframe": self.timeframe,
                    "poll_sec": self.poll_sec,
                    "drop_pct_single": self.drop_pct_single,
                    "drop_pct_window": self.drop_pct_window,
                    "window_min": self.window_min,
                    "ema_fast": self.ema_fast,
                    "ema_slow": self.ema_slow,
                    "vol_shrink_ratio": self.vol_shrink_ratio,
                    "rsi_period": self.rsi_period,
                    "rsi_oversold": self.rsi_oversold,
                    "funding_extreme_neg": self.funding_extreme_neg,
                    "liq_window_sec": self.liq_window_sec,
                    "liq_notional_threshold": self.liq_notional_threshold,
                    "delayed_trigger_pct": self.delayed_trigger_pct,
                    "delayed_window_sec": self.delayed_window_sec,
                    "layer_pcts": self.layer_pcts,
                    "layer_pos_ratio": self.layer_pos_ratio,
                    "total_capital": self.total_capital,
                    "max_account_ratio": self.max_account_ratio,
                    "take_profit_pct": self.take_profit_pct,
                    "hard_stop_extra": self.hard_stop_extra,
                    "sl_time_grace_sec": self.sl_time_grace_sec,
                    "vol_recover_ma_short": self.vol_recover_ma_short,
                    "vol_recover_ma_long": self.vol_recover_ma_long,
                    "vol_recover_ratio": self.vol_recover_ratio,
                    "tick_vol_ratio": self.tick_vol_ratio,
                }
            },
        )


def build_parameters(raw: Dict[str, object]) -> BotParameters:
    """Build :class:`BotParameters` from a dictionary."""

    return BotParameters(**raw)


PARAM_PRESETS: Dict[str, BotParameters] = {
    key: build_parameters(value)
    for key, value in {
        "BTCUSDT": {
            "timeframe": "1m",
            "poll_sec": 2,
            "drop_pct_single": 1.0,
            "drop_pct_window": 3.0,
            "window_min": 5,
            "ema_fast": 20,
            "ema_slow": 60,
            "vol_shrink_ratio": 0.6,
            "rsi_period": 14,
            "rsi_oversold": 25.0,
            "funding_extreme_neg": -0.05,
            "liq_window_sec": 60,
            "liq_notional_threshold": 8_000_000,
            "delayed_trigger_pct": 1.0,
            "delayed_window_sec": 60 * 60 * 12,
            "layer_pcts": [0.8, 1.4, 2.0, 2.6, 3.3],
            "layer_pos_ratio": [0.10, 0.15, 0.20, 0.25, 0.30],
            "total_capital": 1000,
            "max_account_ratio": 0.30,
            "take_profit_pct": 1.0,
            "hard_stop_extra": 0.8,
            "sl_time_grace_sec": 30,
            "vol_recover_ma_short": 5,
            "vol_recover_ma_long": 20,
            "vol_recover_ratio": 1.15,
            "tick_vol_ratio": 1.30,
        },
        "ETHUSDT": {
            "timeframe": "1m",
            "poll_sec": 2,
            "drop_pct_single": 1.0,
            "drop_pct_window": 3.0,
            "window_min": 5,
            "ema_fast": 20,
            "ema_slow": 60,
            "vol_shrink_ratio": 0.6,
            "rsi_period": 14,
            "rsi_oversold": 25.0,
            "funding_extreme_neg": -0.05,
            "liq_window_sec": 60,
            "liq_notional_threshold": 4_000_000,
            "delayed_trigger_pct": 1.0,
            "delayed_window_sec": 60 * 60 * 12,
            "layer_pcts": [0.9, 1.6, 2.3, 3.0, 3.8],
            "layer_pos_ratio": [0.10, 0.15, 0.20, 0.25, 0.30],
            "total_capital": 800,
            "max_account_ratio": 0.30,
            "take_profit_pct": 1.0,
            "hard_stop_extra": 0.9,
            "sl_time_grace_sec": 30,
            "vol_recover_ma_short": 5,
            "vol_recover_ma_long": 20,
            "vol_recover_ratio": 1.15,
            "tick_vol_ratio": 1.30,
        },
        "SOLUSDT": {
            "timeframe": "1m",
            "poll_sec": 2,
            "drop_pct_single": 1.2,
            "drop_pct_window": 3.8,
            "window_min": 5,
            "ema_fast": 20,
            "ema_slow": 60,
            "vol_shrink_ratio": 0.65,
            "rsi_period": 14,
            "rsi_oversold": 24.0,
            "funding_extreme_neg": -0.08,
            "liq_window_sec": 60,
            "liq_notional_threshold": 2_000_000,
            "delayed_trigger_pct": 1.2,
            "delayed_window_sec": 60 * 60 * 10,
            "layer_pcts": [1.0, 1.8, 2.6, 3.5, 4.5],
            "layer_pos_ratio": [0.08, 0.15, 0.22, 0.25, 0.30],
            "total_capital": 600,
            "max_account_ratio": 0.25,
            "take_profit_pct": 1.2,
            "hard_stop_extra": 1.1,
            "sl_time_grace_sec": 25,
            "vol_recover_ma_short": 5,
            "vol_recover_ma_long": 20,
            "vol_recover_ratio": 1.20,
            "tick_vol_ratio": 1.40,
        },
        "BNBUSDT": {
            "timeframe": "1m",
            "poll_sec": 2,
            "drop_pct_single": 1.0,
            "drop_pct_window": 3.2,
            "window_min": 5,
            "ema_fast": 20,
            "ema_slow": 60,
            "vol_shrink_ratio": 0.6,
            "rsi_period": 14,
            "rsi_oversold": 25.0,
            "funding_extreme_neg": -0.06,
            "liq_window_sec": 60,
            "liq_notional_threshold": 2_500_000,
            "delayed_trigger_pct": 1.0,
            "delayed_window_sec": 60 * 60 * 10,
            "layer_pcts": [0.9, 1.6, 2.3, 3.0, 3.8],
            "layer_pos_ratio": [0.10, 0.15, 0.20, 0.25, 0.30],
            "total_capital": 600,
            "max_account_ratio": 0.25,
            "take_profit_pct": 1.0,
            "hard_stop_extra": 0.9,
            "sl_time_grace_sec": 25,
            "vol_recover_ma_short": 5,
            "vol_recover_ma_long": 20,
            "vol_recover_ratio": 1.18,
            "tick_vol_ratio": 1.35,
        },
        "DOGEUSDT": {
            "timeframe": "1m",
            "poll_sec": 2,
            "drop_pct_single": 1.4,
            "drop_pct_window": 4.2,
            "window_min": 5,
            "ema_fast": 20,
            "ema_slow": 60,
            "vol_shrink_ratio": 0.7,
            "rsi_period": 14,
            "rsi_oversold": 23.0,
            "funding_extreme_neg": -0.10,
            "liq_window_sec": 60,
            "liq_notional_threshold": 1_200_000,
            "delayed_trigger_pct": 1.3,
            "delayed_window_sec": 60 * 60 * 8,
            "layer_pcts": [1.2, 2.0, 3.0, 4.2, 5.5],
            "layer_pos_ratio": [0.08, 0.12, 0.20, 0.25, 0.35],
            "total_capital": 400,
            "max_account_ratio": 0.20,
            "take_profit_pct": 1.5,
            "hard_stop_extra": 1.3,
            "sl_time_grace_sec": 20,
            "vol_recover_ma_short": 5,
            "vol_recover_ma_long": 20,
            "vol_recover_ratio": 1.25,
            "tick_vol_ratio": 1.50,
        },
        "XRPUSDT": {
            "timeframe": "1m",
            "poll_sec": 2,
            "drop_pct_single": 1.2,
            "drop_pct_window": 3.6,
            "window_min": 5,
            "ema_fast": 20,
            "ema_slow": 60,
            "vol_shrink_ratio": 0.65,
            "rsi_period": 14,
            "rsi_oversold": 24.0,
            "funding_extreme_neg": -0.08,
            "liq_window_sec": 60,
            "liq_notional_threshold": 1_500_000,
            "delayed_trigger_pct": 1.2,
            "delayed_window_sec": 60 * 60 * 10,
            "layer_pcts": [1.0, 1.8, 2.6, 3.5, 4.5],
            "layer_pos_ratio": [0.08, 0.15, 0.22, 0.25, 0.30],
            "total_capital": 500,
            "max_account_ratio": 0.25,
            "take_profit_pct": 1.2,
            "hard_stop_extra": 1.0,
            "sl_time_grace_sec": 25,
            "vol_recover_ma_short": 5,
            "vol_recover_ma_long": 20,
            "vol_recover_ratio": 1.20,
            "tick_vol_ratio": 1.40,
        },
        "ALTS_MAJOR": {
            "timeframe": "1m",
            "poll_sec": 2,
            "drop_pct_single": 1.2,
            "drop_pct_window": 3.8,
            "window_min": 5,
            "ema_fast": 20,
            "ema_slow": 60,
            "vol_shrink_ratio": 0.65,
            "rsi_period": 14,
            "rsi_oversold": 24.0,
            "funding_extreme_neg": -0.08,
            "liq_window_sec": 60,
            "liq_notional_threshold": 1_000_000,
            "delayed_trigger_pct": 1.2,
            "delayed_window_sec": 60 * 60 * 10,
            "layer_pcts": [1.0, 1.8, 2.6, 3.5, 4.5],
            "layer_pos_ratio": [0.08, 0.15, 0.22, 0.25, 0.30],
            "total_capital": 400,
            "max_account_ratio": 0.25,
            "take_profit_pct": 1.2,
            "hard_stop_extra": 1.1,
            "sl_time_grace_sec": 25,
            "vol_recover_ma_short": 5,
            "vol_recover_ma_long": 20,
            "vol_recover_ratio": 1.20,
            "tick_vol_ratio": 1.45,
        },
    }.items()
}

