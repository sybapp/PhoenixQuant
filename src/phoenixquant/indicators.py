"""Technical indicator utilities used by the PhoenixQuant bot."""
from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np


def ema(values: Sequence[float], period: int) -> np.ndarray:
    """Return the exponential moving average for ``values``."""

    arr = np.asarray(values, dtype=float)
    if len(arr) < period:
        return np.array([])
    k = 2 / (period + 1)
    ema_values = np.zeros_like(arr)
    ema_values[0] = arr[0]
    for idx in range(1, len(arr)):
        ema_values[idx] = arr[idx] * k + ema_values[idx - 1] * (1 - k)
    return ema_values


def rsi(values: Sequence[float], period: int = 14) -> float:
    """Compute the relative strength index for ``values``."""

    arr = np.asarray(values, dtype=float)
    if len(arr) < period + 1:
        return float("nan")
    deltas = np.diff(arr)
    gains = np.clip(deltas, 0, None)
    losses = -np.clip(deltas, None, 0)
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / (avg_loss + 1e-12)
    rsis = [100 - 100 / (1 + rs)]
    for idx in range(period, len(deltas)):
        avg_gain = (avg_gain * (period - 1) + gains[idx]) / period
        avg_loss = (avg_loss * (period - 1) + losses[idx]) / period
        if avg_loss == 0:
            rsis.append(100.0)
            continue
        rs = avg_gain / (avg_loss + 1e-12)
        rsis.append(100 - 100 / (1 + rs))
    return rsis[-1] if rsis else float("nan")


def volume_recovered(
    candles: Iterable[Sequence[float]],
    ma_short: int = 5,
    ma_long: int = 20,
    ratio: float = 1.15,
    tick_ratio: float | None = None,
) -> bool:
    """Return ``True`` when the volume recovery filter passes."""

    vols = [candle[5] for candle in candles]
    if len(vols) < max(ma_short, ma_long) + 1:
        return False
    ma_short_value = float(np.mean(vols[-ma_short:]))
    ma_long_value = float(np.mean(vols[-ma_long:]))
    condition_a = ma_short_value > ma_long_value * ratio
    if tick_ratio is None:
        return condition_a
    condition_b = vols[-1] > ma_long_value * tick_ratio
    return condition_a or condition_b

