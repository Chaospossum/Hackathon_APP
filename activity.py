from typing import Tuple

import numpy as np


def guess_activity(avg_speed_kmh: float) -> str:
    if not np.isfinite(avg_speed_kmh) or avg_speed_kmh < 1.5:
        return 'sitting/idle'
    if avg_speed_kmh < 6.0:
        return 'walking'
    if avg_speed_kmh < 12.0:
        return 'running'
    return 'cycling'


def ema(values: np.ndarray, alpha: float) -> np.ndarray:
    y = np.zeros_like(values, dtype=float)
    if values.size == 0:
        return y
    y[0] = float(values[0])
    for i in range(1, values.size):
        y[i] = alpha * float(values[i]) + (1.0 - alpha) * y[i - 1]
    return y


def predict_speed_next_kmh(speed_kmh: np.ndarray, dt_sec: float, horizon_sec: int = 30) -> float:
    if speed_kmh.size < 6 or dt_sec <= 0:
        return float('nan')
    alpha = min(0.5, max(0.05, dt_sec / 5.0))
    smooth = ema(speed_kmh, alpha)
    x = smooth[1:]
    y = smooth[:-1]
    denom = float(np.dot(y, y)) + 1e-12
    phi = float(np.dot(y, x)) / denom
    steps = max(1, int(round(horizon_sec / max(0.1, dt_sec))))
    s = float(smooth[-1])
    mean_tail = float(np.mean(smooth[max(0, smooth.size - 50):]))
    for _ in range(steps):
        s = phi * s + (1.0 - phi) * mean_tail
    return max(0.0, s)


