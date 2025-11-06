from typing import Tuple

import numpy as np
import pandas as pd


def haversine_distance_km(lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    R = 6371.0
    p1 = np.radians(lat1)
    p2 = np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dl = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2.0) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2.0) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c


def track_distance_speed_from_gps(pos: pd.DataFrame) -> Tuple[float, np.ndarray]:
    lat = pos['latitude'].to_numpy()
    lon = pos['longitude'].to_numpy()
    ts = pos['timestamp'].to_numpy(dtype=np.int64)
    if lat.size < 2:
        return 0.0, np.zeros_like(lat, dtype=float)
    d_km = haversine_distance_km(lat[:-1], lon[:-1], lat[1:], lon[1:])
    d_m = d_km * 1000.0
    dt_s = np.maximum(1e-6, (ts[1:] - ts[:-1]) / 1000.0)
    speed_ms = np.concatenate([[0.0], d_m / dt_s])
    total_km = float(np.nansum(d_km))
    return total_km, speed_ms


def elevation_gain_m(pos: pd.DataFrame) -> float:
    if 'altitude' not in pos.columns:
        return 0.0
    alt = pos['altitude'].to_numpy(dtype=float)
    diffs = np.diff(alt)
    gain = np.sum(np.maximum(0.0, diffs)) if diffs.size else 0.0
    return float(gain)


def floors_from_gain(gain_m: float) -> float:
    return float(gain_m / 3.0)


def avg_speed_kmh(total_km: float, pos: pd.DataFrame) -> float:
    ts = pos['timestamp'].to_numpy(dtype=np.int64)
    if ts.size < 2:
        return float('nan')
    duration_h = (ts[-1] - ts[0]) / 1000.0 / 3600.0
    if duration_h <= 0:
        return float('nan')
    return total_km / duration_h


def minutes_from_pos(pos: pd.DataFrame) -> float:
    ts = pos['timestamp'].to_numpy(dtype=np.int64)
    if ts.size < 2:
        return 0.0
    return (ts[-1] - ts[0]) / 1000.0 / 60.0


def met_for_activity(activity: str, avg_speed_kmh_val: float) -> float:
    if activity == 'sitting/idle':
        return 1.5
    if activity == 'walking':
        return max(3.0, min(5.0, 3.0 + 0.4 * (avg_speed_kmh_val - 3)))
    if activity == 'running':
        return max(7.0, min(12.0, 6.0 + 0.6 * avg_speed_kmh_val))
    if activity == 'cycling':
        return max(6.0, min(12.0, 2.0 + 0.5 * avg_speed_kmh_val))
    if activity == 'stairs':
        # Typical MET for stair climbing is high; keep a conservative fixed value
        return 8.0
    return 3.5


def calories_from_met(met: float, minutes: float, weight_kg: float) -> float:
    return float(met * 3.5 * weight_kg / 200.0 * max(0.0, minutes))


def estimate_life_expectancy_gain(weekly_met_min: float) -> float:
    w = weekly_met_min
    if not np.isfinite(w) or w <= 0:
        return 0.0
    if w < 150:
        return 0.5 * (w / 150.0)
    if w < 300:
        return 1.0 + (w - 150.0) / 150.0 * 2.0
    if w < 600:
        return 3.0 + (w - 300.0) / 300.0 * 1.2
    extra = min(w - 600.0, 900.0)
    return 4.2 + (extra / 900.0) * 2.5


