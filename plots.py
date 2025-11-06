from typing import Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def plot_speed_time(pos: pd.DataFrame, speed_kmh: np.ndarray) -> go.Figure:
    ts = pos['timestamp'].to_numpy(dtype=np.int64)
    if ts.size != speed_kmh.size:
        n = min(ts.size, speed_kmh.size)
        ts = ts[:n]
        speed_kmh = speed_kmh[:n]
    t_s = (ts - ts[0]) / 1000.0
    fig = px.line(x=t_s, y=speed_kmh, labels={'x': 'Time (s)', 'y': 'Speed (km/h)'}, title='Speed vs Time')
    return fig


def plot_elevation(pos: pd.DataFrame) -> Optional[go.Figure]:
    if 'altitude' not in pos.columns:
        return None
    ts = pos['timestamp'].to_numpy(dtype=np.int64)
    t_s = (ts - ts[0]) / 1000.0
    fig = px.line(x=t_s, y=pos['altitude'], labels={'x': 'Time (s)', 'y': 'Altitude (m)'}, title='Elevation vs Time')
    return fig


def plot_map(pos: pd.DataFrame) -> go.Figure:
    fig = px.line_mapbox(
        pos,
        lat='latitude',
        lon='longitude',
        zoom=14,
        height=500,
        title='GPS Track'
    )
    fig.update_layout(mapbox_style='open-street-map')
    return fig


