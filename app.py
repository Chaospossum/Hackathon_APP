import os
import sys
from typing import List, Dict

import numpy as np
import pandas as pd
import streamlit as st

# Ensure parent directory is on sys.path so 'hackathon_app' is importable
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PARENT_DIR = os.path.dirname(_THIS_DIR)
if _PARENT_DIR not in sys.path:
    sys.path.insert(0, _PARENT_DIR)

from hackathon_app.loader import load_sessions
from hackathon_app.metrics import (
    track_distance_speed_from_gps,
    elevation_gain_m,
    floors_from_gain,
    avg_speed_kmh,
    minutes_from_pos,
    met_for_activity,
    calories_from_met,
    estimate_life_expectancy_gain,
)
from hackathon_app.activity import guess_activity, predict_speed_next_kmh
from hackathon_app.plots import plot_map, plot_speed_time, plot_elevation


st.set_page_config(page_title='Fitness Tracker', layout='wide')
st.title('Fitness Tracker')

st.sidebar.header('Inputs')
default_folders = [
    r'c:\\Users\\lts\\Desktop\\activites',
]

@st.cache_data(show_spinner=False)
def cached_load_sessions(folders: List[str]):
    return load_sessions(folders)

with st.sidebar.form('controls'):
    folder_text = st.text_area('Data folders (one per line)', value='\n'.join(default_folders))
    weight = st.number_input('Weight (kg)', min_value=30.0, max_value=200.0, value=75.0, step=1.0)
    submitted = st.form_submit_button('Analyze')

if submitted:
    folders: List[str] = [line.strip() for line in folder_text.splitlines() if line.strip()]
    with st.spinner('Loading sessions...'):
        sessions = cached_load_sessions(folders)
    if not sessions:
        st.error('No sessions found. Check your folder paths and contents.')
        st.stop()

    summary_rows: List[Dict] = []
    per_session_details = []
    for sess in sessions:
        if sess.pos is None or sess.pos.empty:
            continue
        total_km, speed_ms = track_distance_speed_from_gps(sess.pos)
        speed_kmh = speed_ms * 3.6
        gain_m = elevation_gain_m(sess.pos)
        floors = floors_from_gain(gain_m)
        avg_kmh = avg_speed_kmh(total_km, sess.pos)
        minutes = minutes_from_pos(sess.pos)
        # Classify stairs using vertical activity: floors per minute >= 1 and not fast
        floors_per_min = (floors / minutes) if minutes > 0 else 0.0
        if np.isfinite(floors_per_min) and floors_per_min >= 1.0 and (not np.isfinite(avg_kmh) or avg_kmh < 6.0):
            act_raw = 'stairs'
        else:
            act_raw = guess_activity(avg_kmh)

        # Normalize to only four categories: sitting, walking, running, stairs
        def normalize_activity(name: str) -> str:
            n = (name or '').strip().lower()
            if n in ('stairs', 'stair', 'stairclimb', 'stair climbing'):
                return 'stairs'
            if n in ('running', 'run', 'jogging'):
                return 'running'
            if n in ('walking', 'walk'):
                return 'walking'
            if n in ('sitting/idle', 'idle', 'rest', 'sitting'):
                return 'sitting'
            # Map any other activities (e.g., cycling) to walking by default for consistency
            return 'walking'

        act = normalize_activity(act_raw)
        met = met_for_activity(act, avg_kmh)
        # Estimated steps: different factors by activity; fallback for others
        km = total_km
        if act == 'walking':
            est_steps = int(round(km * 1300))  # ~1.3k steps per km
        elif act == 'running':
            est_steps = int(round(km * 1000))  # ~1k steps per km
        elif act == 'stairs':
            est_steps = int(round(floors * 16))  # ~16 steps per floor
        else:
            est_steps = int(round(km * 1200))
        kcal = calories_from_met(met, minutes, weight)
        dt_sec = max(0.1, float(np.median(np.diff(sess.pos['timestamp'].to_numpy(dtype=np.int64))) / 1000.0))
        pred30 = predict_speed_next_kmh(speed_kmh, dt_sec)

        summary_rows.append({
            'session': sess.session_key,
            'activity': act,
            'duration_min': minutes,
            'distance_km': total_km,
            'elevation_gain_m': gain_m,
            'floors': floors,
            'avg_speed_kmh': avg_kmh,
            'met': met,
            'met_minutes': met * minutes,
            'calories_kcal': kcal,
            'estimated_steps': est_steps,
            'pred_next_30s_speed_kmh': pred30,
        })

        per_session_details.append((sess, speed_kmh))

    if not summary_rows:
        st.error('No position logs found in sessions.')
        st.stop()

    summary = pd.DataFrame(summary_rows)

    # KPI header
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    total_min = float(summary['duration_min'].sum())
    total_km = float(summary['distance_km'].sum())
    total_gain = float(summary['elevation_gain_m'].sum())
    total_floors = float(summary['floors'].sum())
    total_kcal = float(summary['calories_kcal'].sum())
    total_met_min = float(summary['met_minutes'].sum())
    weekly_projected = total_met_min * 7.0
    years_gained = estimate_life_expectancy_gain(weekly_projected)
    k1.metric('Total minutes', f"{total_min:.0f}")
    k2.metric('Total km', f"{total_km:.2f}")
    k3.metric('Elevation gain (m)', f"{total_gain:.0f}")
    k4.metric('Floors', f"{total_floors:.0f}")
    k5.metric('Calories (kcal)', f"{total_kcal:.0f}")
    k6.metric('Projected years (est.)', f"{years_gained:.1f}")

    # Health insights
    with st.expander('Health insights'):
        st.markdown(
            f"Weekly MET-min if repeated daily: **{weekly_projected:.0f}**.\n\n"
            f"Estimated life expectancy impact: **~{years_gained:.1f} years** (illustrative).\n\n"
            "Tips: Aim for 500–1000 MET-min/week; mix walking, stairs, and running for balanced cardio. 🫀\n"
        )

    # Tabs for clean layout
    tab_overview, tab_activities, tab_sessions = st.tabs(['Overview', 'Activities', 'Sessions'])

    with tab_overview:
        st.subheader('Session Summaries')
        st.dataframe(
            summary.sort_values('duration_min', ascending=False)
                   .style.format({'distance_km': '{:.2f}', 'calories_kcal': '{:.0f}', 'elevation_gain_m': '{:.0f}'})
                   .hide(axis='index'),
            use_container_width=True,
        )

    with tab_activities:
        st.subheader('Per-Activity Comparison')
        # Activities are already normalized to the four categories
        summary['activity_norm'] = summary['activity']
        grouped = summary.groupby('activity_norm').agg(
            distance_km=('distance_km', 'sum'),
            duration_min=('duration_min', 'sum'),
            calories_kcal=('calories_kcal', 'sum'),
            elevation_gain_m=('elevation_gain_m', 'sum'),
            floors=('floors', 'sum'),
            estimated_steps=('estimated_steps', 'sum'),
            pred_next_30s_speed_kmh=('pred_next_30s_speed_kmh', 'mean'),
        ).reset_index()
        grouped = grouped.rename(columns={'activity_norm': 'activity'})
        st.dataframe(
            grouped.style.format({'distance_km': '{:.2f}', 'calories_kcal': '{:.0f}', 'elevation_gain_m': '{:.0f}', 'pred_next_30s_speed_kmh': '{:.1f}'}).hide(axis='index'),
            use_container_width=True,
        )
        st.caption('Activity colors: walking 🟩, stairs 🟪, running 🟥, cycling 🟦, idle ⬜')

        # Activity distribution pie
        try:
            import plotly.express as px
            pie = px.pie(grouped, values='duration_min', names='activity', title='Time by Activity')
            st.plotly_chart(pie, use_container_width=True)
        except Exception:
            pass

    with tab_sessions:
        st.subheader('Session Visuals')
        for sess, speed_kmh in per_session_details:
            row = summary[summary['session'] == sess.session_key].iloc[0]
            act = str(row['activity'])
            badge = {
                'walking': '🟩 walking',
                'stairs': '🟪 stairs',
                'running': '🟥 running',
                'cycling': '🟦 cycling',
                'sitting/idle': '⬜ idle',
            }.get(act, act)
            st.markdown(f"**{sess.session_key}** — {badge} · {row['distance_km']:.2f} km · {int(row['estimated_steps']):,} steps · {row['elevation_gain_m']:.0f} m gain")
            cols = st.columns(3)
            with cols[0]:
                try:
                    st.plotly_chart(plot_map(sess.pos), use_container_width=True)
                except Exception:
                    st.write('Map plot not available for this session.')
            with cols[1]:
                st.plotly_chart(plot_speed_time(sess.pos, speed_kmh), use_container_width=True)
            with cols[2]:
                elev_fig = plot_elevation(sess.pos)
                if elev_fig is not None:
                    st.plotly_chart(elev_fig, use_container_width=True)
                else:
                    st.write('No elevation data available.')


