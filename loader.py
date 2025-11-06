import os
import zipfile
from dataclasses import dataclass
from typing import List, Dict, Optional

import pandas as pd


@dataclass
class Session:
    folder: str
    session_key: str
    files: List[str]
    pos: Optional[pd.DataFrame]
    accel: Optional[pd.DataFrame]
    orient: Optional[pd.DataFrame]
    angvel: Optional[pd.DataFrame]
    magfield: Optional[pd.DataFrame]


def _ensure_unzipped(folder: str) -> None:
    for name in os.listdir(folder):
        if name.lower().endswith('.zip'):
            zip_path = os.path.join(folder, name)
            out_dir = os.path.join(folder, f"{name}_unzipped")
            if not os.path.isdir(out_dir):
                try:
                    with zipfile.ZipFile(zip_path, 'r') as zf:
                        zf.extractall(out_dir)
                except Exception:
                    pass


def _collect_csvs(roots: List[str]) -> List[str]:
    csvs: List[str] = []
    for root in roots:
        if not os.path.isdir(root):
            continue
        for name in os.listdir(root):
            if name.startswith('sensorlog_') and name.endswith('.csv'):
                csvs.append(os.path.join(root, name))
    # Deduplicate paths
    unique = sorted(set(csvs))
    return unique


def _group_by_session(csvs: List[str]) -> Dict[str, List[str]]:
    groups: Dict[str, List[str]] = {}
    for path in csvs:
        base = os.path.basename(path)
        stem = os.path.splitext(base)[0]
        parts = stem.split('_')
        if len(parts) < 3:
            key = stem
        else:
            key = f"{parts[-2]}_{parts[-1]}"
        groups.setdefault(key, []).append(path)
    return groups


def _read_if_exists(files: List[str], kind: str) -> Optional[pd.DataFrame]:
    for path in files:
        if f"_{kind}_" in os.path.basename(path) or f"sensorlog_{kind}_" in os.path.basename(path):
            try:
                return pd.read_csv(path)
            except Exception:
                return None
    return None


def load_sessions(input_folders: List[str]) -> List[Session]:
    sessions: List[Session] = []
    seen_keys: Dict[str, bool] = {}
    for folder in input_folders:
        if not os.path.isdir(folder):
            continue
        _ensure_unzipped(folder)
        roots = [folder] + [
            os.path.join(folder, d)
            for d in os.listdir(folder)
            if d.endswith('_unzipped') and os.path.isdir(os.path.join(folder, d))
        ]
        csvs = _collect_csvs(roots)
        if not csvs:
            continue
        groups = _group_by_session(csvs)
        for key, files in groups.items():
            sess_key = f"{folder} | {key}"
            if seen_keys.get(sess_key):
                continue
            seen_keys[sess_key] = True
            sess = Session(
                folder=folder,
                session_key=sess_key,
                files=files,
                pos=_read_if_exists(files, 'pos'),
                accel=_read_if_exists(files, 'accel'),
                orient=_read_if_exists(files, 'orient'),
                angvel=_read_if_exists(files, 'angvel'),
                magfield=_read_if_exists(files, 'magfield'),
            )
            sessions.append(sess)
    return sessions


