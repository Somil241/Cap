"""Derive ventilator labels from procedureevents.

Two outputs:
  - labels_vent.parquet : (stay_id, charttime, vent_label) where vent_label = 1
    if mechanical ventilation begins within VENT_LOOKAHEAD_HOURS of charttime.
    Hours at/after vent start are excluded so the classifier doesn't learn the
    onset itself.
  - vent_episodes.parquet : per-stay ventilation episodes with start/end and
    duration_hours, used by the duration (Cox) model.
"""

from __future__ import annotations

import pandas as pd

from config import MIMIC_ICU_DIR, PROCESSED_DIR, VENT_ITEMIDS, VENT_LOOKAHEAD_HOURS
from utils import mimic_file


def _load_vent_episodes() -> pd.DataFrame:
    proc_path, proc_comp = mimic_file(MIMIC_ICU_DIR, "procedureevents")
    cols = ["stay_id", "starttime", "endtime", "itemid"]
    proc = pd.read_csv(proc_path, compression=proc_comp, usecols=cols)
    proc = proc[proc["itemid"].isin(VENT_ITEMIDS)].copy()
    proc["starttime"] = pd.to_datetime(proc["starttime"], errors="coerce")
    proc["endtime"] = pd.to_datetime(proc["endtime"], errors="coerce")
    proc = proc.dropna(subset=["starttime"])
    proc["duration_hours"] = (proc["endtime"] - proc["starttime"]).dt.total_seconds() / 3600
    proc["duration_hours"] = proc["duration_hours"].clip(lower=0.0)
    proc.to_parquet(PROCESSED_DIR / "vent_episodes.parquet", index=False)
    return proc


def create_vent_labels() -> pd.DataFrame:
    feats = pd.read_parquet(PROCESSED_DIR / "features.parquet")
    feats["charttime"] = pd.to_datetime(feats["charttime"])
    feats = feats[["stay_id", "charttime"]].drop_duplicates().sort_values(["stay_id", "charttime"])

    episodes = _load_vent_episodes()
    if episodes.empty:
        labels = feats.copy()
        labels["vent_label"] = 0
        labels.to_parquet(PROCESSED_DIR / "labels_vent.parquet", index=False)
        return labels

    first_start = episodes.groupby("stay_id")["starttime"].min().rename("first_vent_start")
    feats = feats.merge(first_start, on="stay_id", how="left")
    hrs = (feats["first_vent_start"] - feats["charttime"]).dt.total_seconds() / 3600
    feats["vent_label"] = ((hrs >= 0) & (hrs <= VENT_LOOKAHEAD_HOURS)).astype(int)
    # Exclude rows at/after first vent start so we don't leak onset.
    feats = feats[feats["first_vent_start"].isna() | (feats["charttime"] < feats["first_vent_start"])].copy()
    labels = feats[["stay_id", "charttime", "vent_label"]]
    labels.to_parquet(PROCESSED_DIR / "labels_vent.parquet", index=False)
    return labels
