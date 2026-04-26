"""Sepsis-3 label creation.

Suspected infection window: antibiotic order + blood culture (microbiology
event) within 72 h of each other. SOFA increase ≥ SOFA_THRESHOLD vs the
patient's baseline (first hourly snapshot in their stay) flags a sepsis
candidate; the onset is the first hour where both conditions hold.

Label = 1 for hours within SEPSIS_LOOKAHEAD_HOURS *before* onset; hours at or
after onset are excluded so the classifier can't learn the onset itself.

Uses chunked reads + column subsetting so it scales to the full 3.5 GB
prescriptions.csv on a laptop.
"""

from __future__ import annotations

import re

import pandas as pd

from config import (
    ANTIBIOTIC_KEYWORDS,
    CHUNKSIZE,
    MIMIC_HOSP_DIR,
    PROCESSED_DIR,
    SEPSIS_LOOKAHEAD_HOURS,
    SOFA_THRESHOLD,
)
from utils import mimic_file


def _load_antibiotics() -> pd.DataFrame:
    rx_path, rx_comp = mimic_file(MIMIC_HOSP_DIR, "prescriptions")
    pattern = re.compile("|".join(ANTIBIOTIC_KEYWORDS), flags=re.IGNORECASE)
    keep = []
    for chunk in pd.read_csv(
        rx_path,
        compression=rx_comp,
        usecols=["hadm_id", "starttime", "drug"],
        chunksize=CHUNKSIZE,
        low_memory=False,
    ):
        m = chunk["drug"].astype(str).str.contains(pattern, na=False)
        if m.any():
            keep.append(chunk.loc[m, ["hadm_id", "starttime"]])
    if not keep:
        return pd.DataFrame(columns=["hadm_id", "starttime"])
    abx = pd.concat(keep, ignore_index=True).dropna()
    abx["starttime"] = pd.to_datetime(abx["starttime"], errors="coerce")
    return abx.dropna(subset=["starttime"])


def _load_cultures() -> pd.DataFrame:
    micro_path, micro_comp = mimic_file(MIMIC_HOSP_DIR, "microbiologyevents")
    micro = pd.read_csv(
        micro_path,
        compression=micro_comp,
        usecols=["hadm_id", "charttime"],
        low_memory=False,
    ).dropna()
    micro["charttime"] = pd.to_datetime(micro["charttime"], errors="coerce")
    return micro.dropna(subset=["charttime"])


def _suspected_infection_windows() -> pd.DataFrame:
    abx = _load_antibiotics()
    micro = _load_cultures()
    if abx.empty or micro.empty:
        return pd.DataFrame(columns=["hadm_id", "suspected_time"])
    merged = abx.merge(micro, on="hadm_id", how="inner")
    dt_h = (merged["starttime"] - merged["charttime"]).abs().dt.total_seconds() / 3600
    hits = merged.loc[dt_h <= 72, ["hadm_id", "starttime"]]
    hits = hits.rename(columns={"starttime": "suspected_time"})
    # Earliest suspected time per admission is enough.
    hits = hits.groupby("hadm_id", as_index=False)["suspected_time"].min()
    return hits


def create_sepsis_labels() -> pd.DataFrame:
    feats = pd.read_parquet(PROCESSED_DIR / "features.parquet").sort_values(
        ["stay_id", "charttime"]
    )
    feats["charttime"] = pd.to_datetime(feats["charttime"])
    baseline = feats.groupby("stay_id")["sofa_total"].transform("first")
    feats["sofa_delta"] = feats["sofa_total"] - baseline
    feats["sepsis_candidate"] = feats["sofa_delta"] >= SOFA_THRESHOLD

    inf = _suspected_infection_windows()
    feats = feats.merge(inf, on="hadm_id", how="left")
    feats["suspected_time"] = pd.to_datetime(feats["suspected_time"], errors="coerce")
    feats["sepsis_onset"] = (
        feats["sepsis_candidate"]
        & feats["suspected_time"].notna()
        & (feats["charttime"] >= feats["suspected_time"])
    )

    onset = feats[feats["sepsis_onset"]].groupby("stay_id")["charttime"].min().rename("onset_time")
    feats = feats.merge(onset, on="stay_id", how="left")

    hours_to_onset = (feats["onset_time"] - feats["charttime"]).dt.total_seconds() / 3600
    feats["label"] = ((hours_to_onset >= 0) & (hours_to_onset <= SEPSIS_LOOKAHEAD_HOURS)).astype(int)
    feats = feats[(feats["onset_time"].isna()) | (feats["charttime"] < feats["onset_time"])].copy()

    label_df = feats[["stay_id", "charttime", "label"]]
    label_df.to_parquet(PROCESSED_DIR / "labels_sepsis.parquet", index=False)
    pos_rate = float(label_df["label"].mean()) if len(label_df) else 0.0
    n_onsets = int(onset.notna().sum())
    with open(PROCESSED_DIR / "labels_summary.txt", "w", encoding="utf-8") as f:
        f.write(f"positive_rate={pos_rate:.4f}\n")
        f.write(f"n_rows={len(label_df)}\n")
        f.write(f"n_stays_with_onset={n_onsets}\n")
    return label_df
