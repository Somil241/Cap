"""LOS labels.

Joins each (stay_id, charttime) row with the icustay outtime to produce a true
'remaining LOS in hours' target. Clipped at LOS_MAX_DAYS * 24 to control
heavy-tailed outliers.
"""

from __future__ import annotations

import pandas as pd

from config import LOS_MAX_DAYS, MIMIC_ICU_DIR, PROCESSED_DIR
from utils import mimic_file


def create_los_labels() -> pd.DataFrame:
    icu_path, icu_comp = mimic_file(MIMIC_ICU_DIR, "icustays")
    icu = pd.read_csv(icu_path, compression=icu_comp, usecols=["stay_id", "intime", "outtime"])
    icu["intime"] = pd.to_datetime(icu["intime"])
    icu["outtime"] = pd.to_datetime(icu["outtime"])

    feats = pd.read_parquet(PROCESSED_DIR / "features.parquet")[["stay_id", "charttime"]]
    feats["charttime"] = pd.to_datetime(feats["charttime"])
    df = feats.merge(icu, on="stay_id", how="inner")

    remaining = (df["outtime"] - df["charttime"]).dt.total_seconds() / 3600
    df["remaining_los_hours"] = remaining.clip(lower=1, upper=LOS_MAX_DAYS * 24)
    df["los_event"] = (remaining <= LOS_MAX_DAYS * 24).astype(int)

    out = df[["stay_id", "charttime", "remaining_los_hours", "los_event"]]
    out.to_parquet(PROCESSED_DIR / "labels_los.parquet", index=False)
    return out
