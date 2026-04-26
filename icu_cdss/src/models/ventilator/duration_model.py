"""Vent-duration survival model.

Trains a Cox proportional-hazards model on real ventilation episodes from
procedureevents (start/end → duration). Episodes whose endtime is missing or
zero-duration are right-censored. Covariates are taken from the patient-hour
feature snapshot at vent start.
"""

from __future__ import annotations

import json
import pickle

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter

from config import MODELS_DIR, PROCESSED_DIR

NON_FEATURE = {"label", "vent_label", "charttime", "stay_id", "hadm_id", "subject_id",
               "onset_time", "suspected_time", "first_vent_start", "duration_hours", "event"}


def _features_at_start(features: pd.DataFrame, episodes: pd.DataFrame) -> pd.DataFrame:
    """For each vent episode, take the most recent feature row before start."""
    features = features.copy()
    features["charttime"] = pd.to_datetime(features["charttime"])
    out = []
    for stay_id, eps in episodes.groupby("stay_id"):
        f = features[features["stay_id"] == stay_id].sort_values("charttime")
        if f.empty:
            continue
        for _, ep in eps.iterrows():
            row = f[f["charttime"] <= ep["starttime"]].tail(1)
            if row.empty:
                row = f.head(1)  # fall back to earliest snapshot
            row = row.copy()
            row["duration_hours"] = float(ep["duration_hours"]) if pd.notna(ep["duration_hours"]) else 0.0
            row["event"] = 0 if (pd.isna(ep["endtime"]) or ep["duration_hours"] <= 0) else 1
            out.append(row)
    if not out:
        return pd.DataFrame()
    return pd.concat(out, axis=0, ignore_index=True)


class VentilatorDurationModel:
    def train(self, top_k_features: int = 20) -> dict:
        features = pd.read_parquet(PROCESSED_DIR / "features.parquet")
        eps_path = PROCESSED_DIR / "vent_episodes.parquet"
        if not eps_path.exists():
            return {"error": "vent_episodes.parquet missing - run vent_labels first"}
        episodes = pd.read_parquet(eps_path)
        if episodes.empty:
            return {"error": "no vent episodes"}

        df = _features_at_start(features, episodes)
        if df.empty:
            return {"error": "no feature rows aligned to vent starts"}

        df["duration_hours"] = df["duration_hours"].clip(lower=0.5)  # CoxPH needs >0
        feat_cols = [c for c in df.columns if c not in NON_FEATURE and pd.api.types.is_numeric_dtype(df[c])]
        # Drop near-constant columns to keep CoxPH happy.
        feat_cols = [c for c in feat_cols if df[c].fillna(0).nunique() > 1]
        # Top-K most variable features (Cox is slow with hundreds of covariates).
        if len(feat_cols) > top_k_features:
            variances = df[feat_cols].fillna(0).var().sort_values(ascending=False)
            feat_cols = variances.head(top_k_features).index.tolist()

        fit_df = df[feat_cols + ["duration_hours", "event"]].fillna(0).copy()
        cph = CoxPHFitter(penalizer=0.05)
        cph.fit(fit_df, duration_col="duration_hours", event_col="event")
        path = MODELS_DIR / "vent_duration_cox.pkl"
        with open(path, "wb") as f:
            pickle.dump(cph, f)
        with open(MODELS_DIR / "vent_duration_features.json", "w", encoding="utf-8") as f:
            json.dump(feat_cols, f)

        observed = df[df["event"] == 1]["duration_hours"]
        return {
            "model_path": str(path),
            "n_episodes": int(len(df)),
            "n_observed": int(df["event"].sum()),
            "median_duration_hours": float(observed.median()) if len(observed) else 0.0,
            "p10_p90_hours": [
                float(np.percentile(observed, 10)) if len(observed) else 0.0,
                float(np.percentile(observed, 90)) if len(observed) else 0.0,
            ],
            "n_features": len(feat_cols),
        }
