"""LOS survival model (Kaplan-Meier + Weibull AFT).

Fits per-stay survival on the real remaining-LOS target. Uses one
representative row per stay (the earliest snapshot) to avoid pseudo-replication
that would bias hazard estimates.
"""

from __future__ import annotations

import json
import pickle

import pandas as pd
from lifelines import KaplanMeierFitter, WeibullAFTFitter

from config import MODELS_DIR, PROCESSED_DIR

NON_FEATURE = {"label", "vent_label", "remaining_los_hours", "los_event", "duration",
               "charttime", "stay_id", "hadm_id", "subject_id",
               "onset_time", "suspected_time", "first_vent_start"}


class LOSSurvivalModel:
    def train(self, top_k_features: int = 15) -> dict:
        train = pd.read_parquet(PROCESSED_DIR / "train.parquet")
        labels = pd.read_parquet(PROCESSED_DIR / "labels_los.parquet")
        train["charttime"] = pd.to_datetime(train["charttime"])
        labels["charttime"] = pd.to_datetime(labels["charttime"])
        df = train.merge(labels, on=["stay_id", "charttime"], how="inner")
        if df.empty:
            return {"error": "no LOS training rows"}

        # One row per stay: earliest snapshot.
        df = df.sort_values(["stay_id", "charttime"]).groupby("stay_id", as_index=False).first()

        feat_cols = [c for c in df.columns if c not in NON_FEATURE and pd.api.types.is_numeric_dtype(df[c])]
        feat_cols = [c for c in feat_cols if df[c].fillna(0).nunique() > 1]
        if len(feat_cols) > top_k_features:
            variances = df[feat_cols].fillna(0).var().sort_values(ascending=False)
            feat_cols = variances.head(top_k_features).index.tolist()

        df["duration"] = df["remaining_los_hours"].clip(lower=0.5)
        df["event"] = df["los_event"].astype(int)

        km = KaplanMeierFitter()
        km.fit(df["duration"], event_observed=df["event"])
        median_dur = float(km.median_survival_time_)

        aft_input = df[feat_cols + ["duration", "event"]].fillna(0).copy()
        try:
            aft = WeibullAFTFitter(penalizer=0.05)
            aft.fit(aft_input, duration_col="duration", event_col="event")
            aft_path = MODELS_DIR / "los_aft.pkl"
            with open(aft_path, "wb") as f:
                pickle.dump(aft, f)
            with open(MODELS_DIR / "los_aft_features.json", "w", encoding="utf-8") as f:
                json.dump(feat_cols, f)
            return {
                "aft_model": str(aft_path),
                "median_duration_hours": median_dur,
                "n_stays": int(len(df)),
                "n_events": int(df["event"].sum()),
                "n_features": len(feat_cols),
            }
        except Exception as e:
            return {"error": f"AFT fit failed: {e}", "median_duration_hours": median_dur, "n_stays": int(len(df))}
