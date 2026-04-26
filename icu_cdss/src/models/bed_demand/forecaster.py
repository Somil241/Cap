"""Unit-level ICU bed-demand forecaster.

Builds an hourly per-unit occupancy series via a vectorised admission/discharge
delta + cumulative sum (avoids the O(N*hours) per-unit double loop). Then fits
one LightGBM regressor per forecast horizon (24h / 48h / 72h).
"""

from __future__ import annotations

import json

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error

from config import BED_FORECAST_HORIZONS, MIMIC_ICU_DIR, MODELS_DIR, PROCESSED_DIR
from utils import mimic_file


def _hourly_occupancy(icu: pd.DataFrame) -> pd.DataFrame:
    icu = icu.dropna(subset=["intime", "outtime"]).copy()
    icu["intime"] = pd.to_datetime(icu["intime"]).dt.floor("h")
    icu["outtime"] = pd.to_datetime(icu["outtime"]).dt.ceil("h")
    icu = icu[icu["outtime"] > icu["intime"]]
    if icu.empty:
        return pd.DataFrame(columns=["unit", "hour", "occupancy"])

    starts = icu[["first_careunit", "intime"]].rename(columns={"first_careunit": "unit", "intime": "hour"})
    starts["delta"] = 1
    ends = icu[["first_careunit", "outtime"]].rename(columns={"first_careunit": "unit", "outtime": "hour"})
    ends["delta"] = -1
    deltas = pd.concat([starts, ends], ignore_index=True)

    deltas = deltas.groupby(["unit", "hour"], as_index=False)["delta"].sum()
    out_frames = []
    for unit, g in deltas.groupby("unit"):
        g = g.sort_values("hour")
        full_idx = pd.date_range(g["hour"].min(), g["hour"].max(), freq="h")
        s = g.set_index("hour")["delta"].reindex(full_idx, fill_value=0)
        occ = s.cumsum().clip(lower=0)
        out_frames.append(pd.DataFrame({"unit": unit, "hour": occ.index, "occupancy": occ.to_numpy()}))
    if not out_frames:
        return pd.DataFrame(columns=["unit", "hour", "occupancy"])
    df = pd.concat(out_frames, ignore_index=True)
    df["dow"] = df["hour"].dt.dayofweek
    df["hod"] = df["hour"].dt.hour
    return df


class BedDemandForecaster:
    def __init__(self) -> None:
        self.models: dict[int, lgb.Booster] = {}

    def _build(self) -> pd.DataFrame:
        icu_path, icu_comp = mimic_file(MIMIC_ICU_DIR, "icustays")
        icu = pd.read_csv(icu_path, compression=icu_comp, usecols=["first_careunit", "intime", "outtime"])
        return _hourly_occupancy(icu)

    def train(self) -> dict:
        df = self._build()
        if df.empty:
            return {"error": "no occupancy series built"}

        df = df.sort_values(["unit", "hour"]).reset_index(drop=True)
        df["roll24"] = df.groupby("unit")["occupancy"].transform(
            lambda s: s.rolling(24, min_periods=1).mean()
        )
        df["roll7d"] = df.groupby("unit")["occupancy"].transform(
            lambda s: s.rolling(24 * 7, min_periods=1).mean()
        )
        df["adm_24"] = (
            df.groupby("unit")["occupancy"].diff().clip(lower=0)
            .rolling(24, min_periods=1).sum().fillna(0)
        )
        # Save full series for inference convenience.
        df.to_parquet(PROCESSED_DIR / "bed_occupancy_hourly.parquet", index=False)

        metrics: dict = {"horizons": BED_FORECAST_HORIZONS, "rmse": {}, "mape": {}}
        for h in BED_FORECAST_HORIZONS:
            d = df.copy()
            d[f"y_{h}"] = d.groupby("unit")["occupancy"].shift(-h)
            d = d.dropna(subset=[f"y_{h}"])
            split = int(len(d) * 0.8)
            tr = d.iloc[:split]
            te = d.iloc[split:]
            X_cols = ["occupancy", "roll24", "roll7d", "adm_24", "dow", "hod"]
            model = lgb.LGBMRegressor(
                n_estimators=400, learning_rate=0.05, num_leaves=63,
                random_state=42, n_jobs=4, verbose=-1,
            )
            model.fit(tr[X_cols], tr[f"y_{h}"])
            pred = model.predict(te[X_cols])
            metrics["rmse"][str(h)] = float(np.sqrt(mean_squared_error(te[f"y_{h}"], pred)))
            metrics["mape"][str(h)] = float(
                mean_absolute_percentage_error(np.maximum(te[f"y_{h}"], 1), np.maximum(pred, 1))
            )
            self.models[h] = model.booster_
            model.booster_.save_model(str(MODELS_DIR / f"bed_lgbm_{h}.txt"))

        with open(MODELS_DIR / "bed_metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        return metrics
