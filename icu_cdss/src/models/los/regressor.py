"""LOS regressor (LightGBM).

Predicts remaining ICU hours per (stay_id, charttime) using the real
outtime-derived target from labels_los.parquet.
"""

from __future__ import annotations

import json

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, median_absolute_error

from config import MODELS_DIR, PROCESSED_DIR

NON_FEATURE = {"label", "vent_label", "remaining_los_hours", "los_event",
               "charttime", "stay_id", "hadm_id", "subject_id",
               "onset_time", "suspected_time", "first_vent_start"}


def _attach(df: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["charttime"] = pd.to_datetime(df["charttime"])
    labels = labels.copy()
    labels["charttime"] = pd.to_datetime(labels["charttime"])
    return df.merge(labels, on=["stay_id", "charttime"], how="inner")


class LOSRegressor:
    def train(self, max_train_rows: int = 1_500_000) -> dict:
        train = pd.read_parquet(PROCESSED_DIR / "train.parquet")
        val = pd.read_parquet(PROCESSED_DIR / "val.parquet")
        test = pd.read_parquet(PROCESSED_DIR / "test.parquet")
        labels = pd.read_parquet(PROCESSED_DIR / "labels_los.parquet")
        train = _attach(train, labels)
        val = _attach(val, labels)
        test = _attach(test, labels)
        if train.empty:
            return {"error": "no LOS training rows"}
        if len(train) > max_train_rows:
            train = train.sample(max_train_rows, random_state=42).reset_index(drop=True)

        X_cols = [c for c in train.columns if c not in NON_FEATURE and pd.api.types.is_numeric_dtype(train[c])]
        model = lgb.LGBMRegressor(
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=63,
            random_state=42,
            n_jobs=4,
            verbose=-1,
        )
        model.fit(
            train[X_cols].fillna(0).astype("float32"),
            train["remaining_los_hours"],
            eval_set=[(val[X_cols].fillna(0).astype("float32"), val["remaining_los_hours"])] if len(val) else None,
            callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)] if len(val) else None,
        )
        pred_val = model.predict(val[X_cols].fillna(0).astype("float32")) if len(val) else np.array([])
        pred_test = model.predict(test[X_cols].fillna(0).astype("float32")) if len(test) else np.array([])
        metrics = {
            "val_mae_hours": float(mean_absolute_error(val["remaining_los_hours"], pred_val)) if len(val) else None,
            "val_medae_hours": float(median_absolute_error(val["remaining_los_hours"], pred_val)) if len(val) else None,
            "test_mae_hours": float(mean_absolute_error(test["remaining_los_hours"], pred_test)) if len(test) else None,
            "test_medae_hours": float(median_absolute_error(test["remaining_los_hours"], pred_test)) if len(test) else None,
            "n_train": int(len(train)),
            "n_features": len(X_cols),
        }
        model.booster_.save_model(str(MODELS_DIR / "los_lgbm.txt"))
        with open(MODELS_DIR / "los_lgbm_features.json", "w", encoding="utf-8") as f:
            json.dump(X_cols, f)
        with open(MODELS_DIR / "los_metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        return metrics
