"""Per-patient aggregator.

Pulls the most recent feature snapshot for a stay and runs every trained model
to assemble a single CDSS payload, then renders a plain-English summary.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb

from config import LOS_MAX_DAYS, MIMIC_ICU_DIR, MODELS_DIR, PROCESSED_DIR
from explainability import top_shap_features_tree
from models.nlg.summary_generator import generate_summary
from utils import mimic_file


_LOS_MAX_HOURS = float(LOS_MAX_DAYS * 24)
_LOS_MIN_HOURS = 0.5


def _scalar(x) -> float:
    """Coerce a lifelines prediction (Series or numpy scalar) to a Python float."""
    if hasattr(x, "iloc"):
        x = x.iloc[0]
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def _bound_hours(x: float) -> float | None:
    """Clamp a predicted-hours value to the trained label range, or None if non-finite."""
    if x is None or not np.isfinite(x):
        return None
    return float(np.clip(x, _LOS_MIN_HOURS, _LOS_MAX_HOURS))


def _load_xgb(model_path: Path, features_path: Path) -> tuple[xgb.XGBClassifier | None, list[str]]:
    if not model_path.exists() or not features_path.exists():
        return None, []
    model = xgb.XGBClassifier()
    model.load_model(str(model_path))
    feats = json.loads(features_path.read_text(encoding="utf-8"))
    return model, feats


def _load_lgb(model_path: Path) -> lgb.Booster | None:
    if not model_path.exists():
        return None
    return lgb.Booster(model_file=str(model_path))


def _row_for(stay_id: int) -> pd.DataFrame:
    feats = pd.read_parquet(PROCESSED_DIR / "features.parquet")
    rows = feats[feats["stay_id"] == stay_id].sort_values("charttime")
    if rows.empty:
        raise ValueError(f"No features found for stay_id={stay_id}")
    return rows.tail(1)


def _select(row: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = row.copy()
    for c in columns:
        if c not in out.columns:
            out[c] = 0.0
    return out[columns].fillna(0)


def _stay_unit(stay_id: int) -> str:
    icu_path, icu_comp = mimic_file(MIMIC_ICU_DIR, "icustays")
    icu = pd.read_csv(icu_path, compression=icu_comp, usecols=["stay_id", "first_careunit"])
    m = icu[icu["stay_id"] == stay_id]
    return str(m.iloc[0]["first_careunit"]) if len(m) else "ICU"


def _bed_forecast(unit: str) -> dict:
    occ_path = PROCESSED_DIR / "bed_occupancy_hourly.parquet"
    if not occ_path.exists():
        return {"24h": None, "48h": None, "72h": None, "unit": unit}
    occ = pd.read_parquet(occ_path)
    g = occ[occ["unit"] == unit].sort_values("hour")
    if g.empty:
        return {"24h": None, "48h": None, "72h": None, "unit": unit}
    last = g.tail(1)
    X = last[["occupancy", "roll24", "roll7d", "adm_24", "dow", "hod"]].fillna(0).to_numpy()
    out = {"unit": unit}
    for h in (24, 48, 72):
        b = _load_lgb(MODELS_DIR / f"bed_lgbm_{h}.txt")
        out[f"{h}h"] = float(b.predict(X)[0]) if b is not None else None
    return out


def aggregate(stay_id: str | int, timestamp: pd.Timestamp | None = None) -> dict:
    stay_id_int = int(stay_id)
    timestamp = timestamp or pd.Timestamp(datetime.now())
    row = _row_for(stay_id_int)
    patient_id = str(row.iloc[0].get("subject_id", "unknown"))
    unit = _stay_unit(stay_id_int)

    # Sepsis
    sepsis_model, sepsis_feats = _load_xgb(MODELS_DIR / "sepsis_xgb.ubj", MODELS_DIR / "sepsis_xgb_features.json")
    thr_path = MODELS_DIR / "sepsis_threshold.json"
    if thr_path.exists():
        thr_doc = json.loads(thr_path.read_text(encoding="utf-8"))
        alert_thr = float(thr_doc.get("alert", 0.5))
        moderate_thr = float(thr_doc.get("moderate", 0.3))
    else:
        alert_thr, moderate_thr = 0.5, 0.3
    sepsis_block: dict = {
        "probability": None,
        "alert": False,
        "alert_threshold": alert_thr,
        "moderate_threshold": moderate_thr,
        "top_features": [],
    }
    if sepsis_model is not None:
        X = _select(row, sepsis_feats)
        prob = float(sepsis_model.predict_proba(X)[0, 1])
        sepsis_block["probability"] = prob
        sepsis_block["alert"] = prob >= alert_thr
        try:
            sepsis_block["top_features"] = top_shap_features_tree(sepsis_model, X, top_k=5)
        except Exception:
            sepsis_block["top_features"] = []

    # Ventilator
    vent_model, vent_feats = _load_xgb(MODELS_DIR / "vent_classifier.ubj", MODELS_DIR / "vent_classifier_features.json")
    vent_block: dict = {"need_probability": None, "predicted_duration_hours": None, "ci_80": [None, None]}
    if vent_model is not None:
        X = _select(row, vent_feats)
        vent_block["need_probability"] = float(vent_model.predict_proba(X)[0, 1])
    cox_path = MODELS_DIR / "vent_duration_cox.pkl"
    cox_feats_path = MODELS_DIR / "vent_duration_features.json"
    if cox_path.exists() and cox_feats_path.exists():
        try:
            import pickle
            with open(cox_path, "rb") as fh:
                cph = pickle.load(fh)
            feats = json.loads(cox_feats_path.read_text(encoding="utf-8"))
            X = _select(row, feats)
            # lifelines returns either a Series or a numpy scalar depending on
            # input shape — handle both.
            ci_low = _scalar(cph.predict_percentile(X, p=0.9))
            ci_high = _scalar(cph.predict_percentile(X, p=0.1))
            median_h = _scalar(cph.predict_percentile(X, p=0.5))
            if not np.isfinite(median_h):
                try:
                    median_h = _scalar(cph.predict_expectation(X))
                except Exception:
                    median_h = float("nan")
            vent_block["predicted_duration_hours"] = median_h if np.isfinite(median_h) else None
            vent_block["ci_80"] = [
                ci_low if np.isfinite(ci_low) else None,
                ci_high if np.isfinite(ci_high) else None,
            ]
        except Exception:
            pass

    # LOS
    los_block: dict = {"remaining_hours": None, "p10_hours": None, "p90_hours": None}
    los_lgb = _load_lgb(MODELS_DIR / "los_lgbm.txt")
    los_feats_path = MODELS_DIR / "los_lgbm_features.json"
    if los_lgb is not None and los_feats_path.exists():
        feats = json.loads(los_feats_path.read_text(encoding="utf-8"))
        X = _select(row, feats).to_numpy()
        los_block["remaining_hours"] = _bound_hours(float(los_lgb.predict(X)[0]))
    aft_path = MODELS_DIR / "los_aft.pkl"
    aft_feats_path = MODELS_DIR / "los_aft_features.json"
    if aft_path.exists() and aft_feats_path.exists():
        try:
            import pickle
            with open(aft_path, "rb") as fh:
                aft = pickle.load(fh)
            feats = json.loads(aft_feats_path.read_text(encoding="utf-8"))
            X = _select(row, feats)
            # lifelines uses p = fraction surviving, so p=0.9 → early time
            # (lower bound of LOS), p=0.1 → late time (upper bound).
            # Clamp to the trained label range — the AFT's exp(coef·x) is
            # numerically unstable when fed unclipped raw vitals.
            los_block["p10_hours"] = _bound_hours(_scalar(aft.predict_percentile(X, p=0.9)))
            los_block["p90_hours"] = _bound_hours(_scalar(aft.predict_percentile(X, p=0.1)))
        except Exception:
            pass

    return {
        "patient_id": patient_id,
        "stay_id": str(stay_id_int),
        "timestamp": timestamp.isoformat(),
        "sepsis": sepsis_block,
        "bed_demand": _bed_forecast(unit),
        "ventilator": vent_block,
        "los": los_block,
    }


def aggregate_and_generate(stay_id: str | int, timestamp: pd.Timestamp | None = None) -> tuple[dict, str]:
    payload = aggregate(stay_id, timestamp)
    return payload, generate_summary(payload)
