import numpy as np
import pandas as pd

from config import LAB_ITEMIDS, PROCESSED_DIR, VITAL_ITEMIDS


def run_feature_engineering() -> pd.DataFrame:
    df = pd.read_parquet(PROCESSED_DIR / "cohort_hourly.parquet").sort_values(["stay_id", "charttime"])
    feature_cols = [c for c in list(VITAL_ITEMIDS.keys()) + list(LAB_ITEMIDS.keys()) if c in df.columns]

    out = df.copy()
    g = out.groupby("stay_id")
    engineered = {}
    for col in feature_cols:
        engineered[f"{col}_roc_1h"] = g[col].diff(1)
        for w in [1, 6, 24]:
            engineered[f"{col}_mean_{w}h"] = g[col].transform(lambda s: s.rolling(w, min_periods=1).mean())
            engineered[f"{col}_std_{w}h"] = g[col].transform(lambda s: s.rolling(w, min_periods=1).std().fillna(0))

    engineered["shock_index"] = out.get("heart_rate", np.nan) / out.get("sbp", np.nan)
    engineered["sofa_renal"] = (out.get("creatinine", 0) >= 2.0).astype(int) + (out.get("creatinine", 0) >= 3.5).astype(int)
    engineered["sofa_coag"] = (out.get("platelets", 300) < 150).astype(int) + (out.get("platelets", 300) < 100).astype(int)
    engineered["sofa_liver"] = (out.get("bilirubin", 0) >= 2.0).astype(int) + (out.get("bilirubin", 0) >= 6.0).astype(int)
    engineered["sofa_resp"] = (out.get("pao2", 100) / out.get("fio2", 0.5).replace(0, np.nan) < 300).astype(int)
    engineered["sofa_cns"] = (out.get("gcs_total", 15) < 15).astype(int)
    engineered["sofa_cardio"] = (out.get("mbp", 80) < 70).astype(int)
    out = pd.concat([out, pd.DataFrame(engineered)], axis=1)
    out["sofa_total"] = out[[c for c in out.columns if c.startswith("sofa_")]].sum(axis=1)

    out["time_in_icu_hours"] = out.groupby("stay_id").cumcount()
    out["news2_resp"] = (out.get("resp_rate", 16) > 20).astype(int)
    out["news2_spo2"] = (out.get("spo2", 98) < 92).astype(int)
    out["news2_temp"] = ((out.get("temperature", 37) < 36) | (out.get("temperature", 37) > 38)).astype(int)
    out["news2_hr"] = ((out.get("heart_rate", 80) < 50) | (out.get("heart_rate", 80) > 110)).astype(int)
    out["news2_sbp"] = (out.get("sbp", 120) < 90).astype(int)
    out["news2_total"] = out[[c for c in out.columns if c.startswith("news2_")]].sum(axis=1)
    out["cum_fluid_balance"] = 0.0

    out.to_parquet(PROCESSED_DIR / "features.parquet", index=False)
    return out

