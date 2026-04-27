import pandas as pd

from config import GCS_COMPONENTS, LAB_ITEMIDS, MIMIC_HOSP_DIR, MIMIC_ICU_DIR, PROCESSED_DIR, RESAMPLE_FREQ, VITAL_ITEMIDS
from utils import mimic_file


def _pivot_events(df: pd.DataFrame, mapping: dict[str, int], time_col: str, id_col: str) -> pd.DataFrame:
    reverse = {v: k for k, v in mapping.items()}
    select_cols = [id_col, time_col, "itemid", "valuenum"] if id_col == "hadm_id" else [id_col, "hadm_id", time_col, "itemid", "valuenum"]
    df = df[select_cols].copy()
    df = df[df["itemid"].isin(reverse)]
    df["feature"] = df["itemid"].map(reverse)
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df["chart_hour"] = df[time_col].dt.floor("h")
    index_cols = [id_col, "chart_hour"] if id_col == "hadm_id" else [id_col, "hadm_id", "chart_hour"]
    out = (
        df.pivot_table(index=index_cols, columns="feature", values="valuenum", aggfunc="mean")
        .reset_index()
        .sort_values([id_col, "chart_hour"])
    )
    out.columns.name = None
    return out


def run_etl(max_rows: int | None = None) -> pd.DataFrame:
    icu_path, icu_comp = mimic_file(MIMIC_ICU_DIR, "icustays")
    patients_path, patients_comp = mimic_file(MIMIC_HOSP_DIR, "patients")
    icu = pd.read_csv(icu_path, compression=icu_comp)
    patients = pd.read_csv(patients_path, compression=patients_comp)[["subject_id", "anchor_age"]]
    vitals = pd.read_parquet(PROCESSED_DIR / "vitals.parquet")
    labs = pd.read_parquet(PROCESSED_DIR / "labs.parquet")
    if max_rows:
        vitals = vitals.head(max_rows)
        labs = labs.head(max_rows)

    icu = icu.merge(patients, on="subject_id", how="left")
    icu["intime"] = pd.to_datetime(icu["intime"])
    icu["outtime"] = pd.to_datetime(icu["outtime"])
    icu["stay_hours"] = (icu["outtime"] - icu["intime"]).dt.total_seconds() / 3600
    cohort = icu[(icu["anchor_age"] >= 18) & (icu["stay_hours"] >= 4)].copy()

    vitals_h = _pivot_events(vitals, VITAL_ITEMIDS, "charttime", "stay_id")
    labs_h = _pivot_events(labs, LAB_ITEMIDS, "charttime", "hadm_id")

    # Reconstruct true GCS total from its three subscores. min_count=3 means we
    # only emit a sum when all three components were charted in the same hour;
    # otherwise gcs_total stays NaN and is later ffilled per stay.
    present_gcs = [c for c in GCS_COMPONENTS if c in vitals_h.columns]
    if len(present_gcs) == 3:
        vitals_h["gcs_total"] = vitals_h[present_gcs].sum(axis=1, min_count=3)
    elif present_gcs:
        vitals_h["gcs_total"] = vitals_h[present_gcs].sum(axis=1, min_count=len(present_gcs))

    cohort = cohort[["stay_id", "hadm_id", "subject_id", "intime", "outtime"]]
    merged = vitals_h.merge(cohort, on=["stay_id", "hadm_id"], how="inner")
    merged = merged.merge(labs_h, on=["hadm_id", "chart_hour"], how="left", suffixes=("", "_lab"))
    merged = merged[(merged["chart_hour"] >= merged["intime"]) & (merged["chart_hour"] <= merged["outtime"])]

    keep_cols = (
        ["stay_id", "hadm_id", "subject_id", "chart_hour"]
        + list(VITAL_ITEMIDS.keys())
        + (["gcs_total"] if "gcs_total" in merged.columns else [])
        + list(LAB_ITEMIDS.keys())
    )
    keep_cols = [c for c in keep_cols if c in merged.columns]
    merged = merged[keep_cols].sort_values(["stay_id", "chart_hour"])
    merged = merged.rename(columns={"chart_hour": "charttime"})
    hourly = (
        merged.groupby(["stay_id", "hadm_id", "subject_id", "charttime"], as_index=False)
        .mean(numeric_only=True)
        .sort_values(["stay_id", "charttime"])
    )

    vital_cols = [c for c in VITAL_ITEMIDS if c in hourly.columns]
    lab_cols = [c for c in LAB_ITEMIDS if c in hourly.columns]
    hourly[vital_cols] = hourly.groupby("stay_id")[vital_cols].ffill()
    for c in lab_cols:
        hourly[c] = hourly.groupby("stay_id")[c].transform(lambda s: s.fillna(s.median()))

    # Physiologic clipping bounds. Without these, raw MIMIC valuenum corruption
    # (e.g. one row with lactate=1,276,103) poisons rolling means/stds and blows
    # up downstream models — the AFT exp(coef·x) trick turns a single bad mbp
    # value into a 1e18-hour LOS prediction.
    bounds = {
        "heart_rate": (0, 300),
        "spo2": (50, 100),
        "temperature": (25, 45),
        "sbp": (40, 300),
        "dbp": (20, 200),
        "mbp": (20, 200),
        "resp_rate": (0, 80),
        "gcs_eye": (1, 4),
        "gcs_verbal": (1, 5),
        "gcs_motor": (1, 6),
        "gcs_total": (3, 15),
        "wbc": (0, 200),
        "creatinine": (0, 30),
        "lactate": (0, 30),
        "platelets": (0, 2000),
        "bilirubin": (0, 80),
        "pao2": (20, 700),
        "fio2": (0.21, 1.0),
        "sodium": (100, 180),
        "potassium": (1, 10),
    }
    for col, (lo, hi) in bounds.items():
        if col in hourly.columns:
            hourly[col] = hourly[col].clip(lo, hi)

    # Vectorised missingness: mean of NaN-mask across vital columns, then mean per stay.
    if vital_cols:
        nan_frac_per_row = hourly[vital_cols].isna().mean(axis=1)
        miss = nan_frac_per_row.groupby(hourly["stay_id"]).mean()
        keep_stays = miss[miss <= 0.8].index
        hourly = hourly[hourly["stay_id"].isin(keep_stays)].copy()
    hourly.to_parquet(PROCESSED_DIR / "cohort_hourly.parquet", index=False)
    return hourly

