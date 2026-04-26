from pathlib import Path

# Base paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

# Local MIMIC-IV dataset location
MIMIC_ROOT = (PROJECT_ROOT.parent / "mimic-iv-3.1").resolve()
MIMIC_ICU_DIR = MIMIC_ROOT / "icu"
MIMIC_HOSP_DIR = MIMIC_ROOT / "hosp"

# Vital sign itemIDs
VITAL_ITEMIDS = {
    "heart_rate": 220045,
    "spo2": 220277,
    "sbp": 220179,
    "dbp": 220180,
    "mbp": 220052,
    "resp_rate": 220210,
    "temperature": 223761,
    "gcs_total": 223900,
}

# Lab itemIDs
LAB_ITEMIDS = {
    "lactate": 50813,
    "creatinine": 50912,
    "wbc": 51301,
    "platelets": 51265,
    "bilirubin": 50885,
    "pao2": 50821,
    "fio2": 50816,
    "sodium": 50983,
    "potassium": 50971,
}

ANTIBIOTIC_KEYWORDS = [
    "vancomycin",
    "meropenem",
    "piperacillin",
    "cefepime",
    "metronidazole",
    "ciprofloxacin",
    "linezolid",
    "ampicillin",
]

SEPSIS_LOOKAHEAD_HOURS = 6
SOFA_THRESHOLD = 2
LSTM_SEQUENCE_HOURS = 24
RESAMPLE_FREQ = "1h"
# MIMIC-IV-3.1 intime distribution: 10%≈2121, 50%≈2153, 70%≈2169, 85%≈2181, 95%≈2189.
# Cutoffs target ~70/15/15 train/val/test split.
TRAIN_CUTOFF_YEAR = 2169
VAL_CUTOFF_YEAR = 2181
TEST_FROM_YEAR = 2181
BED_FORECAST_HORIZONS = [24, 48, 72]
# itemids in procedureevents: 225792 = invasive ventilation, 225794 = non-invasive ventilation
VENT_ITEMIDS = [225792, 225794]
VENT_LOOKAHEAD_HOURS = 24
LOS_MAX_DAYS = 30
CHUNKSIZE = 200_000


def ensure_dirs() -> None:
    for d in [RAW_DIR, PROCESSED_DIR, MODELS_DIR, OUTPUTS_DIR]:
        d.mkdir(parents=True, exist_ok=True)
    (OUTPUTS_DIR / "shap").mkdir(parents=True, exist_ok=True)
