"""Convenience driver: assumes vitals.parquet and labs.parquet already exist.

Runs ETL → features → labels → datasets → train all 6 models → write
outputs/evaluation_report.json.

Used after data_extractor.run_extraction() completes so the long extraction
phase can be staged separately.
"""

from __future__ import annotations

import json
import sys
import time
import warnings

sys.path.insert(0, ".")
warnings.filterwarnings("ignore")

from config import OUTPUTS_DIR, ensure_dirs
from dataset_builder import build_datasets
from etl_pipeline import run_etl
from feature_engineer import run_feature_engineering
from label_creator import create_sepsis_labels
from los_labels import create_los_labels
from models.bed_demand.forecaster import BedDemandForecaster
from models.los.regressor import LOSRegressor
from models.los.survival_model import LOSSurvivalModel
from models.sepsis.xgboost_model import SepsisXGBoost
from models.ventilator.classifier import VentilatorNeedClassifier
from models.ventilator.duration_model import VentilatorDurationModel
from vent_labels import create_vent_labels


def banner(msg: str) -> None:
    print(f"\n========== {msg} ==========", flush=True)


def main() -> None:
    ensure_dirs()
    results: dict = {}

    banner("ETL")
    t = time.time(); df = run_etl()
    results["etl"] = {"rows": int(len(df)), "stays": int(df["stay_id"].nunique()), "wall_s": round(time.time() - t, 1)}
    print(results["etl"], flush=True)

    banner("FEATURES")
    t = time.time(); fe = run_feature_engineering()
    results["features"] = {"rows": int(len(fe)), "cols": int(fe.shape[1]), "wall_s": round(time.time() - t, 1)}
    print(results["features"], flush=True)

    banner("LABELS")
    t = time.time(); s = create_sepsis_labels()
    sep = {"rows": int(len(s)), "pos_rate": float(s["label"].mean()) if len(s) else 0.0, "wall_s": round(time.time() - t, 1)}
    print("sepsis:", sep, flush=True)
    t = time.time(); v = create_vent_labels()
    vent = {"rows": int(len(v)), "pos_rate": float(v["vent_label"].mean()) if len(v) else 0.0, "wall_s": round(time.time() - t, 1)}
    print("vent:", vent, flush=True)
    t = time.time(); l = create_los_labels()
    los = {"rows": int(len(l)), "wall_s": round(time.time() - t, 1)}
    print("los:", los, flush=True)
    results["labels"] = {"sepsis": sep, "vent": vent, "los": los}

    banner("DATASETS")
    t = time.time(); ds = build_datasets()
    ds["wall_s"] = round(time.time() - t, 1)
    results["datasets"] = ds
    print(json.dumps(ds, indent=2), flush=True)

    banner("TRAIN sepsis_xgb")
    t = time.time()
    try:
        results["sepsis_xgb"] = SepsisXGBoost().train(n_trials=15)
    except Exception as e:
        results["sepsis_xgb"] = {"error": f"{type(e).__name__}: {e}"}
    results["sepsis_xgb"]["wall_s"] = round(time.time() - t, 1)
    print(json.dumps(results["sepsis_xgb"], indent=2, default=str), flush=True)

    banner("TRAIN bed_demand")
    t = time.time()
    try:
        results["bed_demand"] = BedDemandForecaster().train()
    except Exception as e:
        results["bed_demand"] = {"error": f"{type(e).__name__}: {e}"}
    results["bed_demand"]["wall_s"] = round(time.time() - t, 1)
    print(json.dumps(results["bed_demand"], indent=2, default=str), flush=True)

    banner("TRAIN vent_classifier")
    t = time.time()
    try:
        results["vent_classifier"] = VentilatorNeedClassifier().train(n_trials=10)
    except Exception as e:
        results["vent_classifier"] = {"error": f"{type(e).__name__}: {e}"}
    results["vent_classifier"]["wall_s"] = round(time.time() - t, 1)
    print(json.dumps(results["vent_classifier"], indent=2, default=str), flush=True)

    banner("TRAIN vent_duration (Cox)")
    t = time.time()
    try:
        results["vent_duration"] = VentilatorDurationModel().train()
    except Exception as e:
        results["vent_duration"] = {"error": f"{type(e).__name__}: {e}"}
    results["vent_duration"]["wall_s"] = round(time.time() - t, 1)
    print(json.dumps(results["vent_duration"], indent=2, default=str), flush=True)

    banner("TRAIN los_regressor")
    t = time.time()
    try:
        results["los_regressor"] = LOSRegressor().train()
    except Exception as e:
        results["los_regressor"] = {"error": f"{type(e).__name__}: {e}"}
    results["los_regressor"]["wall_s"] = round(time.time() - t, 1)
    print(json.dumps(results["los_regressor"], indent=2, default=str), flush=True)

    banner("TRAIN los_survival (Weibull AFT)")
    t = time.time()
    try:
        results["los_survival"] = LOSSurvivalModel().train()
    except Exception as e:
        results["los_survival"] = {"error": f"{type(e).__name__}: {e}"}
    results["los_survival"]["wall_s"] = round(time.time() - t, 1)
    print(json.dumps(results["los_survival"], indent=2, default=str), flush=True)

    out_path = OUTPUTS_DIR / "evaluation_report.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nReport written -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
