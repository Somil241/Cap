"""End-to-end pipeline driver.

Modes:
  train     -- extract → ETL → features → labels → datasets → train all models
  evaluate  -- print last evaluation_report.json
  infer     -- run aggregator on a stay_id and print a clinical summary
  explore   -- run explore.py only
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime

import pandas as pd

from aggregator import aggregate_and_generate
from config import OUTPUTS_DIR, ensure_dirs
from data_extractor import run_extraction
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


def _step(label: str) -> None:
    print(f"\n========== {label} ==========", flush=True)


def run_train(max_chart_rows: int | None = None, max_lab_rows: int | None = None,
              train_lstm: bool = False) -> dict:
    ensure_dirs()
    results: dict = {}

    _step("EXTRACT vitals + labs")
    run_extraction(max_chart_rows=max_chart_rows, max_lab_rows=max_lab_rows)

    _step("ETL  (cohort + hourly grid)")
    run_etl()

    _step("FEATURE ENGINEERING")
    run_feature_engineering()

    _step("LABELS  (sepsis / vent / LOS)")
    create_sepsis_labels()
    create_vent_labels()
    create_los_labels()

    _step("BUILD train/val/test")
    results["dataset_summary"] = build_datasets()

    _step("TRAIN sepsis XGBoost")
    results["sepsis_xgb"] = SepsisXGBoost().train(n_trials=10)

    if train_lstm:
        _step("TRAIN sepsis LSTM (optional)")
        try:
            from models.sepsis.lstm_model import SepsisLSTM
            results["sepsis_lstm"] = SepsisLSTM().train(epochs=5)
        except Exception as e:  # TF not installed etc.
            results["sepsis_lstm"] = {"error": f"{type(e).__name__}: {e}"}

    _step("TRAIN bed demand forecaster")
    results["bed_demand"] = BedDemandForecaster().train()

    _step("TRAIN ventilator classifier")
    results["vent_classifier"] = VentilatorNeedClassifier().train(n_trials=10)

    _step("TRAIN ventilator duration (Cox)")
    results["vent_duration"] = VentilatorDurationModel().train()

    _step("TRAIN LOS regressor (LightGBM)")
    results["los_regressor"] = LOSRegressor().train()

    _step("TRAIN LOS survival (Weibull AFT)")
    results["los_survival"] = LOSSurvivalModel().train()

    with open(OUTPUTS_DIR / "evaluation_report.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    return results


def run_infer(stay_id: str) -> dict:
    payload, text = aggregate_and_generate(stay_id, pd.Timestamp(datetime.now()))
    with open(OUTPUTS_DIR / f"summary_{stay_id}.txt", "w", encoding="utf-8") as f:
        f.write(text)
    return payload


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", required=True, choices=["train", "evaluate", "infer", "explore"])
    p.add_argument("--stay_id", type=str, default="")
    p.add_argument("--max_chart_rows", type=int, default=None,
                   help="Cap input rows scanned from chartevents.csv (None = full file).")
    p.add_argument("--max_lab_rows", type=int, default=None)
    p.add_argument("--train_lstm", action="store_true")
    args = p.parse_args()

    if args.mode == "train":
        out = run_train(max_chart_rows=args.max_chart_rows,
                        max_lab_rows=args.max_lab_rows,
                        train_lstm=args.train_lstm)
        print(json.dumps(out, indent=2, default=str))
    elif args.mode == "evaluate":
        path = OUTPUTS_DIR / "evaluation_report.json"
        print(path.read_text(encoding="utf-8") if path.exists() else "No evaluation report. Run --mode train first.")
    elif args.mode == "infer":
        if not args.stay_id:
            raise ValueError("--stay_id is required for infer mode")
        print(json.dumps(run_infer(args.stay_id), indent=2, default=str))
    elif args.mode == "explore":
        from explore import run_explore
        run_explore()


if __name__ == "__main__":
    main()
