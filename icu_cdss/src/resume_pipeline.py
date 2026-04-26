"""Resume runner: run from dataset_builder onwards.

Use after run_pipeline.py has produced features.parquet, labels_*.parquet,
vent_episodes.parquet on disk (these stages are slow but idempotent).
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
from models.bed_demand.forecaster import BedDemandForecaster
from models.los.regressor import LOSRegressor
from models.los.survival_model import LOSSurvivalModel
from models.sepsis.xgboost_model import SepsisXGBoost
from models.ventilator.classifier import VentilatorNeedClassifier
from models.ventilator.duration_model import VentilatorDurationModel


def _step(label: str, fn):
    print(f"\n========== {label} ==========", flush=True)
    t = time.time()
    try:
        out = fn()
    except Exception as e:
        out = {"error": f"{type(e).__name__}: {e}"}
    if isinstance(out, dict):
        out["wall_s"] = round(time.time() - t, 1)
    print(json.dumps(out, indent=2, default=str), flush=True)
    return out


def main() -> None:
    ensure_dirs()
    results: dict = {}

    results["datasets"] = _step("DATASETS", build_datasets)
    results["sepsis_xgb"] = _step("TRAIN sepsis_xgb", lambda: SepsisXGBoost().train(n_trials=15))
    results["bed_demand"] = _step("TRAIN bed_demand", lambda: BedDemandForecaster().train())
    results["vent_classifier"] = _step("TRAIN vent_classifier", lambda: VentilatorNeedClassifier().train(n_trials=10))
    results["vent_duration"] = _step("TRAIN vent_duration (Cox)", lambda: VentilatorDurationModel().train())
    results["los_regressor"] = _step("TRAIN los_regressor", lambda: LOSRegressor().train())
    results["los_survival"] = _step("TRAIN los_survival (Weibull AFT)", lambda: LOSSurvivalModel().train())

    out_path = OUTPUTS_DIR / "evaluation_report.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nReport written -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
