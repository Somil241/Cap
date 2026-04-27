import json
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.metrics import f1_score, precision_recall_curve, roc_auc_score

from config import MODELS_DIR, PROCESSED_DIR

NON_FEATURE = {"label", "charttime", "stay_id", "hadm_id", "subject_id", "onset_time", "suspected_time"}


class SepsisXGBoost:
    def __init__(self) -> None:
        self.model: xgb.XGBClassifier | None = None
        self.features: list[str] = []

    def _load(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        train = pd.read_parquet(PROCESSED_DIR / "train_sepsis.parquet")
        val = pd.read_parquet(PROCESSED_DIR / "val.parquet")
        test = pd.read_parquet(PROCESSED_DIR / "test.parquet")
        return train, val, test

    def _feature_cols(self, train: pd.DataFrame) -> list[str]:
        return [
            c for c in train.columns
            if c not in NON_FEATURE and pd.api.types.is_numeric_dtype(train[c])
        ]

    def train(self, n_trials: int = 20) -> dict:
        train, val, test = self._load()
        self.features = self._feature_cols(train)
        # val/test may not have all engineered cols if engineering changed; align.
        for df in (val, test):
            for c in self.features:
                if c not in df.columns:
                    df[c] = 0.0

        X_train = train[self.features].fillna(0).astype("float32")
        y_train = train["label"].astype(int)
        X_val = val[self.features].fillna(0).astype("float32")
        y_val = val["label"].astype(int)
        X_test = test[self.features].fillna(0).astype("float32")
        y_test = test["label"].astype(int)
        spw = max((y_train == 0).sum() / max((y_train == 1).sum(), 1), 1.0)

        def objective(trial: optuna.Trial) -> float:
            params = {
                "n_estimators": 300,
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 10.0),
                "scale_pos_weight": spw,
                "eval_metric": "auc",
                "random_state": 42,
                "tree_method": "hist",
                "n_jobs": 4,
            }
            model = xgb.XGBClassifier(**params)
            model.fit(X_train, y_train)
            pred = model.predict_proba(X_val)[:, 1]
            return roc_auc_score(y_val, pred) if y_val.nunique() > 1 else 0.5

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        best = study.best_params
        self.model = xgb.XGBClassifier(
            **best,
            n_estimators=400,
            scale_pos_weight=spw,
            eval_metric="auc",
            random_state=42,
            tree_method="hist",
            n_jobs=4,
        )
        self.model.fit(X_train, y_train)

        val_prob = self.model.predict_proba(X_val)[:, 1]
        test_prob = self.model.predict_proba(X_test)[:, 1]

        # F1-optimal threshold from val PR curve. With a ~1% positive rate,
        # the default 0.5 is way too strict; without this, alerts are buried.
        opt_thr = 0.5
        opt_f1 = 0.0
        moderate_thr = 0.3
        if y_val.nunique() > 1 and len(y_val) > 0:
            prec, rec, thr = precision_recall_curve(y_val, val_prob)
            f1_curve = (2 * prec * rec) / (prec + rec + 1e-12)
            best_idx = int(np.argmax(f1_curve[:-1])) if len(thr) else 0
            opt_thr = float(thr[best_idx]) if len(thr) else 0.5
            opt_f1 = float(f1_curve[best_idx])
            # Moderate band = highest threshold whose recall is still >= 0.5
            recall_ok = np.where(rec[:-1] >= 0.5)[0]
            if recall_ok.size:
                moderate_thr = float(thr[recall_ok[-1]])

        metrics = {
            "best_params": best,
            "val_auroc": float(roc_auc_score(y_val, val_prob)) if y_val.nunique() > 1 else 0.5,
            "test_auroc": float(roc_auc_score(y_test, test_prob)) if y_test.nunique() > 1 else 0.5,
            "val_f1_at_05": float(f1_score(y_val, (val_prob >= 0.5).astype(int))) if len(y_val) else 0.0,
            "val_f1_at_optimal": opt_f1,
            "optimal_threshold": opt_thr,
            "moderate_threshold": moderate_thr,
            "n_features": len(self.features),
            "n_train": int(len(X_train)),
            "n_val": int(len(X_val)),
            "n_test": int(len(X_test)),
        }
        self.save()
        with open(PROCESSED_DIR / "sepsis_xgb_metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        # Persist thresholds where the aggregator + dashboard can find them.
        with open(MODELS_DIR / "sepsis_threshold.json", "w", encoding="utf-8") as f:
            json.dump({"alert": opt_thr, "moderate": moderate_thr}, f, indent=2)
        return metrics

    def save(self) -> None:
        if self.model is None:
            return
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        self.model.save_model(str(MODELS_DIR / "sepsis_xgb.ubj"))
        with open(MODELS_DIR / "sepsis_xgb_features.json", "w", encoding="utf-8") as f:
            json.dump(self.features, f)

    def load(self) -> None:
        self.model = xgb.XGBClassifier()
        self.model.load_model(str(MODELS_DIR / "sepsis_xgb.ubj"))
        self.features = json.loads(Path(MODELS_DIR / "sepsis_xgb_features.json").read_text(encoding="utf-8"))

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            self.load()
        for c in self.features:
            if c not in X.columns:
                X = X.assign(**{c: 0.0})
        return self.model.predict_proba(X[self.features].fillna(0))[:, 1]
