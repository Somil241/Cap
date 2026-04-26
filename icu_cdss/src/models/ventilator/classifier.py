"""Ventilator-need classifier.

Trains an XGBoost model to predict whether a patient will require mechanical
ventilation within VENT_LOOKAHEAD_HOURS. Uses the real vent labels derived
from procedureevents (see src/vent_labels.py), not the sepsis label.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.metrics import precision_score, recall_score, roc_auc_score

from config import MODELS_DIR, PROCESSED_DIR

NON_FEATURE = {"label", "vent_label", "charttime", "stay_id", "hadm_id", "subject_id",
               "onset_time", "suspected_time", "first_vent_start"}


def _attach_labels(features: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    features = features.copy()
    features["charttime"] = pd.to_datetime(features["charttime"])
    labels = labels.copy()
    labels["charttime"] = pd.to_datetime(labels["charttime"])
    return features.merge(labels, on=["stay_id", "charttime"], how="inner")


class VentilatorNeedClassifier:
    def __init__(self) -> None:
        self.model: xgb.XGBClassifier | None = None
        self.features: list[str] = []

    def _load(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        train = pd.read_parquet(PROCESSED_DIR / "train.parquet")
        val = pd.read_parquet(PROCESSED_DIR / "val.parquet")
        test = pd.read_parquet(PROCESSED_DIR / "test.parquet")
        labels = pd.read_parquet(PROCESSED_DIR / "labels_vent.parquet")
        return _attach_labels(train, labels), _attach_labels(val, labels), _attach_labels(test, labels)

    def _feature_cols(self, df: pd.DataFrame) -> list[str]:
        return [c for c in df.columns if c not in NON_FEATURE and pd.api.types.is_numeric_dtype(df[c])]

    def train(self, n_trials: int = 15, max_train_rows: int = 1_000_000) -> dict:
        train, val, test = self._load()
        if "vent_label" not in train.columns or train.empty:
            return {"error": "no vent labels available", "train_rows": 0}
        self.features = self._feature_cols(train)
        # Memory-bound stratified subsample of training set.
        if len(train) > max_train_rows:
            pos = train[train["vent_label"] == 1]
            neg = train[train["vent_label"] == 0]
            n_pos_keep = min(len(pos), max_train_rows // 5)
            n_neg_keep = max_train_rows - n_pos_keep
            train = pd.concat([
                pos.sample(n_pos_keep, random_state=42) if len(pos) > n_pos_keep else pos,
                neg.sample(min(len(neg), n_neg_keep), random_state=42),
            ], ignore_index=True)
        X_train = train[self.features].fillna(0).astype("float32")
        y_train = train["vent_label"].astype(int)
        X_val = val[self.features].fillna(0).astype("float32")
        y_val = val["vent_label"].astype(int)
        X_test = test[self.features].fillna(0).astype("float32")
        y_test = test["vent_label"].astype(int)
        if y_train.nunique() < 2:
            return {"error": "vent label has single class in train", "train_pos": int(y_train.sum())}
        spw = max((y_train == 0).sum() / max((y_train == 1).sum(), 1), 1.0)

        def objective(trial: optuna.Trial) -> float:
            model = xgb.XGBClassifier(
                n_estimators=300,
                max_depth=trial.suggest_int("max_depth", 3, 8),
                learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                subsample=trial.suggest_float("subsample", 0.7, 1.0),
                colsample_bytree=trial.suggest_float("colsample_bytree", 0.7, 1.0),
                scale_pos_weight=spw,
                eval_metric="auc",
                random_state=42,
                tree_method="hist",
                n_jobs=4,
            )
            model.fit(X_train, y_train)
            p = model.predict_proba(X_val)[:, 1]
            return roc_auc_score(y_val, p) if y_val.nunique() > 1 else 0.5

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        best = xgb.XGBClassifier(
            **study.best_params,
            n_estimators=400,
            scale_pos_weight=spw,
            eval_metric="auc",
            random_state=42,
            tree_method="hist",
            n_jobs=4,
        )
        best.fit(X_train, y_train)
        self.model = best
        best.save_model(str(MODELS_DIR / "vent_classifier.ubj"))
        with open(MODELS_DIR / "vent_classifier_features.json", "w", encoding="utf-8") as f:
            json.dump(self.features, f)

        val_p = best.predict_proba(X_val)[:, 1]
        test_p = best.predict_proba(X_test)[:, 1]
        thresh = 0.5
        m = {
            "best_params": study.best_params,
            "val_auroc": float(roc_auc_score(y_val, val_p)) if y_val.nunique() > 1 else 0.5,
            "test_auroc": float(roc_auc_score(y_test, test_p)) if y_test.nunique() > 1 else 0.5,
            "val_precision_at_05": float(precision_score(y_val, (val_p >= thresh).astype(int), zero_division=0)),
            "val_recall_at_05": float(recall_score(y_val, (val_p >= thresh).astype(int), zero_division=0)),
            "n_features": len(self.features),
            "n_train": int(len(X_train)),
            "train_pos_rate": float(y_train.mean()),
        }
        with open(MODELS_DIR / "vent_classifier_metrics.json", "w", encoding="utf-8") as f:
            json.dump(m, f, indent=2)
        return m

    def load(self) -> None:
        self.model = xgb.XGBClassifier()
        self.model.load_model(str(MODELS_DIR / "vent_classifier.ubj"))
        self.features = json.loads(Path(MODELS_DIR / "vent_classifier_features.json").read_text(encoding="utf-8"))

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            self.load()
        for c in self.features:
            if c not in X.columns:
                X = X.assign(**{c: 0.0})
        return self.model.predict_proba(X[self.features].fillna(0))[:, 1]
