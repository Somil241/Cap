"""Temporal split + SMOTE.

Produces:
  - train.parquet, val.parquet, test.parquet : full feature frames with stay_id /
    hadm_id / subject_id / charttime preserved (used by vent + LOS heads).
  - train_sepsis.parquet : SMOTE-balanced training set for the sepsis classifier
    (numeric features only; identifiers are stripped before resampling so SMOTE
    cannot synthesise fake patient IDs).
"""

from __future__ import annotations

import pandas as pd

from config import PROCESSED_DIR, TEST_FROM_YEAR, TRAIN_CUTOFF_YEAR, VAL_CUTOFF_YEAR

ID_COLS = ["stay_id", "hadm_id", "subject_id"]
NON_FEATURE_COLS = set(ID_COLS) | {"charttime", "label", "onset_time", "suspected_time"}


def _numeric_feature_cols(df: pd.DataFrame) -> list[str]:
    return [
        c for c in df.columns
        if c not in NON_FEATURE_COLS and pd.api.types.is_numeric_dtype(df[c])
    ]


def build_datasets() -> dict:
    features = pd.read_parquet(PROCESSED_DIR / "features.parquet")
    labels = pd.read_parquet(PROCESSED_DIR / "labels_sepsis.parquet")
    df = features.merge(labels, on=["stay_id", "charttime"], how="inner")
    df["charttime"] = pd.to_datetime(df["charttime"])
    yr = df["charttime"].dt.year

    train = df[yr < TRAIN_CUTOFF_YEAR].copy()
    val = df[(yr >= TRAIN_CUTOFF_YEAR) & (yr < VAL_CUTOFF_YEAR)].copy()
    test = df[yr >= TEST_FROM_YEAR].copy()

    train.to_parquet(PROCESSED_DIR / "train.parquet", index=False)
    val.to_parquet(PROCESSED_DIR / "val.parquet", index=False)
    test.to_parquet(PROCESSED_DIR / "test.parquet", index=False)

    feat_cols = _numeric_feature_cols(train)
    summary = {
        "train_rows": int(len(train)),
        "val_rows": int(len(val)),
        "test_rows": int(len(test)),
        "feature_count": len(feat_cols),
        "train_pos_rate": float(train["label"].mean()) if len(train) else 0.0,
        "val_pos_rate": float(val["label"].mean()) if len(val) else 0.0,
        "test_pos_rate": float(test["label"].mean()) if len(test) else 0.0,
    }

    if len(train) > 10 and train["label"].nunique() > 1:
        # Memory-safe pipeline: random-undersample the majority class to a
        # 4:1 neg:pos ratio first, then SMOTE-up to balanced. Without the
        # undersample step the resampled matrix on the full training set can
        # exceed available RAM (millions of rows x 150 cols).
        from imblearn.over_sampling import SMOTE
        from imblearn.under_sampling import RandomUnderSampler

        X_train = train[feat_cols].fillna(0).astype("float32")
        y_train = train["label"].astype(int)
        n_pos = int((y_train == 1).sum())
        n_neg = int((y_train == 0).sum())
        target_neg = min(n_neg, max(n_pos * 4, 1000))
        if n_neg > target_neg and n_pos > 0:
            rus = RandomUnderSampler(random_state=42,
                                     sampling_strategy={0: target_neg, 1: n_pos})
            X_train, y_train = rus.fit_resample(X_train, y_train)

        minority = int(pd.Series(y_train).value_counts().min())
        k = max(1, min(5, minority - 1))
        if minority > 1:
            smote = SMOTE(random_state=42, k_neighbors=k)
            X_res, y_res = smote.fit_resample(X_train, y_train)
        else:
            X_res, y_res = X_train, y_train
        train_sepsis = pd.DataFrame(X_res, columns=feat_cols)
        train_sepsis["label"] = y_res
        summary["train_sepsis_rows"] = int(len(train_sepsis))
        summary["train_sepsis_pos_rate"] = float(train_sepsis["label"].mean())
        summary["train_sepsis_undersampled_negatives"] = int(target_neg)
    else:
        train_sepsis = train[feat_cols + ["label"]].copy()
        summary["train_sepsis_rows"] = int(len(train_sepsis))
        summary["train_sepsis_pos_rate"] = float(train_sepsis["label"].mean()) if len(train_sepsis) else 0.0

    train_sepsis.to_parquet(PROCESSED_DIR / "train_sepsis.parquet", index=False)
    return summary
