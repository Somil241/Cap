import json

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import roc_auc_score

from config import LSTM_SEQUENCE_HOURS, MODELS_DIR, PROCESSED_DIR


class SepsisLSTM:
    def __init__(self) -> None:
        self.model: tf.keras.Model | None = None
        self.features: list[str] = []

    @staticmethod
    def _focal_loss(alpha: float = 0.25, gamma: float = 2.0):
        def loss(y_true, y_pred):
            y_true = tf.cast(y_true, tf.float32)
            eps = tf.keras.backend.epsilon()
            y_pred = tf.clip_by_value(y_pred, eps, 1.0 - eps)
            pt = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
            w = alpha * tf.pow(1 - pt, gamma)
            return tf.reduce_mean(-w * tf.math.log(pt))

        return loss

    def _build(self, feature_dim: int) -> tf.keras.Model:
        inp = tf.keras.Input(shape=(LSTM_SEQUENCE_HOURS, feature_dim))
        x = tf.keras.layers.LSTM(128, return_sequences=True)(inp)
        x = tf.keras.layers.LSTM(64)(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        out = tf.keras.layers.Dense(1, activation="sigmoid")(x)
        model = tf.keras.Model(inp, out)
        model.compile(optimizer="adam", loss=self._focal_loss(), metrics=[tf.keras.metrics.AUC(name="auc")])
        return model

    def _to_sequences(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        self.features = [c for c in df.columns if c not in ["label", "charttime"] and pd.api.types.is_numeric_dtype(df[c])]
        X_rows = df[self.features].fillna(0).to_numpy(dtype=np.float32)
        y_rows = df["label"].astype(np.float32).to_numpy()
        X_seq, y_seq = [], []
        for i in range(LSTM_SEQUENCE_HOURS, len(df)):
            X_seq.append(X_rows[i - LSTM_SEQUENCE_HOURS : i])
            y_seq.append(y_rows[i])
        return np.array(X_seq), np.array(y_seq)

    def train(self, epochs: int = 20) -> dict:
        train = pd.read_parquet(PROCESSED_DIR / "train.parquet")
        val = pd.read_parquet(PROCESSED_DIR / "val.parquet")
        test = pd.read_parquet(PROCESSED_DIR / "test.parquet")
        X_train, y_train = self._to_sequences(train)
        X_val, y_val = self._to_sequences(val)
        X_test, y_test = self._to_sequences(test)

        self.model = self._build(X_train.shape[-1])
        cb = [tf.keras.callbacks.EarlyStopping(monitor="val_auc", mode="max", patience=5, restore_best_weights=True)]
        self.model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=128, callbacks=cb, verbose=1)

        val_prob = self.model.predict(X_val, verbose=0).ravel()
        test_prob = self.model.predict(X_test, verbose=0).ravel()
        metrics = {
            "val_auroc": float(roc_auc_score(y_val, val_prob)) if len(np.unique(y_val)) > 1 else 0.5,
            "test_auroc": float(roc_auc_score(y_test, test_prob)) if len(np.unique(y_test)) > 1 else 0.5,
        }
        self.model.save(MODELS_DIR / "sepsis_lstm.keras")
        with open(MODELS_DIR / "sepsis_lstm_features.json", "w", encoding="utf-8") as f:
            json.dump(self.features, f)
        return metrics

    def load(self) -> None:
        self.model = tf.keras.models.load_model(MODELS_DIR / "sepsis_lstm.keras", compile=False)
        with open(MODELS_DIR / "sepsis_lstm_features.json", "r", encoding="utf-8") as f:
            self.features = json.load(f)

