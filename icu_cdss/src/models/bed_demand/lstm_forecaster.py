import numpy as np
import tensorflow as tf


def build_lstm_forecaster(input_steps: int = 24 * 7, output_steps: int = 24 * 3) -> tf.keras.Model:
    inp = tf.keras.Input(shape=(input_steps, 1))
    x = tf.keras.layers.LSTM(128)(inp)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    out = tf.keras.layers.Dense(output_steps)(x)
    model = tf.keras.Model(inp, out)
    model.compile(optimizer="adam", loss="mse")
    return model


def ensemble_preds(lgb_pred: np.ndarray, lstm_pred: np.ndarray) -> np.ndarray:
    return 0.5 * lgb_pred + 0.5 * lstm_pred

