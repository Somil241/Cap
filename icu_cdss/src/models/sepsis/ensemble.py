import numpy as np
from sklearn.metrics import f1_score


def ensemble_prob(xgb_prob: np.ndarray, lstm_prob: np.ndarray) -> np.ndarray:
    n = min(len(xgb_prob), len(lstm_prob))
    return 0.6 * xgb_prob[:n] + 0.4 * lstm_prob[:n]


def best_f1_threshold(y_true: np.ndarray, prob: np.ndarray) -> float:
    best_t, best_s = 0.5, -1.0
    for t in np.linspace(0.1, 0.9, 81):
        s = f1_score(y_true, (prob >= t).astype(int))
        if s > best_s:
            best_t, best_s = float(t), float(s)
    return best_t

