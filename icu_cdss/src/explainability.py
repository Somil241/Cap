from pathlib import Path

import numpy as np
import shap

from config import OUTPUTS_DIR


def top_shap_features_tree(model, X, top_k: int = 5):
    explainer = shap.TreeExplainer(model)
    values = explainer.shap_values(X)
    if isinstance(values, list):
        values = values[1] if len(values) > 1 else values[0]
    mean_abs = np.abs(values).mean(axis=0)
    idx = np.argsort(mean_abs)[::-1][:top_k]
    return [{"name": X.columns[i], "shap": float(mean_abs[i])} for i in idx]


def save_waterfall(model, X_row, out_name: str) -> str:
    out_dir = OUTPUTS_DIR / "shap"
    out_dir.mkdir(parents=True, exist_ok=True)
    explainer = shap.TreeExplainer(model)
    sv = explainer(X_row)
    shap.plots.waterfall(sv[0], show=False)
    path = Path(out_dir / out_name)
    import matplotlib.pyplot as plt

    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    return str(path)

