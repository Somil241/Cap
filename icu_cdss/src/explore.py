import pandas as pd

from config import LAB_ITEMIDS, MIMIC_HOSP_DIR, MIMIC_ICU_DIR, OUTPUTS_DIR, VITAL_ITEMIDS, ensure_dirs
from utils import mimic_file


def run_explore() -> None:
    ensure_dirs()
    files = [mimic_file(MIMIC_ICU_DIR, "icustays"), mimic_file(MIMIC_ICU_DIR, "chartevents"), mimic_file(MIMIC_HOSP_DIR, "patients"), mimic_file(MIMIC_HOSP_DIR, "labevents"), mimic_file(MIMIC_ICU_DIR, "d_items"), mimic_file(MIMIC_HOSP_DIR, "d_labitems")]
    lines = []
    for p, comp in files:
        try:
            df = pd.read_csv(p, compression=comp, nrows=1000)
            lines.append(f"\n== {p.name} ==")
            lines.append(f"columns={list(df.columns)}")
            lines.append(str(df.dtypes))
            lines.append(f"sample_rows={len(df)}")
        except Exception as e:
            lines.append(f"{p.name}: {e}")

    d_items_path, d_items_comp = mimic_file(MIMIC_ICU_DIR, "d_items")
    d_labs_path, d_labs_comp = mimic_file(MIMIC_HOSP_DIR, "d_labitems")
    d_items = pd.read_csv(d_items_path, compression=d_items_comp)
    d_labs = pd.read_csv(d_labs_path, compression=d_labs_comp)
    lines.append(f"vital_itemids_found={set(VITAL_ITEMIDS.values()).issubset(set(d_items['itemid']))}")
    lines.append(f"lab_itemids_found={set(LAB_ITEMIDS.values()).issubset(set(d_labs['itemid']))}")
    (OUTPUTS_DIR / "exploration_report.txt").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    run_explore()

