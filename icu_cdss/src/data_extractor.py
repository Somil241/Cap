"""Chunked extraction of vitals (chartevents) and labs (labevents).

Reads each huge CSV in fixed chunks, keeps only rows whose itemid is in
config.{VITAL_ITEMIDS, LAB_ITEMIDS}, and writes filtered output to parquet.
The parquet cache is sacred: once vitals.parquet / labs.parquet exist, they
are reused by downstream modules. Pass force=True to rebuild.
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from config import CHUNKSIZE, LAB_ITEMIDS, MIMIC_HOSP_DIR, MIMIC_ICU_DIR, PROCESSED_DIR, VITAL_ITEMIDS
from utils import mimic_file


def _extract_filtered(
    source_path: Path,
    compression: str | None,
    itemids: set[int],
    output_path: Path,
    usecols: list[str],
    max_rows: int | None,
    label: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = None
    seen = 0
    kept = 0
    started = time.time()
    try:
        for chunk in pd.read_csv(
            source_path,
            compression=compression,
            chunksize=CHUNKSIZE,
            usecols=usecols,
            low_memory=False,
        ):
            seen += len(chunk)
            filtered = chunk[chunk["itemid"].isin(itemids)]
            if not filtered.empty:
                table = pa.Table.from_pandas(filtered, preserve_index=False)
                if writer is None:
                    writer = pq.ParquetWriter(output_path, table.schema)
                writer.write_table(table)
                kept += len(filtered)
            if seen % (CHUNKSIZE * 20) == 0:
                elapsed = time.time() - started
                rate = seen / max(elapsed, 1e-6)
                print(f"[{label}] scanned={seen:,} kept={kept:,} "
                      f"rate={rate:,.0f} rows/s elapsed={elapsed:0.1f}s", flush=True)
            if max_rows and seen >= max_rows:
                break
    finally:
        if writer is not None:
            writer.close()
        else:
            pd.DataFrame(columns=usecols).to_parquet(output_path, index=False)
    elapsed = time.time() - started
    print(f"[{label}] DONE scanned={seen:,} kept={kept:,} elapsed={elapsed:0.1f}s -> {output_path}",
          flush=True)


def run_extraction(max_chart_rows: int | None = None,
                   max_lab_rows: int | None = None,
                   force: bool = False) -> None:
    vitals_out = PROCESSED_DIR / "vitals.parquet"
    labs_out = PROCESSED_DIR / "labs.parquet"
    chart_path, chart_comp = mimic_file(MIMIC_ICU_DIR, "chartevents")
    lab_path, lab_comp = mimic_file(MIMIC_HOSP_DIR, "labevents")

    if force or not vitals_out.exists() or vitals_out.stat().st_size < 100_000:
        _extract_filtered(
            chart_path, chart_comp, set(VITAL_ITEMIDS.values()),
            vitals_out,
            usecols=["subject_id", "hadm_id", "stay_id", "itemid", "charttime", "valuenum"],
            max_rows=max_chart_rows, label="vitals",
        )
    else:
        print(f"[vitals] cached -> {vitals_out} ({vitals_out.stat().st_size:,} bytes)")

    if force or not labs_out.exists() or labs_out.stat().st_size < 100_000:
        _extract_filtered(
            lab_path, lab_comp, set(LAB_ITEMIDS.values()),
            labs_out,
            usecols=["subject_id", "hadm_id", "itemid", "charttime", "valuenum"],
            max_rows=max_lab_rows, label="labs",
        )
    else:
        print(f"[labs] cached -> {labs_out} ({labs_out.stat().st_size:,} bytes)")
