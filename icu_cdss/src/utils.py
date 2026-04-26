from pathlib import Path


def mimic_file(base_dir: Path, stem: str) -> tuple[Path, str]:
    gz = base_dir / f"{stem}.csv.gz"
    csv = base_dir / f"{stem}.csv"
    if gz.exists():
        return gz, "gzip"
    if csv.exists():
        return csv, None
    raise FileNotFoundError(f"Missing {stem}.csv(.gz) in {base_dir}")

