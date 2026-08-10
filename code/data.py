from __future__ import annotations

import csv
import hashlib
import io
import json
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


FRENCH_BASE = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp"
FILES = {
    "factors5": "F-F_Research_Data_5_Factors_2x3_daily_CSV.zip",
    "momentum": "F-F_Momentum_Factor_daily_CSV.zip",
    "short_reversal": "F-F_ST_Reversal_Factor_daily_CSV.zip",
    "long_reversal": "F-F_LT_Reversal_Factor_daily_CSV.zip",
    "industries30": "30_Industry_Portfolios_daily_CSV.zip",
}

EXPECTED_SHA256 = {
    "F-F_Research_Data_5_Factors_2x3_daily_CSV.zip": (
        "bcf32ecc9e2bb20383784ac98891e42146a0091eec6ec77d3b5bf0d4e981e3f6"
    ),
    "F-F_Momentum_Factor_daily_CSV.zip": (
        "f4237e2e36dffa13fd7823f55376316a94b5ac663af951dd9eaca8ed2c678bcf"
    ),
    "F-F_ST_Reversal_Factor_daily_CSV.zip": (
        "2114d1ff89842ee703d6c51bbde015c129df2f9edc97cf67fef2d7e278db743b"
    ),
    "F-F_LT_Reversal_Factor_daily_CSV.zip": (
        "0e6ce053ea0c159bb07346cd11c391104297a89adbd7424585fd84f2c7544546"
    ),
    "30_Industry_Portfolios_daily_CSV.zip": (
        "7140a2dbbae2b9fa871ac99c223c4310efd2aa526bb9fa6170b118dbd1d61848"
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(raw_dir: Path) -> dict[str, dict[str, str | int]]:
    raw_dir.mkdir(parents=True, exist_ok=True)
    provenance: dict[str, dict[str, str | int]] = {}
    for key, filename in FILES.items():
        destination = raw_dir / filename
        if not destination.exists() or not zipfile.is_zipfile(destination):
            request = urllib.request.Request(
                f"{FRENCH_BASE}/{filename}",
                headers={"User-Agent": "SAFE-ALPHA research replication"},
            )
            with urllib.request.urlopen(request, timeout=90) as response:
                destination.write_bytes(response.read())
        if not zipfile.is_zipfile(destination):
            raise ValueError(f"Downloaded file is not a ZIP archive: {destination}")
        observed_hash = sha256(destination)
        expected_hash = EXPECTED_SHA256[filename]
        if observed_hash != expected_hash:
            raise ValueError(
                f"Data hash mismatch for {filename}: expected "
                f"{expected_hash}, observed {observed_hash}. "
                "Use the archived replication input rather than a revised file."
            )
        provenance[key] = {
            "filename": filename,
            "url": f"{FRENCH_BASE}/{filename}",
            "bytes": destination.stat().st_size,
            "sha256": observed_hash,
        }
    (raw_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return provenance


def _first_daily_table(path: Path, marker: str | None = None) -> pd.DataFrame:
    with zipfile.ZipFile(path) as archive:
        member = archive.namelist()[0]
        text = archive.read(member).decode("utf-8", errors="replace")
    lines = text.splitlines()
    start = 0
    if marker is not None:
        start = next(i for i, line in enumerate(lines) if marker in line)
    header = next(
        i
        for i in range(start, len(lines) - 1)
        if lines[i].startswith(",")
        and len(lines[i + 1].split(",", 1)[0].strip()) == 8
        and lines[i + 1].split(",", 1)[0].strip().isdigit()
    )
    columns = ["date"] + [
        value.strip() for value in next(csv.reader([lines[header]]))[1:] if value.strip()
    ]
    rows: list[list[str]] = []
    for line in lines[header + 1 :]:
        values = [value.strip() for value in next(csv.reader([line]))]
        if not values or len(values[0]) != 8 or not values[0].isdigit():
            break
        values = values[: len(columns)]
        if len(values) == len(columns):
            rows.append(values)
    frame = pd.DataFrame(rows, columns=columns)
    frame["date"] = pd.to_datetime(frame["date"], format="%Y%m%d")
    frame = frame.set_index("date").apply(pd.to_numeric, errors="coerce") / 100.0
    frame = frame.replace({-0.9999: np.nan, -9.99: np.nan})
    frame = frame.dropna(how="all")
    return frame


def load_panel(raw_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    factors = _first_daily_table(raw_dir / FILES["factors5"])
    momentum = _first_daily_table(raw_dir / FILES["momentum"])
    short_reversal = _first_daily_table(raw_dir / FILES["short_reversal"])
    long_reversal = _first_daily_table(raw_dir / FILES["long_reversal"])
    industries = _first_daily_table(
        raw_dir / FILES["industries30"],
        marker="Average Value Weighted Returns -- Daily",
    )
    factor_panel = pd.concat(
        [
            factors.drop(columns=["RF"]),
            momentum.rename(columns={momentum.columns[0]: "MOM"}),
            short_reversal.rename(columns={short_reversal.columns[0]: "STREV"}),
            long_reversal.rename(columns={long_reversal.columns[0]: "LTREV"}),
        ],
        axis=1,
        join="inner",
    )
    index = factor_panel.index.intersection(industries.index).sort_values()
    factor_panel = factor_panel.loc[index].astype(float)
    industries = industries.loc[index].astype(float)
    risk_free = factors.loc[index, "RF"].astype(float)
    for name, frame in [
        ("factor panel", factor_panel),
        ("industry panel", industries),
    ]:
        if not frame.index.is_monotonic_increasing or not frame.index.is_unique:
            raise ValueError(f"{name} dates must be sorted and unique")
    if industries.shape[1] != 30:
        raise ValueError(
            f"Expected 30 industry portfolios, observed {industries.shape[1]}"
        )
    expected_factors = {"Mkt-RF", "SMB", "HML", "RMW", "CMA", "MOM", "STREV", "LTREV"}
    if set(factor_panel.columns) != expected_factors:
        raise ValueError(
            f"Unexpected factor columns: {sorted(factor_panel.columns)}"
        )
    if pd.Timestamp("2007-01-03") not in index or pd.Timestamp("2026-05-29") not in index:
        raise ValueError("Archived reporting-window endpoints are missing")
    return industries, factor_panel, risk_free


def write_processed(
    processed_dir: Path,
    industries: pd.DataFrame,
    factors: pd.DataFrame,
    risk_free: pd.Series,
) -> None:
    processed_dir.mkdir(parents=True, exist_ok=True)
    industries.to_csv(processed_dir / "industries30_daily.csv", float_format="%.8f")
    factors.to_csv(processed_dir / "factors_daily.csv", float_format="%.8f")
    risk_free.rename("RF").to_csv(
        processed_dir / "risk_free_daily.csv", float_format="%.8f"
    )


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    raw = root / "data" / "raw"
    processed = root / "data" / "processed"
    download(raw)
    industry_returns, factor_returns, rf = load_panel(raw)
    write_processed(processed, industry_returns, factor_returns, rf)
