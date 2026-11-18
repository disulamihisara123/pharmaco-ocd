"""
data_loader.py
Retrieves and parses open-access NCBI GEO transcriptomic series matrix files
and platform annotation tables for OCD and SSRI treatment response cohorts.
"""

import gzip
import os
import urllib.request
from collections import defaultdict
from io import StringIO
from typing import Dict, List, Tuple

import pandas as pd

GEO_BASE_URL = "https://ftp.ncbi.nlm.nih.gov/geo/series"
GEO_ACC_URL = "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi"


def download_file(url: str, dest_path: str) -> str:
    """Download a file if not already cached locally."""
    if os.path.exists(dest_path):
        print(f"[CACHE] File already exists: {dest_path}")
        return dest_path

    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    print(f"[DOWNLOADING] {url} -> {dest_path}")
    urllib.request.urlretrieve(url, dest_path)
    print(f"[COMPLETED] Download finished: {dest_path}")
    return dest_path


def get_geo_series_url(geo_accession: str) -> str:
    """Construct the NCBI GEO FTP URL for a given series matrix accession."""
    prefix = geo_accession[:-3] + "nnn" if len(geo_accession) > 5 else "GSEnnn"
    filename = f"{geo_accession}_series_matrix.txt.gz"
    return f"{GEO_BASE_URL}/{prefix}/{geo_accession}/matrix/{filename}"


def get_geo_platform_url(gpl_accession: str) -> str:
    """Construct the NCBI GEO acc.cgi URL for standalone platform annotation data."""
    return f"{GEO_ACC_URL}?acc={gpl_accession}&targ=self&form=text&view=data"


def download_platform_annotation(gpl_accession: str, raw_dir: str) -> str:
    """Download and gzip-cache a standalone GEO platform (GPL) annotation table."""
    dest_path = os.path.join(raw_dir, f"{gpl_accession}_platform.txt.gz")
    if os.path.exists(dest_path):
        print(f"[CACHE] Platform annotation already exists: {dest_path}")
        return dest_path

    os.makedirs(raw_dir, exist_ok=True)
    url = get_geo_platform_url(gpl_accession)
    print(f"[DOWNLOADING] {url} -> {dest_path}")
    with urllib.request.urlopen(url, timeout=60) as resp, gzip.open(dest_path, "wb") as gz_out:
        while True:
            chunk = resp.read(65536)
            if not chunk:
                break
            gz_out.write(chunk)
    print(f"[COMPLETED] Platform annotation saved: {dest_path}")
    return dest_path


def parse_platform_table(file_path: str) -> pd.DataFrame:
    """Parse a GEO platform annotation table (.txt or .txt.gz) into a DataFrame."""
    table_lines: List[str] = []
    in_table = False
    opener = gzip.open if file_path.endswith(".gz") else open

    with opener(file_path, "rt", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("!platform_table_begin"):
                in_table = True
                continue
            if line.startswith("!platform_table_end"):
                break
            if in_table:
                table_lines.append(line)

    if not table_lines:
        raise ValueError(f"No !platform_table_begin block found in {file_path}")

    return pd.read_csv(StringIO("".join(table_lines)), sep="\t", low_memory=False)


def _extract_characteristic_columns(
    char_rows: List[List[str]], n_samples: int
) -> Dict[str, List[str]]:
    """Parse repeated !Sample_characteristics_ch1 rows into named metadata columns."""
    parsed: Dict[str, List[str]] = {}
    for row_idx, values in enumerate(char_rows, start=1):
        padded = values + [""] * max(0, n_samples - len(values))
        row_pairs = []
        for val in padded[:n_samples]:
            if ":" in val:
                k, v = val.split(":", 1)
                row_pairs.append((k.strip(), v.strip()))
            else:
                row_pairs.append((f"characteristics_{row_idx}", val.strip()))

        keys_in_row = {k for k, v in row_pairs if v != "" or k != f"characteristics_{row_idx}"}
        if len(keys_in_row) == 1:
            col_name = next(iter(keys_in_row))
            target_key = col_name if col_name not in parsed else f"{col_name}_{row_idx}"
            parsed[target_key] = [v for _, v in row_pairs]
        else:
            # Mixed attribute keys across samples in the same row
            for sample_i, (k, v) in enumerate(row_pairs):
                if k not in parsed:
                    parsed[k] = [""] * n_samples
                parsed[k][sample_i] = v
    return parsed


def parse_series_matrix(file_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Parse an NCBI GEO Series Matrix file into:
    1. An expression DataFrame (rows: probe IDs, columns: GSM sample IDs)
    2. A metadata DataFrame (rows: GSM sample IDs, columns: sample attributes)
    """
    metadata_rows: Dict[str, List[List[str]]] = defaultdict(list)
    data_lines: List[str] = []
    is_data = False

    opener = gzip.open if file_path.endswith(".gz") else open
    with opener(file_path, "rt", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("!series_matrix_table_begin"):
                is_data = True
                continue
            if line.startswith("!series_matrix_table_end"):
                break
            if is_data:
                data_lines.append(line)
            elif line.startswith("!Sample_"):
                parts = line.rstrip("\r\n").split("\t")
                key = parts[0].replace("!Sample_", "")
                values = [p.strip('"') for p in parts[1:]]
                metadata_rows[key].append(values)

    if not metadata_rows or not data_lines:
        raise ValueError(f"Invalid or incomplete GEO series matrix file: {file_path}")

    # Determine sample count from geo_accession if available
    n_samples = len(metadata_rows["geo_accession"][0]) if "geo_accession" in metadata_rows else max(
        len(v) for rows in metadata_rows.values() for v in rows
    )

    meta_dict: Dict[str, List[str]] = {}
    for key, rows in metadata_rows.items():
        if key == "characteristics_ch1":
            char_cols = _extract_characteristic_columns(rows, n_samples)
            for c_key, c_vals in char_cols.items():
                safe_key = c_key if c_key not in meta_dict else f"char_{c_key}"
                meta_dict[safe_key] = c_vals
        elif len(rows) == 1:
            vals = rows[0] + [""] * max(0, n_samples - len(rows[0]))
            meta_dict[key] = vals[:n_samples]
        else:
            for idx, vals in enumerate(rows, start=1):
                padded = vals + [""] * max(0, n_samples - len(vals))
                meta_dict[f"{key}_{idx}"] = padded[:n_samples]

    df_meta = pd.DataFrame(meta_dict)
    if "geo_accession" in df_meta.columns:
        df_meta.set_index("geo_accession", inplace=True)

    # Parse Expression Data
    df_expr = pd.read_csv(StringIO("".join(data_lines)), sep="\t", index_col=0, low_memory=False)
    df_expr.index = df_expr.index.astype(str).str.strip('"')
    df_expr.columns = [str(c).strip('"') for c in df_expr.columns]
    df_expr = df_expr.apply(pd.to_numeric, errors="coerce")

    return df_expr, df_meta


if __name__ == "__main__":
    print("Pharmaco-OCD Data Loader initialized.")
