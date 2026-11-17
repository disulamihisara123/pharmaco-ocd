"""
data_loader.py
Retrieves and parses open-access NCBI GEO transcriptomic series matrix files
and platform annotation tables for OCD and SSRI treatment response cohorts.
"""

import os
import gzip
import shutil
import urllib.request
import pandas as pd
from typing import Tuple, Dict, Any

GEO_BASE_URL = "https://ftp.ncbi.nlm.nih.gov/geo/series"

def download_file(url: str, dest_path: str) -> str:
    """Download a file with progress reporting if not already cached."""
    if os.path.exists(dest_path):
        print(f"[CACHE] File already exists: {dest_path}")
        return dest_path
    
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    print(f"[DOWNLOADING] {url} -> {dest_path}")
    urllib.request.urlretrieve(url, dest_path)
    print(f"[COMPLETED] Download finished: {dest_path}")
    return dest_path

def get_geo_series_url(geo_accession: str) -> str:
    """
    Constructs the NCBI GEO FTP URL for a given series accession (e.g., GSE60190).
    Format: ftp://ftp.ncbi.nlm.nih.gov/geo/series/GSE60nnn/GSE60190/matrix/GSE60190_series_matrix.txt.gz
    """
    prefix = geo_accession[:-3] + "nnn" if len(geo_accession) > 5 else "GSEnnn"
    filename = f"{geo_accession}_series_matrix.txt.gz"
    return f"{GEO_BASE_URL}/{prefix}/{geo_accession}/matrix/{filename}"

def parse_series_matrix(file_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Parses an NCBI GEO Series Matrix file into:
    1. An expression dataframe (rows: probe IDs, columns: GSM sample IDs)
    2. A metadata dataframe (rows: GSM sample IDs, columns: sample attributes)
    """
    metadata_lines = []
    data_lines = []
    is_data = False
    
    opener = gzip.open if file_path.endswith('.gz') else open
    with opener(file_path, 'rt', encoding='utf-8', errors='ignore') as f:
        for line in f:
            if line.startswith('!series_matrix_table_begin'):
                is_data = True
                continue
            if line.startswith('!series_matrix_table_end'):
                break
            if is_data:
                data_lines.append(line)
            elif line.startswith('!Sample_'):
                metadata_lines.append(line.strip())
                
    # Parse Metadata
    meta_dict = {}
    for line in metadata_lines:
        parts = line.split('\t')
        key = parts[0].replace('!Sample_', '')
        values = [p.strip('"') for p in parts[1:]]
        meta_dict[key] = values
        
    df_meta = pd.DataFrame(meta_dict)
    if 'geo_accession' in df_meta.columns:
        df_meta.set_index('geo_accession', inplace=True)
        
    # Parse Expression Data
    from io import StringIO
    data_io = StringIO("".join(data_lines))
    df_expr = pd.read_csv(data_io, sep='\t', index_col=0)
    
    return df_expr, df_meta

if __name__ == "__main__":
    # Test stub for GSE60190
    print("Pharmaco-OCD Data Loader initialized.")
