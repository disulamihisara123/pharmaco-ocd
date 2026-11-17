"""
preprocess.py
Standardization, quality control, log2 normalization, and probe-to-gene mapping
for psychiatric pharmacogenomic datasets.
"""

import numpy as np
import pandas as pd
from typing import Optional, List

def check_and_apply_log2(df_expr: pd.DataFrame) -> pd.DataFrame:
    """
    Checks if expression data is already on a log2 scale.
    If the 99th percentile value > 100 or max > 250, applies log2(x + 1) transform.
    """
    max_val = np.nanmax(df_expr.values)
    p99 = np.nanpercentile(df_expr.values, 99)
    
    if max_val > 100 or p99 > 50:
        print(f"[TRANSFORM] Data appears unlogged (Max={max_val:.2f}, P99={p99:.2f}). Applying log2(x + 1)...")
        # Shift min if negative values exist
        min_val = np.nanmin(df_expr.values)
        offset = abs(min_val) + 1 if min_val < 0 else 1
        return np.log2(df_expr + offset)
    else:
        print(f"[QC] Data appears already log2-transformed (Max={max_val:.2f}, P99={p99:.2f}).")
        return df_expr

def filter_low_variance_probes(df_expr: pd.DataFrame, top_percentile: float = 0.5) -> pd.DataFrame:
    """
    Filters out uninformative, low-variance probes/genes.
    Retains the top `top_percentile` proportion of features based on interquartile range (IQR) or variance.
    """
    variances = df_expr.var(axis=1)
    threshold = variances.quantile(1.0 - top_percentile)
    retained_df = df_expr.loc[variances >= threshold]
    print(f"[VARIANCE FILTER] Reduced from {len(df_expr)} to {len(retained_df)} high-variance features.")
    return retained_df

def map_probes_to_genes(df_expr: pd.DataFrame, annotation_file: str, probe_col: str, gene_col: str) -> pd.DataFrame:
    """
    Maps microarray platform probe IDs to official HUGO Gene Symbols.
    Aggregates multi-probe expression per gene using the median.
    """
    # Load platform annotation (GPL)
    df_annot = pd.read_csv(annotation_file, sep='\t', comment='#', usecols=[probe_col, gene_col])
    df_annot.dropna(subset=[gene_col], inplace=True)
    
    # Strip any extra metadata or multiple symbols (take primary symbol)
    df_annot['Gene'] = df_annot[gene_col].apply(lambda x: str(x).split('///')[0].strip())
    annot_map = dict(zip(df_annot[probe_col], df_annot['Gene']))
    
    df_mapped = df_expr.copy()
    df_mapped['Gene'] = df_mapped.index.map(annot_map)
    df_mapped.dropna(subset=['Gene'], inplace=True)
    
    # Collapse duplicate gene probes via median expression
    df_gene_level = df_mapped.groupby('Gene').median()
    print(f"[MAPPING] Successfully mapped probes to {len(df_gene_level)} unique gene symbols.")
    return df_gene_level
