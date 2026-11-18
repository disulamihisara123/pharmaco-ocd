"""
preprocess.py
Standardization, quality control, log2 normalization, probe-to-gene mapping,
sample QC outlier detection, and cross-cohort harmonization for psychiatric
pharmacogenomic datasets.
"""

import os
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from data_loader import parse_platform_table

COMMON_GENE_COLUMNS = [
    "Symbol",
    "GENE_SYMBOL",
    "Gene Symbol",
    "Gene_Symbol",
    "ILMN_Gene",
    "gene_symbol",
]


def clean_missing_probes(
    df_expr: pd.DataFrame, max_missing_rate: float = 0.20
) -> pd.DataFrame:
    """Drop probes exceeding `max_missing_rate` and median-impute sparse NaNs."""
    missing_rates = df_expr.isna().mean(axis=1)
    df_clean = df_expr.loc[missing_rates <= max_missing_rate].copy()
    dropped = len(df_expr) - len(df_clean)
    if dropped > 0:
        print(f"[MISSING FILTER] Dropped {dropped} probes with >{max_missing_rate*100:.0f}% missing values.")

    if df_clean.isna().any().any():
        row_medians = df_clean.median(axis=1)
        df_clean = df_clean.T.fillna(row_medians).T
        print("[IMPUTATION] Imputed remaining sparse NaNs with probe medians.")
    return df_clean


def check_and_apply_log2(df_expr: pd.DataFrame) -> pd.DataFrame:
    """
    Check if expression data is already on a log2 scale.
    If max > 100 or 99th percentile > 50, apply log2(x + offset) transform.
    """
    max_val = float(np.nanmax(df_expr.values))
    min_val = float(np.nanmin(df_expr.values))
    p99 = float(np.nanpercentile(df_expr.values, 99))

    if max_val > 100 or p99 > 50:
        print(f"[TRANSFORM] Data appears unlogged (Min={min_val:.2f}, Max={max_val:.2f}, P99={p99:.2f}). Applying log2...")
        offset = abs(min_val) + 1.0 if min_val <= 0 else 1.0
        return np.log2(df_expr + offset)

    print(f"[QC] Data appears already log2-scaled (Min={min_val:.2f}, Max={max_val:.2f}, P99={p99:.2f}).")
    return df_expr


def filter_low_variance_probes(
    df_expr: pd.DataFrame, top_percentile: float = 0.5
) -> pd.DataFrame:
    """Retain the top `top_percentile` proportion of features by variance."""
    variances = df_expr.var(axis=1)
    threshold = variances.quantile(1.0 - top_percentile)
    retained_df = df_expr.loc[variances >= threshold]
    print(f"[VARIANCE FILTER] Reduced from {len(df_expr)} to {len(retained_df)} high-variance features.")
    return retained_df


def _load_annotation_table(annotation: Union[str, pd.DataFrame]) -> pd.DataFrame:
    """Load a platform annotation table from a DataFrame or GEO platform file."""
    if isinstance(annotation, pd.DataFrame):
        return annotation.copy()
    if not os.path.exists(annotation):
        raise FileNotFoundError(f"Annotation file not found: {annotation}")
    try:
        return parse_platform_table(annotation)
    except ValueError:
        return pd.read_csv(annotation, sep="\t", comment="#", low_memory=False)


def _detect_gene_column(df_annot: pd.DataFrame, gene_col: Optional[str]) -> str:
    """Resolve the gene symbol column name in a platform annotation table."""
    if gene_col is not None:
        if gene_col not in df_annot.columns:
            raise KeyError(f"Column '{gene_col}' not found in annotation columns: {list(df_annot.columns)}")
        return gene_col
    for candidate in COMMON_GENE_COLUMNS:
        if candidate in df_annot.columns:
            return candidate
    raise KeyError(f"Could not auto-detect gene symbol column in: {list(df_annot.columns)}")


def map_probes_to_genes(
    df_expr: pd.DataFrame,
    annotation: Union[str, pd.DataFrame],
    probe_col: str = "ID",
    gene_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    Map microarray platform probe IDs to official HUGO Gene Symbols.
    Removes control probes and collapses duplicate probes per gene via median.
    """
    df_annot = _load_annotation_table(annotation)
    resolved_gene_col = _detect_gene_column(df_annot, gene_col)

    # Filter out Agilent/platform control probes when annotated
    if "CONTROL_TYPE" in df_annot.columns:
        ctrl_vals = df_annot["CONTROL_TYPE"].astype(str).str.upper()
        if (ctrl_vals == "FALSE").any():
            df_annot = df_annot.loc[ctrl_vals == "FALSE"].copy()

    df_sub = df_annot[[probe_col, resolved_gene_col]].dropna(subset=[probe_col, resolved_gene_col]).copy()
    df_sub[probe_col] = df_sub[probe_col].astype(str).str.strip()

    # Extract primary gene symbol from multi-mapped annotations
    df_sub["Gene"] = (
        df_sub[resolved_gene_col]
        .astype(str)
        .str.split("///|//|;")
        .str[0]
        .str.strip()
    )
    invalid_tokens = {"", "---", "NA", "NAN", "NONE", "NULL"}
    df_sub = df_sub.loc[~df_sub["Gene"].str.upper().isin(invalid_tokens)]

    annot_map = dict(zip(df_sub[probe_col], df_sub["Gene"]))

    df_mapped = df_expr.copy()
    df_mapped.index = df_mapped.index.astype(str).str.strip()
    df_mapped["Gene"] = df_mapped.index.map(annot_map)
    df_mapped = df_mapped.dropna(subset=["Gene"])

    df_gene_level = df_mapped.groupby("Gene").median(numeric_only=True)
    print(f"[MAPPING] Mapped {len(df_mapped)} probes -> {len(df_gene_level)} unique gene symbols (col='{resolved_gene_col}').")
    return df_gene_level


def compute_sample_qc_metrics(
    df_expr: pd.DataFrame, z_threshold: float = 2.5
) -> pd.DataFrame:
    """
    Compute sample-level quality metrics (median, IQR, missing rate, mean
    inter-sample Pearson correlation) and flag statistical outlier samples.
    """
    corr_matrix = df_expr.corr(method="pearson")
    # Exclude self-correlation (diagonal = 1.0) when computing mean r
    n_samples = corr_matrix.shape[0]
    if n_samples > 1:
        mean_corr = (corr_matrix.sum(axis=1) - 1.0) / (n_samples - 1)
    else:
        mean_corr = pd.Series(1.0, index=df_expr.columns)

    corr_std = mean_corr.std()
    z_corr = (mean_corr - mean_corr.mean()) / corr_std if corr_std > 0 else pd.Series(0.0, index=df_expr.columns)

    qc_df = pd.DataFrame({
        "sample_median": df_expr.median(axis=0),
        "sample_iqr": df_expr.quantile(0.75, axis=0) - df_expr.quantile(0.25, axis=0),
        "missing_rate": df_expr.isna().mean(axis=0),
        "mean_sample_corr": mean_corr,
        "corr_zscore": z_corr,
    })
    qc_df["qc_flag"] = (qc_df["corr_zscore"] < -z_threshold) | (qc_df["missing_rate"] > 0.20)
    return qc_df


def standardize_cohort_zscore(df_gene_expr: pd.DataFrame) -> pd.DataFrame:
    """Apply gene-wise Z-score standardization within a single cohort."""
    means = df_gene_expr.mean(axis=1)
    stds = df_gene_expr.std(axis=1).replace(0, np.nan)
    df_z = df_gene_expr.sub(means, axis=0).div(stds, axis=0)
    return df_z.dropna(how="any")


def harmonize_cohorts(
    cohort_expr_dict: Dict[str, pd.DataFrame],
    zscore: bool = True,
) -> Tuple[Dict[str, pd.DataFrame], List[str]]:
    """
    Align multiple gene-level expression matrices to their shared HUGO gene
    symbol intersection, optionally Z-score standardizing within each cohort.
    """
    if not cohort_expr_dict:
        raise ValueError("cohort_expr_dict cannot be empty")

    common_genes = sorted(
        set.intersection(*(set(df.index) for df in cohort_expr_dict.values()))
    )
    print(f"[HARMONIZE] Found {len(common_genes)} shared genes across {len(cohort_expr_dict)} cohorts.")

    harmonized: Dict[str, pd.DataFrame] = {}
    for name, df in cohort_expr_dict.items():
        aligned = df.loc[common_genes].copy()
        if zscore:
            aligned = standardize_cohort_zscore(aligned)
        harmonized[name] = aligned

    # Re-intersect in case zero-variance genes were dropped during Z-scoring
    final_genes = sorted(
        set.intersection(*(set(df.index) for df in harmonized.values()))
    )
    for name in harmonized:
        harmonized[name] = harmonized[name].loc[final_genes]

    print(f"[HARMONIZE] Final harmonized feature space: {len(final_genes)} genes.")
    return harmonized, final_genes
