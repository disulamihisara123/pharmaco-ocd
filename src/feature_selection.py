"""
feature_selection.py
Pathway-guided and ElasticNet stability feature selection for identifying a
compact 10-15 gene SSRI response biomarker panel in Pharmaco-OCD.
"""

import inspect
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests

PATHWAY_GENE_SETS: Dict[str, List[str]] = {
    "Serotonergic": [
        "SLC6A4", "HTR1A", "HTR1B", "HTR2A", "HTR2C", "HTR3A", "HTR7",
        "TPH1", "TPH2", "MAOA", "MAOB", "DDC", "GNAQ", "PLCB1",
    ],
    "Glutamatergic": [
        "SLC1A1", "SLC1A2", "SLC1A3", "DLGAP3", "DLGAP1", "GRIN2A", "GRIN2B",
        "GRIA1", "GRIA2", "GRM5", "GRM8", "BDNF", "NTRK2", "SHANK3", "HOMER1", "CAMK2A",
    ],
    "Neuroinflammatory": [
        "IL6", "IL1B", "TNF", "NFKB1", "PTGS2", "IDO1", "KYNU",
        "NR3C1", "FKBP5", "CRH", "COMT", "DRD2", "DRD4", "ADRA2A",
    ],
}

GENE_TO_PATHWAY: Dict[str, str] = {
    gene: pathway
    for pathway, genes in PATHWAY_GENE_SETS.items()
    for gene in genes
}


def get_gene_pathway(gene: str) -> str:
    """Return the functional biological pathway label for a HUGO gene symbol."""
    return GENE_TO_PATHWAY.get(gene, "OCD Neuropathology")


def get_all_pathway_genes(available_genes: Optional[Sequence[str]] = None) -> List[str]:
    """Return unique curated pathway genes, optionally filtered to available features."""
    ordered = list(dict.fromkeys(GENE_TO_PATHWAY.keys()))
    if available_genes is None:
        return ordered
    avail_set = set(available_genes)
    return [g for g in ordered if g in avail_set]


def make_elasticnet_logreg(
    C: float = 0.5,
    l1_ratio: float = 0.5,
    max_iter: int = 3000,
    random_state: int = 42,
) -> LogisticRegression:
    """
    Construct a version-safe ElasticNet LogisticRegression across scikit-learn
    1.3-1.7 (where penalty='elasticnet' is required) and >=1.8 (where penalty is deprecated).
    """
    sig = inspect.signature(LogisticRegression.__init__)
    penalty_param = sig.parameters.get("penalty")
    kwargs = {
        "C": C,
        "l1_ratio": l1_ratio,
        "solver": "saga" if l1_ratio > 0 else "lbfgs",
        "max_iter": max_iter,
        "random_state": random_state,
    }
    if penalty_param is not None and penalty_param.default != "deprecated":
        kwargs["penalty"] = "elasticnet" if 0.0 < l1_ratio < 1.0 else ("l1" if l1_ratio == 1.0 else "l2")
    return LogisticRegression(**kwargs)


def compute_ocd_differential_expression(
    df_gene_expr: pd.DataFrame,
    df_meta: pd.DataFrame,
    case_mask: Optional[pd.Series] = None,
    ctrl_mask: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """
    Compute Welch t-test differential expression between OCD cases and pure
    controls in GSE60190 (strictly independent of SSRI treatment outcome labels).
    """
    aligned_meta = df_meta.loc[df_gene_expr.columns]
    if case_mask is None:
        case_mask = aligned_meta["ocd"] == "OCD"
    if ctrl_mask is None:
        ctrl_mask = aligned_meta["dx"] == "Control"

    case_cols = aligned_meta.index[case_mask]
    ctrl_cols = aligned_meta.index[ctrl_mask]

    X_case = df_gene_expr[case_cols].T
    X_ctrl = df_gene_expr[ctrl_cols].T

    t_stats, p_vals = stats.ttest_ind(X_case, X_ctrl, equal_var=False, axis=0, nan_policy="omit")
    delta = X_case.mean(axis=0).values - X_ctrl.mean(axis=0).values
    p_vals = np.nan_to_num(p_vals, nan=1.0)
    _, fdr, _, _ = multipletests(p_vals, method="fdr_bh")

    deg_df = pd.DataFrame({
        "delta_mean": delta,
        "t_stat": np.nan_to_num(t_stats, nan=0.0),
        "abs_t": np.abs(np.nan_to_num(t_stats, nan=0.0)),
        "pval": p_vals,
        "fdr": fdr,
        "pathway": [get_gene_pathway(g) for g in df_gene_expr.index],
    }, index=df_gene_expr.index)
    deg_df.index.name = "Gene"
    return deg_df.sort_values("pval", ascending=True)


def build_candidate_pool(
    deg_df: pd.DataFrame,
    available_genes: Sequence[str],
    n_ocd_top: int = 200,
) -> List[str]:
    """Combine top OCD neuropathology DEGs with curated SSRI/OCD pathway genes."""
    avail_set = set(available_genes)
    ocd_top = [g for g in deg_df.index[:n_ocd_top] if g in avail_set]
    pathway_genes = get_all_pathway_genes(available_genes)
    return sorted(set(ocd_top) | set(pathway_genes))


class PathwayElasticNetSelector(BaseEstimator, TransformerMixin):
    """
    Leakage-free sklearn transformer that standardizes features on the training
    fold and retains the top-k genes ranked by ElasticNet Logistic Regression |coef|.
    """

    def __init__(
        self,
        k: int = 15,
        C: float = 0.4,
        l1_ratio: float = 0.7,
        max_iter: int = 3000,
        random_state: int = 42,
    ):
        self.k = k
        self.C = C
        self.l1_ratio = l1_ratio
        self.max_iter = max_iter
        self.random_state = random_state

    def fit(self, X: pd.DataFrame, y: np.ndarray):
        self.feature_names_in_ = list(X.columns) if hasattr(X, "columns") else [f"f{i}" for i in range(X.shape[1])]
        X_arr = np.asarray(X, dtype=float)
        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X_arr)

        clf = make_elasticnet_logreg(
            C=self.C,
            l1_ratio=self.l1_ratio,
            max_iter=self.max_iter,
            random_state=self.random_state,
        )
        clf.fit(X_scaled, y)
        coefs = np.abs(clf.coef_[0])
        n_keep = min(self.k, X_arr.shape[1])
        self.selected_indices_ = np.argsort(coefs)[::-1][:n_keep]
        self.selected_features_ = [self.feature_names_in_[i] for i in self.selected_indices_]
        self.importances_ = coefs
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        X_arr = np.asarray(X, dtype=float)
        X_scaled = self.scaler_.transform(X_arr)
        return X_scaled[:, self.selected_indices_]


def run_stability_selection(
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    candidate_genes: Sequence[str],
    deg_df: Optional[pd.DataFrame] = None,
    k: int = 15,
    n_resamples: int = 60,
    train_size: float = 0.80,
    C: float = 0.4,
    l1_ratio: float = 0.7,
    min_pathway_per_axis: int = 1,
    random_state: int = 42,
) -> Tuple[List[str], pd.DataFrame]:
    """
    Identify a stable 10-15 gene biomarker panel via donor-grouped ElasticNet
    resampling, preserving representation across the three functional axes.
    """
    cols = [g for g in candidate_genes if g in X.columns]
    X_cand = X[cols]

    splitter = GroupShuffleSplit(n_splits=n_resamples, train_size=train_size, random_state=random_state)
    weights_matrix = np.zeros((n_resamples, len(cols)), dtype=float)
    selected_counts = np.zeros(len(cols), dtype=float)

    for idx, (tr_idx, _) in enumerate(splitter.split(X_cand, y, groups)):
        y_tr = y[tr_idx]
        if len(np.unique(y_tr)) < 2:
            continue
        X_tr = StandardScaler().fit_transform(X_cand.iloc[tr_idx].values)
        clf = make_elasticnet_logreg(C=C, l1_ratio=l1_ratio, random_state=random_state + idx)
        clf.fit(X_tr, y_tr)
        w = clf.coef_[0]
        weights_matrix[idx] = w
        top_idx = np.argsort(np.abs(w))[::-1][:k]
        selected_counts[top_idx] += 1.0

    summary = pd.DataFrame({
        "Gene": cols,
        "Pathway": [get_gene_pathway(g) for g in cols],
        "Selection_Freq": selected_counts / n_resamples,
        "Mean_Abs_Weight": np.mean(np.abs(weights_matrix), axis=0),
        "Signed_Weight": np.mean(weights_matrix, axis=0),
    })

    if deg_df is not None:
        summary["OCD_t_stat"] = [float(deg_df.loc[g, "t_stat"]) if g in deg_df.index else 0.0 for g in cols]
        summary["OCD_pval"] = [float(deg_df.loc[g, "pval"]) if g in deg_df.index else 1.0 for g in cols]

    summary = summary.sort_values(["Selection_Freq", "Mean_Abs_Weight"], ascending=False).reset_index(drop=True)

    # Ensure at least `min_pathway_per_axis` top representative(s) from each biological axis
    chosen: List[str] = []
    if min_pathway_per_axis > 0:
        for axis in PATHWAY_GENE_SETS:
            axis_rows = summary[summary["Pathway"] == axis].head(min_pathway_per_axis)
            for g in axis_rows["Gene"]:
                if g not in chosen and len(chosen) < k:
                    chosen.append(g)

    for g in summary["Gene"]:
        if len(chosen) >= k:
            break
        if g not in chosen:
            chosen.append(g)

    panel_df = summary[summary["Gene"].isin(chosen)].copy()
    panel_df = panel_df.sort_values(["Selection_Freq", "Mean_Abs_Weight"], ascending=False).reset_index(drop=True)
    return list(panel_df["Gene"]), panel_df

