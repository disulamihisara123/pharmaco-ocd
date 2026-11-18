"""
models.py
Classifier benchmarking, donor-grouped cross-validation, bootstrap confidence
intervals, risk tier stratification, and model persistence for Pharmaco-OCD.
"""

import os
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import LeaveOneGroupOut, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

from feature_selection import (
    PathwayElasticNetSelector,
    get_gene_pathway,
    make_elasticnet_logreg,
)

RISK_THRESHOLDS = {
    "low_risk_min_prob": 0.65,       # p >= 0.65 -> Low Risk of Non-Response (Responder)
    "moderate_risk_min_prob": 0.35,  # 0.35 <= p < 0.65 -> Moderate Risk
}


def get_candidate_classifiers(random_state: int = 42) -> Dict[str, Any]:
    """Return baseline and candidate ML classifiers tuned for small-N transcriptomics."""
    return {
        "Dummy_Baseline": DummyClassifier(strategy="stratified", random_state=random_state),
        "ElasticNet_LogReg": make_elasticnet_logreg(C=0.5, l1_ratio=0.5, random_state=random_state),
        "L2_LogReg": make_elasticnet_logreg(C=0.5, l1_ratio=0.0, random_state=random_state),
        "RandomForest": RandomForestClassifier(
            n_estimators=200, max_depth=4, min_samples_leaf=2, random_state=random_state
        ),
        "XGBoost": XGBClassifier(
            n_estimators=80,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=random_state,
        ),
        "SVM_RBF": SVC(C=1.0, kernel="rbf", probability=True, random_state=random_state),
    }


def stratify_risk_tier(prob_responder: float) -> Dict[str, str]:
    """
    Map calibrated SSRI responder probability to SRS FR-03 clinical risk tiers.
    """
    if prob_responder >= RISK_THRESHOLDS["low_risk_min_prob"]:
        return {
            "tier": "Low Risk of Non-Response",
            "prediction": "Predicted SSRI Responder",
            "guidance_code": "STANDARD_SSRI",
        }
    if prob_responder >= RISK_THRESHOLDS["moderate_risk_min_prob"]:
        return {
            "tier": "Moderate Risk of Non-Response",
            "prediction": "Indeterminate / Intermediate Response",
            "guidance_code": "CLOSE_MONITORING",
        }
    return {
        "tier": "High Risk of Non-Response",
        "prediction": "Predicted SSRI Treatment-Resistant",
        "guidance_code": "EARLY_AUGMENTATION_REVIEW",
    }


def bootstrap_metric_ci(
    y_true: np.ndarray,
    y_score: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    n_boot: int = 500,
    random_state: int = 42,
) -> Tuple[float, float, float]:
    """Compute point estimate and 95% stratified-safe bootstrap confidence interval."""
    rng = np.random.default_rng(random_state)
    point = float(metric_fn(y_true, y_score))
    boot_vals: List[float] = []
    n = len(y_true)

    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        boot_vals.append(float(metric_fn(y_true[idx], y_score[idx])))

    if not boot_vals:
        return point, point, point
    ci_low, ci_high = np.percentile(boot_vals, [2.5, 97.5])
    return point, float(ci_low), float(ci_high)


def _get_cv_splitter(cv_strategy: str, n_splits: int = 5):
    if cv_strategy.lower() == "logo":
        return LeaveOneGroupOut()
    return StratifiedGroupKFold(n_splits=n_splits)


def _predict_pos_proba(estimator: Any, X_arr: np.ndarray) -> np.ndarray:
    if hasattr(estimator, "predict_proba"):
        return estimator.predict_proba(X_arr)[:, 1]
    scores = estimator.decision_function(X_arr)
    return 1.0 / (1.0 + np.exp(-scores))


def evaluate_leakage_free_cv(
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    candidate_genes: Sequence[str],
    classifiers: Optional[Dict[str, Any]] = None,
    k: int = 15,
    cv_strategy: str = "logo",
    n_splits: int = 5,
    selector_C: float = 0.4,
    selector_l1_ratio: float = 0.7,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    """
    Run strict zero-leakage donor-grouped CV where scaling and ElasticNet gene
    selection occur strictly inside each training fold.
    """
    if classifiers is None:
        classifiers = get_candidate_classifiers(random_state=random_state)

    cols = [g for g in candidate_genes if g in X.columns]
    X_sub = X[cols]
    splitter = _get_cv_splitter(cv_strategy, n_splits=n_splits)

    oof_probs: Dict[str, np.ndarray] = {name: np.zeros(len(y), dtype=float) for name in classifiers}
    fold_times: Dict[str, List[float]] = {name: [] for name in classifiers}

    for tr_idx, va_idx in splitter.split(X_sub, y, groups):
        X_tr, y_tr = X_sub.iloc[tr_idx], y[tr_idx]
        X_va = X_sub.iloc[va_idx]

        selector = PathwayElasticNetSelector(
            k=k, C=selector_C, l1_ratio=selector_l1_ratio, random_state=random_state
        )
        X_tr_sel = selector.fit_transform(X_tr, y_tr)
        X_va_sel = selector.transform(X_va)

        for name, base_clf in classifiers.items():
            clf = clone(base_clf)
            t0 = time.perf_counter()
            clf.fit(X_tr_sel, y_tr)
            oof_probs[name][va_idx] = _predict_pos_proba(clf, X_va_sel)
            fold_times[name].append(time.perf_counter() - t0)

    rows = []
    for name, probs in oof_probs.items():
        auc, auc_lo, auc_hi = bootstrap_metric_ci(y, probs, roc_auc_score, random_state=random_state)
        pr_auc = float(average_precision_score(y, probs))
        preds = (probs >= 0.5).astype(int)
        bacc = float(balanced_accuracy_score(y, preds))
        f1 = float(f1_score(y, preds, zero_division=0))
        brier = float(brier_score_loss(y, probs))

        donor_df = pd.DataFrame({"donor": groups, "y": y, "p": probs}).groupby("donor").mean()
        donor_auc = float(roc_auc_score(donor_df["y"], donor_df["p"]))

        rows.append({
            "Model": name,
            "Sample_ROC_AUC": auc,
            "ROC_AUC_95CI": f"[{auc_lo:.3f}, {auc_hi:.3f}]",
            "Donor_ROC_AUC": donor_auc,
            "PR_AUC": pr_auc,
            "Balanced_Acc": bacc,
            "F1_Score": f1,
            "Brier_Score": brier,
            "Avg_Fold_Sec": float(np.mean(fold_times[name])),
        })

    results_df = pd.DataFrame(rows).sort_values("Donor_ROC_AUC", ascending=False).reset_index(drop=True)
    return results_df, oof_probs


def evaluate_external_cohort(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_ext: pd.DataFrame,
    y_ext: np.ndarray,
    selected_genes: Sequence[str],
    ext_groups: Optional[np.ndarray] = None,
    classifiers: Optional[Dict[str, Any]] = None,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    """Train models on the fixed biomarker panel and evaluate on an unseen external cohort."""
    if classifiers is None:
        classifiers = get_candidate_classifiers(random_state=random_state)

    cols = list(selected_genes)
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_train[cols].values)
    X_ex_s = scaler.transform(X_ext[cols].values)

    rows = []
    ext_probs: Dict[str, np.ndarray] = {}

    for name, base_clf in classifiers.items():
        clf = clone(base_clf)
        clf.fit(X_tr_s, y_train)
        probs = _predict_pos_proba(clf, X_ex_s)
        ext_probs[name] = probs
        preds = (probs >= 0.5).astype(int)

        row = {
            "Model": name,
            "Ext_Sample_ROC_AUC": float(roc_auc_score(y_ext, probs)),
            "Ext_PR_AUC": float(average_precision_score(y_ext, probs)),
            "Ext_Balanced_Acc": float(balanced_accuracy_score(y_ext, preds)),
            "Ext_Brier_Score": float(brier_score_loss(y_ext, probs)),
        }
        if ext_groups is not None:
            donor_df = pd.DataFrame({"donor": ext_groups, "y": y_ext, "p": probs}).groupby("donor").mean()
            row["Ext_Donor_ROC_AUC"] = float(roc_auc_score(donor_df["y"], donor_df["p"]))
        rows.append(row)

    return pd.DataFrame(rows).sort_values("Ext_Sample_ROC_AUC", ascending=False).reset_index(drop=True), ext_probs


def train_and_save_final_model(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    selected_genes: Sequence[str],
    estimator: Optional[Any] = None,
    output_path: str = "models/pharmaco_ocd_classifier.joblib",
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Fit the production StandardScaler + Classifier pipeline on the selected
    10-15 gene biomarker panel, benchmark latency, and persist via joblib.
    """
    if estimator is None:
        estimator = get_candidate_classifiers()["XGBoost"]

    cols = list(selected_genes)
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", clone(estimator)),
    ])
    pipeline.fit(X_train[cols], y_train)

    # Benchmark single-patient CPU inference latency (SRS NFR-01 < 2.5s)
    sample_row = X_train[cols].iloc[[0]]
    t0 = time.perf_counter()
    _ = pipeline.predict_proba(sample_row)
    latency_ms = (time.perf_counter() - t0) * 1000.0

    bundle = {
        "pipeline": pipeline,
        "selected_genes": cols,
        "pathway_map": {g: get_gene_pathway(g) for g in cols},
        "risk_thresholds": RISK_THRESHOLDS,
        "inference_latency_ms": round(latency_ms, 3),
        "metadata": metadata or {},
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    joblib.dump(bundle, output_path)
    print(f"[MODEL SAVED] {output_path} | Genes={len(cols)} | Inference Latency={latency_ms:.2f} ms")
    return bundle


def load_model_bundle(path: str = "models/pharmaco_ocd_classifier.joblib") -> Dict[str, Any]:
    """Load a persisted Pharmaco-OCD model bundle."""
    return joblib.load(path)

