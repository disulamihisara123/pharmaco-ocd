# Pharmaco-OCD: Explainable Clinical Decision Support System

An Explainable Machine Learning Clinical Decision Support System (CDSS) for Predicting SSRI Treatment Response in Obsessive-Compulsive Disorder (OCD) using Transcriptomic Biomarkers.

## Project Overview
- **Degree:** BSc (Hons) in Technology Management (University of Plymouth / NSBM Green University)
- **Domain Intersections:** Bioinformatics, Human Genomics, Machine Learning, Clinical Decision Support Systems (CDSS), Health Informatics.
- **Compute Constraints:** Zero paid cloud servers required; CPU-efficient local computation and Google Colab free tier compatible.

## Project Structure
```
pharmaco-ocd/
├── data/
│   ├── raw/             # Original downloaded NCBI GEO series matrices and platform annotation GPLs
│   ├── processed/       # Cleaned, log2-normalized, gene-symbol-mapped expression matrices
│   └── metadata/        # Patient clinical metadata (responder/non-responder labels, demographic data)
├── notebooks/           # Jupyter notebooks for data acquisition, EDA, and model experimentation
├── src/                 # Modular Python source code
│   ├── data_loader.py   # NCBI GEO download and parsing routines
│   ├── preprocess.py    # Probe-to-gene mapping, log2 normalization, and batch QC
│   ├── feature_selection.py # ElasticNet/LASSO and pathway-guided feature prioritization
│   ├── models.py        # Classifier training, cross-validation, and metrics evaluation
│   ├── explainability.py# SHAP and Pathway-SHAP calculation modules
│   └── app.py           # Streamlit-based Clinical Decision Support System portal
├── reports/
│   └── figures/         # Volcano plots, ROC-AUC curves, SHAP summary plots, and pathway bubbles
├── requirements.txt     # Python dependencies
└── README.md
```

## Datasets Targeted
1. **NCBI GEO GSE60190:** Genome-wide gene expression profiling in OCD cohorts.
2. **NCBI GEO GSE25974 / GSE76826 / GSE49667:** SSRI/antidepressant treatment response transcriptomic profiling (Responder vs. Non-Responder).
3. **Psychiatric Genomics Consortium (PGC):** OCD GWAS summary statistics and candidate gene lists (*SLC1A1*, *SLC6A4*, *HTR2A*, *BDNF*, *COMT*, *DLGAP3*, *SAPAP3*, *IL6*).

## Project Milestones (5 Months)
- **M1 (Month 1):** Harmonised Dataset & Quality Control Report
- **M2 (Month 3):** 10–15 Gene Biomarker Panel & Validated ML Classifier
- **M3 (Month 4):** Functional CDSS Web Application & PDF Export Engine
- **M4 (Month 5):** System Usability Scale (SUS) Evaluation, Governance Audit & Final Dissertation
