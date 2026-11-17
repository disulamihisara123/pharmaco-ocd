# System Requirements Specification (SRS)
## Project: Pharmaco-OCD Clinical Decision Support System

### 1. Introduction & Purpose
This document specifies the software and clinical requirements for **Pharmaco-OCD**, an explainable Clinical Decision Support System (CDSS) for stratifying OCD patients by predicted response to first-line Selective Serotonin Reuptake Inhibitor (SSRI) therapy.

### 2. Stakeholders & User Roles
1. **Consultant Psychiatrist / Prescribing Clinician:** Enters patient biomarker profile; views calibrated response probability and guideline recommendations; exports clinical PDF.
2. **Clinical Pharmacogenomic Laboratory Scientist:** Uploads raw assay batches and generates standardized interpretation dossiers.
3. **Healthcare Administrator:** Evaluates outpatient remission rates, average trial-and-error treatment duration, and medication budget optimization.
4. **Research Informatician:** Accesses open-source biomarker discovery scripts and pathway enrichment outputs.

### 3. Functional Requirements (FR)
- **FR-01 (Data Ingestion):** The system shall allow clinicians to enter expression levels for a compact 10–15 gene panel manually or via CSV file upload with automated range-checking.
- **FR-02 (Demonstration Profiles):** The system shall provide pre-configured benchmark patient profiles (e.g., "Typical Responder", "Treatment-Resistant Profile") for immediate demonstration without requiring manual input.
- **FR-03 (Predictive Stratification):** The inference engine shall calculate an SSRI response probability score (0.0 to 1.0) and categorize the patient into three calibrated risk tiers:
  - *Low Risk of Non-Response* (Predicted Responder)
  - *Moderate Risk of Non-Response*
  - *High Risk of Non-Response* (Predicted Treatment-Resistant)
- **FR-04 (Mechanistic Explainability):** The system shall generate patient-level SHAP waterfall visualizations and aggregate contributions into functional biological pathways (Serotonergic, Glutamatergic, Neuroinflammatory).
- **FR-05 (Clinical Guidance Mapping):** The system shall provide rule-based decision trees mapped to NICE and APA OCD clinical guidelines (e.g., standard SSRI titration vs. early augmentation review).
- **FR-06 (PDF Dossier Generation):** The system shall generate a 1-page standardized, downloadable PDF consultation summary containing patient pseudonym, risk gauge, top contributing genes, and audit metadata.

### 4. Non-Functional Requirements (NFR)
- **NFR-01 (Performance & Latency):** Inference calculation, SHAP plot generation, and PDF rendering must execute in under **2.5 seconds** on consumer hardware (standard 8 GB RAM laptop CPU).
- **NFR-02 (Zero Cloud Infrastructure Cost):** The system shall operate 100% locally or on free serverless cloud tiers (Streamlit Community Cloud / Hugging Face Spaces) with zero dedicated GPU or rented server costs.
- **NFR-03 (Data Privacy & Governance):** The system shall adhere to GDPR and HIPAA data-minimization principles:
  - Zero storage of raw patient genomic sequence files.
  - Ephemeral processing with pseudonymized patient identifier IDs.
  - Local audit logging of consultation timestamps without storing identifiable personal health data.
- **NFR-04 (Usability):** The user interface shall achieve a System Usability Scale (SUS) benchmark score of **$\ge 68$** in empirical evaluations.
