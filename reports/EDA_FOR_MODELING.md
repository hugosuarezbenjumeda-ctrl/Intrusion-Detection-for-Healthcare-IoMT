# EDA for Modeling: CICIoMT2024 IDS

## Purpose
This document translates EDA findings into concrete modeling decisions for the thesis objective:
- Binary flow-based IDS (`label`: benign=0, attack=1)
- Reproducible training/evaluation pipeline
- Evaluation focused on operational IDS tradeoffs (false positives, recall under low FPR, robustness, explainability)

Primary EDA source bundle:
- `/home/capstone15/reports/eda_advanced_20260305_231027/`

## Data Scope and Reality Check
From merged datasets (`metadata_train.csv` + `metadata_test.csv`):
- Total rows: `9,320,045`
- Features: `45` numeric flow features
- Metadata columns: `20`
- Class distribution: `93.02% attack` / `6.98% benign`
- Split imbalance:
  - Train attack ratio: `92.15%`
  - Test attack ratio: `97.07%`
  - Test-train attack-rate delta: `+4.91%`

Interpretation:
- Accuracy will be misleadingly high for trivial attack-biased models.
- Thresholding and FPR-constrained metrics are mandatory.
- Validation protocol must be careful due to split drift.

## Distribution Findings That Matter for Modeling

### 1. Protocol concentration
Counts by `protocol_hint`:
- `wifi`: `8,640,752` rows
- `mqtt`: `524,217` rows
- `bluetooth`: `155,076` rows

Implication:
- A single global model can overfit dominant WiFi attack patterns.
- Always report metrics overall and by protocol slice.

### 2. Attack family concentration
Top families:
- `DDoS`: `6,097,614` (`65.42%` of all rows)
- `DoS`: `2,290,990` (`24.58%`)
- `Recon`: `131,402`
- `Other` (Bluetooth): `125,009`

Implication:
- Macro performance across families must be tracked.
- Long-tail families can be ignored by naive training objectives.

### 3. Top separative features (sampled EDA)
Highest |Cohen's d| examples:
- `psh_flag_number`: `-3.75`
- `Variance`: `-3.71`
- `rst_count`: `-3.25`
- `ack_flag_number`: `-2.95`
- `Max`, `Magnitue`, `Std`, `Radius`

Implication:
- These features are strong discriminators and likely key for baseline performance.
- Some are strongly correlated (`Std`, `Radius`, `Max`, `Magnitue`), so regularization or tree-based models are preferred over unregularized linear assumptions.

## Modeling Plan Derived from EDA

### A. Dataset usage policy
- Primary train/test files:
  - `/home/capstone15/data/ciciomt2024/merged/metadata_train.csv`
  - `/home/capstone15/data/ciciomt2024/merged/metadata_test.csv`
- Do not reshuffle train/test boundaries.
- Create a validation split only from train (deterministic seed), stratified by:
  - `label`
  - `protocol_hint`
  - `attack_family` (coarse stratification where feasible)

### B. Feature policy
- Start with 45 numeric flow features as core model inputs.
- Keep metadata fields for slicing/evaluation and error analysis.
- Avoid using high-risk identifiers (`source_relpath`, `sample_name`) as direct model inputs to prevent leakage-by-origin.

### C. Baseline model matrix
Run all with fixed seed and same split protocol:
1. Logistic Regression (class-weighted)
2. Random Forest (class-weighted)
3. Gradient Boosting candidate (XGBoost/LightGBM if available; else HistGradientBoosting)

Track per model:
- PR-AUC
- ROC-AUC
- F1
- Precision
- Recall
- FPR
- Confusion matrix
- Recall at fixed FPR targets (`<=1%`, `<=0.5%` optional)

### D. Threshold policy
For each model:
- Optimize threshold on validation for operational target:
  - Primary: maximize recall under `FPR <= 1%`
  - Secondary: max F1 threshold
- Freeze threshold before test evaluation.

### E. Slice-level evaluation (mandatory)
Report metrics by:
- Protocol: WiFi / MQTT / Bluetooth
- Attack family: DDoS / DoS / Recon / Other / Spoofing / Malformed

Reason:
- EDA shows strong concentration and drift; overall scores alone are not trustworthy.

### F. Robustness and explainability priorities
Robustness:
- Start perturbation sensitivity on top EDA-ranked features (`psh_flag_number`, `Variance`, `rst_count`, `ack_flag_number`).
- Keep perturbations plausible (non-negative, bounded).

Explainability:
- Global importance first (feature ranking)
- Local explanations on:
  - False positives in benign-heavy slices (especially MQTT/Bluetooth benign)
  - False negatives for minority attack families

## Experiment Checklist (Use Before Every Run)
1. Confirm deterministic seed is set.
2. Confirm train/validation/test boundaries and no leakage fields in feature matrix.
3. Confirm class-weighting strategy is logged.
4. Confirm threshold selection protocol (validation only).
5. Confirm output artifacts are saved:
   - metrics JSON/CSV
   - confusion matrices
   - PR/ROC curves
   - per-slice tables
   - threshold report

## Suggested First Modeling Milestone
Deliver a baseline comparison report with:
- 3 models (LR, RF, boosting)
- Global + per-protocol + per-family metrics
- Threshold at `FPR <= 1%`
- Initial robustness stress test on top 4 EDA features
- Explainability summary for best model

## Reference Files
- EDA narrative: `/home/capstone15/reports/eda_advanced_20260305_231027/REPORT.md`
- EDA tables: `/home/capstone15/reports/eda_advanced_20260305_231027/tables/`
- EDA plots: `/home/capstone15/reports/eda_advanced_20260305_231027/plots/`
