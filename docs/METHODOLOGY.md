# Methodology

## Audit track A â€” Adult Census income proxy

The app loads the local Adult Census CSV, trims text fields, treats `?`/empty values as missing, drops incomplete rows and duplicates, and defines the target as `income == ">50K"`. Race and gender are removed from the model feature matrix and retained for post-hoc group evaluation. This does not eliminate proxy effects: other socioeconomic features may correlate with protected groups.

Categorical variables use most-frequent imputation and one-hot encoding. Numeric variables use median imputation and standard scaling. A stratified 80/20 train/test split with random seed 42 is used for Logistic Regression and Random Forest. The app reports accuracy, balanced accuracy, F1, ROC-AUC, confusion matrices, and group-level selection rate, TPR, and FPR. It also reports the maximum/minimum selection-rate gap and the minimum/maximum rate ratio.

A rate gap is descriptive. The 0.80 ratio is treated only as a screening flag for further analysis, not a legal determination. Since the label is income, TPR/equal-opportunity calculations are relative to the income proxy, not job qualification.

## Audit track B â€” paired LLM screening outputs

The app reads the saved `fairhire_llm_audit.csv` file and validates the expected columns. For each paired observation, it compares Candidate A and Candidate B decisions and scores. The score difference is A minus B. It summarizes decision flips, non-zero score differences, model-level results, attribute-level results, and between-model score comparisons where available.

The validated file contains 96 observations from Rounds 2â€“5 (48 Gemini and 48 GPT-6). Round 1 was excluded because the previously available records conflicted. These outputs are imported records; the dashboard does not generate new model results automatically.

## Resume workflow

The local baseline extracts the presence of required skill phrases and calculates keyword coverage. It is intentionally transparent and does not infer candidate quality from identity. This is a workflow demonstration, not a validation study. The optional Gemini integration requires a separately configured API key and network access. No real applicant resume should be uploaded or stored in this public repository.