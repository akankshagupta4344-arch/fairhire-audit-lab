# FairHire Audit Lab

**A Streamlit dashboard for exploring demographic disparities in machine-learning predictions and auditing consistency in AI-assisted resume screening.**

FairHire combines two deliberately separate audit tracks:

1. **Income-proxy classifier audit:** Logistic Regression and Random Forest models predict whether Adult Census income exceeds $50K. Their performance and predicted-positive rates are compared across gender and race groups.
2. **Paired LLM resume-screening audit:** saved Gemini and GPT-6 outputs are compared across paired synthetic candidate profiles where an identity-associated attribute is varied while the job-related profile is held comparable.

> **Interpretation boundary:** Adult Census is socioeconomic census data, not a resume dataset. Income above $50K is not a hiring label. The classifier results describe disparities on this proxy task; they are not direct evidence of hiring discrimination. The paired LLM results apply only to the recorded profiles, prompts, models, and runs in the included sample.

## Dashboard

The Streamlit app contains seven sections:

- Final dashboard â€” headline results from both audit tracks.
- Dataset & EDA â€” data cleaning, missingness, distributions, and descriptive statistics.
- Income-proxy model audit â€” model metrics and group-level selection rates, TPR, and FPR.
- Paired LLM results â€” A/B score and decision comparisons from the saved audit CSV.
- Resume input & AI screening â€” a demonstration workflow using synthetic or anonymized resumes.
- Counterfactual prompt lab â€” prepares a paired prompt for an identity-attribute comparison.
- Export â€” downloads result tables for further analysis.

## Methodology

### 1. Data preparation and EDA

The Adult Census data is cleaned by trimming whitespace, treating `?` and empty strings as missing values, dropping incomplete rows, and removing duplicates. The final app expects a local `adult.csv` containing the columns used by the app, including `income`, `race`, and `gender`.

### 2. Proxy classifiers

- **Target:** income greater than $50K (`>50K`).
- **Features:** remaining census attributes; `race` and `gender` are held out for post-hoc group auditing rather than included as model input features.
- **Split:** stratified 80/20 train/test split, `random_state=42`.
- **Preprocessing:** most-frequent imputation and one-hot encoding for categorical data; median imputation and standard scaling for numeric data.
- **Models:** Logistic Regression (`max_iter=1200`) and Random Forest (`n_estimators=180`, `min_samples_leaf=5`, `random_state=42`).
- **Performance measures:** accuracy, balanced accuracy, F1-score, and ROC-AUC.
- **Group measures:** predicted-positive/selection rate, true-positive rate, false-positive rate, maximum-minus-minimum selection-rate gap, and minimum-to-maximum selection-rate ratio.

### 3. Paired LLM audit

The included `fairhire_llm_audit.csv` contains 96 validated observations from Rounds 2â€“5 (48 Gemini and 48 GPT-6 observations). Each row records the varied attribute, paired candidate values, model, run, decisions, scores, and rationales. Round 1 was excluded because the available records conflicted. The app summarizes the saved outputs; it does not silently generate or fabricate experiment results.

### 4. Resume demo

The local baseline measures exact required-skill keyword coverage. It is a transparent demonstration baseline, not a trained hiring model and not proof of fairness. The optional live Gemini path depends on external API access. Use synthetic or anonymized resumes only, and treat outputs as advisory with human review.

## Results snapshot

The final recorded proxy-model run reported:

| Model | Accuracy | Balanced accuracy | F1-score | ROC-AUC |
|---|---:|---:|---:|---:|
| Logistic Regression | 84.47% | 75.91% | 65.30% | 0.9039 |
| Random Forest | 86.18% | 77.39% | 68.26% | 0.9169 |

The maximum-minus-minimum predicted-positive-rate gaps were:

| Model | Gender gap | Race gap |
|---|---:|---:|
| Logistic Regression | 17.51 percentage points | 25.67 percentage points |
| Random Forest | 17.29 percentage points | 22.85 percentage points |

In the 96 saved paired LLM observations, no within-pair decision changes or non-zero score differences were recorded. This is a result for this limited test sample only; it does not establish that either model is universally unbiased. A between-model disagreement was also observed in one profile/round, illustrating that models can differ even when within-model paired outputs remain unchanged.

## Run locally (Windows)

1. Install Python 3.10+ and Git for Windows.
2. Clone/download this repository and place the Adult Census CSV at the project root as `adult.csv` (see [`data/README.md`](data/README.md)).
3. Open PowerShell in the project folder and run:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run fairhire_final_app.py
```

The app opens in your browser at the local Streamlit address printed in the terminal.

## Repository contents

```text
fairhire-audit-lab/
â”œâ”€â”€ fairhire_final_app.py
â”œâ”€â”€ fairhire_llm_audit.csv
â”œâ”€â”€ requirements.txt
â”œâ”€â”€ data/
â”‚   â””â”€â”€ README.md
â”œâ”€â”€ docs/
â”‚   â”œâ”€â”€ METHODOLOGY.md
â”‚   â””â”€â”€ RESULTS.md
â””â”€â”€ scripts/
    â””â”€â”€ download_adult.py
```

## Responsible use and limitations

- Adult Census income is a proxy prediction target, not a hiring outcome or measure of candidate merit.
- The paired LLM experiment is limited by the number of profiles, attributes, prompts, model versions, and recorded runs.
- A lack of observed differences in a small sample is not proof of absence of bias.
- Proxy variables and correlations may influence predictions even when protected attributes are excluded from the feature matrix.
- Do not use this classroom prototype to make or automate real employment decisions.
- Use synthetic or anonymized resumes; do not commit personal applicant data, API keys, tokens, or private credentials.

## Dataset citation

Becker, B., & Kohavi, R. (1996). *Adult* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5XW20

## Acknowledgement

Developed as an academic project to demonstrate exploratory data analysis, model evaluation, group-level disparity measurement, and paired counterfactual testing in an inspectable dashboard.