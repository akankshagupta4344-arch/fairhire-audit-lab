# Results and interpretation

## Proxy classifier performance

| Model | Accuracy | Balanced accuracy | F1 | ROC-AUC |
|---|---:|---:|---:|---:|
| Logistic Regression | 84.47% | 75.91% | 65.30% | 0.9039 |
| Random Forest | 86.18% | 77.39% | 68.26% | 0.9169 |

Random Forest scored higher on the listed overall test metrics. Overall metrics do not describe how outcomes are distributed across groups.

## Predicted-positive-rate gaps

| Model | Gender gap | Race gap |
|---|---:|---:|
| Logistic Regression | 17.51 percentage points | 25.67 percentage points |
| Random Forest | 17.29 percentage points | 22.85 percentage points |

These are maximum-minus-minimum group selection-rate gaps on the income-proxy task. They are not hiring selection rates.

## Paired LLM observations

The validated Rounds 2â€“5 file contains 96 observations: 48 Gemini and 48 GPT-6. It records zero within-pair decision changes and zero non-zero score differences. This means the paired outputs were unchanged in this particular dataset and set of runs; it does not prove general model fairness. A between-model disagreement was also observed for one profile/round, so model-to-model consistency should be evaluated separately from within-model counterfactual consistency.

## Main interpretation

1. Better aggregate predictive performance does not guarantee similar outcomes across demographic groups.
2. Group-level differences should be examined alongside, not replaced by, accuracy metrics.
3. Paired counterfactual tests provide useful evidence about sensitivity to selected identity-related fields, but conclusions are limited by sample size and experimental design.
4. Resume keyword coverage is only a transparent baseline; it is not an LLM result or evidence that a hiring workflow is unbiased.

## Limitations

The Adult Census dataset is not a resume dataset. Income is not a hiring label. The paired LLM sample is limited to the profiles, attributes, prompts, models, and runs represented in the file. The project is a classroom prototype and must not be used for real hiring decisions.