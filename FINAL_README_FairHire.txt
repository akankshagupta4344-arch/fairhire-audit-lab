FAIRHIRE AUDIT LAB — FINAL RUN INSTRUCTIONS

Put these three files into the SAME folder:
  fairhire_final_app.py
  adult.csv
  fairhire_llm_audit.csv

Windows PowerShell:
  cd "$env:USERPROFILE\OneDrive\Desktop\eda ai bias"
  .\.venv\Scripts\Activate.ps1
  streamlit run fairhire_final_app.py

If Streamlit says the app file is missing, first save the code as fairhire_final_app.py in that folder.
If your existing .venv already runs Streamlit, do not reinstall packages just before the demo.
If creating a new environment, use:
  python -m venv .venv
  .\.venv\Scripts\Activate.ps1
  python -m pip install -r requirements_final.txt
  streamlit run fairhire_final_app.py

What is in the site:
  1. Final dashboard: summary of both audit tracks.
  2. Dataset & EDA: data cleaning, distributions, group income rates, and age statistics.
  3. Income-proxy model audit: Logistic Regression and Random Forest group metrics.
  4. Paired LLM results: validated R2-R5 recorded outputs with score/decision comparisons.
  5. Resume input & AI screening: upload PDF/DOCX/TXT or paste a resume; assess against a job description with a two-stage Gemini workflow, or run an independent counterfactual A/B test.
  6. Counterfactual prompt lab: generate a paired screening prompt for external testing.
  7. Export: download cleaned data and summary metrics.

IMPORTANT INTERPRETATION:
- The Adult Census dataset is socioeconomic census data, not actual resumes and not hiring outcomes.
- The paired LLM CSV is the recorded output for R2-R5 (96 pair observations). It has zero within-pair changes in these records.
- Do not alter scores to manufacture bias. The local proxy models show group-level outcome disparities, which can be reported as a separate descriptive audit.
- The existing R1 records were inconsistent between the old workbook and recorded observations, so they are intentionally not merged into the validated R2-R5 CSV.

## Module 5: Resume input and AI screening workflow (UPDATED)
The current app's fifth module accepts PDF, DOCX, or TXT resumes (or pasted resume text) and a job description. It offers: (1) a two-stage Gemini workflow that structures resume evidence and then evaluates evidence against job requirements, and (2) a counterfactual A/B mode that runs two independent Gemini calls with the same resume and one changed identity-associated field. Results include scores, decisions, matched/missing skills, explanations, reviewer flags, and a downloadable CSV.

The app uses the Gemini generateContent REST API and defaults to gemini-3.8-flash. A live call requires a valid Gemini API key entered in the page's password field. The key is not stored in the source file. Because an API key was exposed in an earlier chat, revoke it and create a fresh key before using live mode. Never paste API keys into chat or commit them to source control.

If no key is entered, Batch mode falls back to the transparent local keyword-coverage baseline. That baseline is not an LLM and cannot test LLM bias. The paired A/B test requires a live model key to measure model response differences. Use synthetic or anonymized resumes for classroom demonstration. Do not use this prototype to automatically reject real applicants.

## Updated sidebar pages
1. Final dashboard
2. Dataset & EDA
3. Income-proxy model audit
4. Paired LLM results (reads fairhire_llm_audit.csv)
5. Resume input & AI screening (file intake + two-stage LLM workflow + independent A/B mode)
6. Counterfactual prompt lab
7. Export

## Install only the new document parsers if missing
Inside the existing working virtual environment: `pip install pypdf python-docx`. Do not reinstall or upgrade the rest of the environment if the current app already runs.
