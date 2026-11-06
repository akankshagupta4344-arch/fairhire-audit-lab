# Data setup

The repository intentionally does **not** include `adult.csv`. Downloading it locally avoids duplicating the full dataset in the Git repository. The Streamlit app expects an `adult.csv` file in the project root with columns including `income`, `race`, and `gender`.

From the project root, run:

```powershell
python -m pip install ucimlrepo
python scripts/download_adult.py
```

This downloads UCI Adult (Dataset ID 2), combines its features and target, renames `sex` to `gender`, and writes `adult.csv` at the project root. The data file is git-ignored and stays on your computer.

Source and citation: Becker, B., & Kohavi, R. (1996). *Adult* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5XW20