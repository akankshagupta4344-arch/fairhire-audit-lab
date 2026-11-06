"""Download the UCI Adult dataset into the format expected by FairHire Audit Lab."""
from pathlib import Path
import pandas as pd
from ucimlrepo import fetch_ucirepo


def main() -> None:
    dataset = fetch_ucirepo(id=2)
    features = dataset.data.features.copy()
    targets = dataset.data.targets
    if targets is None or targets.empty:
        raise RuntimeError("UCI Adult dataset did not return a target column.")
    frame = features.copy()
    frame["income"] = targets.iloc[:, 0]
    frame.columns = [str(column).strip().lower() for column in frame.columns]
    if "sex" in frame.columns and "gender" not in frame.columns:
        frame = frame.rename(columns={"sex": "gender"})
    required = {"income", "race", "gender"}
    missing = required.difference(frame.columns)
    if missing:
        raise RuntimeError(f"Downloaded dataset is missing expected columns: {sorted(missing)}")
    output = Path(__file__).resolve().parents[1] / "adult.csv"
    frame.to_csv(output, index=False)
    print(f"Saved {len(frame):,} rows and {len(frame.columns)} columns to: {output}")
    print("The CSV is intentionally excluded from Git by .gitignore.")


if __name__ == "__main__":
    main()