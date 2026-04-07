import pandas as pd
from pathlib import Path

root = Path("/Users/aly/Documents/University_of_Waterloo/Winter 2025/Research/code/NFL/ML/dataset_interpolated_fixed")

csv_files = list(root.rglob("*.csv"))
print(f"Found {len(csv_files)} CSV files")

for path in csv_files:
    df = pd.read_csv(path)
    if "home_has_possession" not in df.columns:
        print(f"  SKIP (no column): {path}")
        continue
    df["home_has_possession"] = df["home_has_possession"].map({True: 1, False: -1, "True": 1, "False": -1})
    df.to_csv(path, index=False)

print("Done.")
