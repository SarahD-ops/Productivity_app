"""
explore_data.py
---------------
Quick exploratory script to inspect the Lifestyle Habit Dataset.
Prints: shape, per-column data types, and missing-value counts.
"""

import pandas as pd

DATA_PATH = "data/Lifestyle_Habit_Dataset.csv"

def main():
    df = pd.read_csv(DATA_PATH)

    print("=" * 60)
    print(f"  Dataset Shape: {df.shape[0]} rows  x  {df.shape[1]} columns")
    print("=" * 60)

    print("\n--- Data Types ---")
    print(df.dtypes.to_string())

    print("\n--- Missing Values (count + % of total rows) ---")
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    missing_df = pd.DataFrame({
        "Missing Count": missing,
        "Missing %": missing_pct
    })
    # Show only columns that actually have missing values
    has_missing = missing_df[missing_df["Missing Count"] > 0]
    if has_missing.empty:
        print("  No missing values found!")
    else:
        print(has_missing.to_string())

    print("\n--- Quick Statistical Summary (numeric columns) ---")
    print(df.describe().T.to_string())

    print("\n--- Unique Value Counts (categorical columns) ---")
    cat_cols = df.select_dtypes(include=["object"]).columns
    for col in cat_cols:
        print(f"  {col}: {df[col].nunique()} unique values  -> {df[col].unique()[:5]} ...")

    print("\n--- First 5 Rows ---")
    print(df.head().to_string())

if __name__ == "__main__":
    main()
