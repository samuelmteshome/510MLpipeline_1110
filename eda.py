"""Explore the training patients using the same split as prepare.py."""
import pandas as pd

from prepare import BINARY, CONTINUOUS, DATA, load_split


def main():
    X, _, y, _ = load_split()
    training = pd.read_csv(DATA).loc[X.index]
    print(f"Training patients: {len(X)}, recorded deaths: {y.sum()}")
    print(f"Missing cells: {training.isna().sum().sum()}")
    print(f"Duplicate complete records: {training.duplicated().sum()}")
    print("Follow-up days (min, median, max):",
          training["time"].agg(["min", "median", "max"]).tolist())

    stats = X[CONTINUOUS].describe().T[["min", "25%", "50%", "75%", "max"]]
    stats["median_no_event"] = X.loc[y == 0, CONTINUOUS].median()
    stats["median_event"] = X.loc[y == 1, CONTINUOUS].median()
    print("\nContinuous measurements\n", stats.to_string())

    print("\nRecorded deaths / patients by binary feature value")
    for feature in BINARY:
        groups = y.groupby(X[feature]).agg(["sum", "count"])
        print(feature, " ".join(
            f"{value}: {row['sum']}/{row['count']}"
            for value, row in groups.iterrows()
        ))


if __name__ == "__main__":
    main()
