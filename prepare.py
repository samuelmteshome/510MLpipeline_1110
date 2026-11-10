"""Prepare the classroom train/test split and preprocessing step."""

from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


DATA = Path(__file__).parent / "data" / "heart_failure.csv"

# "time" (follow-up length) is left out on purpose: it is only known after
# the outcome, so using it would leak the answer into the model.
CONTINUOUS = [
    "age", "creatinine_phosphokinase", "ejection_fraction", "platelets",
    "serum_creatinine", "serum_sodium",
]
BINARY = ["anaemia", "diabetes", "high_blood_pressure", "sex", "smoking"]


def load_split():
    records = pd.read_csv(DATA)
    features = records[CONTINUOUS + BINARY]
    target = records["DEATH_EVENT"]
    return train_test_split(
        features, target, test_size=0.2, random_state=42, stratify=target
    )


def make_preprocessor():
    return ColumnTransformer([
        ("scale", StandardScaler(), CONTINUOUS),
        ("binary", "passthrough", BINARY),
    ])


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_split()
    preprocess = make_preprocessor()
    train = preprocess.fit_transform(X_train)
    test = preprocess.transform(X_test)
    for name, matrix, labels in [
        ("Train", train, y_train), ("Test", test, y_test)
    ]:
        print(f"{name}: {matrix.shape[0]} patients, {matrix.shape[1]} features, "
              f"{labels.sum()} recorded deaths")
