"""Evaluate the saved logistic regression pipeline on the shared test split."""
import argparse
from pathlib import Path

import joblib
from sklearn.metrics import average_precision_score, confusion_matrix, precision_score, recall_score

from prepare import load_split


def report(model, features, target):
    event_column = list(model.classes_).index(1)
    scores = model.predict_proba(features)[:, event_column]
    predictions = scores >= 0.5
    tn, fp, fn, tp = confusion_matrix(target, predictions, labels=[0, 1]).ravel()

    print(f"Patients: {len(target)} | Recorded deaths: {target.sum()}")
    print(f"Average precision: {average_precision_score(target, scores):.3f}"
          f" (prevalence reference: {target.mean():.3f})")
    print("At a 0.50 probability threshold:")
    print(f"  Precision: {precision_score(target, predictions, zero_division=0):.3f}")
    print(f"  Recall:    {recall_score(target, predictions, zero_division=0):.3f}")
    print(f"  Deaths detected: {tp} | Deaths missed: {fn}")
    print(f"  False alarms: {fp} | Correct negatives: {tn}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("artifacts/model.joblib"))
    args = parser.parse_args()
    if not args.model.is_file():
        parser.error(f"Missing {args.model}. Run train.py or pass --model PATH.")
    model = joblib.load(args.model)
    _, X_test, _, y_test = load_split()
    report(model, X_test, y_test)


if __name__ == "__main__":
    main()
