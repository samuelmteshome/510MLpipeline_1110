"""Compare the two trained classifiers on the shared held-out patients."""
import json
from pathlib import Path

import joblib
import matplotlib
import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay, accuracy_score, confusion_matrix, precision_score,
    recall_score,
)

from prepare import load_split

matplotlib.use("Agg")
from matplotlib import pyplot as plt


def main():
    models = {
        "logistic_regression": Path("models/model.joblib"),
        "baseline": Path("models/baseline.joblib"),
    }
    for path in models.values():
        if not path.is_file():
            raise SystemExit(f"Missing {path}. Run python train.py first.")

    _, X_test, _, y_test = load_split()
    metrics = {}
    for name, path in models.items():
        predictions = joblib.load(path).predict(X_test)
        metrics[name] = {
            "accuracy": float(accuracy_score(y_test, predictions)),
            "precision": float(precision_score(y_test, predictions, zero_division=0)),
            "recall": float(recall_score(y_test, predictions, zero_division=0)),
            "confusion_matrix": confusion_matrix(y_test, predictions, labels=[0, 1]).tolist(),
        }

    results = Path("results")
    results.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 4), layout="constrained")
    display = ConfusionMatrixDisplay(
        confusion_matrix=np.asarray(metrics["logistic_regression"]["confusion_matrix"]),
        display_labels=["No recorded death", "Recorded death"],
    )
    display.plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
    ax.set_title(f"Logistic regression · {len(y_test)} test patients")
    fig.savefig(results / "confusion_matrix.png", dpi=160)
    plt.close(fig)
    summary = json.dumps({"test_patients": len(y_test), **metrics}, indent=2)
    (results / "metrics.json").write_text(summary + "\n", encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main()
