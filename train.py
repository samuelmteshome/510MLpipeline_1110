"""Train the classifier and a baseline, then save both for evaluate.py."""

from pathlib import Path

import joblib
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline

from prepare import load_split, make_preprocessor


MODELS = Path(__file__).parent / "artifacts"


def build_model():
    return make_pipeline(make_preprocessor(), LogisticRegression(max_iter=1000))


def build_baseline():
    # Always predicts the most common class (no recorded death).
    return DummyClassifier(strategy="most_frequent")


def main():
    X_train, _, y_train, _ = load_split()
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for name, model in [("baseline", build_baseline()), ("model", build_model())]:
        scores = cross_val_score(model, X_train, y_train, cv=folds, scoring="accuracy")
        print(f"{name}: 5-fold CV accuracy on training set = "
              f"{scores.mean():.1%} (+/- {scores.std():.1%})")

        model.fit(X_train, y_train)
        MODELS.mkdir(exist_ok=True)
        joblib.dump(model, MODELS / f"{name}.joblib")

    print(f"Saved models to {MODELS}/")


if __name__ == "__main__":
    main()
