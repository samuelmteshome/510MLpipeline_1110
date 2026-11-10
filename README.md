# Heart failure clinical records | AIPI 510

Prepare 299 patient records for binary classification. The target, `DEATH_EVENT`, indicates whether a patient died during their observed follow-up. Follow-up duration varies by patient. We exclude `time` because the final duration is unknown at the start of follow-up.

## Run in class

From the repository directory on macOS or Linux:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python prepare.py
```

The original CSV is bundled so the exercise runs offline after dependency installation. The script creates a repeatable 80/20 split with similar outcome proportions in each group. It learns the means and standard deviations of six numeric features from the training patients, applies them to both groups, and passes five binary features through unchanged.

## Add your classifier

Once your partner chooses a scikit-learn classifier, use its instance as `classifier` in the following snippet. Keeping the preprocessing inside the pipeline also lets you cross-validate on the training set without leaking validation-fold statistics.

```python
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import make_pipeline
from prepare import load_split, make_preprocessor

X_train, X_test, y_train, y_test = load_split()
model = make_pipeline(make_preprocessor(), classifier)
model.fit(X_train, y_train)
print(confusion_matrix(y_test, model.predict(X_test), labels=[0, 1]))
```

Rows of the matrix are observed outcomes and columns are predictions. Class 1 means a recorded death.

## Data credit

*Heart Failure Clinical Records* (2020), UCI Machine Learning Repository. The CSV is redistributed unchanged under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). [Dataset DOI](https://doi.org/10.24432/C5Z89R) · [Chicco & Jurman (2020)](https://doi.org/10.1186/s12911-020-1023-5)
