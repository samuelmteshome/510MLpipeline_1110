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

## Explore the training data

Run `python eda.py` to reproduce the class counts, ranges, quartiles, and outcome comparisons. The 239 training records have no missing cells or duplicate complete rows. Of these patients, 77 (32.2%) had a recorded death. Follow-up ranges from 4 to 280 days, so the label describes an event during observation rather than risk by a common deadline.

| Training median | No recorded death (162) | Recorded death (77) |
| --- | ---: | ---: |
| Ejection fraction (%) | 38 | 30 |
| Serum creatinine (mg/dL) | 1.0 | 1.4 |
| Age (years) | 60 | 62 |
| Serum sodium (mEq/L) | 137 | 136 |

**Class experiment.** Start with ejection fraction and serum creatinine. Their group medians suggest a two-feature hypothesis that is easy to explain. Compare that pair with all 11 baseline features using the same classifier and identical stratified five-fold splits on the training set. For the pair, select those two columns and put a `StandardScaler` inside the pipeline. For all features, use `make_preprocessor()`. Keep the 60 test patients for one evaluation after choosing the specification.

Serum creatinine has a long upper tail (median 1.1, maximum 9.4), as does creatinine phosphokinase (median 253, maximum 7,861). Keep these observations. If using a linear classifier, a log transform of the skewed measurements is a reasonable follow-up experiment. The binary variables show a mix of patterns. For example, recorded mortality is 34/89 (38%) with high blood pressure and 43/150 (29%) without it, while both sex groups are near 32%. These unadjusted rates alone cannot establish independent predictive value.

**Scoring.** Always predicting no event would score 67.8% training accuracy and catch zero deaths. Compare models by cross-validated average precision, using the 32.2% event prevalence as a reference. Report recall and the confusion matrix at the chosen decision threshold to show how many deaths were missed and how many false alarms occurred.

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
