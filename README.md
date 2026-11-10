**Test accuracy (logistic regression): 70.0%**

# Heart failure clinical records | AIPI 510

Prepare 299 patient records for binary classification. The target, `DEATH_EVENT`, indicates whether a patient died during their observed follow-up. Follow-up duration varies by patient. We exclude `time` because the final duration is unknown at the start of follow-up.

## Run in class

From the repository directory on macOS or Linux:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python prepare.py
python train.py
python evaluate.py
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

## Results

On 60 held-out patients, logistic regression correctly classified 42 and the always-survived baseline correctly classified 41.

| Model | Test accuracy | Precision for death | Recall for death |
| --- | ---: | ---: | ---: |
| Logistic regression | 70.0% | 53.8% | 36.8% |
| Always-survived baseline | 68.3% | 0.0% | 0.0% |

Logistic regression detected 7 of 19 recorded deaths, missed 12, and raised 6 false alarms. The baseline missed all 19 deaths.

![Logistic regression confusion matrix. Actual outcomes in rows and predictions in columns.](results/confusion_matrix.png)

Training saves both models under `artifacts/`. Evaluation writes the [full metrics](results/metrics.json) and updates the accuracy on the first line of this README.

## Data credit

*Heart Failure Clinical Records* (2020), UCI Machine Learning Repository. The CSV is redistributed unchanged under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). [Dataset DOI](https://doi.org/10.24432/C5Z89R) · [Chicco & Jurman (2020)](https://doi.org/10.1186/s12911-020-1023-5)
