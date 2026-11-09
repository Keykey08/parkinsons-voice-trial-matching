# %% [markdown]
# # Project 2, Stage 1b: the leakage experiment
# Same models, two ways of splitting the data:
#   - by RECORDING (StratifiedKFold): a person's recordings can sit in train AND test (leaky)
#   - by PERSON (GroupKFold): all of a person's recordings stay on one side (honest)
# WHY: each person gave 6-7 recordings, so a leaky split lets the model recognise the voice.
# Expected output (run on 9 Nov 2026): LogReg AUC 0.91 leaky vs 0.77 honest; RF 0.96 vs 0.78.

# %%
from pathlib import Path
import pandas as pd
from sklearn.model_selection import GroupKFold, StratifiedKFold, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier

ROOT = Path(__file__).resolve().parents[1] if "__file__" in globals() else Path.cwd()
df = pd.read_csv(ROOT / "data" / "raw" / "parkinsons.data")
person = df["name"].str.extract(r"(S\d+)")[0]          # subject ID, e.g. S01
X, y = df.drop(columns=["name", "status"]), df["status"]

models = {
    "Always predict Parkinson's": DummyClassifier(strategy="most_frequent"),
    "Logistic regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced")),
    "Random forest": RandomForestClassifier(n_estimators=300, random_state=0, class_weight="balanced"),
}
rows = []
for name, m in models.items():
    honest = cross_validate(m, X, y, groups=person, cv=GroupKFold(5), scoring=["roc_auc", "balanced_accuracy"])
    leaky = cross_validate(m, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=0), scoring=["roc_auc", "balanced_accuracy"])
    rows.append({"model": name,
                 "AUC by recording (leaky)": leaky["test_roc_auc"].mean(), "AUC by person (honest)": honest["test_roc_auc"].mean(),
                 "BalAcc by recording": leaky["test_balanced_accuracy"].mean(), "BalAcc by person": honest["test_balanced_accuracy"].mean()})
res = pd.DataFrame(rows).round(2)
print(res.to_string(index=False))
res.to_csv(ROOT / "reports" / "01b_leakage_experiment.csv", index=False)
