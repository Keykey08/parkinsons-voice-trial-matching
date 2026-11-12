# %% [markdown]
# # Project 2, Stage 2b: the leakage experiment on Istanbul (252 people)
#
# **Question:** does the Oxford finding (split by recording inflates AUC from ~0.78
# to ~0.96) repeat on a dataset 8x larger, from another hospital? And how much of
# the signal is just "is this a man?"
#
# **What this script does**
#   1. Loads Istanbul, drops person 37's duplicate recording (see 03_profile_istanbul.py).
#   2. Four feature sets:
#        - Sex only: the floor. Sex alone gave AUC 0.60 in the profile.
#        - Shared 16: the 16 features Istanbul shares with Oxford (PPE excluded:
#          it points the opposite way in the two datasets, so it is not the same measure).
#        - Shared 16 + sex.
#        - All 752 Istanbul features.
#   3. Two models: logistic regression and random forest.
#   4. Two ways of splitting, each repeated 10 times with different random seeds:
#        - by RECORDING (leaky): a person's recordings can sit in train and test.
#        - by PERSON (honest): all of a person's recordings stay on one side.
#   5. Scores at person level (average the person's recordings), overall and
#      separately for men and women.
#
# **Outputs:** reports/04_leakage_istanbul.csv, reports/04_leakage_istanbul.txt,
#              reports/04_leakage_istanbul.png
#
# Run: from the project folder, `python src/04_leakage_istanbul.py`, or cell by cell
# in VS Code. Takes a few minutes (the 752-feature random forest is the slow part).
# Module links: 4_02 logistic regression, 5_01 random forest, 6_03 ML in practice.

# %%
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, balanced_accuracy_score

ROOT = Path(__file__).resolve().parents[1] if "__file__" in globals() else Path.cwd()
RAW, REPORTS = ROOT / "data" / "raw", ROOT / "reports"
REPORTS.mkdir(exist_ok=True)
N_REPEATS = 10   # WHY 10: one random split of 252 people can be lucky or unlucky. Repeating
                 # with 10 seeds shows how much the score moves just because of the split.

LOG = []
def say(*parts):
    line = " ".join(str(p) for p in parts); print(line); LOG.append(line)

# %% [markdown]
# ## 1. Load and clean

# %%
ist = pd.read_csv(RAW / "uci470_istanbul" / "pd_speech_features.csv", header=1)
ist = ist.drop_duplicates().reset_index(drop=True).copy()     # person 37's copied recording
ist["F0_Hz"] = 1 / ist["meanPeriodPulses"]                     # pitch in Hz, comparable to Oxford
say(f"Istanbul after dropping the duplicate: {len(ist)} recordings, {ist['id'].nunique()} people")
# gender: 1 = male, 0 = female (checked against the UCI page: 130 men, 122 women)

SHARED16 = ["F0_Hz", "locPctJitter", "locAbsJitter", "rapJitter", "ppq5Jitter", "ddpJitter",
            "locShimmer", "locDbShimmer", "apq3Shimmer", "apq5Shimmer", "apq11Shimmer", "ddaShimmer",
            "meanNoiseToHarmHarmonicity", "meanHarmToNoiseHarmonicity", "RPDE", "DFA"]
ALL752 = [c for c in ist.columns if c not in ("id", "gender", "class", "F0_Hz")]
FEATURE_SETS = {
    "Sex only": ["gender"],
    "Shared 16": SHARED16,
    "Shared 16 + sex": SHARED16 + ["gender"],
    "All 752": ALL752,
}
say("Feature counts:", {k: len(v) for k, v in FEATURE_SETS.items()})

y = ist["class"].to_numpy()
person = ist["id"].to_numpy()
sex = ist.groupby("id")["gender"].first()       # one value per person
label = ist.groupby("id")["class"].first()

# %% [markdown]
# ## 2. Models
# - Logistic regression: scaled first (pitch is ~150, jitter ~0.001). `class_weight="balanced"`
#   makes the 64 healthy people count as much as the 188 with PD. C=0.1 is a stronger
#   penalty than the default (1.0). WHY: with 752 features and 200 people, an unpenalised
#   model memorises noise. Same setting for every feature set, so the comparison is fair.
# - Random forest: 300 trees, balanced classes, at least 3 rows per leaf (stops trees
#   memorising single recordings).

# %%
def make_model(name):
    if name == "Logistic regression":
        return make_pipeline(StandardScaler(),
                             LogisticRegression(C=0.1, max_iter=5000, class_weight="balanced"))
    return RandomForestClassifier(n_estimators=300, min_samples_leaf=3, class_weight="balanced",
                                  n_jobs=-1, random_state=0)

# %% [markdown]
# ## 3. Out-of-fold predictions
# For each split, every recording gets a prediction from a model that never trained on it
# ("out-of-fold"). Then we average each person's recordings into ONE person-level score.
# WHY person level: a clinician screens a person, not a recording. It also stops one
# person with 3 recordings counting 3 times in the score.

# %%
def oof_predictions(model_name, cols, splitter):
    X = ist[cols].to_numpy()
    pred = np.zeros(len(ist))
    for tr, te in splitter.split(X, y, groups=person):
        m = make_model(model_name).fit(X[tr], y[tr])
        pred[te] = m.predict_proba(X[te])[:, 1]
    return pred

def person_scores(pred):
    p = pd.Series(pred).groupby(person).mean()          # one probability per person
    out = {"AUC": roc_auc_score(label, p),
           "Balanced acc": balanced_accuracy_score(label, p > 0.5)}
    for s, nm in [(1, "AUC men"), (0, "AUC women")]:
        m = sex == s
        out[nm] = roc_auc_score(label[m], p[m])
    return out

# %% [markdown]
# ## 4. Run everything
# 4 feature sets x 2 models x 2 splits x 10 repeats x 5 folds.

# %%
rows = []
for fs, cols in FEATURE_SETS.items():
    for model_name in ["Logistic regression", "Random forest"]:
        if fs == "Sex only" and model_name == "Random forest":
            continue    # one yes/no feature: a forest adds nothing over logistic regression
        for seed in range(N_REPEATS):
            splits = {
                "By recording (leaky)": StratifiedKFold(5, shuffle=True, random_state=seed),
                "By person (honest)": StratifiedGroupKFold(5, shuffle=True, random_state=seed),
            }
            for split_name, splitter in splits.items():
                sc = person_scores(oof_predictions(model_name, cols, splitter))
                rows.append({"features": fs, "model": model_name, "split": split_name, "seed": seed, **sc})
        print("done:", fs, model_name)

res = pd.DataFrame(rows)
# WHY blank these: within men only (or women only) "sex" is the same for everyone, so a sex-only
# model has nothing to rank people by. Any AUC there is noise from fold-to-fold differences.
res.loc[res.features == "Sex only", ["AUC men", "AUC women"]] = np.nan
res.to_csv(REPORTS / "04_leakage_istanbul.csv", index=False)

# %% [markdown]
# ## 5. Summary: mean across the 10 repeats, with the lowest and highest repeat
# The range shows how much the score depends on which people happened to land in which fold.
# It is NOT a full confidence interval (that would also need to account for the 252 people
# being a sample of all patients), so read it as "at least this much uncertainty".

# %%
def summarise(col):
    g = res.groupby(["features", "model", "split"])[col]
    return (g.mean().round(3).astype(str) + " [" + g.min().round(2).astype(str) + "-" + g.max().round(2).astype(str) + "]")

summary = pd.DataFrame({c: summarise(c) for c in ["AUC", "Balanced acc", "AUC men", "AUC women"]})
order = list(FEATURE_SETS)
summary = summary.reset_index()
summary["features"] = pd.Categorical(summary["features"], order, ordered=True)
summary = summary.sort_values(["features", "model", "split"])
say("\nPerson-level results, mean [min-max] over 10 repeats of 5-fold cross-validation:")
say(summary.to_string(index=False))

# %% [markdown]
# ## 6. The two headline comparisons

# %%
mean = res.groupby(["features", "model", "split"])["AUC"].mean()
say("\nLeakage gap (AUC by recording minus AUC by person):")
for fs in order:
    for mn in ["Logistic regression", "Random forest"]:
        if (fs, mn, "By person (honest)") in mean:
            gap = mean[(fs, mn, "By recording (leaky)")] - mean[(fs, mn, "By person (honest)")]
            say(f"  {fs:<16} {mn:<20} {mean[(fs, mn, 'By recording (leaky)')]:.3f} -> "
                f"{mean[(fs, mn, 'By person (honest)')]:.3f}  (gap {gap:+.3f})")

floor = mean[("Sex only", "Logistic regression", "By person (honest)")]
say(f"\nFloor to beat: sex only, honest split, AUC {floor:.3f} "
    "(0.60 if computed directly on all 252 people as in 03; slightly lower here because each fold's "
    "model sees a slightly different share of PD among men and women)")
say("Oxford for comparison (01b_baselines_parkinsons.py, 32 people, all 22 features): "
    "LR 0.91 -> 0.77, RF 0.96 -> 0.78")

# %% [markdown]
# ## 7. Chart: leaky vs honest AUC for each model and feature set

# %%
honest = res[res.split == "By person (honest)"].groupby(["features", "model"])["AUC"].agg(["mean", "min", "max"])
leaky = res[res.split == "By recording (leaky)"].groupby(["features", "model"])["AUC"].mean()
labels = [f"{f}\n{m}" for f, m in honest.index]
x = np.arange(len(labels))
fig, ax = plt.subplots(figsize=(10, 4.5))
ax.bar(x - 0.2, leaky.reindex(honest.index), 0.4, color="#c9c9c9", label="Split by recording (leaky)")
ax.bar(x + 0.2, honest["mean"], 0.4, color="#2b5d8a", label="Split by person (honest)")
ax.errorbar(x + 0.2, honest["mean"], yerr=[honest["mean"] - honest["min"], honest["max"] - honest["mean"]],
            fmt="none", ecolor="black", capsize=3, lw=1)
ax.axhline(0.5, color="black", lw=0.8, ls=":")
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
ax.set_ylim(0.4, 1.0); ax.set_ylabel("Person-level AUC")
ax.set_title("Istanbul, 252 people: leaky vs honest evaluation (bars on honest = min-max over 10 repeats)")
ax.legend(frameon=False, fontsize=8, loc="upper left")
fig.tight_layout(); fig.savefig(REPORTS / "04_leakage_istanbul.png", dpi=150)

(REPORTS / "04_leakage_istanbul.txt").write_text("\n".join(LOG))
print("\nSaved reports/04_leakage_istanbul.csv, .txt, .png")
