# %% [markdown]
# # Project 2, Stage 2c: external validation, Istanbul <-> Oxford
#
# **Question:** does a voice model built in one hospital still work in another?
# Within-hospital scores (04, and 01b for Oxford) only tell you how well a model works
# on new people FROM THE SAME CLINIC, recorded with the same microphone and software.
# A screening tool would be used somewhere else. This is the harder, more honest test.
#
# **Design**
#   - Features: the 16 measures both datasets share (PPE excluded, see 03).
#   - Two directions: train on Istanbul (252 people), test on Oxford (32), and the reverse.
#   - Two preprocessing choices:
#       a) "Train scaling only": standardise with the TRAINING hospital's averages and
#          apply them to the test hospital. This is the naive approach.
#       b) "Each hospital scaled to itself": standardise each dataset with its OWN averages.
#          WHY: 03 showed jitter is 3-9x smaller and shimmer ~2.5x larger in Istanbul,
#          because of different analysis software (MDVP vs Praat). Scaling each dataset to
#          itself removes that software offset. It uses the test hospital's feature
#          averages, but NEVER its labels, and both datasets are 75% PD, so the averages
#          are not tilted by a different patient mix.
#   - Models: logistic regression and random forest, same settings as 04.
#   - Scored per person (average of their recordings), with a bootstrap 95% interval
#     that resamples PEOPLE in the test set.
#   - Benchmark: within-hospital honest AUC on the same 16 features (computed here for
#     Oxford, taken from 04 for Istanbul).
#
# **Outputs:** reports/05_external_validation.csv, .txt, .png
# Module links: 4_02, 5_01, 6_03 (data shift, evaluation).

# %%
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, balanced_accuracy_score

ROOT = Path(__file__).resolve().parents[1] if "__file__" in globals() else Path.cwd()
RAW, REPORTS = ROOT / "data" / "raw", ROOT / "reports"
REPORTS.mkdir(exist_ok=True)
rng = np.random.default_rng(0)

LOG = []
def say(*parts):
    line = " ".join(str(p) for p in parts); print(line); LOG.append(line)

# %% [markdown]
# ## 1. Load both datasets into the same shape
# Same 16 features, renamed to one common name each, plus person ID and label.

# %%
MATCH = [  # (common name, Oxford column, Istanbul column), from 03_feature_matching.csv
    ("pitch_hz",      "MDVP:Fo(Hz)",      "F0_Hz"),
    ("jitter_local",  "MDVP:Jitter(%)",   "locPctJitter"),
    ("jitter_abs",    "MDVP:Jitter(Abs)", "locAbsJitter"),
    ("jitter_rap",    "MDVP:RAP",         "rapJitter"),
    ("jitter_ppq5",   "MDVP:PPQ",         "ppq5Jitter"),
    ("jitter_ddp",    "Jitter:DDP",       "ddpJitter"),
    ("shimmer_local", "MDVP:Shimmer",     "locShimmer"),
    ("shimmer_db",    "MDVP:Shimmer(dB)", "locDbShimmer"),
    ("shimmer_apq3",  "Shimmer:APQ3",     "apq3Shimmer"),
    ("shimmer_apq5",  "Shimmer:APQ5",     "apq5Shimmer"),
    ("shimmer_apq11", "MDVP:APQ",         "apq11Shimmer"),
    ("shimmer_dda",   "Shimmer:DDA",      "ddaShimmer"),
    ("nhr",           "NHR",              "meanNoiseToHarmHarmonicity"),
    ("hnr",           "HNR",              "meanHarmToNoiseHarmonicity"),
    ("rpde",          "RPDE",             "RPDE"),
    ("dfa",           "DFA",              "DFA"),
]
FEATS = [m[0] for m in MATCH]

oxf_raw = pd.read_csv(RAW / "parkinsons.data")
oxf = pd.DataFrame({c: oxf_raw[o] for c, o, _ in MATCH})
oxf["person"], oxf["label"] = oxf_raw["name"].str.extract(r"(S\d+)")[0], oxf_raw["status"]

ist_raw = pd.read_csv(RAW / "uci470_istanbul" / "pd_speech_features.csv", header=1).drop_duplicates().copy()
ist_raw["F0_Hz"] = 1 / ist_raw["meanPeriodPulses"]
ist = pd.DataFrame({c: ist_raw[i].to_numpy() for c, _, i in MATCH})
ist["person"], ist["label"] = ist_raw["id"].to_numpy(), ist_raw["class"].to_numpy()
ist["sex"] = ist_raw["gender"].to_numpy()

DATA = {"Oxford": oxf, "Istanbul": ist}
for k, d in DATA.items():
    p = d.groupby("person")["label"].first()
    say(f"{k}: {len(d)} recordings, {len(p)} people, {p.mean():.0%} PD")

# %% [markdown]
# ## 2. Helpers

# %%
def zscore(train_df, test_df, mode):
    if mode == "Train scaling only":
        mu, sd = train_df[FEATS].mean(), train_df[FEATS].std()
        return (train_df[FEATS] - mu) / sd, (test_df[FEATS] - mu) / sd
    # "Each hospital scaled to itself"
    return ((train_df[FEATS] - train_df[FEATS].mean()) / train_df[FEATS].std(),
            (test_df[FEATS] - test_df[FEATS].mean()) / test_df[FEATS].std())

def make_model(name):
    if name == "Logistic regression":
        return LogisticRegression(C=0.1, max_iter=5000, class_weight="balanced")
    return RandomForestClassifier(n_estimators=300, min_samples_leaf=3, class_weight="balanced",
                                  n_jobs=-1, random_state=0)

def person_auc(df, pred):
    p = pd.Series(pred, index=df.index).groupby(df["person"]).mean()
    y = df.groupby("person")["label"].first()
    return roc_auc_score(y, p), p, y

def boot_ci(p, y, n=2000):
    # WHY resample people: the uncertainty comes from WHICH people happen to be in the
    # test set. With only 8 healthy people in Oxford, this interval will be wide.
    ids, aucs = np.arange(len(y)), []
    for _ in range(n):
        b = rng.choice(ids, len(ids), replace=True)
        if y.iloc[b].nunique() == 2:
            aucs.append(roc_auc_score(y.iloc[b], p.iloc[b]))
    return np.percentile(aucs, [2.5, 97.5])

# %% [markdown]
# ## 3. Benchmark: within-Oxford honest AUC on the same 16 features
# Same method as 04 (split by person, 5 folds, 10 repeats), so the numbers compare.

# %%
within = {}
for mn in ["Logistic regression", "Random forest"]:
    aucs = []
    for seed in range(10):
        pred = np.zeros(len(oxf))
        for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(oxf, oxf.label, groups=oxf.person):
            Xtr, Xte = zscore(oxf.iloc[tr], oxf.iloc[te], "Train scaling only")
            pred[te] = make_model(mn).fit(Xtr, oxf.label.iloc[tr]).predict_proba(Xte)[:, 1]
        aucs.append(person_auc(oxf, pred)[0])
    within[("Oxford", mn)] = np.mean(aucs)
# From 04_leakage_istanbul (Shared 16, honest split): copied, not recomputed, to keep this script fast.
within[("Istanbul", "Logistic regression")] = 0.772
within[("Istanbul", "Random forest")] = 0.784
say("\nWithin-hospital honest AUC, 16 shared features:",
    {f"{k[0]} / {k[1]}": round(v, 3) for k, v in within.items()})

# %% [markdown]
# ## 4. Cross-hospital: train on one, test on the other

# %%
rows = []
for train_name, test_name in [("Istanbul", "Oxford"), ("Oxford", "Istanbul")]:
    tr_df, te_df = DATA[train_name], DATA[test_name]
    for mode in ["Train scaling only", "Each hospital scaled to itself"]:
        Xtr, Xte = zscore(tr_df, te_df, mode)
        for mn in ["Logistic regression", "Random forest"]:
            pred = make_model(mn).fit(Xtr, tr_df.label).predict_proba(Xte)[:, 1]
            auc, p, y = person_auc(te_df, pred)
            lo, hi = boot_ci(p, y)
            row = {"train": train_name, "test": test_name, "scaling": mode, "model": mn,
                   "AUC": auc, "CI low": lo, "CI high": hi,
                   # WHY these two: AUC only checks the RANKING of people. A screening tool also
                   # needs a cut-off (here 0.5). If the scale shift moves everyone's score, the
                   # ranking can survive while the cut-off breaks.
                   "Balanced acc at 0.5": balanced_accuracy_score(y, p > 0.5),
                   "Share flagged PD": (p > 0.5).mean(),
                   "within-hospital AUC of test site": within[(test_name, mn)]}
            if test_name == "Istanbul":   # only Istanbul has sex
                s = te_df.groupby("person")["sex"].first()
                row["AUC men"] = roc_auc_score(y[s == 1], p[s == 1])
                row["AUC women"] = roc_auc_score(y[s == 0], p[s == 0])
            rows.append(row)

res = pd.DataFrame(rows)
res.to_csv(REPORTS / "05_external_validation.csv", index=False)
say("\nCross-hospital person-level AUC (95% bootstrap interval over test people):")
say(res.round(3).to_string(index=False))

# %% [markdown]
# ## 5. Which features carry over? Logistic regression coefficients, trained on each site
# Same sign in both = the feature means the same thing in both hospitals.

# %%
coefs = {}
for name, d in DATA.items():
    X = (d[FEATS] - d[FEATS].mean()) / d[FEATS].std()
    coefs[name] = pd.Series(make_model("Logistic regression").fit(X, d.label).coef_[0], index=FEATS)
coefs = pd.DataFrame(coefs).round(2)
coefs["same sign"] = np.sign(coefs["Oxford"]) == np.sign(coefs["Istanbul"])
say("\nStandardised logistic regression coefficients (positive = pushes towards PD):")
say(coefs.to_string())
say(f"Same sign in both hospitals: {int(coefs['same sign'].sum())} of {len(coefs)}")

# %% [markdown]
# ## 6. Chart

# %%
t = res[res.model == "Logistic regression"]
fig, ax = plt.subplots(figsize=(8, 4))
labels = [f"{r.train} -> {r.test}\n{r.scaling}" for r in t.itertuples()]
x = np.arange(len(t))
ax.errorbar(x, t.AUC, yerr=[t.AUC - t["CI low"], t["CI high"] - t.AUC], fmt="o", color="#2b5d8a", capsize=4,
            label="Cross-hospital AUC (95% interval)")
ax.scatter(x, t["within-hospital AUC of test site"], marker="_", s=600, color="#d98c2b",
           label="Within-hospital AUC of the test site")
ax.axhline(0.5, color="black", lw=0.8, ls=":")
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
ax.set_ylim(0.2, 1.0); ax.set_ylabel("Person-level AUC")
ax.set_title("Does the voice model travel? Logistic regression, 16 shared features")
ax.legend(frameon=False, fontsize=8, loc="lower left")
fig.tight_layout(); fig.savefig(REPORTS / "05_external_validation.png", dpi=150)

(REPORTS / "05_external_validation.txt").write_text("\n".join(LOG))
print("\nSaved reports/05_external_validation.csv, .txt, .png")
