# %% [markdown]
# # Project 2, Stage 2e: threshold recalibration across hospitals
# Code written by: Claude (Cowork) at Kerem's request, 12 Nov 2026.
#
# **Question:** 05 showed that a voice model trained in one hospital still RANKS people
# fairly well in the other (AUC 0.71 to 0.83), but the 0.5 cut-off breaks: the Oxford-trained
# random forest flags 94% of Istanbul as PD when the truth is 75%. Can the cut-off be
# repaired, and how many labelled people from the new hospital does that take?
#
# **Design**
#   - Same 16 shared features, same two models and same "train scaling only" preprocessing
#     as 05, so the scores here are the scores from 05.
#   - Four ways of choosing the cut-off on the new hospital:
#       a) Fixed 0.5: the default, what 05 reported.
#       b) Tuned at home: the cut-off that gives the best balanced accuracy on the TRAINING
#          hospital (out-of-fold, split by person). Uses no data from the new hospital.
#       c) Local calibration with k people: label k random people in the new hospital,
#          choose the cut-off that is best on those k, then score everyone ELSE.
#          Repeated 500 times with different random people, because with small k the
#          answer depends heavily on who you happened to label.
#       d) Oracle: the best possible cut-off using all the new hospital's labels.
#          This is a ceiling, not a method. It tells you the most a cut-off can ever fix.
#   - Metric: balanced accuracy per person = average of (share of PD people caught) and
#     (share of healthy people cleared). 0.5 = useless, 1.0 = perfect.
#     WHY not plain accuracy: both datasets are 75% PD, so "everyone has PD" scores 75%.
#
# **Outputs:** reports/07_threshold_recalibration.csv, .txt, .png
# Module links: 5_01, 6_03 (evaluation, data shift).

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
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1] if "__file__" in globals() else Path.cwd()
RAW, REPORTS = ROOT / "data" / "raw", ROOT / "reports"
REPORTS.mkdir(exist_ok=True)
rng = np.random.default_rng(0)
N_DRAWS = 500

LOG = []
def say(*parts):
    line = " ".join(str(p) for p in parts); print(line); LOG.append(line)

# %% [markdown]
# ## 1. Load both datasets (identical to 05)

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

DATA = {"Oxford": oxf, "Istanbul": ist}
for k, d in DATA.items():
    p = d.groupby("person")["label"].first()
    say(f"{k}: {len(d)} recordings, {len(p)} people, {int(p.sum())} PD, {int((1 - p).sum())} healthy")

# %% [markdown]
# ## 2. Helpers

# %%
def make_model(name):
    if name == "Logistic regression":
        return LogisticRegression(C=0.1, max_iter=5000, class_weight="balanced")
    return RandomForestClassifier(n_estimators=300, min_samples_leaf=3, class_weight="balanced",
                                  n_jobs=-1, random_state=0)

def scale(train_df, test_df):
    # WHY training averages only: this is 05's "train scaling only". Nothing from the
    # new hospital is used to prepare the features.
    mu, sd = train_df[FEATS].mean(), train_df[FEATS].std()
    return (train_df[FEATS] - mu) / sd, (test_df[FEATS] - mu) / sd

def per_person(df, pred):
    """One score and one label per person (average of that person's recordings)."""
    p = pd.Series(pred, index=df.index).groupby(df["person"]).mean()
    y = df.groupby("person")["label"].first()
    return p.to_numpy(), y.to_numpy()

def bal_acc(p, y, cut):
    flag = p > cut
    caught = flag[y == 1].mean()          # share of PD people flagged (sensitivity)
    cleared = (~flag)[y == 0].mean()      # share of healthy people not flagged (specificity)
    return (caught + cleared) / 2

def best_cut(p, y):
    """Cut-off with the highest balanced accuracy on (p, y).
    WHY midpoints: any cut-off between two neighbouring scores flags the same people,
    so only the gaps between sorted scores need checking. If several are equally good,
    take the middle one so the choice is not pushed to an extreme."""
    s = np.unique(p)
    cands = np.concatenate([[s[0] - 1e-6], (s[:-1] + s[1:]) / 2, [s[-1] + 1e-6]])
    scores = np.array([bal_acc(p, y, c) for c in cands])
    good = cands[scores >= scores.max() - 1e-12]
    return float(np.median(good))

# %% [markdown]
# ## 3. Run: train on one hospital, choose the cut-off four ways, score on the other

# %%
rows, curves = [], []
for train_name, test_name in [("Oxford", "Istanbul"), ("Istanbul", "Oxford")]:
    tr_df, te_df = DATA[train_name], DATA[test_name]
    for mn in ["Logistic regression", "Random forest"]:
        # Scores on the new hospital (same as 05)
        Xtr, Xte = scale(tr_df, te_df)
        p, y = per_person(te_df, make_model(mn).fit(Xtr, tr_df.label).predict_proba(Xte)[:, 1])
        n, truth = len(y), y.mean()

        # (b) cut-off tuned at home: out-of-fold scores on the training hospital, split by person
        oof = np.zeros(len(tr_df))
        for a, b in StratifiedGroupKFold(5, shuffle=True, random_state=0).split(tr_df, tr_df.label, groups=tr_df.person):
            Xa, Xb = scale(tr_df.iloc[a], tr_df.iloc[b])
            oof[b] = make_model(mn).fit(Xa, tr_df.label.iloc[a]).predict_proba(Xb)[:, 1]
        home_cut = best_cut(*per_person(tr_df, oof))

        base = {"train": train_name, "test": test_name, "model": mn, "AUC": roc_auc_score(y, p)}
        def add(method, k, cut, ba, lo=np.nan, hi=np.nan, flagged=np.nan):
            rows.append({**base, "method": method, "k labelled": k, "cut-off": cut,
                         "balanced acc": ba, "BA low": lo, "BA high": hi, "share flagged PD": flagged})

        add("a) fixed 0.5", 0, 0.5, bal_acc(p, y, 0.5), flagged=(p > 0.5).mean())
        add("b) tuned at home", 0, home_cut, bal_acc(p, y, home_cut), flagged=(p > home_cut).mean())

        # (c) local calibration with k labelled people from the new hospital
        ks = [10, 20, 40, 80] if n > 100 else [8, 12, 16]
        for k in ks:
            bas, cuts, flags = [], [], []
            while len(bas) < N_DRAWS:
                idx = rng.permutation(n)
                cal, rest = idx[:k], idx[k:]
                # WHY skip: a cut-off cannot be tuned on a sample with only PD or only healthy people
                if len(np.unique(y[cal])) < 2 or len(np.unique(y[rest])) < 2:
                    continue
                c = best_cut(p[cal], y[cal])
                bas.append(bal_acc(p[rest], y[rest], c)); cuts.append(c); flags.append((p[rest] > c).mean())
            add("c) local calibration", k, float(np.median(cuts)), float(np.mean(bas)),
                *np.percentile(bas, [2.5, 97.5]), flagged=float(np.mean(flags)))
            curves.append((f"{train_name} -> {test_name}", mn, k, np.mean(bas), *np.percentile(bas, [2.5, 97.5])))

        # (d) oracle: best cut-off using every label in the new hospital (a ceiling, not a method)
        oc = best_cut(p, y)
        add("d) oracle (all labels)", n, oc, bal_acc(p, y, oc), flagged=(p > oc).mean())
        say(f"\n{train_name} -> {test_name}, {mn}: {n} people, truth {truth:.0%} PD, AUC {roc_auc_score(y, p):.3f}")

res = pd.DataFrame(rows)
res.to_csv(REPORTS / "07_threshold_recalibration.csv", index=False)
say("\nBalanced accuracy per person on the new hospital, by how the cut-off was chosen")
say("(for local calibration: mean over 500 random draws, with the middle 95% of draws)")
say(res.drop(columns=["AUC"]).round(3).to_string(index=False))

# %% [markdown]
# ## 4. Chart: how much does labelling k local people buy?

# %%
fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
for ax, direction in zip(axes, ["Oxford -> Istanbul", "Istanbul -> Oxford"]):
    tr, te = direction.split(" -> ")
    for mn, col in [("Logistic regression", "#2b5d8a"), ("Random forest", "#d98c2b")]:
        c = [r for r in curves if r[0] == direction and r[1] == mn]
        ks = [r[2] for r in c]; m = np.array([r[3] for r in c])
        lo = np.array([r[4] for r in c]); hi = np.array([r[5] for r in c])
        ax.errorbar(ks, m, yerr=[m - lo, hi - m], fmt="o-", color=col, capsize=3, label=f"{mn}: local calibration")
        sub = res[(res.train == tr) & (res.test == te) & (res.model == mn)]
        ax.axhline(sub[sub.method == "a) fixed 0.5"]["balanced acc"].iloc[0], color=col, ls=":", lw=1)
        ax.axhline(sub[sub.method == "d) oracle (all labels)"]["balanced acc"].iloc[0], color=col, ls="--", lw=1)
    ax.axhline(0.5, color="black", lw=0.6)
    ax.set_title(f"Train {tr}, use in {te}", fontsize=10)
    ax.set_xlabel("People labelled in the new hospital (k)")
axes[0].set_ylabel("Balanced accuracy on the remaining people")
axes[0].legend(frameon=False, fontsize=7, loc="lower right")
fig.suptitle("Repairing the cut-off. Dotted = fixed 0.5, dashed = oracle ceiling, bars = middle 95% of 500 draws", fontsize=9)
fig.tight_layout(); fig.savefig(REPORTS / "07_threshold_recalibration.png", dpi=150)

(REPORTS / "07_threshold_recalibration.txt").write_text("\n".join(LOG) + "\n")
print("\nSaved reports/07_threshold_recalibration.csv, .txt, .png")

# %% [markdown]
# ## 5. Kerem: your turn
# 1. Which hospital direction gains most from moving the cut-off, and why? (Look at
#    "share flagged PD" for method a against the truth of 75%.)
# 2. Does the tuned-at-home cut-off (b) help or hurt compared with 0.5? What does that say
#    about using one hospital's cut-off in another?
# 3. How wide is the 95% range at the smallest k? Would you trust a cut-off set on that few people?
# 4. The oracle is not much above 0.7 anywhere. What does that tell you about what a cut-off
#    can and cannot fix?
