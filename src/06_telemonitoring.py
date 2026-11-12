# %% [markdown]
# # Project 2, Stage 2d: telemonitoring, within-person test
# Code written by: Claude Code at Kerem's request (task sheet: reports/06_TASK_SHEET.md).
# Kerem's own parts: the two PREDICTIONS below and the write-up in reports/06_findings.md.
# Question: does a person's voice get worse when their own symptom score gets worse?
# Data: Parkinson's Telemonitoring (Tsanas et al., 2010), UCI, 42 people, ~6 months each.
# Run cell by cell in VS Code (Shift+Enter).

# %%
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, GroupKFold, cross_val_predict
from sklearn.metrics import r2_score, mean_absolute_error

ROOT = Path(__file__).resolve().parents[1] if "__file__" in globals() else Path.cwd()
DATA = ROOT / "data" / "raw" / "telemonitoring" / "parkinsons_updrs.data"

VOICE = ["Jitter(%)", "Jitter(Abs)", "Jitter:RAP", "Jitter:PPQ5", "Jitter:DDP",
         "Shimmer", "Shimmer(dB)", "Shimmer:APQ3", "Shimmer:APQ5", "Shimmer:APQ11", "Shimmer:DDA",
         "NHR", "HNR", "RPDE", "DFA", "PPE"]

# %% [markdown]
# ## Step 1: load and meet the data

# %%
# WHY: read_csv turns the file into a table (DataFrame); one row = one voice recording.
df = pd.read_csv(DATA)
# WHY: "#" is awkward in code, so rename. rename() returns a NEW table, so assign it back.
df = df.rename(columns={"subject#": "subject"})

# CAREFUL: here sex 0 = male, 1 = female. This is the OPPOSITE of Istanbul's "gender".
# Mixing them up would silently flip every sex result.

print("shape:", df.shape)                          # rows (recordings) x columns
print("people:", df["subject"].nunique())          # how many different people
print(df.head())                                   # eyeball the first rows

# %%
# WHY: one row per PERSON shows what a person looks like before any modelling.
# agg(name=(column, function)) lets us name each output column.
people = df.groupby("subject").agg(
    n=("age", "size"),                  # number of recordings
    age=("age", "first"),               # age is constant within a person
    sex=("sex", "first"),
    first_day=("test_time", "min"),
    last_day=("test_time", "max"),
    updrs_min=("total_UPDRS", "min"),
    updrs_max=("total_UPDRS", "max"),
)
print(people.describe())
print("sex counts (0 = male, 1 = female):")
print(people["sex"].value_counts())

# %% [markdown]
# ## Step 2: how was the symptom score made?

# %%
# WHY: "linearly interpolated" (from the .names file) means: the clinician measured UPDRS
# only at baseline, ~3 months and ~6 months, and the scores for every recording in
# between were filled in along straight lines. So within a person the score is
# (almost) just a function of time, not a fresh measurement each time.

# WHY nunique: if the score were measured each time, a person would have ~140 different values.
n_unique = df.groupby("subject")["total_UPDRS"].nunique()
print("unique total_UPDRS values per person:")
print(n_unique.describe())

# %%
# WHY: fit a straight line of UPDRS on time for each person and see how much of the
# variation the line explains (R squared). If the score was interpolated, R squared is high.
def line_fit(x, y):
    """Return (slope, r_squared) of a straight-line fit y = a + b*x."""
    slope, intercept = np.polyfit(x, y, 1)
    resid = y - np.polyval([slope, intercept], x)       # what the line does not explain
    r2 = 1 - resid.var() / y.var()                      # 1 - residual variance / total variance
    return slope, r2

r2_list = []
for pid, g in df.groupby("subject"):
    _, r2 = line_fit(g["test_time"].to_numpy(), g["total_UPDRS"].to_numpy())
    r2_list.append(r2)

print("median R squared of UPDRS ~ time across people:", round(float(np.median(r2_list)), 3))
# DRAFT by Claude, for Kerem to review and rewrite in his own words:
# Median R squared 0.83 means a straight line against time explains most of each person's
# UPDRS "trajectory", because the scores between clinic visits were interpolated. So any
# voice feature that just drifts over time would correlate with UPDRS by construction.

# %% [markdown]
# ## Step 3: the leakage trap, regression version
# Write your prediction for "age + sex, random split" here BEFORE running:
# MY PREDICTION:

# %%
# WHY: the same trap as scripts 04/05. A random split puts recordings from the SAME person
# in both train and test, so the model can recognise the person (age + sex is nearly a
# person ID) and read off their average score. GroupKFold keeps each person entirely in
# train or entirely in test, which is the honest test.
y = df["total_UPDRS"].to_numpy()
groups = df["subject"].to_numpy()

feature_sets = {
    "voice only":       VOICE,
    "age + sex only":   ["age", "sex"],
    "voice + age + sex": VOICE + ["age", "sex"],
}
splits = {
    "random KFold":      KFold(5, shuffle=True, random_state=0),
    "GroupKFold person": GroupKFold(5),
}

rows = []
for fs_name, cols in feature_sets.items():
    X = df[cols].to_numpy()
    for sp_name, cv in splits.items():
        model = RandomForestRegressor(200, min_samples_leaf=3, random_state=0, n_jobs=-1)
        # groups is ignored by KFold and used by GroupKFold, so one call works for both
        pred = cross_val_predict(model, X, y, cv=cv, groups=groups)
        rows.append({"features": fs_name, "split": sp_name,
                     "R2": r2_score(y, pred), "MAE": mean_absolute_error(y, pred)})

# WHY: the baseline to beat is "predict the average for everyone".
baseline_mae = mean_absolute_error(y, np.full_like(y, y.mean()))
res3 = pd.DataFrame(rows).round(3)
print(res3.to_string(index=False))
print("MAE of predicting the average for everyone:", round(baseline_mae, 3))
# DRAFT by Claude, for Kerem to review and rewrite in his own words:
# With a random split the same person appears in train and test, and age + sex is almost a
# person ID, so the forest just recalls that person's typical score (R2 0.84). Split by
# person, age + sex and voice both lose to predicting the average (negative R2, MAE above
# 8.66), so nothing learned transfers to new people.

# %% [markdown]
# ## Step 4: within-person slopes
# Write your expected sign of the correlation here BEFORE running:
# MY PREDICTION:

# %%
# WHY standardise: the 16 measures are on very different scales (jitter ~0.005, HNR ~21).
# Subtracting the mean and dividing by the std puts them all on the same footing.
Z = (df[VOICE] - df[VOICE].mean()) / df[VOICE].std()

# WHY one index: 16 separate tests give 16 chances of a lucky result. We fix ONE main
# outcome in advance. Higher = rougher voice. HNR is subtracted because higher HNR = cleaner.
jitter_cols  = [c for c in VOICE if c.startswith("Jitter")]
shimmer_cols = [c for c in VOICE if c.startswith("Shimmer")]
rough_parts = (Z[jitter_cols].mean(axis=1)
               + Z[shimmer_cols].mean(axis=1)
               + Z["NHR"] + Z["RPDE"] - Z["HNR"])
# NOTE: I averaged within jitter and within shimmer first so those two families do not
# outweigh NHR, RPDE and HNR just because they have 5 and 6 columns. This is a design
# choice; the task sheet says "average of standardised jitter, shimmer, NHR, plus RPDE,
# minus HNR". DFA and PPE are left out as instructed.
df["roughness"] = rough_parts / 5

# %%
# WHY per-person slopes: each person is compared only with themselves, so age, sex,
# microphone and home cancel out. time in months = test_time / 30.
slope_rows = []
for pid, g in df.groupby("subject"):
    months = (g["test_time"] / 30).to_numpy()
    s_updrs, _ = line_fit(months, g["total_UPDRS"].to_numpy())
    s_voice, _ = line_fit(months, g["roughness"].to_numpy())
    slope_rows.append({"subject": pid, "updrs_slope": s_updrs, "voice_slope": s_voice})
slopes = pd.DataFrame(slope_rows)
print(slopes.describe().round(3))

# %%
# WHY Spearman: it works on ranks, so one extreme person cannot dominate the result.
rho, p = stats.spearmanr(slopes["updrs_slope"], slopes["voice_slope"])
print(f"Spearman rho = {rho:.3f}, p = {p:.3f}, n = {len(slopes)} people")
# DRAFT by Claude, for Kerem to review and rewrite in his own words:
# Spearman rho = -0.050, p = 0.752, n = 42: no detectable link between how fast a person's
# UPDRS changed and how fast their voice roughness changed. Rough 95% interval for rho is
# about -0.35 to +0.26, so a strong link is ruled out but a weak one is not. Each UPDRS slope
# rests on about three real clinic exams, so the slopes are noisy (low power).

# %%
# ROBUSTNESS (exploratory): the main index above averages within jitter and within shimmer
# first (my design choice). Here is the plain average the task sheet describes, to check
# the result does not depend on that choice.
plain_cols = jitter_cols + shimmer_cols + ["NHR", "RPDE"]
df["roughness_plain"] = (Z[plain_cols].sum(axis=1) - Z["HNR"]) / (len(plain_cols) + 1)

plain_rows = []
for pid, g in df.groupby("subject"):
    months = (g["test_time"] / 30).to_numpy()
    s_u, _ = line_fit(months, g["total_UPDRS"].to_numpy())
    s_v, _ = line_fit(months, g["roughness_plain"].to_numpy())
    plain_rows.append((s_u, s_v))
plain = np.array(plain_rows)
rho2, p2 = stats.spearmanr(plain[:, 0], plain[:, 1])
print(f"Plain-average index: Spearman rho = {rho2:.3f}, p = {p2:.3f}")
# Result when Claude ran it: rho = -0.037, p = 0.814, same story as the main index.
# The individual families (jitter, shimmer, NHR, HNR, RPDE) were also all non-significant
# (rho between -0.15 and +0.06). Those are secondary, 5 extra tests, so no weight on them.

# %% [markdown]
# ## Step 5: write reports/06_findings.md in your own words
