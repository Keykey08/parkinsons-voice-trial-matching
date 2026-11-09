# %% [markdown]
# # Project 2 (Neuro / health), Stage 1: load and profile the Parkinson's voice data
#
# **Question (part A, classical ML):** can simple voice measurements tell
# apart people with Parkinson's disease from healthy people?
# **Part B (GenAI, Stage 2):** match a described patient to recruiting
# Parkinson's clinical trials with embeddings + an LLM explanation
# (see 02_fetch_trials.py).
#
# Data: UCI "Parkinsons" (Little et al., 2007), CC BY 4.0, see ../../DATA_SOURCES.md
# Module links: 2_05 EDA, 4_02 logistic regression, 5_01 RF/GBDT, 6_03 imbalance.

# %%
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1] if "__file__" in globals() else Path.cwd()
df = pd.read_csv(ROOT / "data" / "raw" / "parkinsons.data")
out = [f"Shape: {df.shape[0]} recordings x {df.shape[1]} columns"]

# %% [markdown]
# ## 1. Who is in the data?
# Each row is ONE voice recording, but each PERSON gave about 6 recordings.
# The `name` column looks like phon_R01_S01_3 = subject S01, recording 3.
# WHY this matters most: if recordings of the same person land in both the
# training and the test set, the model can recognise the VOICE, not the
# disease, and the accuracy will be fake. That is leakage again (6_03).
# Fix for Stage 2: split by person (GroupKFold), never by row.

# %%
df["subject"] = df["name"].str.extract(r"(S\d+)")[0]
per_subject = df.groupby("subject").agg(recordings=("name", "size"), status=("status", "max"))
out.append(f"People: {per_subject.shape[0]}  (Parkinson's: {int(per_subject.status.sum())}, healthy: {int((per_subject.status == 0).sum())})")
out.append(f"Recordings per person: min {per_subject.recordings.min()}, max {per_subject.recordings.max()}")
out.append("CHECK: parkinsons.names says 31 people (23 with PD), but the file has "
           f"{per_subject.shape[0]} subject IDs ({int(per_subject.status.sum())} with PD). "
           "Either one person has two IDs or the documentation is off. Open question; ask in seminar.")
out.append(f"Recordings labelled Parkinson's: {df.status.mean():.1%}  -> imbalanced; accuracy alone would mislead")

# %% [markdown]
# ## 2. Data quality checks

# %%
out.append(f"Missing values: {int(df.isna().sum().sum())}")
out.append(f"Duplicate rows: {int(df.duplicated().sum())}")
num = df.drop(columns=["name", "subject"])
corr = num.corr().abs()
pairs = [(a, b, corr.loc[a, b]) for i, a in enumerate(corr.columns) for b in corr.columns[i+1:] if corr.loc[a, b] > 0.95 and "status" not in (a, b)]
out.append(f"Feature pairs with |correlation| > 0.95: {len(pairs)}  -> many jitter/shimmer measures repeat the same info")
# WHY it matters: highly correlated inputs make logistic regression
# coefficients unstable (hard to interpret), a point from 4_01 stats vs ML.

# %% [markdown]
# ## 3. Which features separate the groups? (quick look, not a model)

# %%
diff = (num.groupby("status").mean().T.assign(ratio=lambda d: d[1] / d[0]).sort_values("ratio"))
out.append("\nMean by group (0 = healthy, 1 = Parkinson's), most different first:\n"
           + diff.reindex(diff["ratio"].sub(1).abs().sort_values(ascending=False).index).head(8).round(3).to_string())

fig, axes = plt.subplots(1, 3, figsize=(11, 3.5))
for ax, col in zip(axes, ["PPE", "spread1", "MDVP:Fo(Hz)"]):
    df.boxplot(column=col, by="status", ax=ax); ax.set_title(col); ax.set_xlabel("0 = healthy, 1 = Parkinson's")
plt.suptitle(""); plt.tight_layout()
(ROOT / "reports").mkdir(exist_ok=True)
plt.savefig(ROOT / "reports" / "01_feature_boxplots.png", dpi=150); plt.close()

(ROOT / "reports" / "01_profile.txt").write_text("\n".join(out)); print("\n".join(out))
