# %% [markdown]
# # Project 2, Stage 2a: profile the Istanbul dataset (UCI 470, Sakar et al. 2019)
# Goal: understand the second dataset BEFORE modelling it, and check whether it can be
# compared with Oxford (UCI 174) at all.
# Five parts:
#   1. Basic profile (size, people, classes, missing, feature families)
#   2. The duplicate row: which person, what to do about it
#   3. Feature-matching table: which Istanbul columns measure the same thing as Oxford's
#   4. Scale check: same name, same numbers? (Oxford used MDVP software, Istanbul used Praat)
#   5. Sex confounder check: is "sex" quietly doing the work of "Parkinson's"?
# Outputs go to reports/03_*.  Run cell by cell in VS Code (Shift+Enter).

# %%
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
from sklearn.metrics import roc_auc_score

# WHY this ROOT line: works both when run as a script and cell by cell in VS Code.
ROOT = Path(__file__).resolve().parents[1] if "__file__" in globals() else Path.cwd()
RAW = ROOT / "data" / "raw"
REPORTS = ROOT / "reports"
REPORTS.mkdir(exist_ok=True)

# WHY a log: everything printed is also saved, so the numbers in the walkthrough can be
# traced back to a file rather than to my memory of a terminal.
LOG = []
def say(*parts):
    line = " ".join(str(p) for p in parts)
    print(line)
    LOG.append(line)

# %% [markdown]
# ## 1. Load and basic profile

# %%
# WHY header=1: row 0 of the CSV is a "family" banner (Baseline Features, MFCC, Wavelet...),
# row 1 holds the real column names.
ist_raw = pd.read_csv(RAW / "uci470_istanbul" / "pd_speech_features.csv", header=1)
banner = pd.read_csv(RAW / "uci470_istanbul" / "pd_speech_features.csv", header=None, nrows=1).iloc[0]
oxf = pd.read_csv(RAW / "parkinsons.data")
oxf["person"] = oxf["name"].str.extract(r"(S\d+)")[0]   # e.g. phon_R01_S01_1 -> S01

say("=== 1. BASIC PROFILE ===")
say("Istanbul shape (rows, columns):", ist_raw.shape)
say("People:", ist_raw["id"].nunique(), "| ids run", ist_raw["id"].min(), "to", ist_raw["id"].max())
say("Recordings per person:", ist_raw.groupby("id").size().value_counts().to_dict())

# WHY check this: if a person had both labels, the label would describe a recording, not a person.
labels_per_person = ist_raw.groupby("id")["class"].nunique()
say("People with more than one label:", int((labels_per_person > 1).sum()))

people = ist_raw.groupby("id")[["class", "gender"]].first()
say("People by class (1 = PD):", people["class"].value_counts().sort_index().to_dict(),
    f"-> {people['class'].mean():.0%} PD (Oxford: {oxf.groupby('person')['status'].first().mean():.0%})")
say("Missing values:", int(ist_raw.isna().sum().sum()), "| constant columns:", int((ist_raw.nunique() <= 1).sum()))

# Feature families: the banner names a family at its first column, blanks after it -> forward-fill.
families = banner.ffill().iloc[2:-1]          # skip id, gender at the start and class at the end
families.index = ist_raw.columns[2:-1]
say("Features by family:", families.str.strip().value_counts().to_dict())

# WHY count near-identical pairs: Oxford had 19 pairs above 0.95. Many more here would mean
# the 752 features carry far less independent information than their number suggests.
X_only = ist_raw.drop(columns=["id", "gender", "class"])
corr = X_only.corr().abs().to_numpy()
iu = np.triu_indices_from(corr, k=1)
say("Feature pairs with |correlation| > 0.95:", int((corr[iu] > 0.95).sum()), "of", len(iu[0]))

# %% [markdown]
# ## 2. The duplicate row

# %%
say("\n=== 2. DUPLICATE ROW ===")
dup_mask = ist_raw.duplicated(keep=False)
say("Rows that are exact copies of another row:", ist_raw.index[dup_mask].tolist(),
    "| person id:", ist_raw.loc[dup_mask, "id"].unique().tolist())
say("Duplicates if we ignore the id column (copies ACROSS people?):",
    int(ist_raw.drop(columns="id").duplicated().sum()))
# WHY show the person's 3 rows: tells us whether the copy replaced a missing recording
# (two copies + one different row) or something else.
d_id = ist_raw.loc[dup_mask, "id"].iloc[0]
say(ist_raw.loc[ist_raw["id"] == d_id, ["id", "gender", "class", "PPE", "DFA", "RPDE", "numPulses"]].to_string())

# DECISION: drop the second copy. WHY: in a split by RECORDING, one copy could land in train and
# the other in test, so the model would be tested on a row it has literally seen. That is the
# purest possible leak. This person keeps 2 recordings instead of 3.
ist = ist_raw.drop_duplicates().reset_index(drop=True).copy()   # .copy() avoids a pandas speed warning
say("After dropping the copy:", ist.shape[0], "rows,", ist["id"].nunique(), "people")

# %% [markdown]
# ## 3. Feature-matching table, Oxford vs Istanbul
# Oxford has 22 features, Istanbul 752. Only some measure the same quantity.

# %%
# WHY pitch needs a conversion: Oxford gives fundamental frequency in Hz (MDVP:Fo).
# Istanbul gives the mean period between glottal pulses in seconds. Frequency = 1 / period.
ist["F0_Hz"] = 1 / ist["meanPeriodPulses"]

MATCH = [  # (family, Oxford column, Istanbul column, note)
    ("Pitch",   "MDVP:Fo(Hz)",      "F0_Hz",                      "Istanbul: 1 / meanPeriodPulses"),
    ("Jitter",  "MDVP:Jitter(%)",   "locPctJitter",               "local jitter; both stored as a fraction"),
    ("Jitter",  "MDVP:Jitter(Abs)", "locAbsJitter",               "seconds"),
    ("Jitter",  "MDVP:RAP",         "rapJitter",                  "3-point average"),
    ("Jitter",  "MDVP:PPQ",         "ppq5Jitter",                 "5-point average"),
    ("Jitter",  "Jitter:DDP",       "ddpJitter",                  "= 3 x RAP by definition"),
    ("Shimmer", "MDVP:Shimmer",     "locShimmer",                 "local shimmer"),
    ("Shimmer", "MDVP:Shimmer(dB)", "locDbShimmer",               "decibels"),
    ("Shimmer", "Shimmer:APQ3",     "apq3Shimmer",                "3-point"),
    ("Shimmer", "Shimmer:APQ5",     "apq5Shimmer",                "5-point"),
    ("Shimmer", "MDVP:APQ",         "apq11Shimmer",               "MDVP APQ is the 11-point version"),
    ("Shimmer", "Shimmer:DDA",      "ddaShimmer",                 "= 3 x APQ3 by definition"),
    ("Noise",   "NHR",              "meanNoiseToHarmHarmonicity", "noise-to-harmonics ratio"),
    ("Noise",   "HNR",              "meanHarmToNoiseHarmonicity", "harmonics-to-noise, dB"),
    ("Nonlinear", "RPDE",           "RPDE",                       "Little et al. measure"),
    ("Nonlinear", "DFA",            "DFA",                        "Little et al. measure"),
    ("Nonlinear", "PPE",            "PPE",                        "Little et al. measure"),
]

def person_level(df, person_col, label_col, feat):
    """Average each person's recordings first. WHY: 3 (or 6-7) recordings of one voice are not
    independent evidence, so effect sizes are computed on one row per person."""
    return df.groupby(person_col).agg(x=(feat, "mean"), y=(label_col, "first"))

def cohens_d(p):
    """Standardised PD-minus-healthy difference: (mean_PD - mean_healthy) / pooled SD.
    WHY: puts every feature on a common scale, so 'bigger in PD' can be compared across datasets
    even when the raw units differ."""
    a, b = p.loc[p.y == 1, "x"], p.loc[p.y == 0, "x"]
    sd = np.sqrt(((len(a) - 1) * a.var() + (len(b) - 1) * b.var()) / (len(a) + len(b) - 2))
    return (a.mean() - b.mean()) / sd

rows = []
for fam, o_col, i_col, note in MATCH:
    po = person_level(oxf, "person", "status", o_col)
    pi = person_level(ist, "id", "class", i_col)
    d_o, d_i = cohens_d(po), cohens_d(pi)
    rows.append({
        "family": fam, "oxford": o_col, "istanbul": i_col, "note": note,
        "median_oxford": oxf[o_col].median(), "median_istanbul": ist[i_col].median(),
        "ratio_ist_over_oxf": ist[i_col].median() / oxf[o_col].median(),
        "d_PD_vs_healthy_oxford": round(d_o, 2), "d_PD_vs_healthy_istanbul": round(d_i, 2),
        "same_direction": bool(np.sign(d_o) == np.sign(d_i)),
    })
match = pd.DataFrame(rows)
match.to_csv(REPORTS / "03_feature_matching.csv", index=False)

say("\n=== 3. FEATURE MATCHING (person-level effect sizes) ===")
with pd.option_context("display.width", 200, "display.float_format", "{:.4g}".format):
    say(match.drop(columns="note").to_string(index=False))
say("Features where PD vs healthy points the SAME way in both datasets:",
    int(match["same_direction"].sum()), "of", len(match))

# %% [markdown]
# ## 4. Scale check: are the numbers comparable?
# Oxford: Kay MDVP software (2007). Istanbul: Praat (2019). Same names, possibly different maths.

# %%
say("\n=== 4. SCALE CHECK ===")
# WHY these identities: if they hold exactly, the software computes the derived measure the
# textbook way. Oxford: exact. Istanbul: check.
for name, top, bottom in [("ddpJitter / rapJitter", "ddpJitter", "rapJitter"),
                          ("ddaShimmer / apq3Shimmer", "ddaShimmer", "apq3Shimmer")]:
    r = ist[top] / ist[bottom]
    say(f"Istanbul {name}: min {r.min():.3f}, median {r.median():.3f}, max {r.max():.3f},",
        f"share within 1% of 3: {((r - 3).abs() < 0.03).mean():.1%}")
# WHY: rapJitter values are tiny (about 0.001) and stored to few decimals, so rounding alone
# can move the ratio away from 3. Check how many decimals the raw values carry.
say("Smallest non-zero rapJitter:", ist.loc[ist.rapJitter > 0, "rapJitter"].min())

# WHY ratio of medians: 1.0 means same scale. Far from 1 means a model trained on one dataset
# would see the other's values as extreme, so raw features cannot be transferred directly.
say("Median ratio Istanbul / Oxford (1 = same scale):")
for _, r in match.iterrows():
    flag = "  <-- different scale" if not 0.67 < r.ratio_ist_over_oxf < 1.5 else ""
    say(f"  {r.istanbul:28s} {r.ratio_ist_over_oxf:7.2f}{flag}")

# Plot: one small panel per shared feature, Oxford vs Istanbul, each panel its own axis.
# WHY separate axes: the point is the shape and location of each distribution, and the raw
# units differ by feature, so a shared axis would squash most panels flat.
BLUE, ORANGE = "#2a78d6", "#eb6834"   # Oxford, Istanbul (fixed throughout)
fig, axes = plt.subplots(3, 6, figsize=(16, 8))
for ax, (_, r) in zip(axes.flat, match.iterrows()):
    bp = ax.boxplot([oxf[r.oxford], ist[r.istanbul]], widths=0.5, patch_artist=True,
                    showfliers=False, medianprops={"color": "black", "linewidth": 1.5})
    for patch, c in zip(bp["boxes"], [BLUE, ORANGE]):
        patch.set_facecolor(c); patch.set_alpha(0.8); patch.set_edgecolor("white")
    ax.set_xticks([1, 2], ["Oxford", "Istanbul"], fontsize=8)
    ax.set_title(r.istanbul, fontsize=9)
    ax.tick_params(axis="y", labelsize=7)
    ax.grid(axis="y", alpha=0.3)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
axes.flat[-1].axis("off")
fig.suptitle("Shared voice features: Oxford (MDVP, 195 recordings) vs Istanbul (Praat, 755 recordings). "
             "Boxes = middle 50%, outliers hidden", fontsize=11)
fig.tight_layout()
fig.savefig(REPORTS / "03_scale_check.png", dpi=150)
plt.show()

# %% [markdown]
# ## 5. Sex confounder check
# A confounder here = something linked to BOTH the label and the voice. Sex is the obvious one:
# men have lower pitch, and if men are also more often PD in this sample, a model could learn
# "low voice = PD" and look clever while really detecting sex.

# %%
say("\n=== 5. SEX CONFOUNDER ===")
people = ist.groupby("id")[["class", "gender"]].first()   # one row per person (gender 1 = male)
tab = pd.crosstab(people["gender"].map({0: "women", 1: "men"}), people["class"].map({0: "healthy", 1: "PD"}))
say(tab.to_string())
say("Share PD: men", f"{people.loc[people.gender == 1, 'class'].mean():.0%},",
    "women", f"{people.loc[people.gender == 0, 'class'].mean():.0%}")
chi2, p, _, _ = chi2_contingency(tab)
say(f"Chi-square test, sex vs PD (person level): chi2 = {chi2:.2f}, p = {p:.4f}")

# WHY this AUC: it is the score a 'model' gets by knowing only the person's sex.
# Any real model must clearly beat this, otherwise it may just be reading sex from the voice.
say(f"AUC from sex alone (person level): {roc_auc_score(people['class'], people['gender']):.2f}")

# For each shared feature: how strongly does it separate men from women, vs PD from healthy?
# And does the PD effect survive WITHIN each sex? If it does, sex is not the whole story.
rows = []
for fam, o_col, i_col, note in MATCH:
    p_all = ist.groupby("id").agg(x=(i_col, "mean"), y=("class", "first"), g=("gender", "first"))
    sex_d = cohens_d(p_all.assign(y=p_all.g))                       # men vs women
    rows.append({"feature": i_col,
                 "d_men_vs_women": round(sex_d, 2),
                 "d_PD_all": round(cohens_d(p_all), 2),
                 "d_PD_men_only": round(cohens_d(p_all[p_all.g == 1]), 2),
                 "d_PD_women_only": round(cohens_d(p_all[p_all.g == 0]), 2)})
sex = pd.DataFrame(rows)
sex.to_csv(REPORTS / "03_sex_confounder.csv", index=False)
say(sex.to_string(index=False))

# Oxford has NO sex column, so this check cannot be repeated there. Pitch is the best clue.
say("Istanbul median pitch: men", f"{ist.loc[ist.gender == 1, 'F0_Hz'].median():.0f} Hz,",
    "women", f"{ist.loc[ist.gender == 0, 'F0_Hz'].median():.0f} Hz |",
    "Oxford median pitch:", f"{oxf['MDVP:Fo(Hz)'].median():.0f} Hz")

# %%
(REPORTS / "03_profile_istanbul.txt").write_text("\n".join(LOG) + "\n")
say("\nSaved: reports/03_profile_istanbul.txt, 03_feature_matching.csv, 03_sex_confounder.csv, 03_scale_check.png")
