# %% [markdown]
# # 08: match 20 fictional patients to recruiting Parkinson's trials
#
# Stage 2 of Part B. Input: `data/raw/ctgov_parkinsons_recruiting.csv` (from
# `02_fetch_trials.py`) and `data/patients_fictional.json` (20 invented patients,
# written BEFORE any trial data was fetched, so they could not be tailored to
# the trials).
#
# WHY this design: a trial matcher has two different jobs that must not be mixed.
# 1. HARD RULES: age limits and sex are stated eligibility facts. A text model
#    should never be allowed to overrule them, so they are applied as filters first.
# 2. RELEVANCE: among trials the patient could legally enter, which five are most
#    worth a clinician's time? That is a ranking problem on free text.
# The ranking here uses TF-IDF vectors with cosine similarity: every trial and
# every patient becomes a (sparse) vector of word weights, where a word scores
# high if it is frequent in this document but rare across all trials. "Dyskinesia"
# separates trials; "patient" does not. A plain keyword-overlap baseline is run
# alongside, because if TF-IDF cannot beat counting shared words, the extra
# machinery is not earning its keep. A neural sentence encoder is the natural
# upgrade and is discussed in the README; the pipeline is written so only the
# vectorizer would change.
#
# This is a retrieval exercise. It does NOT determine eligibility (most criteria
# live in free text the filters do not parse: medication rules, comorbidities,
# prior surgery), and it is not medical advice. See reports/ETHICS_NOTE.md.

# %% 0. setup
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parents[1] if "__file__" in globals() else Path.cwd()
RNG = np.random.default_rng(0)
TOP_K = 5

trials = pd.read_csv(ROOT / "data" / "raw" / "ctgov_parkinsons_recruiting.csv")
with open(ROOT / "data" / "patients_fictional.json") as f:
    patients = pd.DataFrame(json.load(f)["patients"])

print(f"trials fetched: {len(trials)}  patients: {len(patients)}")

# %% 1. clean the trials table
# WHY: retrieval quality is bounded by the text you index. A trial with no
# eligibility text cannot be matched on eligibility, so it is dropped and the
# drop is counted (silent drops are how datasets lie to you - lesson of 03).

for col in ["title", "summary", "eligibility"]:
    if col in trials.columns:
        trials[col] = trials[col].fillna("")
trials["doc"] = (trials["title"] + ". " + trials.get("summary", "") + ". " + trials["eligibility"]).str.strip()

n0 = len(trials)
trials = trials[trials["eligibility"].str.len() > 100].reset_index(drop=True)
print(f"dropped {n0 - len(trials)} trials with missing/trivial eligibility text -> {len(trials)} remain")


def age_to_years(s):
    """'65 Years' -> 65.0, '18 Months' -> 1.5, missing -> None."""
    if not isinstance(s, str) or not s.strip():
        return None
    m = re.match(r"([\d.]+)\s*(Year|Month|Week|Day)", s, flags=re.I)
    if not m:
        return None
    v, unit = float(m.group(1)), m.group(2).lower()
    return v if unit == "year" else v / {"month": 12, "week": 52, "day": 365}[unit]


trials["min_age_y"] = trials["min_age"].map(age_to_years)
trials["max_age_y"] = trials["max_age"].map(age_to_years)

# %% 2. hard filters: age and sex
# WHY FIRST: a 43-year-old woman must never be shown a trial for men over 60,
# however similar the text looks. Stated limits are facts, not similarities.
# Missing limits are treated as "no limit", which matches how ClinicalTrials.gov
# displays them.


def eligible_mask(patient):
    ok_min = trials["min_age_y"].isna() | (patient["age"] >= trials["min_age_y"])
    ok_max = trials["max_age_y"].isna() | (patient["age"] <= trials["max_age_y"])
    sex = trials["sex"].fillna("ALL").str.upper()
    ok_sex = (sex == "ALL") | (sex == patient["sex"].upper())
    return ok_min & ok_max & ok_sex


# %% 3. TF-IDF retrieval vs keyword baseline
# WHY sublinear_tf and bigrams: eligibility texts repeat boilerplate ("informed
# consent", "male or female"); log-scaled counts stop repetition dominating, and
# bigrams let "deep brain" or "impulse control" count as units.
vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True, min_df=2)
X_trials = vec.fit_transform(trials["doc"])
X_patients = vec.transform(patients["description"])
sim = cosine_similarity(X_patients, X_trials)  # patients x trials

STOP = set(ENGLISH_STOP_WORDS) | {"parkinson", "parkinsons", "disease", "patient", "patients", "study", "trial", "year", "years", "old"}


def tokens(text):
    return {w for w in re.findall(r"[a-z]{3,}", text.lower()) if w not in STOP}


def keyword_overlap(p_text, t_text):
    return len(tokens(p_text) & tokens(t_text))


rows = []
overlaps = []
for i, p in patients.iterrows():
    mask = eligible_mask(p).to_numpy()
    n_elig = int(mask.sum())

    s = np.where(mask, sim[i], -1.0)
    top_tfidf = np.argsort(-s)[:TOP_K]

    kw = np.array([keyword_overlap(p["description"], d) if m else -1 for d, m in zip(trials["doc"], mask)])
    top_kw = np.argsort(-kw)[:TOP_K]

    overlaps.append(len(set(top_tfidf) & set(top_kw)))
    for rank, j in enumerate(top_tfidf, 1):
        rows.append({
            "patient": p["id"], "age": p["age"], "sex": p["sex"], "eligible_trials": n_elig,
            "rank": rank, "nct_id": trials.loc[j, "nct_id"], "title": trials.loc[j, "title"],
            "phase": trials.loc[j, "phase"], "min_age": trials.loc[j, "min_age"],
            "max_age": trials.loc[j, "max_age"], "tfidf_score": round(float(s[j]), 4),
            "in_keyword_top5": j in set(top_kw),
        })

matches = pd.DataFrame(rows)
matches.to_csv(ROOT / "reports" / "08_trial_matches.csv", index=False)

print(f"\nmean eligible trials per patient after hard filters: {matches.groupby('patient')['eligible_trials'].first().mean():.0f}")
print(f"TF-IDF top-5 vs keyword top-5, shared trials per patient: mean {np.mean(overlaps):.1f} of 5")
print(matches.groupby("patient")["tfidf_score"].max().describe().round(3))

# %% 4. write the review file
# WHY a review file: the task was never "trust the ranking". Every match gets a
# human check against the full eligibility text. The explanations are drafted by
# an LLM (Claude) in reports/08_matches_for_review.md and each one is marked
# until Kerem has verified it against the eligibility text himself.
lines = ["# 08 trial matches: top 5 per fictional patient\n",
         "Ranking: TF-IDF cosine similarity after hard age/sex filters. ",
         "`in_keyword_top5` marks trials the plain keyword baseline also found.\n"]
for pid, g in matches.groupby("patient", sort=True):
    p = patients.set_index("id").loc[pid]
    lines.append(f"\n## {pid} ({p['age']}, {p['sex']}, dx {p['years_since_diagnosis']}y): {p['description'][:120]}...\n")
    for _, r in g.iterrows():
        lines.append(f"{r['rank']}. **{r['nct_id']}** (score {r['tfidf_score']}, phase {r['phase'] or 'n/a'}, "
                     f"ages {r['min_age'] or '-'} to {r['max_age'] or '-'}, kw={'y' if r['in_keyword_top5'] else 'n'}): {r['title']}")
with open(ROOT / "reports" / "08_trial_matches.md", "w") as f:
    f.write("\n".join(lines) + "\n")
print("\nwrote reports/08_trial_matches.csv and reports/08_trial_matches.md")
