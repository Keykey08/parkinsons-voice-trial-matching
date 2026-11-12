# Task sheet 06: does a person's voice get worse when their own symptoms get worse?

You write the code. Each step has: what to do, why, a hint, and a checkpoint to compare your answer against. Do one step, run it, check it, then move on. If you're stuck for more than 15 minutes on one step, ask me and paste your code and the error.

**File:** `02_neuro_parkinsons_trials/src/06_telemonitoring.py` (starter already created: imports and cell headers only)
**Data:** `data/raw/telemonitoring/parkinsons_updrs.data`
**Read first:** `data/raw/telemonitoring/parkinsons_updrs.names` (5 minutes, it explains every column)

---

## The question

So far we compared *different people*: people with Parkinson's vs healthy people. Anything else that differs between those groups (age, sex, recording room) could fake the result. This dataset follows **42 people with Parkinson's for 6 months**, with about 140 home recordings each, plus a clinician's symptom score (UPDRS: higher = worse). If a person's voice worsens when *their own* score worsens, age, sex and microphone can't explain it, because those don't change inside one person.

---

## Step 1: load and meet the data (module 2_05)

**Do:**
1. Load the file with `pd.read_csv`. Rename `subject#` to `subject` (the `#` makes it awkward to type).
2. Print the shape, the number of unique subjects, and `.head()`.
3. Make a person-level table with `groupby("subject").agg(...)`: number of recordings, age, sex, first and last `test_time`, lowest and highest `total_UPDRS`.
4. Print `.describe()` of that table.

**Why:** you need to know what one row is (one recording) and what one person looks like before you model anything. Same habit as script 03.

**Hint:** `agg(n=("age", "size"), age=("age", "first"), ...)` lets you name each output column.

**Checkpoint:** 5,875 rows, 42 people, about 140 recordings each (101 to 168), ages 36 to 85, follow-up of about 6 months. Sex: 28 coded 0 and 14 coded 1.

**Careful:** here `sex` 0 = male, 1 = female. That's the **opposite** of Istanbul's `gender`. Write that in a comment. Mixing them up would silently flip every sex result.

---

## Step 2: how was the symptom score made? (the trap)

**Do:**
1. The `.names` file says UPDRS is "linearly interpolated". Look up what that means in one sentence and write it as a comment.
2. For each person, count how many *different* `total_UPDRS` values they have (`nunique`).
3. For each person, fit a straight line of `total_UPDRS` against `test_time` (`np.polyfit(x, y, 1)`) and compute how much of the variation the line explains (R squared). Report the median across people.

**Why:** a doctor only examined each patient at baseline, 3 months and 6 months. The ~140 scores per person in between are **filled in with straight lines**, not measured. So within one person the "symptom score" is basically a function of time. Any voice feature that drifts over time (e.g. people getting used to the device) will look correlated with symptoms even if it has nothing to do with Parkinson's. You have to know this before you interpret anything.

**Hint:** R squared = 1 minus (variance of the residuals / variance of y). Residuals = y minus the fitted line (`np.polyval`).

**Checkpoint:** median R squared around 0.8. Write one sentence on what that means for the analysis.

---

## Step 3: the leakage trap, version 3 (module 6_03)

**Do:** predict `total_UPDRS` with a random forest (`RandomForestRegressor(200, min_samples_leaf=3, random_state=0)`) using `cross_val_predict`, three feature sets x two splits:

| Features | Split |
|---|---|
| 16 voice measures only | `KFold(5, shuffle=True)` (random) and `GroupKFold(5)` by subject |
| age + sex only | both |
| voice + age + sex | both |

Score each with R squared and MAE (mean absolute error, in UPDRS points). Also compute the MAE of just predicting the average for everyone.

**Why:** the third time you test the same trap, in a new form. This time it's a regression (predicting a number), not a yes/no.

**Before you run it, write down your prediction:** what R squared will "age + sex only" get with a random split? Think about how many people share the same age and sex.

**Checkpoint:** I won't give the numbers. Compare your random vs person results and explain the gap in two sentences. If your age + sex random-split result looks impressive, ask yourself what the model has actually learned.

---

## Step 4: the real test, within each person (econometrics you know)

**Do:**
1. Standardise each of the 16 voice measures across the whole dataset (subtract mean, divide by standard deviation).
2. Build one **voice roughness index**: the average of the standardised jitter (5), shimmer (6) and NHR measures, plus RPDE, minus HNR (because higher HNR = cleaner). Leave DFA and PPE out. One number per recording, higher = rougher voice.
3. For **each person**, fit a straight line against `test_time / 30` (time in months) for (a) `total_UPDRS` and (b) the roughness index. Keep the two slopes. You get a table with 42 rows: one UPDRS slope and one voice slope per person.
4. Spearman correlation (`scipy.stats.spearmanr`) between the two slope columns, across the 42 people.

**Why this design:** each person is compared only with themselves (their slope), so age, sex, microphone and home are cancelled out. Then we ask: do the people whose symptoms **rose faster** also have voices that **roughened faster**? Anything that drifts over time for everyone (like getting used to the device) affects both groups, so it doesn't create a correlation between the slopes. It's the same idea as difference-in-differences: compare changes, not levels.

**Why one index instead of 16 tests:** 16 separate tests give you 16 chances to find something by luck. Decide on ONE main outcome before looking (the index), and treat the 16 individual features as secondary.

**Before you run it, write down:** what correlation do you expect if the medical theory is right? Positive or negative?

**Checkpoint:** I won't give the number. Whatever you get, write: the correlation, the p-value, and one sentence saying what it means. A null result here is a real finding. Think about why it might happen (hint: step 2).

---

## Step 5: write it up (your words, not mine)

5 to 8 sentences in `reports/06_findings.md`:
- What the data is and the interpolation problem
- The leakage result from step 3
- The within-person result from step 4
- One limitation you'd tell a WISE interviewer
- What you'd need to test this properly (hint: what data would fix step 2?)

Then add your row to `AI_USE_LOG.md` (what you asked me, what you wrote yourself).

---

## Concepts you'll need (look up if unsure)
- `groupby` + `agg` (2_01, 2_03)
- `np.polyfit` for a straight-line slope
- R squared and MAE
- `KFold` vs `GroupKFold` (6_03)
- Spearman correlation: like normal correlation but on ranks, so one extreme person can't dominate
