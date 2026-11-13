# Results ledger: Parkinson's project, scripts 01b to 07

Structure built by Claude Code on 12 Nov 2026. The three right-hand columns are Kerem's to fill.

## How to use it

One row = one number you might be asked about. For each row:

1. Run the script and find the number in your own terminal output.
2. Write it in **My value**. If it differs from the reference, do not "fix" it: write what you got and ask why.
3. Write **What it means** in one sentence, in your own words, without looking at the walkthrough.
4. Put the date in **Verified** only when 1 to 3 are done.

A result moves to the verified table in `KEREM_PROJECTS_AI_CONTEXT.md` only when its row here is complete.

**Reference** = the value in the saved report files (produced when Claude ran the scripts). It is there to compare against, not to copy.

**Run status** = what your pasted terminal output showed on 12 Nov 2026:
- `ran` = your run printed this number and it matched the reference.
- `not reached` = your run stopped before this point.
- `not run` = this number has not appeared in any run of yours yet.
- `Claude ran` = Claude ran it for you; you still need to run it once yourself.

---

## Day 1 gate check, 12 Nov 2026 (run by Claude while Kerem was in a seminar)

Claude re-ran 01b, 03, 04 and 05 on your scripts and data, unchanged. Outputs and console logs are in `reports/claude_run_2026-11-12/`. Your original report files were not touched (except 01b's CSV, which came out identical).

- 01b and 03: identical to the saved reports.
- 04: same to 2 decimals. A few 3rd-decimal values differ by 0.001.
- 05: identical except 05.4 (RF Istanbul to Oxford): 0.812 vs 0.807 saved.
- Seeds are fixed, so the small differences are most likely a different scikit-learn version (Claude: 1.9.1). No conclusion changes.

Rows marked `Claude ran` still need one run by you before you fill **My value**. For 03, close the plot window instead of pressing Ctrl-C. For 04, wait for all 7 `done:` lines (about 5 minutes).

---

## 01b_baselines_parkinsons.py: Oxford leakage experiment

Source of reference values: `reports/01b_leakage_experiment.csv` and `.txt`. 195 recordings, 5-fold.

Run status `Claude ran` = Claude ran it on 12 Nov 2026 at Kerem's request while he was in a seminar. It reproduced the saved CSV exactly. It still needs one run by Kerem before it counts as verified.

| # | Result | Reference | Run status | My value | What it means (my words) | Verified |
|---|---|---|---|---|---|---|
| 01b.1 | Always predict PD, AUC | 0.50 | Claude ran | | | |
| 01b.2 | Logistic regression AUC: by recording (leaky) vs by person (honest) | 0.91 vs 0.77 | Claude ran | | | |
| 01b.3 | Random forest AUC: by recording vs by person | 0.96 vs 0.78 | Claude ran | | | |
| 01b.4 | Balanced accuracy, LR: by recording vs by person | 0.78 vs 0.64 | Claude ran | | | |
| 01b.5 | Balanced accuracy, RF: by recording vs by person | 0.89 vs 0.64 | Claude ran | | | |

---

## 03_profile_istanbul.py: what is in the Istanbul data

Source of reference values: `reports/03_profile_istanbul.txt`

| # | Result | Reference | Run status | My value | What it means (my words) | Verified |
|---|---|---|---|---|---|---|
| 03.1 | Rows and columns in the raw file | 756 x 755 | ran | | | |
| 03.2 | People, and recordings per person | 252 people, 3 each | ran | | | |
| 03.3 | Share of people with PD | 75% (188 of 252) | ran | | | |
| 03.4 | Duplicate row | person 37, rows 111 and 113; 755 rows after dropping | ran | | | |
| 03.5 | Features pointing the same way in Oxford and Istanbul | 16 of 17 | ran | | | |
| 03.6 | PPE effect size, Oxford vs Istanbul | 1.68 vs -0.24 (opposite) | ran | | | |
| 03.7 | Jitter scale, Istanbul / Oxford | 0.11 to 0.32 | ran | | | |
| 03.8 | Shimmer scale, Istanbul / Oxford | 2.19 to 2.60 | ran | | | |
| 03.9 | ddaShimmer / apq3Shimmer within 1% of 3 | 100% | ran | | | |
| 03.10 | Share PD, men vs women | 82% vs 66% | Claude ran | | | |
| 03.11 | Chi-square, sex vs PD | chi2 7.59, p 0.0059 | Claude ran | | | |
| 03.12 | AUC from sex alone | 0.60 | Claude ran | | | |
| 03.13 | Pitch effect size: all, men only, women only | -0.72, -0.23, -0.82 | Claude ran | | | |

Why 03.10 to 03.13 are "not reached": the run was stopped with Ctrl-C at the plot window. Close the plot window instead and the script continues.

---

## 04_leakage_istanbul.py: leaky vs honest split

Source of reference values: `reports/04_leakage_istanbul.txt`. All AUCs are person level, mean over 10 repeats of 5-fold.

| # | Result | Reference | Run status | My value | What it means (my words) | Verified |
|---|---|---|---|---|---|---|
| 04.1 | Sex only, honest AUC | 0.568 | Claude ran | | | |
| 04.2 | Shared 16, logistic regression: leaky -> honest | 0.792 -> 0.772 (gap 0.020) | Claude ran | | | |
| 04.3 | Shared 16, random forest: leaky -> honest | 0.860 -> 0.784 (gap 0.077) | Claude ran | | | |
| 04.4 | Shared 16 + sex, honest AUC (LR / RF) | 0.772 / 0.781 | Claude ran | | | |
| 04.5 | All 752, logistic regression: leaky -> honest | 0.933 -> 0.832 (gap 0.101) | Claude ran | | | |
| 04.6 | All 752, random forest: leaky -> honest | 0.953 -> 0.850 (gap 0.102) | Claude ran | | | |
| 04.7 | Shared 16 honest AUC within men / within women (LR) | 0.743 / 0.764 | Claude ran | | | |
| 04.8 | Shared 16 honest balanced accuracy (LR / RF) | 0.689 / 0.712 | Claude ran | | | |

Why "not run": the run was stopped with Ctrl-C while the random forest was still training. It takes several minutes. Wait for all 7 `done:` lines.

---

## 05_external_validation.py: train in one hospital, test in the other

Source of reference values: `reports/05_external_validation.txt`. 16 shared features, "train scaling only" rows unless stated.

| # | Result | Reference | Run status | My value | What it means (my words) | Verified |
|---|---|---|---|---|---|---|
| 05.1 | Within-Oxford honest AUC (LR / RF) | 0.732 / 0.681 | ran | | | |
| 05.2 | Within-Istanbul honest AUC (LR / RF) | 0.772 / 0.784 | ran | | | |
| 05.3 | Istanbul -> Oxford AUC, LR, with 95% interval | 0.828 [0.645 to 0.978] | ran | | | |
| 05.4 | Istanbul -> Oxford AUC, RF, with 95% interval | 0.807 [0.623 to 0.950] (Claude's run: 0.812) | ran | | | |
| 05.5 | Oxford -> Istanbul AUC, LR, with 95% interval | 0.710 [0.630 to 0.782] | ran | | | |
| 05.6 | Oxford -> Istanbul AUC, RF, with 95% interval | 0.777 [0.713 to 0.835] | ran | | | |
| 05.7 | Share flagged PD, Oxford-trained RF on Istanbul (truth 75%) | 94% | ran | | | |
| 05.8 | Balanced accuracy at 0.5, Oxford -> Istanbul (LR / RF) | 0.645 / 0.586 | ran | | | |
| 05.9 | Per-hospital scaling: Oxford -> Istanbul AUC (LR / RF) | 0.719 / 0.765 | ran | | | |
| 05.10 | LR coefficients with the same sign in both hospitals | 10 of 16 | ran | | | |

---

## 06_telemonitoring.py: within-person test

No saved report file. Reference values are from your own terminal output of 12 Nov 2026, except 06.9.

| # | Result | Reference | Run status | My value | What it means (my words) | Verified |
|---|---|---|---|---|---|---|
| 06.1 | Rows, people | 5,875 rows, 42 people | ran | | | |
| 06.2 | Recordings per person | 101 to 168, mean 140 | ran | | | |
| 06.3 | Sex coding and counts | 28 coded 0 (male), 14 coded 1 (female) | ran | | | |
| 06.4 | Distinct total_UPDRS values per person | 14 to 39, median 27 | ran | | | |
| 06.5 | Median R squared, UPDRS against time | 0.83 | ran | | | |
| 06.6 | Voice only: random split vs person split (R squared) | 0.381 vs -0.225 | ran | | | |
| 06.7 | Age + sex only: random split vs person split (R squared) | 0.843 vs -0.803 | ran | | | |
| 06.8 | MAE: voice only by person vs predicting the average | 9.488 vs 8.656 | ran | | | |
| 06.9 | Spearman rho, UPDRS slope vs voice slope (main index) | -0.050, p 0.752, n 42 | ran | | | |
| 06.10 | Same, plain-average index (robustness) | -0.037, p 0.814 (Claude's run only) | not run | | | |

Still open for 06: the two `MY PREDICTION:` lines in the script, the three "DRAFT by Claude" comments, and the DRAFT banner on `06_findings.md`.

---

## 07_threshold_recalibration.py: can the cut-off be repaired in a new hospital?

Source of reference values: `reports/07_threshold_recalibration.txt`. Script written and run by Claude on 12 Nov 2026 (scikit-learn 1.9.1). Balanced accuracy per person on the new hospital.

| # | Result | Reference | Run status | My value | What it means (my words) | Verified |
|---|---|---|---|---|---|---|
| 07.1 | Oxford -> Istanbul, RF: fixed 0.5 vs local calibration k=10 / k=80 | 0.586 vs 0.670 / 0.713 | Claude ran | | | |
| 07.2 | Oxford -> Istanbul, RF: share flagged PD at 0.5 vs at k=80 (truth 75%) | 94% vs 57% | Claude ran | | | |
| 07.3 | Oxford -> Istanbul, LR: fixed 0.5 vs local calibration k=10 / k=80 | 0.645 vs 0.622 / 0.631 | Claude ran | | | |
| 07.4 | Oxford -> Istanbul: cut-off tuned at home (LR / RF) | 0.653 / 0.631 | Claude ran | | | |
| 07.5 | Istanbul -> Oxford: fixed 0.5 (LR / RF) | 0.646 / 0.708 | Claude ran | | | |
| 07.6 | Istanbul -> Oxford: local calibration k=16 (LR / RF), with middle 95% of draws | 0.689 [0.500 to 0.893] / 0.646 [0.436 to 0.813] | Claude ran | | | |
| 07.7 | Oracle ceiling, all four cases | 0.671, 0.741, 0.792, 0.771 | Claude ran | | | |

The script ends with four questions for you (section 5). Answer them before quoting any 07 number.

---

## Questions I could not answer yet

Write any row you cannot explain here, then ask.

- 

---

## 08_match_trials.py: trial matcher (run by Claude, 13 Nov 2026)

Corpus fetched 13 Nov 2026 with `02_fetch_trials.py`. Verify a row by re-running both scripts and reading three patients' top trials on ClinicalTrials.gov.

| # | Result | Reference | Run status | My value | What it means (my words) | Verified |
|---|---|---|---|---|---|---|
| 08.1 | Recruiting PD trials fetched / kept after dropping trivial eligibility text | 671 / 670 | Claude ran | | | |
| 08.2 | Trials with a UK site | 48 | Claude ran | | | |
| 08.3 | Eligible trials per patient after age/sex filters (min to max) | 527 to 658 | Claude ran | | | |
| 08.4 | TF-IDF top-5 overlap with keyword top-5, mean per patient | 0.9 of 5 | Claude ran | | | |
| 08.5 | Best TF-IDF score per patient (min / median / max) | 0.064 / 0.102 / 0.232 | Claude ran | | | |
| 08.6 | Clear retrieval failure cases to be able to explain | P18 (breast-cancer trials), P17 (top hit likely ineligible), P14 (pacemaker vs stimulation trials) | Claude ran | | | |

Outputs: `reports/08_trial_matches.csv`, `08_trial_matches.md`, `08_explanations_DRAFT.md` (banner stays until you verify three patients against the registry).
