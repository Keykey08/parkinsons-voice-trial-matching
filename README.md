# Parkinson's from the voice: how much of the reported accuracy is real?

**Status, 13 Nov 2026:** Part A (voice evidence, scripts 01 to 07) is complete. Part B (matching fictional patients to clinical trials) is built end to end: 20 fictional patient profiles, hard-filter + TF-IDF matcher (`08`), ethics note. The one step not yet run is the trial download itself (`02`), pending network access to ClinicalTrials.gov from the analysis environment; the script is ready and runs anywhere with ordinary internet.

> Research project only. Not a diagnostic tool. Nothing here detects or diagnoses Parkinson's. Only public research datasets and fictional patients are used.

## The question

Papers report that a handful of voice measurements can separate people with Parkinson's (PD) from healthy people with very high accuracy. This project asks how much of that survives four honest tests:

1. Split the data by person, not by recording.
2. Repeat it in a second hospital.
3. Train in one hospital and test in the other.
4. Follow the same people over time and ask whether their voice moves with their own symptoms.

## Data

| Dataset | Who | Size | Used in |
|---|---|---|---|
| Oxford (UCI 174, Little et al. 2007) | 24 PD, 8 healthy | 195 recordings, 6 to 7 per person | 01, 01b, 05, 07 |
| Istanbul (UCI 470, Sakar et al. 2019) | 188 PD, 64 healthy | 755 recordings after removing one duplicate, 3 per person | 03, 04, 05, 07 |
| Telemonitoring (UCI 189, Tsanas et al. 2010) | 42 PD, followed about 6 months | 5,875 home recordings | 06 |

All three are CC BY 4.0. Full citations are in `DATA_SOURCES.md`.

## Findings

### 1. The split decides the score (Oxford, `01b`)

Each person gave several recordings. If some go to training and some to testing, the model has already met that person.

| Model | AUC, split by recording (leaky) | AUC, split by person (honest) |
|---|---|---|
| Always predict PD | 0.50 | 0.50 |
| Logistic regression | 0.91 | 0.77 |
| Random forest | 0.96 | 0.78 |

Honest balanced accuracy is about 0.64.

### 2. The same thing happens in a second hospital, and gets worse with more features (Istanbul, `04`)

Person-level AUC, mean of 10 repeats of 5-fold.

| Features | Model | Leaky | Honest | Gap |
|---|---|---|---|---|
| 16 shared with Oxford | Logistic regression | 0.792 | 0.772 | 0.020 |
| 16 shared with Oxford | Random forest | 0.860 | 0.784 | 0.077 |
| All 752 | Logistic regression | 0.933 | 0.832 | 0.101 |
| All 752 | Random forest | 0.953 | 0.850 | 0.102 |

The honest AUC on the 16 shared features (0.77 to 0.78) matches Oxford.

### 3. Sex is a confounder, but it does not explain the result (Istanbul, `03` and `04`)

82% of the men in the Istanbul data have PD against 66% of the women (chi-square 7.59, p = 0.006). Sex alone gives AUC 0.57 to 0.60. Adding sex to the voice features changes nothing, and the voice model still works inside each sex (AUC 0.74 for men, 0.76 for women, logistic regression).

`03` also found that one feature (PPE) points in opposite directions in the two datasets, and that jitter and shimmer sit on different scales because the two studies used different analysis software. PPE is excluded from all cross-hospital work.

### 4. The ranking travels between hospitals, the cut-off does not (`05`)

16 shared features, training-hospital scaling only, 95% bootstrap interval over test people.

| Train | Test | Model | AUC | 95% interval |
|---|---|---|---|---|
| Istanbul | Oxford | Logistic regression | 0.828 | 0.645 to 0.978 |
| Istanbul | Oxford | Random forest | 0.807 | 0.623 to 0.950 |
| Oxford | Istanbul | Logistic regression | 0.710 | 0.630 to 0.782 |
| Oxford | Istanbul | Random forest | 0.777 | 0.713 to 0.835 |

The Oxford intervals are wide because Oxford has only 8 healthy people. At the default 0.5 cut-off the Oxford-trained random forest flags 94% of Istanbul as PD when the truth is 75%, and balanced accuracy falls to 0.59.

### 5. Moving the cut-off repairs only part of that (`07`)

Balanced accuracy per person on the new hospital. "Local calibration" labels k random people in the new hospital, sets the cut-off on them and scores everyone else (mean of 500 draws).

| Train -> test | Model | Fixed 0.5 | Cut-off tuned at home | Local, smallest k | Local, largest k | Oracle ceiling |
|---|---|---|---|---|---|---|
| Oxford -> Istanbul | Logistic regression | 0.645 | 0.653 | 0.622 (k=10) | 0.631 (k=80) | 0.671 |
| Oxford -> Istanbul | Random forest | 0.586 | 0.631 | 0.670 (k=10) | 0.713 (k=80) | 0.741 |
| Istanbul -> Oxford | Logistic regression | 0.646 | 0.688 | 0.678 (k=8) | 0.689 (k=16) | 0.792 |
| Istanbul -> Oxford | Random forest | 0.708 | 0.688 | 0.664 (k=8) | 0.646 (k=16) | 0.771 |

Recalibration clearly helps in one case of four (the Oxford-trained random forest, 0.59 to about 0.70). Elsewhere it makes little difference or hurts, and with few labelled people the result swings widely between draws (for Oxford the middle 95% of draws runs from 0.50 to above 0.80). Even a perfect cut-off stays between 0.67 and 0.79, so the cut-off is not the main limit. The ranking is.

### 6. Within the same person, no detectable link between voice and symptoms (Telemonitoring, `06`)

The symptom score (UPDRS) in this dataset was examined at about three clinic visits and filled in along straight lines in between. A straight line against time explains a median of 83% of each person's score.

- Predicting UPDRS with a random split looks strong (R squared 0.84 from age and sex alone, because age plus sex nearly identifies the person). Split by person, every feature set does worse than predicting the average for everyone (voice only: MAE 9.49 against 8.66).
- Main test: across the 42 people, the slope of UPDRS and the slope of a voice roughness index have Spearman correlation -0.05 (p = 0.75).

This is a null result from a low-powered design. It does not show that voice and symptoms are unrelated. It shows that this dataset, with interpolated scores, cannot detect it.

## Part B: the trial matcher (`02`, `08`)

A retrieval exercise built on the lesson of Part A: keep hard facts and soft similarity separate.

- `data/patients_fictional.json`: 20 fictional UK patient profiles covering newly diagnosed to advanced disease, DBS, cognitive impairment, young onset and common comorbidities. Written **before** any trial data was fetched, so they could not be tailored to the trials (same pre-commitment discipline as the `06` predictions).
- `02_fetch_trials.py`: downloads recruiting Parkinson's trials from the public ClinicalTrials.gov registry (API v2) with title, summary, eligibility text, age limits and sex.
- `08_match_trials.py`: applies the stated hard criteria first (age range, sex) as filters a text model can never overrule, then ranks the remaining trials by TF-IDF cosine similarity, with a plain keyword-overlap baseline alongside. TF-IDF rather than a neural encoder is deliberate: every score decomposes into which words matched, so a ranking can be audited by hand. The vectorizer is the one swappable line if a neural upgrade is wanted.
- Outputs: `reports/08_trial_matches.csv` and `.md`, with every match carrying its NCT number so claims trace to the registry. Per-match explanations are drafted by an LLM and marked unverified until checked by a person against the eligibility text.
- `reports/ETHICS_NOTE.md`: why the patients are fictional, what the matcher is not (an eligibility decision), and where this kind of tool causes harm if misused.

## Limits

- Per-person age is not in the Oxford or Istanbul data files. The Sakar et al. 2019 paper reports cohort ages (PD 33 to 87, mean 65.1, SD 10.9; controls 41 to 82, mean 61.1, SD 8.9), so the groups roughly overlap but PD runs about 4 years older, and age cannot be adjusted for at person level.
- Recording protocol in Istanbul per the paper: sustained /a/, three repetitions, 44.1 kHz, collected at the Department of Neurology, Cerrahpasa Faculty of Medicine, Istanbul University. Medication state at recording time is unreported in both classification datasets.
- Oxford has 8 healthy people. Every Oxford test number carries a wide interval.
- Both classification datasets are 75% PD, far from any screening population.
- The telemonitoring symptom scores are interpolated, not measured at each recording.
- The roughness index in `06` and the calibration design in `07` are choices made for this project, not standard instruments.

## Verification status and AI use

The code in this repository was written with AI assistance (Claude, Anthropic). `AI_USE_LOG.md` records each session: what was asked, what the tool produced and what I checked.

`reports/RESULTS_LEDGER.md` lists every number above with its source file and whether I have reproduced it in my own run. As of 13 Nov 2026 I have run 05 and 06 myself, and part of 03. Scripts 01b, 04 and 07 and the rest of 03 have so far been run by Claude only. The draft of `reports/06_findings.md` was written by Claude and is marked as such until I rewrite it.

## Reproduce

```
pip install -r requirements.txt
python src/01b_baselines_parkinsons.py
python src/03_profile_istanbul.py
python src/04_leakage_istanbul.py      # about 5 minutes
python src/05_external_validation.py
python src/06_telemonitoring.py
python src/07_threshold_recalibration.py
python src/02_fetch_trials.py          # needs internet access to clinicaltrials.gov
python src/08_match_trials.py
```

Each script is split into `# %%` cells with a WHY comment per step, and writes its outputs to `reports/`. Seeds are fixed. Small third-decimal differences between scikit-learn versions are expected. The `07` numbers above come from scikit-learn 1.9.1, where the Istanbul to Oxford random forest AUC is 0.812, not the 0.807 saved by `05` under an earlier version.
