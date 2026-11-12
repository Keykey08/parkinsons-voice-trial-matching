# Walkthrough: `04_leakage_istanbul.py` and `05_external_validation.py`

Run `03_profile_istanbul.py` first (it explains the dataset). Then run 04 (about 4 minutes) and 05 (under a minute), cell by cell, with this sheet open. Check every number against your own run.

## The story in four sentences

1. On 252 Istanbul patients, splitting by recording still inflates the score, and the inflation gets bigger the more features you use (all 752 features: AUC 0.95 leaky vs 0.85 honest).
2. Using only the 16 voice measures both hospitals share, the honest AUC is 0.77 to 0.78. That's almost exactly Oxford's 0.78, from a different country, different software and 8x more people.
3. The signal isn't just sex: sex alone gives about 0.57, and the model scores about equally within men only and women only (0.74 to 0.77).
4. A model trained in one hospital still ranks people sensibly in the other (AUC 0.71 to 0.83). But its yes/no cut-off doesn't transfer reliably, and the Oxford test has only 8 healthy people, so its interval is very wide.

---

# Script 04: leakage on Istanbul

**Cell 1, load.** Drops person 37's duplicated recording (755 rows, 252 people). Converts the Istanbul pulse period into pitch in Hz (`1 / meanPeriodPulses`) so it matches Oxford.

**Feature sets, and why each one is there:**

| Set | Features | Why |
|---|---|---|
| Sex only | 1 | The floor. If a model can't beat "is this a man?", it has learned nothing about the voice |
| Shared 16 | 16 | The measures Istanbul shares with Oxford, minus PPE (it points the opposite way, so it isn't the same measure). Needed for 05 |
| Shared 16 + sex | 17 | Does adding sex help? If yes, sex carries information the voice doesn't |
| All 752 | 752 | Everything Istanbul has. The "throw it all in" approach |

**Cell 2, models.**
- Logistic regression has `C=0.1`, a stronger penalty than the default. With 752 features and about 200 people per training fold, an unpenalised model would memorise noise. Same setting everywhere, so the comparison is fair.
- The random forest uses `min_samples_leaf=3`, which stops trees memorising single recordings.

**Cell 3, out-of-fold predictions, then person level.** Each recording is predicted by a model that never saw it. Then each person's (usually 3) predictions are averaged into one score. Two reasons:
- A clinic screens a person, not a recording.
- Otherwise each person counts three times.

**Cell 4, the grid.** 4 feature sets x 2 models x 2 splits x 10 repeats x 5 folds. The 10 repeats use different random seeds, because one split of 252 people can be lucky or unlucky.

**Cell 5, results (person-level AUC, mean over 10 repeats, [lowest to highest repeat]):**

| Features | Model | Leaky (by recording) | **Honest (by person)** | Honest, men | Honest, women |
|---|---|---|---|---|---|
| Sex only | LR | 0.57 | **0.57** | n/a | n/a |
| Shared 16 | LR | 0.79 | **0.77** [0.76-0.79] | 0.74 | 0.76 |
| Shared 16 | RF | 0.86 | **0.78** [0.77-0.80] | 0.76 | 0.77 |
| Shared 16 + sex | LR | 0.80 | **0.77** | 0.75 | 0.75 |
| Shared 16 + sex | RF | 0.86 | **0.78** | 0.76 | 0.76 |
| All 752 | LR | 0.93 | **0.83** [0.81-0.86] | 0.80 | 0.85 |
| All 752 | RF | 0.95 | **0.85** [0.83-0.86] | 0.80 | 0.87 |

**What to take from it:**

1. **Leakage grows with features.** The gap between leaky and honest is 0.02 for logistic regression on 16 features, 0.08 for a random forest on 16, and 0.10 on 752. Why: the more measurements you have, the easier it is to fingerprint one person's voice. A flexible model (forest) with many features is the best fingerprinter. So "always split by person" matters most exactly when you're using the most powerful setup.
2. **Same answer in two countries.** Shared 16, honest: 0.77 to 0.78 in Istanbul, 0.77 to 0.78 in Oxford (01b, 22 features). That replication is your strongest new claim.
3. **Sex isn't the story.** Adding sex changes nothing (0.772 to 0.772). Within men only and women only, the voice still separates PD from healthy (0.74 to 0.77). Sex alone is 0.57 here (0.60 when computed on everyone at once in 03, a bit lower out-of-fold).
4. **752 features beat 16 (0.85 vs 0.78)**, but only within Istanbul. Those extra features (wavelets, MFCCs, TQWT) don't exist in Oxford, so you can't check whether they travel.

**Why the "n/a":** within men only, a sex-only model gives everyone the same score, so there's nothing to rank. The script blanks it rather than report a meaningless number.

---

# Script 05: does the model travel between hospitals?

**The idea.** Script 04 tests new people from the same clinic. A real screening tool would be used in a different clinic, with a different microphone and software. So: train on one hospital, test on the other.

**Cell 1.** Puts both datasets into the same 16 columns with common names (e.g. `MDVP:Fo(Hz)` and `F0_Hz` both become `pitch_hz`).

**Two scaling choices (cell 2), and why they matter.**
- *Train scaling only:* standardise with the training hospital's averages. Naive.
- *Each hospital scaled to itself:* standardise each dataset with its own averages. This removes the software offset found in 03 (jitter 3-9x smaller and shimmer ~2.5x larger in Istanbul). It uses the test hospital's averages but **never its labels**. And both datasets are 75% PD, so those averages aren't tilted by a different mix of patients. Say this out loud if asked whether it's cheating.

**Cell 3, benchmark.** Within-Oxford honest AUC on the same 16 features: 0.73 (LR), 0.68 (RF). Lower than with all 22 Oxford features (0.77 to 0.78 in 01b), mostly because PPE, Oxford's strongest feature, is gone.

**Cell 4, results (person-level AUC, 95% bootstrap interval resampling test people):**

| Train -> Test | Scaling | Model | AUC [95% interval] | Balanced acc at 0.5 | Share flagged PD (truth 75%) | Within-hospital AUC of test site |
|---|---|---|---|---|---|---|
| Istanbul -> Oxford | Train only | LR | 0.83 [0.65-0.98] | 0.65 | 84% | 0.73 |
| Istanbul -> Oxford | Train only | RF | 0.81 [0.63-0.96] | 0.71 | 56% | 0.68 |
| Istanbul -> Oxford | Own | LR | 0.80 [0.58-0.97] | 0.73 | 59% | 0.73 |
| Istanbul -> Oxford | Own | RF | 0.80 [0.60-0.97] | 0.71 | 81% | 0.68 |
| Oxford -> Istanbul | Train only | LR | 0.71 [0.63-0.78] | 0.65 | 81% | 0.77 |
| Oxford -> Istanbul | Train only | RF | 0.78 [0.71-0.84] | 0.59 | 94% | 0.78 |
| Oxford -> Istanbul | Own | LR | 0.72 [0.65-0.79] | 0.65 | 60% | 0.77 |
| Oxford -> Istanbul | Own | RF | 0.77 [0.69-0.83] | 0.55 | 94% | 0.78 |

**What to take from it:**

1. **The ranking travels.** Every cross-hospital AUC is well above 0.5. A model built in Istanbul ranks Oxford patients about as well as Oxford's own model does, and maybe better, because it learned from 252 people instead of about 25. But the Oxford interval runs from about 0.6 to 0.98, so you **cannot** claim "better". You can only say "no worse".
2. **Oxford -> Istanbul loses a bit** (0.71 to 0.78 vs 0.77 to 0.78 within Istanbul). That's expected: a model trained on 32 people is fragile.
3. **The cut-off doesn't travel.** AUC only checks the order of people. A screening tool needs a cut-off. The random forest trained in Oxford flags 94% of Istanbul people as PD when the truth is 75%, giving a balanced accuracy of 0.55 to 0.59. The ranking survived the move between hospitals but the threshold broke. In practice, any tool moved to a new clinic needs re-calibrating there.
4. **Per-hospital scaling didn't consistently help AUC.** For logistic regression it moved the share flagged closer to reality in one direction (81% to 60% vs truth 75%, which overshoots the other way). It's not a clean fix. Report it as mixed.

**Cell 5, coefficients.** 10 of 16 features have the same sign in both hospitals. The flips are mostly *within* the jitter and shimmer families (e.g. APQ3, APQ5 and DDA flip while APQ11 and dB don't). That's the Stage 1 lesson again: when features are near-copies of each other, the model splits the credit between them arbitrarily, so individual signs are unstable even when the family as a whole points the same way. Don't interpret single coefficients here.

---

## Limits to name before anyone asks

- **Oxford's 8 healthy people** make every Oxford-side number uncertain. That's why the intervals are shown.
- **Different software (MDVP vs Praat):** same names don't guarantee identical calculations. PPE showed this clearly, and the others may differ in subtler ways.
- **Two hospitals, two countries, both from around 2007 to 2018.** That's not enough to claim the model works "anywhere".
- **Oxford has no sex column,** so the sex check only exists for Istanbul.
- **The 10-repeat range in 04 is not a full confidence interval.** It shows how much the split matters, not how much a new sample of 252 people would differ.
- **Research only.** Never "diagnoses Parkinson's".

## Check yourself (out loud)

1. Why does the leaky-vs-honest gap grow from 0.02 to 0.10 as you add features and switch to a forest?
2. What does "scaled to itself" use from the test hospital, and why is it not leakage?
3. AUC 0.78 but balanced accuracy 0.55: how can both be true for the same model?
4. Why can't you claim the Istanbul-trained model is *better* on Oxford than Oxford's own model?
5. Why are individual jitter coefficients unreliable, but the jitter family as a whole is fine?

## Exercises

1. In 04, drop the two exact-copy features (ddpJitter = 3 x rapJitter, ddaShimmer = 3 x apq3Shimmer) and rerun Shared 16. Does AUC change? Predict the answer before you run it.
2. In 05, set `C=1.0` instead of 0.1. Does the Istanbul -> Oxford result move?
3. Pick a cut-off in Istanbul that flags 75% as PD, then apply that same cut-off to Oxford. Write down what share gets flagged.
