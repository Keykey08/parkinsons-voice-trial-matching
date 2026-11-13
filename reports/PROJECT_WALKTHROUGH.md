# The whole project, explained so you can defend it

Written by Claude for Kerem, 13 Nov 2026. Read with `RESULTS_LEDGER.md` open. Every number here is in the ledger with its source file. Sections end with the interview questions you should expect.

---

## 0. The one-paragraph story

Papers claim voice recordings can detect Parkinson's with 90%+ accuracy. I took three public datasets and ran four honest tests: split by person instead of by recording, repeat in a second hospital, train in one hospital and test in the other, and track the same people over time. The headline accuracy is mostly an artifact of letting the model see the same person in training and test. What survives is real but modest: person-level AUC around 0.77-0.85 inside a hospital, 0.71-0.83 across hospitals, with a decision threshold that does not transfer at all and can only partly be repaired. Within one person over six months, I could not detect voice tracking symptoms, but the dataset's symptom scores are interpolated, so that test was underpowered from the start.

That paragraph is your elevator answer. Everything below is the evidence for each clause.

---

## 1. Script 01b: the leakage experiment (Oxford data)

**The question.** Each of the 32 Oxford people gave about 6 recordings. If recordings from the same person land in both training and test, is the model detecting Parkinson's, or recognising the person?

**The method.** Same features, same models, two cross-validation splits: `KFold` shuffled by recording (leaky) vs `GroupKFold` by person (honest). The ONLY thing that changes is the split. That's what makes it an experiment rather than an anecdote: one variable moves, everything else is pinned.

**The numbers (ledger 01b.1-01b.5).** Random forest AUC 0.96 leaky vs 0.78 honest. Logistic regression 0.91 vs 0.77. Balanced accuracy honest: ~0.64. The "always predict PD" floor is AUC 0.50.

**Why AUC and not accuracy.** 75% of recordings are PD, so "always say PD" already gets 75% accuracy. AUC asks: if I hand you one random PD recording and one random healthy recording, how often does the model score the PD one higher? Chance = 0.5 regardless of class balance. Balanced accuracy (mean of sensitivity and specificity) is the same idea for a hard yes/no at a threshold.

**Why the gap happens.** A voice is like a fingerprint: pitch level, breathiness, habits. With 6 recordings per person, a flexible model memorises "this voice = that label" and cashes it in when the same voice shows up in test. The random forest gains more from leakage (0.96 vs 0.91) precisely because it is more flexible: more capacity to memorise individuals.

**Interview questions.**
- "Why is splitting by recording wrong?" Because the unit of independence is the person, not the recording. Test data must be independent of training data; two recordings of one voice are not independent.
- "Is 0.78 good?" It beats chance clearly, but it's a long way from the 0.96 a naive reader takes from the paper's setup. And with 8 healthy people, every Oxford number has a wide interval.

---

## 2. Script 03: profiling the second hospital (Istanbul)

**The question.** Before you pool or compare two datasets, you have to ask: are they measuring the same thing? 03 is pure due diligence - no models, just looking.

**What it found, and why each finding matters.**
- *756 rows, 252 people, 3 recordings each; one duplicate row (person 37) dropped.* Row counts and duplicates first, always. A duplicate across CV folds is leakage in miniature.
- *16 of 17 shared features point the same direction as Oxford* (PD = more jitter, more shimmer, lower HNR...). This is the replication check: the biology shows up the same way in a second country, second language, second device. That's the strongest pro-signal evidence in the project.
- *PPE points the OPPOSITE way and sits on a different scale* → excluded from all cross-hospital work. One feature failing direction-consistency is exactly what due diligence is for.
- *Jitter 3-9x smaller, shimmer ~2.5x larger in Istanbul.* Different analysis software (MDVP vs Praat) = different units for the "same" feature. Consequence: you can never feed raw Istanbul values into an Oxford-trained model; everything cross-hospital must be standardised, and the scaling choice becomes an experimental variable (that's 05.9).
- *Sex is confounded with the label*: 82% of men have PD vs 66% of women (chi2 7.59, p 0.006). Sex alone gives AUC 0.60. A voice model could just be a sex detector - which is why 04 tests within each sex separately.

**Interview question.** "What's the first thing you do with a new dataset?" Profile it before modelling: counts, duplicates, scales, direction-consistency with prior data, and candidate confounders. 03 found a flipped feature, a units mismatch and a confounder - any one of which would have silently corrupted the modelling scripts.

---

## 3. Script 04: leakage again, plus the confounder check (Istanbul)

**The question.** Does the Oxford leakage result replicate at 8x the sample size, and does the sex confounder explain the signal?

**The numbers (person-level AUC, mean of 10 repeats of 5-fold).**
| | Leaky | Honest | Gap |
|---|---|---|---|
| 16 shared features, LR | 0.792 | 0.772 | 0.020 |
| 16 shared features, RF | 0.860 | 0.784 | 0.077 |
| All 752 features, LR | 0.933 | 0.832 | 0.101 |
| All 752 features, RF | 0.953 | 0.850 | 0.102 |

**Three lessons in one table.**
1. The leak replicates, and it *scales with feature count*: 752 features give the model far more material to fingerprint a person (0.10 of AUC is pure leakage). More features = more leakage surface, not just more signal.
2. With only 3 recordings per person (vs Oxford's 6), the honest-vs-leaky gap is smaller on 16 features. Fewer repeats per person = less to memorise. The leak size depends on dataset structure - which is why you can't just subtract a fixed "leakage tax".
3. Honest 16-feature AUC (0.77-0.78) *matches Oxford's honest result*. Two hospitals, two languages, same honest number. That consistency is the finding.
4. Sex does not explain it: sex-only honest AUC 0.57; within men only 0.74-0.76; within women 0.76-0.77. The voice signal survives inside each sex.

**Why "10 repeats of 5-fold".** One 5-fold split is one random partition; its AUC carries partition luck. Repeating with 10 seeds and averaging turns a lottery ticket into an estimate, and the spread across repeats tells you how much partition luck there was.

**Interview question.** "How do you know the model isn't just detecting sex?" Three ways, all run: sex alone (0.57, weak), adding sex to voice features (changes nothing), and fitting within each sex separately (signal persists). That triangulation pattern - test the confounder alone, added, and stratified - works for any confounder.

---

## 4. Script 05: train in one hospital, test in the other

**The question.** Cross-validation inside one hospital still shares everything else: microphone, room, software, population. The real test of "does this generalise" is: train in hospital A, test on hospital B, having never seen B.

**The numbers (16 shared features, train-hospital scaling).**
| Train → Test | LR | RF |
|---|---|---|
| Istanbul → Oxford | 0.828 [0.645-0.978] | 0.807 [0.623-0.950] |
| Oxford → Istanbul | 0.710 [0.630-0.782] | 0.777 [0.713-0.835] |

**How to read this honestly.**
- The *ranking transfers*: AUC 0.71-0.83 across hospitals is in the same region as within-hospital honest AUC. The signal is not a laboratory artifact.
- The Oxford-test intervals are enormous because Oxford has only 8 healthy people; a bootstrap over test people swings on whether those 8 land well. Never quote 0.83 without its [0.65-0.98].
- The *threshold does not transfer*: at the default 0.5 cut-off, the Oxford-trained RF flags 94% of Istanbul as PD (truth: 75%), balanced accuracy 0.59. Why: the two feature distributions sit on different scales (03's finding), so the model's score distribution shifts, and a cut-off tuned to one distribution lands nonsense on the other.
- 05.9/05.10: per-hospital scaling changes results modestly, and only 10 of 16 LR coefficients keep the same sign across hospitals - individual feature weights are unstable even when the overall ranking holds.

**Interview question.** "Your AUC is 0.83 across hospitals - so it works?" AUC measures ranking, not decisions. A deployed screener must say yes/no, which needs a threshold, and the threshold is exactly what broke. That distinction (discrimination vs calibration) is the whole point of 07.

---

## 5. Script 07: can the threshold be repaired?

**The question.** If the ranking travels but the cut-off doesn't, the practical question is: how much labelled data from the NEW hospital do you need to fix the cut-off, and how far does that get you?

**The design.** "Local calibration with k people": label k random people in the new hospital, place the cut-off using only them, score everyone else; repeat 500 draws. Compare against: fixed 0.5, cut-off tuned at home (train hospital), and an oracle that knows the best cut-off in hindsight (the ceiling).

**The numbers (balanced accuracy per person).**
- Oxford→Istanbul RF: 0.586 at fixed 0.5 → 0.670 with k=10 → 0.713 with k=80 (oracle 0.741). Recalibration genuinely repairs this case - and the share flagged PD falls from 94% to 57%.
- Oxford→Istanbul LR: 0.645 → 0.622/0.631. No help; fixed 0.5 was already near its ceiling (0.671).
- Istanbul→Oxford, either model: little or no help, and k=16 draws swing from 0.50 to 0.89 - with 16 people, your calibration set is itself a lottery.
- Oracle ceilings 0.67-0.79 everywhere: even a perfect cut-off leaves you far from the leaky-split fantasy numbers. **The cut-off is not the main limit. The ranking is.**

**Interview question.** "What would you tell a hospital that wants to deploy this?" Expect ranking performance around 0.75-0.8 AUC, budget labelled local data for calibration, and even then the ceiling is ~0.75 balanced accuracy - useful as triage at best, nowhere near diagnostic. And the variance at small k means a pilot on 16 patients can mislead in either direction.

---

## 6. Script 06: within-person over time (yours - this section only frames it)

You ran 06 and have the numbers in the ledger (06.1-06.10). The two ideas to hold onto:

1. **The interpolation trap**: UPDRS was measured ~3 times and straight-lined in between (median R² against time 0.83). So within one person, "symptoms" ≈ "time". Anything that drifts with time correlates with "symptoms" for free.
2. **The within-person design**: comparing each person's two slopes (UPDRS vs roughness index) cancels age, sex, microphone and home - the same difference-in-differences logic from your econometrics. Your result: rho = -0.05, p = 0.75. A null, from a design the interpolation had already hobbled - which is itself the finding: this dataset cannot answer the within-person question.

Still yours to do, in your own words: the two MY PREDICTION lines, the three draft comments, rewriting `06_findings.md`, and removing its DRAFT banner. Don't skip the predictions - "I wrote my prediction before running" is the most credible sentence in the whole repo.

---

## 7. Part B: the trial matcher (02, patients file, 08)

**What it is.** 20 fictional UK patient profiles (written before any trial data was fetched - so they couldn't be tailored to the trials), matched against recruiting Parkinson's trials from ClinicalTrials.gov.

**The architecture, and why it's two stages.**
1. *Hard filters first*: age limits and sex are stated eligibility facts. A text model must never outrank them - a 43-year-old woman must never see a men-over-60 trial, however similar the text.
2. *TF-IDF cosine ranking* on what remains. TF-IDF turns each document into a vector of word weights: a word weighs more the more it appears in this document and the less it appears across all documents. "Dyskinesia" is informative; "patient" is not. Cosine similarity then measures the angle between the patient vector and each trial vector. It is a (sparse) embedding - every document becomes a point in a vector space and nearness = relatedness - just not a *neural* one. The honest comparison is run alongside: a plain shared-keyword count. If TF-IDF can't beat counting words, it isn't earning its complexity.
3. *LLM explanations last, checked by a human*: each top-5 match gets a drafted explanation that must be verified against the actual eligibility text before anyone quotes it. See `ETHICS_NOTE.md` for why that discipline is not optional in a medical context.

**Why not neural embeddings here?** They'd likely rank better (synonyms: "stiffness" vs "rigidity"), but they're a black box ranking in a context where every output needs to be traceable to eligibility text. TF-IDF scores decompose into exactly which words matched - you can audit a ranking by hand. For v1 of a safety-relevant demo, auditable beats better. The upgrade path (swap the vectorizer) is one line and is noted in the script.

**Interview question.** "Why fictional patients?" Real matching means processing real health data under UK GDPR with no need to. And writing them *before* fetching is the same pre-registration discipline as the 06 predictions: commit before you peek.

---

## 8. The confounders, closed and open

- **Sex**: measured, confounded, and ruled out as the explanation (04).
- **Age**: not in the Oxford/Istanbul data files. The Sakar 2019 paper reports cohort ages: PD 33-87 (mean 65.1 ± 10.9), controls 41-82 (mean 61.1 ± 8.9) - so the groups roughly overlap but PD runs ~4 years older, and per-person age is still unavailable. Verdict: partially closed at cohort level, open at person level.
- **Recording setup**: Sakar 2019 reports sustained /a/, three repetitions, 44.1 kHz, collected at the Department of Neurology, Cerrahpaşa Faculty of Medicine, Istanbul University. Same protocol for both groups - but medication state at recording time is unreported in both datasets. Open.
- **Base rates**: both datasets are ~75% PD. A real screening population is nothing like that, so every balanced-accuracy number here would shift in deployment. Open by construction.

---

## 9. The five sentences to own before any interview

1. "The 0.96 AUC is leakage: split by person and it's 0.78 - I ran that experiment, both hospitals."
2. "The signal replicates across hospitals (16 of 17 features same direction; cross-hospital AUC 0.71-0.83) but the threshold does not transfer - the Oxford model flags 94% of Istanbul as PD."
3. "Local recalibration repairs the threshold only sometimes, and the oracle ceiling is ~0.75 balanced accuracy - the ranking, not the cut-off, is the limit."
4. "Within-person, the telemonitoring data can't answer the question because the symptom scores are interpolated - my null result is a statement about the data, not about Parkinson's."
5. "Every number is in a ledger with the script that produced it, and the AI-assisted parts are logged - ask me about any row."
