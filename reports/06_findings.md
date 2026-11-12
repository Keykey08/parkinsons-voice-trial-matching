# 06 findings: does a person's voice get worse when their own symptoms get worse?

> **DRAFT written by Claude Code at Kerem's request, not yet reviewed.** Kerem: read every sentence, check each number against your own run of `src/06_telemonitoring.py`, and rewrite it in your own words before using it anywhere. The task sheet asked for this in your words.

The data is the Parkinson's Telemonitoring set (UCI): 42 people with Parkinson's, about 140 home voice recordings each over roughly 6 months (5,875 recordings), with a clinician symptom score (total UPDRS, higher = worse). The score looks measured but was only taken at baseline, about 3 months and 6 months, and the values in between were filled in along straight lines. Each person has only 14 to 39 distinct UPDRS values, and a straight line against time explains a median of 83% of each person's score.

In step 3, a random forest predicting UPDRS looked good when recordings from the same person were split across train and test (R squared 0.84 for age + sex alone, 0.91 with voice added). Splitting by person removed all of it: every feature set fell below zero R squared, and the MAE (9.5 for voice only, 11.3 to 11.7 with age and sex) was worse than predicting the average for everyone (8.66). Age and sex are close to a person ID, so the random split let the model recall each person's typical score.

In step 4, I compared each person only with themselves: the slope of their UPDRS against the slope of a voice roughness index. Across the 42 people the Spearman correlation was -0.05 (p = 0.75). A plain-average version of the index gave -0.04 (p = 0.81). The rough 95% interval for rho is about -0.35 to +0.26, so a strong link is ruled out but a weak one is not.

Limitation I would tell a WISE interviewer: this is a null result from a design with low power. Each UPDRS slope rests on about three real clinic exams plus interpolation, so it is a noisy measure, and the roughness index was my own choice. A null here does not show that voice and symptoms are unrelated. It shows this dataset cannot detect it. It is also a research exercise, not a diagnostic or monitoring tool.

To test this properly I would need symptom scores actually measured at the same time as each recording (for example a weekly clinician or self-report score), not interpolated, and more people.
