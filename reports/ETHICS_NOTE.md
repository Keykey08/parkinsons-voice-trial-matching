# Ethics note: the trial matcher (Part B)

Written for this repository; applies to `02_fetch_trials.py`, `data/patients_fictional.json` and `08_match_trials.py`.

## Why every patient is fictional

Matching real people to clinical trials means processing health data, the most protected category under UK GDPR, and creating a record of a named person's diagnosis. Nothing in this project needs that. The 20 profiles were invented to cover the situations a matcher must handle (newly diagnosed, advanced disease, DBS, cognitive impairment, pregnancy planning, comorbidities), and they were written **before** any trial data was fetched, so they could not be quietly tailored to make the matcher look good. That ordering is recorded in `AI_USE_LOG.md`.

## What the matcher is, and is not

It is a **retrieval** tool: given a patient description, it ranks publicly registered trials by text similarity, after filtering on the two stated hard criteria (age range, sex). It is not an eligibility decision. Most real criteria live in free text the filters never parse: medication washout rules, prior surgery, kidney function, pacemakers, pregnancy. A ranked list is a starting point for a clinician and patient to read the actual criteria together, nothing more.

## Where it could cause harm if misused

- **False hope and wasted effort**: a high similarity score reads like "you qualify". It does not mean that. Every output file in this repo says so.
- **Silent exclusion**: a retrieval model trained or weighted on text can systematically bury trials for groups who describe symptoms differently, or whose conditions are under-described in registries. A null check (does the keyword baseline surface different trials?) is run in `08` precisely because disagreement between two rankers is a flag to read, not to average away.
- **Cognitive impairment and consent**: two fictional profiles (P08, P11) were included because consent capacity is a real eligibility dimension. A text matcher sees only words; it must never be the thing that decides whether such a patient is approached.
- **LLM explanations**: the per-match explanations are drafted by an LLM and are only as good as the eligibility text they cite. Each one is marked as unverified until checked by a person against the registry entry. An unchecked LLM explanation in a medical context is a liability, not a feature.

## Data handling

Trial records come from the public ClinicalTrials.gov registry (API v2) and carry their NCT numbers so every claim can be traced to the registry entry. No scraping of participant-level or personal data of any kind. The fetched CSV and all outputs contain registry text only.
