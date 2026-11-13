# 08 match explanations

> **DRAFT written by Claude (the LLM step of the pipeline) on 13 Nov 2026, not yet verified.** Kerem: before quoting any of this, open at least three patients' top trials on ClinicalTrials.gov (search the NCT number) and check each claim against the actual eligibility text. The ethics note says why this check is not optional. Remove this banner only after you have done it.

Corpus: 671 recruiting Parkinson's trials fetched 13 Nov 2026 (ClinicalTrials.gov API v2), 670 kept after dropping 1 with trivial eligibility text. Hard age/sex filters leave 527 to 658 candidates per patient, so the TF-IDF ranking is doing almost all the work. Of each patient's TF-IDF top-5, on average only 0.9 trials also appear in the keyword baseline's top-5: the two rankers disagree far more than they agree, which is itself a reason to read matches, not trust them.

## Where the ranking clearly worked

- **P03 and P19 (young onset)**: both get the Young Onset PD Subtypes study (NCT07752355) first, plus genetic registries (PD GENEration, Parkinson's Families Project) for P03, who mentions a LRRK2 mutation, and a young-onset ACT psychological intervention for P19. These are the right neighbourhoods.
- **P06 (DBS + speech problems)**: top-5 is DBS-speech studies (NCT07455760, NCT05182892) and LSVT-LOUD voice therapy (NCT07730073). Direct hit on both halves of her description.
- **P07 (impulse control disorder)**: three of five are ICD-specific trials, including rTMS for ICD (NCT06237868) and an ICD-prevention algorithm (NCT07505394).
- **P09 and P17 (RBD / prodromal)**: both surface RBD and prodromal-PD cohort studies (NCT07790744, NCT06582121). For P17, note the eligibility risk below.
- **P11 (advanced, end of life, swallowing)**: continuous apomorphine at end of life (NCT07257861) and expiratory muscle strength training for swallowing (NCT07932561) are exactly the supportive-care territory his family asked about.
- **P16 (painful dystonia, apomorphine)**: spinal cord stimulation for painful camptocormia (NCT06291051) and two apomorphine-infusion studies fit her description well.

## Where it failed, and why that matters

- **P18 (PD, breast cancer in remission)**: three of her top-4 are metastatic breast cancer drug trials (NCT06966700, NCT06312176, NCT06841354). She is in remission and described as wanting low-risk PD trials. "Breast cancer" is a rare, high-weight phrase, so it dominated the cosine similarity and buried her PD relevance. This is the silent-failure mode the ethics note warns about: a score of 0.095 looks the same whether the match is right or absurd.
- **P17 (prodromal, not yet diagnosed)**: the top hit (Ginger-PD, NCT07882108, score 0.23, the highest in the whole run) matched on gut words (constipation, gut function) but almost certainly requires a confirmed PD diagnosis he does not have. High score, likely ineligible: text similarity cannot read "suspected" as an eligibility fact.
- **P14 (pacemaker)**: her top hits are stimulation studies (cerebellar stimulation, vagal nerve stimulation). A pacemaker typically excludes her from exactly these, and from MRI protocols. The hard filters only know age and sex; device contraindications live in free text the pipeline does not parse.
- **P15 (daytime sleepiness)** and **P16's #1 (RE104 adjustment disorder)**: sleep and mood vocabulary pulled in partially off-target trials; plausible-looking, wrong on reading.

## The sentence to carry into an interview

The matcher retrieves the right neighbourhood for most patients, but its worst errors score as confidently as its best hits, so the ranking is a reading list for a clinician, never a decision; and the TF-IDF vs keyword disagreement (0.9 of 5 shared) shows how much the choice of ranker alone changes who is shown what.
