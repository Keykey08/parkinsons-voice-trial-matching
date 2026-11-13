# %% [markdown]
# # Project 2, Part B data: recruiting Parkinson's trials from ClinicalTrials.gov
# Source: ClinicalTrials.gov API v2 (US National Library of Medicine). Public data;
# cite ClinicalTrials.gov, the retrieval date and the NCT numbers you use. See DATA_SOURCES.md.
#
# WHY these fields: the matcher (08) ranks trials by text similarity against a
# patient description, so it needs the free text (title, brief summary,
# eligibility criteria). min/max age and sex are kept separately because they are
# hard eligibility facts and are applied as filters, never as text.

# %%
from pathlib import Path
import requests
import pandas as pd

ROOT = Path(__file__).resolve().parents[1] if "__file__" in globals() else Path.cwd()
URL = "https://clinicaltrials.gov/api/v2/studies"
params = {"query.cond": "Parkinson Disease", "filter.overallStatus": "RECRUITING", "pageSize": 100}
rows, token = [], None
while True:
    if token:
        params["pageToken"] = token
    r = requests.get(URL, params=params, timeout=30)
    r.raise_for_status()
    j = r.json()
    for s in j.get("studies", []):
        p = s.get("protocolSection", {})
        ident = p.get("identificationModule", {})
        elig = p.get("eligibilityModule", {})
        design = p.get("designModule", {})
        rows.append({
            "nct_id": ident.get("nctId"),
            "title": ident.get("briefTitle"),
            "study_type": design.get("studyType"),
            "phase": ",".join(design.get("phases", []) or []),
            "enrollment": (design.get("enrollmentInfo", {}) or {}).get("count"),
            "conditions": "; ".join(p.get("conditionsModule", {}).get("conditions", []) or []),
            "summary": p.get("descriptionModule", {}).get("briefSummary"),
            "min_age": elig.get("minimumAge"),
            "max_age": elig.get("maximumAge"),
            "sex": elig.get("sex"),
            "eligibility": elig.get("eligibilityCriteria"),
            "countries": ",".join(sorted({l.get("country", "") for l in p.get("contactsLocationsModule", {}).get("locations", []) or []})),
        })
    token = j.get("nextPageToken")
    if not token:
        break

df = pd.DataFrame(rows)
out = ROOT / "data" / "raw" / "ctgov_parkinsons_recruiting.csv"
df.to_csv(out, index=False)
print(f"saved {df.shape[0]} trials x {df.shape[1]} columns -> {out}")
print(df["study_type"].value_counts(dropna=False))
print(df.head(3)[["nct_id", "title", "phase", "min_age", "max_age", "sex"]])
