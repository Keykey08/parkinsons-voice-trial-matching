# %% [markdown]
# # Project 2, Part B data: recruiting Parkinson's trials from ClinicalTrials.gov
# NOT RUN YET. Run this on your own Mac (needs internet + `pip install requests`).
# Source: ClinicalTrials.gov API v2 (US National Library of Medicine). Public data;
# cite ClinicalTrials.gov and the trial NCT numbers you use. See DATA_SOURCES.md.
#
# WHY these fields: the eligibility criteria text is what the embedding model
# will compare against a patient description in Stage 2.

# %%
from pathlib import Path
import requests
import pandas as pd

ROOT = Path(__file__).resolve().parents[1] if "__file__" in globals() else Path.cwd()
URL = "https://clinicaltrials.gov/api/v2/studies"
params = {"query.cond": "Parkinson Disease", "filter.overallStatus": "RECRUITING", "pageSize": 100}
rows, token = [], None
while True:
    if token: params["pageToken"] = token
    r = requests.get(URL, params=params, timeout=30); r.raise_for_status(); j = r.json()
    for s in j.get("studies", []):
        p = s.get("protocolSection", {})
        rows.append({
            "nct_id": p.get("identificationModule", {}).get("nctId"),
            "title": p.get("identificationModule", {}).get("briefTitle"),
            "phase": ",".join(p.get("designModule", {}).get("phases", []) or []),
            "min_age": p.get("eligibilityModule", {}).get("minimumAge"),
            "max_age": p.get("eligibilityModule", {}).get("maximumAge"),
            "sex": p.get("eligibilityModule", {}).get("sex"),
            "eligibility": p.get("eligibilityModule", {}).get("eligibilityCriteria"),
            "countries": ",".join(sorted({l.get("country", "") for l in p.get("contactsLocationsModule", {}).get("locations", []) or []})),
        })
    token = j.get("nextPageToken")
    if not token: break
df = pd.DataFrame(rows)
df.to_csv(ROOT / "data" / "raw" / "ctgov_parkinsons_recruiting.csv", index=False)
print(df.shape); print(df.head())
