# Versioned rules gold production

Gold for the scored FAR split is derived **programmatically from eCFR Title 48 Chapter 1 XML**.

1. Fetch part-level XML from `https://www.ecfr.gov/api/versioner/v1/full/{date}/title-48.xml?chapter=1&part={part}` for snapshots listed in `far_fetch.json`.
2. Parse `DIV*` nodes with `TYPE=SECTION`. Consecutive snapshots with identical normalized text are merged into one validity interval (`valid_from` / `valid_to`).
3. **As-of gold:** a numeric/unit span that appears in the in-force version and in **no other** retained version of that clause.
4. **Changelog gold:** spans (or sentences) present in version N but not N+1, and vice versa.
5. FAMR fixtures remain `ci_smoke` only and are **not** part of the scored test split.

U.S. FAR/eCFR text is a U.S. government work (17 U.S.C. § 105).

## Human review (sample)

I read the in-force eCFR text for these test items against the committed gold:

- `vr-far-asof-far-19.1505-v2` — **pass**. As of 2023-01-18 the clause says SBA will make a determination "Within 15 calendar days" of the contracting officer's notification.
- `vr-far-asof-far-19.1506-v1` / `v4` — **pass**. WOSB manufacturing ceiling moved from `$6.5` million (2019 snapshot) to `$8.5` million (2026 snapshot); each figure is unique to that interval.
- `vr-far-asof-far-19.1507-v2` — **pass**. Limitation on subcontracting: "at least 50 percent of the cost incurred for personnel".
- `vr-far-chg-far-13.000-v3-v4` — **pass**. Distinctive amounts `$7.5` vs `$9` (simplified acquisition related) appear in consecutive versions.
- `vr-far-chg-far-13.003-v1-v2` — **pass**. `$3,500` vs `$7.5` on consecutive snapshots.

Automated check: every test `must_include` span occurs in the as-of (or D1∪D2) clause text. Weak changelog items whose truncated spans were identical were dropped by the builder.

Builder meta: `{"fetched_at": "2026-11-16T20:09:21+00:00", "n_xml": 90, "n_changing_clauses": 317, "n_docs": 825, "n_candidate_questions": 373}`

## Spot-check sample

### 1. `vr-far-asof-far-19.1505-v2`

- split: `test`
- gold method: `diff_unique_numeric_span`
- question: As of 2023-01-18, what figure does FAR 19.1505 (19.1505 Set-aside procedures.) specify that is unique to the version then in force?
- gold: `15 calendar days`
- must_include: `['15 calendar days']`

### 2. `vr-far-asof-far-19.708-v2`

- split: `test`
- gold method: `diff_unique_numeric_span`
- question: As of 2023-12-01, what figure does FAR 19.708 (19.708 Contract clauses.) specify that is unique to the version then in force?
- gold: `$750,000`
- must_include: `['$750,000']`

### 3. `vr-far-asof-far-22.102-2-v2`

- split: `test`
- gold method: `diff_unique_numeric_span`
- question: As of 2021-01-19, what figure does FAR 22.102-2 (22.102-2 Administration.) specify that is unique to the version then in force?
- gold: `$10,000`
- must_include: `['$10,000']`

### 4. `vr-far-asof-far-22.1505-v3`

- split: `test`
- gold method: `diff_unique_numeric_span`
- question: As of 2023-01-18, what figure does FAR 22.1505 (22.1505 Solicitation provision and contract clause.) specify that is unique to the version then in force?
- gold: `$50,000`
- must_include: `['$50,000']`

### 5. `vr-far-asof-far-22.502-v1`

- split: `test`
- gold method: `diff_unique_numeric_span`
- question: As of 2021-07-03, what figure does FAR 22.502 (22.502 Definitions.) specify that is unique to the version then in force?
- gold: `$25`
- must_include: `['$25']`

### 6. `vr-far-asof-far-36.104-v3`

- split: `test`
- gold method: `diff_unique_numeric_span`
- question: As of 2025-01-17, what figure does FAR 36.104 (36.104 Policy.) specify that is unique to the version then in force?
- gold: `$35`
- must_include: `['$35']`

### 7. `vr-far-asof-far-42.1502-v3`

- split: `test`
- gold method: `diff_unique_numeric_span`
- question: As of 2026-11-12, what figure does FAR 42.1502 (42.1502 Policy.) specify that is unique to the version then in force?
- gold: `$900,000`
- must_include: `['$900,000']`

### 8. `vr-far-chg-far-13.006-v1-far-13.006-v2`

- split: `test`
- gold method: `consecutive_version_diff`
- question: What changed in FAR 13.006 between 2021-07-03 and 2025-01-17?
- gold: `(f) 52.223-6, Drug-Free Workplace, except for individuals. became (f) 52.226-7, Drug-Free Workplace, except for individuals.`
- must_include: `['(f) 52.223-6, Drug-Free Workplace, excep', '(f) 52.226-7, Drug-Free Workplace, excep']`

### 9. `vr-far-chg-far-13.501-v2-far-13.501-v3`

- split: `test`
- gold method: `consecutive_version_diff`
- question: What changed in FAR 13.501 between 2023-12-01 and 2026-11-12?
- gold: `$750,000 became $900,000`
- must_include: `['$750,000', '$900,000']`

### 10. `vr-far-chg-far-16.102-v1-far-16.102-v2`

- split: `test`
- gold method: `consecutive_version_diff`
- question: What changed in FAR 16.102 between 2020-07-03 and 2023-01-18?
- gold: `2306(a) and 41 U.S.C. became 3321(a) and 41 U.S.C.`
- must_include: `['2306(a) and 41 U.S.C.', '3321(a) and 41 U.S.C.']`

### 11. `vr-far-chg-far-19.1303-v3-far-19.1303-v4`

- split: `test`
- gold method: `consecutive_version_diff`
- question: What changed in FAR 19.1303 between 2022-01-03 and 2023-01-18?
- gold: `(b) If the SBA determines that a concern is a HUBZone small business concern, it became (b) If SBA determines that a concern is a HUBZone small business, it will design`
- must_include: `['(b) If the SBA determines that a concern', '(b) If SBA determines that a concern is ']`

### 12. `vr-far-chg-far-19.1504-v1-far-19.1504-v2`

- split: `test`
- gold method: `consecutive_version_diff`
- question: What changed in FAR 19.1504 between 2020-07-03 and 2023-01-18?
- gold: `(see subpart 8.6); or (2) AbilityOne participating non-profit agencies for the b became This subpart does not apply to— (a) Requirements that an 8(a) contractor is curr`
- must_include: `['(see subpart 8.6); or (2) AbilityOne par', 'This subpart does not apply to— (a) Requ']`
