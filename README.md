# Jev biomedical evidence screening benchmark

Public data and analysis for **Evaluating Jev for biomedical evidence screening: discrimination, calibration and high-recall workload**.

**Author:** Xiang Chen, Clinical Medical College, North China University of Science and Technology. ORCID: https://orcid.org/0009-0005-6464-7377 .

## Scope of this release

Version 1.0.0 contains 17,191 review-record pairs from ten SYNERGY reviews, with 392 eventual inclusions. It provides frozen Jev and comparator scores, the ten LR features, text-presence/quality indicators, the frozen prompt/protocol, the archived statistical analysis source, a portable public-data analysis adapter, and figure-generation code. No API key or model access is needed to reproduce the reported statistics.

This is a **frozen-prediction evaluation release**. It is not an end-to-end model rerun package. The historical Jev API caller, original baseline text-feature generation code, original full inference environment and proprietary Jev model weights are not included. No reconstructed caller is represented as the original implementation. Baseline method settings and the exact LR feature list are documented in the protocol and manuscript. The versioned returned model was `jev-1.13.0`.

## Reproduce the analysis

Use Python 3.12 and run these commands from this repository directory:

```sh
python -m pip install -r requirements.txt
python reproduce_analysis.py --out reproduced --bootstrap 5000
python verify_reproduction.py --out reproduced
```

The public adapter was executed with all 5,000 bootstrap replicates and 10,000 review resamples. Thirteen archived CSV tables and the calibration summary match within numerical tolerance; see `REPRODUCTION_CHECK.json`. Row bootstrap seed: 20260921. Review-resampling seed: 20260922. The archived analysis used NumPy 2.3.5 and pandas 3.0.1. The public adapter replaces original-file checks with public export hash/key checks and substitutes Boolean text-presence flags for missingness tests. Its mathematical analysis is retained from `archived_reproduce_analysis.py`.

`archived_reproduce_analysis.py` preserves the historical analysis source; it expects the original nine-file source mapping and is not the runnable entry point for this sanitized export. Original-input hashes and integrity checks are retained in `reference_sha256_manifest.csv` and `reference_integrity_checks.csv`; public export hashes are separately listed in `PUBLIC_MANIFEST.json`.

To regenerate the figures:

```sh
python -m pip install -r requirements-figures.txt
python make_figures.py
```

This writes PNG, SVG, PDF and EPS figures to `figures/`. Figure code is the publication-stage implementation and uses frozen aggregate tables. It does not alter or refit predictions.

## File guide

- `frozen_records.csv`: identifiers, final-inclusion labels, frozen predictions/features, quality indicators and selected method metadata; no bibliographic text
- `review_sources.csv`: review identifiers and provenance links for locating source review context
- `jev_full_protocol.json`: exact frozen instructions and choices; local input paths removed
- `zero_cost_baseline_protocol.json`: comparator definition and LR predictors; local input paths removed
- `reference_*.csv` and `reference_*.json`: archived numerical outputs
- `DATA_DICTIONARY.md`: public-field interpretation
- `ESM_1.pdf` and `ESM_2.zip`: journal supplementary results and numerical companion

## Interpretation boundaries

Labels are final full-text inclusion status, not independently adjudicated title/abstract screening decisions. Strong AUROC/AP does not imply lower high-recall workload or safe automatic exclusion. Primary workload retains complete boundary ties; random-tie expectations are label-informed retrospective quantities. LR is feature-based leave-one-review-out logistic regression. Review-level separation does not establish publication-identity independence. Frozen-prediction sensitivity analyses do not retrain LR. The benchmark is selected, and uncertainty across ten reviews is limited.

Candidate text, review abstracts and verbatim review eligibility passages are not redistributed. Source metadata and labels originate from SYNERGY V1 (https://doi.org/10.34894/HE6NAQ), upstream commit `dc2dadfdbb98eb1b4259604789abd640aa3b693e`. See https://github.com/asreview/synergy-dataset for source retrieval and attribution. `review_sources.csv` records source review links. Reacquiring text for new inference may not recreate the historical text snapshot.

## Citation and licensing

Use `CITATION.cff` and cite the SYNERGY dataset independently. Code: MIT. Original derived numeric outputs: CC BY 4.0; source SYNERGY identifiers/labels retain their upstream CC0 status. See `DATA_LICENSE.md`. This repository does not grant rights to third-party publications or the Jev model. The release tag and commit identify a versioned public artifact; no DOI is claimed for this GitHub release.
