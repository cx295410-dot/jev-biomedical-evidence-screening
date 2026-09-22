# Public record table

One row per review-record pair, 17,191 rows and 392 final inclusions. `dataset_id` identifies the review; `record_id` is the source OpenAlex identifier. These are not guaranteed unique publication identities across reviews. `ground_truth` is final review inclusion (0/1), not an independent initial-screening label. DOI/PMID/year are source metadata.

`title_present` and `abstract_present` preserve missingness without redistributing text. `duplicate_group`, `duplicate_status` and `abstract_index_gap_n` are frozen quality fields. Empty duplicate-group values are preserved.

`jev_decision` is the native returned decision; `p_include` and `p_exclude` are frozen probabilities. `latency_ms`, `model`, `api_status`, `prompt_version` and `prompt_sha256` are method metadata. Request identifiers, token telemetry, errors and timestamps are omitted.

`tfidf_score`, `bm25_score`, `minilm_score` and respective `_rank_pct` fields retain the frozen comparator scores. Percentiles use average rank and (rank-1)/(N-1). `bm25_z` is a standardized lexical score. `lr_loro_p_include`, `lr_loro_decision`, `lr_training_n` and `lr_training_positive_n` retain feature-based leave-one-review-out LR outputs and training counts.

The ten LR features are listed in zero_cost_baseline_protocol.json. Token-count and lexical-overlap fields are features rather than source text. `abstract_missing` is the baseline feature; presence flags are retained independently from the master record table.

All metric proportions use the 0-1 scale. The analysis code defines AUROC, non-interpolated AP, native binary metrics, calibration, complete-boundary-tie workload and random-tie expectations. `reference_*.csv` and `reference_*.json` are the archived analysis outputs. `PUBLIC_MANIFEST.json` hashes the distributed files; `reference_sha256_manifest.csv` preserves original-input provenance and therefore does not describe the sanitized public export.
