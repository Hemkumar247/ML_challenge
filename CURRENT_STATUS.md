# Current implementation status and next steps

Last updated: 2026-09-26

## Executive status

The repository contains a complete end-to-end implementation and packaging workflow,
but it does **not yet contain final predictions**. The full composite-blocking run was
interrupted before the training benchmark completed. Therefore
`matching_results.tsv`, `candidate_pairs.tsv`, a validator `PASS`, and a leaderboard
score do not yet exist and must not be claimed.

The most important next action is to run the full workflow on the intended 16 GB Apple
Silicon Mac, inspect the recorded blocking recall and validation macro F0.5, and iterate
only if those measurements show a real bottleneck.

## What has been completed

- Read and mapped the organizer's full problem statement, required ZIP layout, exact TSV
  schemas, validator behavior, fair-play rules, and macro-F0.5 singleton scoring.
- Profiled all seven supplied TSVs locally: 24,229,173 source records (26,435,994 rows
  including the ground-truth file) and roughly 2.5 GB of raw files.
- Measured 2,206,821 training S1 entities and 123,247 training singletons (5.5848%).
  Labeled S1 entities average 3.4613 linked S2/S3 records.
- Implemented Unicode-aware normalization without a fixed country vocabulary. France is
  retained as an unseen open-set label at inference.
- Implemented out-of-core DuckDB ingestion, tokenization, blocking, native similarity
  features, XGBoost hard-negative training, deterministic entity-level train/validation
  splits, macro-F0.5 threshold search, inference, output aggregation, and metrics.
- Implemented exact name/address signatures, bounded typo-shape signatures, and hashed
  composite signatures from pairs among the twelve longest typed tokens.
- Bounded final candidates per S1 and target source, so `candidate_pairs.tsv` is the exact
  set scored by the model rather than an early unfiltered block.
- Added the official validator, automated result review, methodology generation, ZIP
  packaging, a macOS runner, and a detailed autonomous-agent prompt.
- Compiled all Python modules and smoke-tested real-data ingestion, blocking, feature
  generation, label joining, and append behavior on small real subsets.

## Evidence collected so far

### Dataset profile

| Item | Count |
| --- | ---: |
| Train S1 | 2,206,821 |
| Train S2 | 5,034,616 |
| Train S3 | 5,285,603 |
| Test S1 | 1,732,544 |
| Test S2 | 4,887,273 |
| Test S3 | 5,082,316 |
| Training truth links | 7,638,365 |
| Training singletons | 123,247 (5.5848%) |

### Positive-pair blocking analysis

On 172,731 labeled positive pairs sampled at the S1 entity level:

- at least one normalized exact name/address token survived in 99.9444%;
- at least one character 5-gram survived in 99.9213%;
- a composite pair among the twelve deterministic longest typed tokens survived in
  98.4919% before adding the other blocking routes.

This motivated composite token-pair signatures: single Faker-generated words occur too
often across millions of businesses, while a preserved typed pair is much more selective.

### Rejected baseline

The first full-target benchmark used only rare individual/shape signatures. It produced a
very compact average of 6.22 S2 and 6.12 S3 candidates per sampled S1, but captured only
117,474 of 228,040 truth links: **51.514647% candidate-pair recall**. That approach was
correctly rejected because blocking recall is the maximum possible downstream recall.

### Composite redesign and resource constraint

The initial composite implementation attempted one global self-join and exceeded the
6 GB DuckDB limit. A local list-combination implementation still accumulated too much in
one transaction. The current code shards signature generation into bounded approximately
600,000-record partitions and commits incrementally. During the interrupted full Source 2
run, resident memory stayed around 3–5 GB and the database advanced to roughly 3 GB,
showing that sharding resolved the immediate memory-growth pattern.

The latest code also avoids globally grouping the very large, already-selective composite
hash table when calculating document frequencies. Individual and typo-shape signatures
still receive IDF weights; composite pairs receive a strong fixed retrieval weight. This
latest optimization has passed a focused real-data smoke test (2,000 S1 records against
50,000 S2 records: 5,996 final candidates, successful feature generation, and 30 labeled
positives present) but has **not yet completed the full benchmark**, so its final
recall/runtime remains to be measured honestly.

## Why these methods were chosen

1. **Blocking before classification:** all-pairs comparison would require trillions of
   comparisons. The challenge also ranks candidate efficiency explicitly.
2. **Several complementary signatures:** exact tokens handle most records cheaply;
   token shapes recover bounded typo variants; composite pairs make common synthetic name
   and address vocabularies discriminative without broad fuzzy search.
3. **Country-scoped but open-set retrieval:** cross-country matches are implausible, but
   country values are never hard-coded to US/India, so France works without retraining a
   categorical encoder.
4. **Hard-negative XGBoost model:** the candidates that survive blocking are the useful
   confusing negatives. Gradient-boosted trees model nonlinear evidence combinations on
   CPU and use an eligible Apache-2.0 implementation.
5. **Entity-level threshold optimization:** the organizer scores macro F0.5 per S1,
   including exact credit for correct singletons. A fixed 0.5 threshold would not optimize
   that objective.
6. **Out-of-core SQL:** pandas-only processing is not appropriate for 25 million rows and
   hundreds of millions of temporary signatures. DuckDB provides bounded memory and disk
   spilling with reproducible SQL.

## Main constraints encountered

- Raw data is ~2.5 GB and cannot be committed to ordinary GitHub storage.
- Composite blocking produces a multi-GB temporary index even though the final candidate
  list is small.
- A 6 GB DuckDB ceiling exposed per-thread and single-transaction memory pressure. The
  defaults now cap DuckDB at eight threads, and signature generation is sharded.
- The full run takes hours rather than minutes. Starting duplicate runs would waste memory
  and disk.
- France has no training labels, so the approach must rely on language-agnostic string
  evidence and cannot tune a France-specific threshold.
- External lookup, geocoding, business registries, search engines, and enrichment APIs are
  prohibited. No API key is required or allowed for the implemented solution.
- A top-five placement cannot be guaranteed without completed offline metrics and portal
  leaderboard evidence.

## Exact process to carry out next

Use the intended 16 GB Mac and follow [`RUN_ON_MAC.md`](RUN_ON_MAC.md). The shortest path is:

```bash
git clone https://github.com/Hemkumar247/ML_challenge.git
cd ML_challenge
# Copy the seven original TSVs into student_resource/dataset/train and dataset/test.
chmod +x scripts/run_macos.sh
TEAM_NAME="Your actual team name" \
TEAM_MEMBERS="Actual member names" \
MEMORY_LIMIT=8GB THREADS=8 \
./scripts/run_macos.sh
```

If macOS reports high memory pressure or kills the job, rerun with
`MEMORY_LIMIT=6GB THREADS=6`. Keep at least 35–50 GB of free disk and use `caffeinate` as
documented in the runbook.

On Windows PowerShell, run the equivalent workflow from the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r student_resource\code\business_entity_resolution\requirements.txt
python student_resource\code\business_entity_resolution\src\pipeline.py --stage all --rebuild --memory-limit 8GB --threads 8
python student_resource\utils\validate_submission.py --matching student_resource\output\matching_results.tsv --candidate student_resource\output\candidate_pairs.tsv --test-dir student_resource\dataset\test
python scripts\review_results.py --metrics student_resource\output\run_metrics.json --output student_resource\output\result_review.md
python scripts\package_submission.py --resource-root student_resource --team-name "Your actual team name" --team-members "Actual member names"
```

The workflow must finish with all of these:

1. `candidate_pair_recall_on_sample` measured in `output/run_metrics.json`;
2. `validation_macro_f0_5` and learned decision threshold recorded;
3. both required TSVs generated;
4. the official validator printing `PASS`;
5. the final submission ZIP built and inspected.

### Decision gate after the first complete run

- If candidate recall is below 0.98, do not tune XGBoost yet. Inspect missed labeled pairs
  and adjust composite coverage, frequency limits, or the per-source candidate cap.
- If candidate recall is at least 0.98 but F0.5 is weak, tune comparison features, model
  parameters, probability calibration, and the entity-level threshold.
- If false merges dominate, prioritize precision: generic-name candidates with conflicting
  address/digit evidence and the decision threshold are the first places to inspect.
- If validation is strong but the public score is weak, investigate distribution shift and
  unseen-France robustness without using external identity data.

Make one coherent change per experiment and retain the deterministic split/seed. Do not
tune repeatedly against the public leaderboard.

## Submission artifacts

After a successful run:

- Upload `student_resource/output/matching_results.tsv` to the live leaderboard.
- Submit `student_resource/<team_name>_submission.zip` as the final package.
- Keep `candidate_pairs.tsv`, `run_metrics.json`, validator output, artifact SHA-256 hashes,
  Git commit hash, and the portal receipt together for auditability.

The generated outputs, temporary database, model, raw datasets, and ZIP are intentionally
ignored by Git. They are large/generated or organizer-provided artifacts, not source code.
