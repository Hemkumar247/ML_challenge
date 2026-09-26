# Runbook: Apple Silicon Mac (M5, 16 GB RAM)

This runbook takes the repository from a fresh clone to the leaderboard file and final
submission ZIP. The pipeline uses only the supplied challenge data. Do not give an agent
permission to query web services, maps, registries, search engines, or external business
databases for entity information; the challenge explicitly prohibits that.

## 1. Prepare the Mac

Keep the Mac connected to power, disable automatic sleep for the duration of the run, and
make at least 35–50 GB of free disk space available. The raw data is about 2.5 GB, but
DuckDB needs additional working space. A GPU is not required.

Install Apple's command-line tools, Homebrew (if absent), Python 3.12, and OpenMP:

```bash
xcode-select --install
brew install python@3.12 libomp
```

Clone the repository:

```bash
git clone https://github.com/Hemkumar247/ML_challenge.git
cd ML_challenge
```

## 2. Place the data

The large TSV files are not stored in Git. Copy the unmodified challenge files into this
exact layout:

```text
student_resource/dataset/train/train_source1.tsv
student_resource/dataset/train/train_source2.tsv
student_resource/dataset/train/train_source3.tsv
student_resource/dataset/train/train_ground_truth.tsv
student_resource/dataset/test/test_source1.tsv
student_resource/dataset/test/test_source2.tsv
student_resource/dataset/test/test_source3.tsv
```

Check that all seven files are present:

```bash
find student_resource/dataset -type f -name '*.tsv' -print
```

Do not rename, edit, open-and-resave, or convert the TSV files. They must remain tab
separated.

## 3. Run the entire workflow

Make the runner executable (needed only if the executable bit was lost during transfer),
then launch it:

```bash
chmod +x scripts/run_macos.sh
TEAM_NAME="Your Team Name" \
TEAM_MEMBERS="Name One, Name Two" \
./scripts/run_macos.sh
```

Defaults are deliberately safe for a 16 GB Mac: 8 GB DuckDB memory and up to 8 threads.
Override them only if needed:

```bash
MEMORY_LIMIT=6GB THREADS=6 TEAM_NAME="Your Team" TEAM_MEMBERS="Names" ./scripts/run_macos.sh
```

The runner creates `.venv`, installs pinned packages, performs training and inference,
validates both TSVs, writes a results review, fills the methodology document, and builds
the final ZIP. It may run for hours on the full dataset. Progress timestamps are printed
to the terminal. Avoid running a second copy concurrently.

To keep the Mac awake from another terminal:

```bash
caffeinate -dimsu
```

Stop `caffeinate` with Control-C after the pipeline ends.

## 4. If the run is interrupted or fails

Read the last error before changing anything. Common responses:

- `Out of Memory` or the process is killed: rerun with `MEMORY_LIMIT=6GB THREADS=6`.
- `No space left on device`: free space; keep at least 35 GB available, then rerun.
- missing TSV error: restore the exact directory and filenames shown above.
- XGBoost/OpenMP library error: run `brew install libomp`, reactivate `.venv`, and retry.
- corrupt/stale work database after a forced stop: the runner uses `--rebuild` and safely
  replaces the generated database on the next complete run.

The `.work/` directory is disposable generated state. Never delete the source dataset.

## 5. Inspect the results

The command must finish with validator output containing `PASS`. Then review:

```bash
cat student_resource/output/result_review.md
cat student_resource/output/run_metrics.json
```

Use this order when deciding what to change:

| Observed result | Meaning | Next action |
| --- | --- | --- |
| Candidate pair recall below 0.98 | Blocking loses too many true pairs | Improve/add blocking signatures or raise candidate limits before changing the model |
| Recall at least 0.98 but validation F0.5 is weak | Candidates exist, ranking/classification is weak | Improve comparison features, hard negatives, model tuning, or validation design |
| Validation is good but public score is much lower | Likely distribution shift, overfit, or threshold instability | Compare country/source slices, especially unseen France; prefer robust features and conservative thresholds |
| Many false positives | Harmful for F0.5, which favors precision | Increase threshold and inspect same-name/different-address cases |
| Many false negatives with high precision | Model is too conservative | Reduce threshold slightly, then confirm macro F0.5 on the fixed validation split |
| Candidate lists are huge | Slow and may introduce hard false positives | Tighten frequency caps or candidate ranking while preserving labeled recall |

Change one coherent idea at a time, retain the fixed hash-based split and seed, and compare
`validation_macro_f0_5`, `candidate_pair_recall_on_sample`, singleton behavior, runtime,
and candidate count. Do not tune repeatedly against the public leaderboard.

After code changes, rerun the complete command. Never submit an output from a different
code/model version than the version included in the ZIP.

## 6. Final checks and submission

The scored file is:

```text
student_resource/output/matching_results.tsv
```

Upload that TSV to the challenge portal during the competition. It must be plain UTF-8,
tab-separated text; do not open it in Excel or Numbers and resave it.

The separately required final package is:

```text
student_resource/<team_name>_submission.zip
```

Inspect its contents before upload:

```bash
unzip -l student_resource/*_submission.zip
```

It must contain `output/matching_results.tsv`, `output/candidate_pairs.tsv`, the runnable
`code/business_entity_resolution/` tree, and the filled `Documentation_template.md`.
Keep the ZIP, the two TSVs, `run_metrics.json`, the Git commit hash, and the portal receipt
together. Record the public score without treating it as ground truth for further tuning.
