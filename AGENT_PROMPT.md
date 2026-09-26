# Copy-paste prompt for Antigravity or another autonomous coding agent

Copy everything inside the block below into the agent while its working directory is the
cloned `ML_challenge` repository on the Apple Silicon Mac.

```text
You are responsible for running, validating, reviewing, and packaging the ML Challenge
2026 Business Entity Resolution solution in this repository. Work autonomously, but do
not weaken validation and do not use any external entity data.

Machine and environment:
- Apple Silicon MacBook M5 with 16 GB unified memory.
- Repository root is the current working directory.
- The seven organizer-provided TSV files should already be under
  student_resource/dataset/train and student_resource/dataset/test.
- Use Python 3.12 and a repository-local .venv.
- Stay within an 8 GB DuckDB memory limit and at most 8 threads initially. If the OS kills
  the process or memory pressure is severe, retry with 6 GB and 6 threads.
- Ensure at least 35 GB of free disk before the full run.

Non-negotiable challenge rules:
1. Use only the supplied train/test files. Never query or scrape the internet, business
   registries, maps, geocoders, commercial ER services, APIs, or outside datasets.
2. Do not upload the challenge data to GitHub or any service.
3. The model must remain challenge-eligible (the current XGBoost approach is eligible).
4. Preserve the unseen-country behavior: country is an open string label; never restrict
   logic to only US and India, and retain every France record.
5. F0.5 is precision-heavy and macro-averaged over S1 entities, including singletons.
6. Do not fabricate metrics, output rows, or a successful status. Report exact evidence.

Required workflow:
A. Read README.md, RUN_ON_MAC.md, student_resource/README.md, the pipeline README, and
   all Python source under student_resource/code/business_entity_resolution/src before
   editing anything. Confirm git status and preserve unrelated user changes.
B. Verify the exact seven TSV filenames, their tab-separated headers, available disk,
   Python architecture/version, and that no other pipeline instance is running.
C. Run the supported workflow rather than creating predictions manually:
     chmod +x scripts/run_macos.sh
     TEAM_NAME="REPLACE_WITH_TEAM_NAME" \
     TEAM_MEMBERS="REPLACE_WITH_MEMBER_NAMES" \
     MEMORY_LIMIT=8GB THREADS=8 ./scripts/run_macos.sh
   Replace the two placeholders before executing. If team details are unavailable, ask
   once; do not ship placeholder values in the final documentation.
D. Monitor timestamped logs. Do not start duplicate full runs. If it fails, diagnose the
   actual exception, make the smallest justified code/configuration fix, run focused
   checks, and then rerun the complete reproducible workflow. For memory failure use
   6GB/6 threads. For missing libomp install it with Homebrew. For low disk stop and ask
   the user to free space; do not delete personal files.
E. Require all of the following artifacts:
   - student_resource/output/matching_results.tsv
   - student_resource/output/candidate_pairs.tsv
   - student_resource/output/run_metrics.json
   - student_resource/output/result_review.md
   - student_resource/<sanitized_team_name>_submission.zip
F. Require the validator to print PASS. Independently check that matching_results.tsv has
   the exact two-column header, exactly one row per test S1 ID, unique target IDs per row,
   only existing S2/S3 IDs, no duplicate S1 rows, and that every final match is present
   in candidate_pairs.tsv. Do not alter a TSV by hand to silence a failure; fix the code
   that generated it and rerun.
G. Read run_metrics.json and result_review.md. Use this diagnostic priority:
   1) If candidate_pair_recall_on_sample < 0.98, improve blocking first. Inspect missed
      labeled pairs offline using only training data. Consider additional bounded fuzzy
      signatures or a larger final candidate cap, but measure reduction and memory.
   2) If blocking recall >= 0.98 but macro validation F0.5 is weak, work on comparison
      features, hard-negative selection, calibration/model parameters, and thresholding.
   3) If false positives dominate, increase precision: inspect shared generic tokens,
      conflicting digits/addresses, and raise the threshold only when fixed-split macro
      F0.5 improves.
   4) If false negatives dominate while precision is strong, lower the threshold slightly
      or add robust name/address features, again choosing by fixed validation macro F0.5.
   5) Check per-country and per-source behavior using training labels. France has no train
      labels, so use language-agnostic normalization and never infer quality from external
      lookup.
   6) Make one coherent experimental change per run. Preserve the deterministic split and
      seed. Revert changes that do not improve reliable offline evidence.
H. Any optimization must stay out-of-core and feasible on 16 GB RAM. Do not replace the
   pipeline with an all-pairs join or load every large TSV into pandas. Do not trade away
   candidate recall merely to make the run look faster.
I. Once satisfied, rerun from clean generated state using run_macos.sh, validate again,
   rebuild the ZIP, and inspect it with `unzip -l`. The ZIP must contain:
     output/matching_results.tsv
     output/candidate_pairs.tsv
     code/business_entity_resolution/src/*
     code/business_entity_resolution/README.md
     code/business_entity_resolution/requirements.txt
     Documentation_template.md (filled with real metrics)
J. Return a concise final report with: commit hash, exact command/configuration, runtime,
   validation F0.5, blocking recall, learned threshold, candidate count, predicted links,
   singleton rate, validator PASS evidence, artifact sizes/SHA-256 hashes, and exact paths.
   Clearly distinguish the leaderboard TSV from the final ZIP. If anything failed, say so
   and do not call the submission ready.

The primary outcome is a genuinely generated and validated
student_resource/output/matching_results.tsv. The second required outcome is the final
reproducibility ZIP. Do not stop after giving recommendations if the safe in-scope next
step can still be executed.
```

