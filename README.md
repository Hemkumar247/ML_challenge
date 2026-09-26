# ML Challenge 2026 — Business Entity Resolution

This repository contains a scalable, challenge-compliant entity-resolution pipeline and
an end-to-end macOS run/package workflow. It produces the two required TSV files and the
final submission ZIP without external entity lookup.

## Start here

- [`CURRENT_STATUS.md`](CURRENT_STATUS.md) — what is implemented, measured evidence,
  constraints encountered, and the exact remaining work. Read this first.
- [`RUN_ON_MAC.md`](RUN_ON_MAC.md) — exact setup, execution, validation, interpretation,
  and submission steps for an Apple Silicon Mac with 16 GB RAM.
- [`AGENT_PROMPT.md`](AGENT_PROMPT.md) — a detailed copy-paste prompt for an autonomous
  coding agent such as Antigravity.
- [`student_resource/code/business_entity_resolution/README.md`](student_resource/code/business_entity_resolution/README.md)
  — pipeline architecture and manual commands.

The challenge dataset is intentionally excluded from Git because the seven TSV files are
about 2.5 GB in total. Put the original files under `student_resource/dataset/train/` and
`student_resource/dataset/test/` before running.

## One-command run on macOS

```bash
TEAM_NAME="Your Team Name" \
TEAM_MEMBERS="Name One, Name Two" \
./scripts/run_macos.sh
```

Successful completion creates:

- `student_resource/output/matching_results.tsv` — upload this to the leaderboard.
- `student_resource/output/candidate_pairs.tsv` — blocking audit file.
- `student_resource/output/result_review.md` — automatic results interpretation.
- `student_resource/<team_name>_submission.zip` — final reproducible package.

Do not commit or publish the supplied datasets, generated work database, predictions, or
submission ZIP. The `.gitignore` already excludes them.
