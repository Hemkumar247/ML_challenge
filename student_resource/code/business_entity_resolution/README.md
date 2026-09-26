# Hybrid business entity resolution

This pipeline is designed for the full challenge dataset (roughly 25 million rows),
not a toy in-memory sample. It uses DuckDB for out-of-core preprocessing and joins,
multi-pass rare-signature blocking, hard-negative supervised learning with XGBoost,
and entity-level macro-F0.5 threshold selection.

## Run from `student_resource/`

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r code\business_entity_resolution\requirements.txt
python code\business_entity_resolution\src\pipeline.py --stage all --rebuild
python utils\validate_submission.py --matching output\matching_results.tsv --candidate output\candidate_pairs.tsv --test-dir dataset\test
```

Linux/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r code/business_entity_resolution/requirements.txt
python code/business_entity_resolution/src/pipeline.py --stage all --rebuild
python utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir dataset/test
```

The default DuckDB memory ceiling is 6 GB. On a machine with less RAM, pass
`--memory-limit 3GB --threads 4`; on a larger machine, `--memory-limit 12GB` is a
reasonable acceleration. Intermediate data is stored under `.work/`. No API key,
GPU, internet lookup, or external business data is used.

The outputs are:

- `output/matching_results.tsv` — upload this to the leaderboard.
- `output/candidate_pairs.tsv` — include this in the final submission ZIP.
- `output/run_metrics.json` — validation score, learned threshold, blocking recall,
  candidate statistics, and feature importance.

Training and inference can be separated:

```powershell
python code\business_entity_resolution\src\pipeline.py --stage train --rebuild
python code\business_entity_resolution\src\pipeline.py --stage infer
```

Do not delete `.work/xgb_matcher.json` between those two commands.
