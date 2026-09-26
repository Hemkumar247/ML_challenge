#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
RESOURCE_ROOT="${REPO_ROOT}/student_resource"
PYTHON_BIN="${PYTHON_BIN:-python3.12}"
MEMORY_LIMIT="${MEMORY_LIMIT:-8GB}"
THREADS="${THREADS:-8}"
TEAM_NAME="${TEAM_NAME:-}"
TEAM_MEMBERS="${TEAM_MEMBERS:-}"

if [[ -z "${TEAM_NAME}" || -z "${TEAM_MEMBERS}" ]]; then
  echo "ERROR: Set TEAM_NAME and TEAM_MEMBERS before running."
  echo 'Example: TEAM_NAME="My Team" TEAM_MEMBERS="A, B" ./scripts/run_macos.sh'
  exit 2
fi

required=(
  "dataset/train/train_source1.tsv"
  "dataset/train/train_source2.tsv"
  "dataset/train/train_source3.tsv"
  "dataset/train/train_ground_truth.tsv"
  "dataset/test/test_source1.tsv"
  "dataset/test/test_source2.tsv"
  "dataset/test/test_source3.tsv"
)
for relative in "${required[@]}"; do
  if [[ ! -f "${RESOURCE_ROOT}/${relative}" ]]; then
    echo "ERROR: Missing ${RESOURCE_ROOT}/${relative}"
    exit 2
  fi
done

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN=python3
  else
    echo "ERROR: Python 3 is missing. Install it with: brew install python@3.12 libomp"
    exit 2
  fi
fi

cd "${REPO_ROOT}"
if [[ ! -x .venv/bin/python ]]; then
  "${PYTHON_BIN}" -m venv .venv
fi
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r student_resource/code/business_entity_resolution/requirements.txt

echo "Running full pipeline with memory=${MEMORY_LIMIT}, threads=${THREADS}"
python student_resource/code/business_entity_resolution/src/pipeline.py \
  --stage all \
  --rebuild \
  --memory-limit "${MEMORY_LIMIT}" \
  --threads "${THREADS}"

cd "${RESOURCE_ROOT}"
python utils/validate_submission.py \
  --matching output/matching_results.tsv \
  --candidate output/candidate_pairs.tsv \
  --test-dir dataset/test

cd "${REPO_ROOT}"
python scripts/review_results.py \
  --metrics student_resource/output/run_metrics.json \
  --output student_resource/output/result_review.md
python scripts/package_submission.py \
  --resource-root student_resource \
  --team-name "${TEAM_NAME}" \
  --team-members "${TEAM_MEMBERS}"

echo
echo "READY"
echo "Leaderboard file: ${RESOURCE_ROOT}/output/matching_results.tsv"
SAFE_TEAM_NAME="$(python -c 'import re,sys; print(re.sub(r"[^A-Za-z0-9._-]+", "_", sys.argv[1]).strip("_") + "_submission.zip")' "${TEAM_NAME}")"
echo "Final package:     ${RESOURCE_ROOT}/${SAFE_TEAM_NAME}"
