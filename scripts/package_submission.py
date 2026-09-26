#!/usr/bin/env python3
"""Validate outputs, fill methodology from real metrics, and build the final ZIP."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from datetime import date
from pathlib import Path


def safe_name(value: str) -> str:
    result = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_")
    if not result:
        raise ValueError("Team name must contain at least one letter or number")
    return result


def metric(metrics: dict, key: str, digits: int = 6) -> str:
    value = metrics.get(key)
    return f"{value:.{digits}f}" if isinstance(value, (float, int)) else "Not recorded"


def methodology(team: str, members: str, metrics: dict) -> str:
    config = metrics.get("configuration", {})
    blocking = metrics.get("blocking", {})
    candidate_lines = []
    for name, values in sorted(blocking.items()):
        if isinstance(values, dict):
            candidate_lines.append(
                f"- **{name}:** {values.get('candidates', 'not recorded')} candidates; "
                f"{values.get('average_candidates', 'not recorded')} average per query"
            )
    if not candidate_lines:
        candidate_lines = ["- Candidate statistics were not recorded."]
    feature_importance = metrics.get("feature_importance", {})
    top = sorted(feature_importance.items(), key=lambda item: item[1], reverse=True)[:10]
    top_text = ", ".join(f"{name} ({value:.4f})" for name, value in top) or "Not recorded"
    return f"""# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** {team}  
**Team Members:** {members}  
**Submission Date:** {date.today().isoformat()}

## 1. Executive Summary

We use an out-of-core hybrid entity-resolution pipeline: multi-pass rare-signature
blocking followed by an XGBoost classifier over normalized name/address similarities.
The decision threshold is selected directly for entity-level macro F0.5 on a deterministic
validation split, reflecting the challenge's precision-heavy objective and singleton rules.

## 2. Methodology

### 2.1 Problem Analysis

Records contain legal-suffix variation, punctuation and word-order changes, typographical
noise, incomplete or reordered addresses, transliteration differences, and missing fields.
Country is treated as an open string label and used only to constrain implausible cross-
country comparisons; no fixed US/India one-hot vocabulary is used. No external data,
geocoder, registry, API, or identity lookup is used.

### 2.2 Solution Strategy

DuckDB performs out-of-core ingestion, Unicode-aware canonicalization, tokenization,
blocking, and feature computation. Blocking combines exact name/address tokens, pairs of
the longest informative typed tokens, and bounded typo-tolerant token-shape signatures.
Candidates are ranked and capped before the model. XGBoost is trained on labeled positives
and the difficult negatives that survive blocking.

**Approach Type:** Blocking + supervised gradient-boosted classifier  
**Core Innovation:** Bounded rare composite signatures create scalable hard candidates,
while the final threshold is optimized for macro F0.5 including singleton entities.

## 3. Candidate Generation (Blocking)

- Exact normalized name and address token signatures.
- Composite signatures from pairs among the twelve longest deterministic typed tokens.
- Typo-tolerant signatures based on token prefix, suffix, and length bucket.
- Signatures are country-scoped and high-frequency signatures are discarded.
- Candidates are pre-ranked by exact/shape evidence, limited to
  {config.get('max_candidates_per_source', 'not recorded')} per target source.
- Labeled-sample candidate-pair recall: **{metric(metrics, 'candidate_pair_recall_on_sample')}**.

Recorded candidate statistics:

{chr(10).join(candidate_lines)}

## 4. Matching Model

The classifier is XGBoost using histogram tree construction. Features include exact
normalized equality, Jaro-Winkler similarity, normalized Levenshtein similarity, character
and token Jaccard similarities, name/address token overlaps, digit overlap/conflict signal,
length ratios, missing-address signal, blocking evidence, and joint field similarities.

- Training reference sample: {config.get('train_percent', 'not recorded')}%
- Validation reference sample: {config.get('validation_percent', 'not recorded')}%
- Random seed: {config.get('random_seed', 'not recorded')}
- Learned decision threshold: **{metric(metrics, 'decision_threshold')}**
- Most influential recorded features: {top_text}

## 5. Results & Error Analysis

- **Validation macro F0.5:** {metric(metrics, 'validation_macro_f0_5')}
- **Candidate recall:** {metric(metrics, 'candidate_pair_recall_on_sample')}
- **Validation candidate pairs:** {metrics.get('validation_candidate_pairs', 'Not recorded')}
- **Test candidate pairs scored:** {metrics.get('test_candidate_pairs_scored', 'Not recorded')}
- **Predicted links:** {metrics.get('predicted_matches', 'Not recorded')}
- **Predicted singleton rate:** {metric(metrics, 'predicted_singleton_rate')}

The principal false-positive risk is a generic/shared business name paired with conflicting
or incomplete address evidence. False negatives are expected when all informative tokens
are corrupted or removed before blocking. Because F0.5 weights precision more heavily,
threshold selection explicitly penalizes false merges and rewards correct empty predictions.

## 6. Conclusion

The solution is reproducible on CPU with bounded memory, preserves every test S1 entity,
and produces both the scored match file and the exact final candidate set. The pipeline is
language-agnostic enough to process the unseen France label without hard-coded country
categories or prohibited external enrichment.

## Appendix A. Code Artefacts

`code/business_entity_resolution/src/pipeline.py` is the end-to-end entry point. Its README
contains exact Windows/macOS/Linux commands. Dependencies are pinned in `requirements.txt`.
The supplied validator checks the generated TSV files before packaging.
"""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resource-root", type=Path, default=Path("student_resource"))
    parser.add_argument("--team-name", required=True)
    parser.add_argument("--team-members", required=True)
    args = parser.parse_args()
    root = args.resource_root.resolve()
    matching = root / "output" / "matching_results.tsv"
    candidates = root / "output" / "candidate_pairs.tsv"
    metrics_path = root / "output" / "run_metrics.json"
    for path in (matching, candidates, metrics_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required generated artifact is missing: {path}")

    validator = root / "utils" / "validate_submission.py"
    subprocess.run(
        [
            sys.executable,
            str(validator),
            "--matching",
            str(matching),
            "--candidate",
            str(candidates),
            "--test-dir",
            str(root / "dataset" / "test"),
        ],
        check=True,
    )

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    archive = root / f"{safe_name(args.team_name)}_submission.zip"
    with tempfile.TemporaryDirectory(prefix="ml_challenge_package_") as tmp:
        stage = Path(tmp)
        (stage / "output").mkdir()
        shutil.copy2(matching, stage / "output" / matching.name)
        shutil.copy2(candidates, stage / "output" / candidates.name)
        shutil.copytree(
            root / "code" / "business_entity_resolution",
            stage / "code" / "business_entity_resolution",
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
        (stage / "Documentation_template.md").write_text(
            methodology(args.team_name, args.team_members, metrics), encoding="utf-8"
        )
        if archive.exists():
            archive.unlink()
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
            for path in sorted(stage.rglob("*")):
                if path.is_file():
                    zf.write(path, path.relative_to(stage).as_posix())

    print(f"Built {archive}")
    print(f"  matching_results.tsv sha256: {sha256(matching)}")
    print(f"  candidate_pairs.tsv sha256:  {sha256(candidates)}")
    print(f"  archive sha256:              {sha256(archive)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

