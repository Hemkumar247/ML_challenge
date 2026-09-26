#!/usr/bin/env python3
"""Turn pipeline metrics into a concise, evidence-based next-action report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def fmt(value: object, digits: int = 4) -> str:
    if isinstance(value, (float, int)):
        return f"{value:.{digits}f}"
    return "not recorded"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    metrics = json.loads(args.metrics.read_text(encoding="utf-8"))
    recall = metrics.get("candidate_pair_recall_on_sample")
    score = metrics.get("validation_macro_f0_5")
    threshold = metrics.get("decision_threshold")
    predicted = metrics.get("predicted_matches")
    singleton_rate = metrics.get("predicted_singleton_rate")
    candidate_count = metrics.get("test_candidate_pairs_scored")

    actions: list[str] = []
    if not isinstance(recall, (float, int)):
        actions.append("Blocking recall was not recorded; do not tune the classifier until it is measured.")
    elif recall < 0.98:
        actions.append(
            "Blocking is the first bottleneck (recall below 0.98). Inspect missed labeled pairs and "
            "improve bounded candidate generation before tuning XGBoost or its threshold."
        )
    elif isinstance(score, (float, int)) and score < 0.80:
        actions.append(
            "Blocking recall is usable, but validation F0.5 is weak. Focus on hard negatives, "
            "comparison features, calibration, and threshold stability."
        )
    else:
        actions.append(
            "No obvious blocking failure is visible. Prioritize slice-level error analysis and "
            "robustness rather than broad architectural changes."
        )
    if isinstance(threshold, (float, int)) and (threshold < 0.15 or threshold > 0.98):
        actions.append(
            "The learned threshold is near an extreme. Check calibration and validation sample size "
            "before trusting small threshold changes."
        )
    actions.append(
        "Because macro F0.5 rewards precision and correct singletons, review false merges before "
        "trying to recover marginal extra links."
    )
    actions.append(
        "Use only organizer data. Evaluate source/country slices offline and keep France handling "
        "language-agnostic; external lookup is prohibited."
    )

    feature_importance = metrics.get("feature_importance", {})
    top_features = sorted(feature_importance.items(), key=lambda item: item[1], reverse=True)[:10]
    feature_lines = [f"- `{name}`: {value:.6f}" for name, value in top_features]
    if not feature_lines:
        feature_lines = ["- Not recorded."]

    text = "\n".join(
        [
            "# Result review",
            "",
            "## Key metrics",
            "",
            f"- Validation macro F0.5: {fmt(score, 6)}",
            f"- Candidate-pair recall on labeled sample: {fmt(recall, 6)}",
            f"- Learned decision threshold: {fmt(threshold, 6)}",
            f"- Test candidate pairs scored: {candidate_count if candidate_count is not None else 'not recorded'}",
            f"- Predicted links: {predicted if predicted is not None else 'not recorded'}",
            f"- Predicted singleton rate: {fmt(singleton_rate, 6)}",
            "",
            "## Recommended next actions",
            "",
            *[f"{index}. {action}" for index, action in enumerate(actions, 1)],
            "",
            "## Top feature importances",
            "",
            *feature_lines,
            "",
            "A public leaderboard score is feedback, not a substitute for a fixed offline validation "
            "protocol. Make one coherent change per experiment and keep the deterministic split/seed.",
            "",
        ]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text, encoding="utf-8")
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

