from __future__ import annotations

import numpy as np
import pandas as pd


def macro_f05(
    pairs: pd.DataFrame,
    truth_counts: np.ndarray,
    entity_mask: np.ndarray,
    threshold: float,
) -> float:
    selected = pairs["probability"].to_numpy() >= threshold
    qids = pairs["query_rid"].to_numpy(dtype=np.int64)
    labels = pairs["label"].to_numpy(dtype=np.int8)
    size = truth_counts.size
    predicted = np.bincount(qids[selected], minlength=size)
    true_positive = np.bincount(qids[selected], weights=labels[selected], minlength=size)
    truth = truth_counts[entity_mask]
    pred = predicted[entity_mask]
    tp = true_positive[entity_mask]
    score = np.zeros_like(truth, dtype=np.float64)
    singleton = truth == 0
    score[singleton & (pred == 0)] = 1.0
    regular = (truth > 0) & (pred > 0) & (tp > 0)
    precision = np.divide(tp[regular], pred[regular])
    recall = np.divide(tp[regular], truth[regular])
    score[regular] = 1.25 * precision * recall / (0.25 * precision + recall)
    return float(score.mean())


def tune_threshold(
    pairs: pd.DataFrame, truth_counts: np.ndarray, validation_qids: np.ndarray
) -> tuple[float, float, list[dict[str, float]]]:
    entity_mask = np.zeros(truth_counts.size, dtype=bool)
    entity_mask[validation_qids] = True
    fixed = np.linspace(0.10, 0.995, 180)
    quantiles = np.quantile(pairs["probability"], np.linspace(0.50, 0.999, 100))
    thresholds = np.unique(np.concatenate([fixed, quantiles]))
    history = []
    for threshold in thresholds:
        score = macro_f05(pairs, truth_counts, entity_mask, float(threshold))
        history.append({"threshold": float(threshold), "macro_f0_5": score})
    best = max(history, key=lambda x: x["macro_f0_5"])
    return best["threshold"], best["macro_f0_5"], history
