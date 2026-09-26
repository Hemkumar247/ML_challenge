#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from blocking import create_candidates, create_tokens, drop_blocking_intermediates
from config import FEATURE_COLUMNS, PipelineConfig
from database import connect, load_ground_truth, load_records, sql_path
from evaluation import tune_threshold
from features import create_feature_table
from model import fit_model, load_model, save_model


def log(message: str) -> None:
    print(time.strftime("[%H:%M:%S]"), message, flush=True)


def validate_inputs(config: PipelineConfig, stage: str) -> None:
    needed: list[Path] = []
    if stage in {"all", "train"}:
        needed += [
            config.train_dir / "train_source1.tsv",
            config.train_dir / "train_source2.tsv",
            config.train_dir / "train_source3.tsv",
            config.train_dir / "train_ground_truth.tsv",
        ]
    if stage in {"all", "infer"}:
        needed += [
            config.test_dir / "test_source1.tsv",
            config.test_dir / "test_source2.tsv",
            config.test_dir / "test_source3.tsv",
        ]
    missing = [str(p) for p in needed if not p.is_file()]
    if missing:
        raise FileNotFoundError("Missing required dataset files:\n" + "\n".join(missing))


def _query_filter(config: PipelineConfig, split: str) -> str:
    bucket = "mod(hash(qr.entity_id), 100)"
    if split == "sample":
        return f"{bucket} < {config.train_percent + config.validation_percent}"
    if split == "train":
        return f"{bucket} < {config.train_percent}"
    if split == "validation":
        return (
            f"{bucket} >= {config.train_percent} AND "
            f"{bucket} < {config.train_percent + config.validation_percent}"
        )
    raise ValueError(split)


def train(config: PipelineConfig, con: duckdb.DuckDBPyConnection, metrics: dict) -> tuple[object, float]:
    log("Loading sampled training reference records and ground truth")
    n_s1 = load_records(con, "train_s1", config.train_dir / "train_source1.tsv")
    load_ground_truth(con, config.train_dir / "train_ground_truth.tsv")
    con.execute("DROP TABLE IF EXISTS train_query")
    con.execute(
        f"CREATE TABLE train_query AS SELECT * FROM train_s1 qr WHERE {_query_filter(config, 'sample')}"
    )
    n_query = con.execute("SELECT count(*) FROM train_query").fetchone()[0]
    log(f"Training sample: {n_query:,} of {n_s1:,} S1 entities")
    n_query_tokens = create_tokens(con, "train_query", "train_query_tokens")
    log(f"Created {n_query_tokens:,} query blocking signatures")

    metrics["training_reference_rows"] = n_s1
    metrics["training_sample_rows"] = n_query
    metrics["blocking"] = {}

    for source in (2, 3):
        log(f"Loading training Source {source}")
        n_target = load_records(
            con, f"train_s{source}", config.train_dir / f"train_source{source}.tsv"
        )
        log(f"Source {source}: {n_target:,} target rows; building signatures")
        token_table = f"train_s{source}_tokens"
        n_tokens = create_tokens(con, f"train_s{source}", token_table)
        log(f"Source {source}: {n_tokens:,} target signatures; retrieving candidates")
        block_metrics = create_candidates(
            con,
            "train_query",
            "train_query_tokens",
            f"train_s{source}",
            token_table,
            "train_candidates",
            source,
            config,
            append=source != 2,
        )
        metrics["blocking"][f"train_source{source}"] = block_metrics
        log(
            f"Source {source}: {block_metrics['candidates']:,} candidates "
            f"({block_metrics['average_candidates']:.2f}/query)"
        )
        create_feature_table(
            con,
            f"train_candidates_part_{source}",
            "train_query",
            f"train_s{source}",
            "train_features",
            training=True,
            append=source != 2,
        )
        con.execute(f"DROP TABLE IF EXISTS train_s{source}")
        drop_blocking_intermediates(con, token_table, "train_candidates", source)

    total_truth = con.execute(
        """
        SELECT count(*)
        FROM truth_pairs tp
        JOIN train_query q ON q.entity_id = tp.source1_entity_id
        """
    ).fetchone()[0]
    captured = con.execute("SELECT sum(label) FROM train_features").fetchone()[0] or 0
    pair_recall = captured / total_truth if total_truth else 1.0
    metrics["candidate_pair_recall_on_sample"] = pair_recall
    metrics["sample_truth_pairs"] = total_truth
    metrics["sample_truth_pairs_captured"] = captured
    log(f"Candidate recall on labeled sample: {pair_recall:.6%} ({captured:,}/{total_truth:,})")

    columns = ", ".join(FEATURE_COLUMNS)
    train_filter = _query_filter(config, "train").replace("qr.entity_id", "source1_entity_id")
    val_filter = _query_filter(config, "validation").replace("qr.entity_id", "source1_entity_id")
    log("Loading hard-negative candidate features for XGBoost")
    train_df = con.execute(
        f"SELECT {columns}, label FROM train_features WHERE {train_filter}"
    ).fetch_df()
    val_df = con.execute(
        f"SELECT query_rid, {columns}, label FROM train_features WHERE {val_filter}"
    ).fetch_df()
    metrics["training_pairs"] = len(train_df)
    metrics["validation_candidate_pairs"] = len(val_df)
    metrics["training_positive_rate"] = float(train_df["label"].mean())
    log(f"Fitting model on {len(train_df):,} hard candidates")
    model = fit_model(train_df, val_df, config.random_seed)
    save_model(model, config.model_path)

    val_df["probability"] = model.predict_proba(val_df[FEATURE_COLUMNS])[:, 1]
    max_rid = con.execute("SELECT max(rid) FROM train_s1").fetchone()[0]
    truth_counts = np.zeros(int(max_rid) + 1, dtype=np.int16)
    truth_df = con.execute(
        """
        SELECT q.rid AS query_rid, count(tp.target_id)::INTEGER AS truth_count
        FROM train_query q
        LEFT JOIN truth_pairs tp ON tp.source1_entity_id = q.entity_id
        GROUP BY q.rid
        """
    ).fetch_df()
    truth_counts[truth_df["query_rid"].to_numpy(dtype=np.int64)] = truth_df[
        "truth_count"
    ].to_numpy(dtype=np.int16)
    val_qids = con.execute(
        f"SELECT rid FROM train_query qr WHERE {_query_filter(config, 'validation')}"
    ).fetchnumpy()["rid"].astype(np.int64)
    threshold, score, history = tune_threshold(val_df, truth_counts, val_qids)
    metrics["validation_macro_f0_5"] = score
    metrics["decision_threshold"] = threshold
    metrics["threshold_search"] = sorted(history, key=lambda x: x["macro_f0_5"], reverse=True)[:20]
    metrics["feature_importance"] = {
        name: float(value)
        for name, value in sorted(
            zip(FEATURE_COLUMNS, model.feature_importances_), key=lambda x: x[1], reverse=True
        )
    }
    log(f"Best entity-level validation macro F0.5={score:.6f} at threshold={threshold:.6f}")
    con.execute("DROP TABLE IF EXISTS train_features")
    con.execute("DROP TABLE IF EXISTS train_candidates")
    con.execute("DROP TABLE IF EXISTS train_query_tokens")
    con.execute("DROP TABLE IF EXISTS train_query")
    con.execute("DROP TABLE IF EXISTS train_s1")
    return model, threshold


def _prepare_prediction_tables(con: duckdb.DuckDBPyConnection) -> None:
    con.execute("DROP TABLE IF EXISTS test_predictions")
    con.execute(
        """
        CREATE TABLE test_predictions (
            query_rid BIGINT,
            target_source UTINYINT,
            target_entity_id VARCHAR,
            probability DOUBLE
        )
        """
    )
    con.execute("DROP TABLE IF EXISTS test_candidate_ids")
    con.execute(
        """
        CREATE TABLE test_candidate_ids (
            query_rid BIGINT,
            target_source UTINYINT,
            target_entity_id VARCHAR,
            candidate_rank UTINYINT
        )
        """
    )


def _predict_feature_table(
    config: PipelineConfig,
    con: duckdb.DuckDBPyConnection,
    model: object,
    table: str,
) -> int:
    select_cols = ", ".join(FEATURE_COLUMNS)
    cursor = con.execute(
        f"SELECT query_rid, target_source, target_entity_id, {select_cols} FROM {table}"
    )
    writer = duckdb.connect(str(config.db_path))
    total = 0
    try:
        while True:
            frame = cursor.fetch_df_chunk(64)
            if frame.empty:
                break
            probability = model.predict_proba(frame[FEATURE_COLUMNS])[:, 1]
            output = frame[["query_rid", "target_source", "target_entity_id"]].copy()
            output["probability"] = probability
            writer.register("prediction_chunk", output)
            writer.execute("INSERT INTO test_predictions SELECT * FROM prediction_chunk")
            writer.unregister("prediction_chunk")
            total += len(output)
            if total % 1_000_000 < len(output):
                log(f"Scored {total:,} candidate pairs")
    finally:
        writer.close()
    return total


def infer(
    config: PipelineConfig,
    con: duckdb.DuckDBPyConnection,
    model: object,
    threshold: float,
    metrics: dict,
) -> None:
    log("Loading test reference source")
    n_s1 = load_records(con, "test_s1", config.test_dir / "test_source1.tsv")
    n_qtokens = create_tokens(con, "test_s1", "test_query_tokens")
    log(f"Test: {n_s1:,} reference rows and {n_qtokens:,} blocking signatures")
    _prepare_prediction_tables(con)
    metrics.setdefault("blocking", {})
    metrics["test_reference_rows"] = n_s1
    total_scored = 0

    for source in (2, 3):
        log(f"Loading test Source {source}")
        n_target = load_records(
            con, f"test_s{source}", config.test_dir / f"test_source{source}.tsv"
        )
        token_table = f"test_s{source}_tokens"
        n_tokens = create_tokens(con, f"test_s{source}", token_table)
        log(f"Source {source}: {n_target:,} rows, {n_tokens:,} signatures; retrieving")
        block_metrics = create_candidates(
            con,
            "test_s1",
            "test_query_tokens",
            f"test_s{source}",
            token_table,
            "test_candidates",
            source,
            config,
            append=source != 2,
        )
        metrics["blocking"][f"test_source{source}"] = block_metrics
        log(
            f"Source {source}: {block_metrics['candidates']:,} candidates "
            f"({block_metrics['average_candidates']:.2f}/query)"
        )
        con.execute(
            f"""
            INSERT INTO test_candidate_ids
            SELECT c.query_rid, c.target_source, t.entity_id, c.candidate_rank
            FROM test_candidates_part_{source} c
            JOIN test_s{source} t ON t.rid = c.target_rid
            """
        )
        create_feature_table(
            con,
            f"test_candidates_part_{source}",
            "test_s1",
            f"test_s{source}",
            "test_features_part",
            training=False,
            append=False,
        )
        total_scored += _predict_feature_table(config, con, model, "test_features_part")
        con.execute("DROP TABLE IF EXISTS test_features_part")
        con.execute(f"DROP TABLE IF EXISTS test_s{source}")
        drop_blocking_intermediates(con, token_table, "test_candidates", source)

    metrics["test_candidate_pairs_scored"] = total_scored
    config.output_dir.mkdir(parents=True, exist_ok=True)
    matching = config.output_dir / "matching_results.tsv"
    candidates = config.output_dir / "candidate_pairs.tsv"
    log("Writing matching_results.tsv")
    con.execute(
        f"""
        COPY (
            SELECT q.entity_id AS source1_entity_id,
                   coalesce(string_agg(p.target_entity_id, ',' ORDER BY p.target_source, p.target_entity_id)
                       FILTER (WHERE p.probability >= {float(threshold)}), '') AS matched_entity_ids
            FROM test_s1 q
            LEFT JOIN test_predictions p ON p.query_rid = q.rid
            GROUP BY q.rid, q.entity_id
            ORDER BY q.rid
        ) TO '{sql_path(matching)}' (HEADER, DELIMITER '\\t', QUOTE '')
        """
    )
    log("Writing candidate_pairs.tsv")
    con.execute(
        f"""
        COPY (
            SELECT q.entity_id AS source1_entity_id,
                   coalesce(string_agg(c.target_entity_id, ',' ORDER BY c.target_source, c.candidate_rank), '')
                       AS candidate_entity_ids
            FROM test_s1 q
            LEFT JOIN test_candidate_ids c ON c.query_rid = q.rid
            GROUP BY q.rid, q.entity_id
            ORDER BY q.rid
        ) TO '{sql_path(candidates)}' (HEADER, DELIMITER '\\t', QUOTE '')
        """
    )
    predicted = con.execute(
        "SELECT count(*) FROM test_predictions WHERE probability >= ?", [threshold]
    ).fetchone()[0]
    nonempty = con.execute(
        "SELECT count(DISTINCT query_rid) FROM test_predictions WHERE probability >= ?", [threshold]
    ).fetchone()[0]
    metrics["predicted_matches"] = predicted
    metrics["predicted_non_singletons"] = nonempty
    metrics["predicted_singletons"] = n_s1 - nonempty
    metrics["predicted_singleton_rate"] = (n_s1 - nonempty) / n_s1
    log(f"Predicted {predicted:,} links; {(n_s1 - nonempty):,} singleton S1 records")


def read_saved_metadata(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} is missing. Run --stage train or --stage all before --stage infer."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    script = Path(__file__).resolve()
    default_root = script.parents[3]
    parser = argparse.ArgumentParser(description="Scalable business entity-resolution pipeline")
    parser.add_argument("--resource-root", type=Path, default=default_root)
    parser.add_argument("--work-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--stage", choices=("all", "train", "infer"), default="all")
    parser.add_argument("--threads", type=int, default=max(1, min(8, (os.cpu_count() or 8) - 1)))
    parser.add_argument("--memory-limit", default="6GB")
    parser.add_argument("--train-percent", type=int, default=8)
    parser.add_argument("--validation-percent", type=int, default=2)
    parser.add_argument("--max-candidates-per-source", type=int, default=10)
    parser.add_argument("--rebuild", action="store_true", help="Discard the exact work database first")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.resource_root.resolve()
    config = PipelineConfig(
        resource_root=root,
        work_dir=(args.work_dir or root / ".work").resolve(),
        output_dir=(args.output_dir or root / "output").resolve(),
        threads=args.threads,
        memory_limit=args.memory_limit,
        train_percent=args.train_percent,
        validation_percent=args.validation_percent,
        max_candidates_per_source=args.max_candidates_per_source,
    )
    validate_inputs(config, args.stage)
    if args.rebuild and config.db_path.is_file():
        config.db_path.unlink()
    metrics: dict = {
        "configuration": {
            "threads": config.threads,
            "memory_limit": config.memory_limit,
            "train_percent": config.train_percent,
            "validation_percent": config.validation_percent,
            "max_token_frequency": config.max_token_frequency,
            "max_shape_frequency": config.max_shape_frequency,
            "prefilter_candidates": config.prefilter_candidates,
            "max_candidates_per_source": config.max_candidates_per_source,
            "random_seed": config.random_seed,
        }
    }
    con = connect(config)
    try:
        if args.stage in {"all", "train"}:
            model, threshold = train(config, con, metrics)
            config.output_dir.mkdir(parents=True, exist_ok=True)
            config.metadata_path.write_text(
                json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8"
            )
        else:
            metrics = read_saved_metadata(config.metadata_path)
            model = load_model(config.model_path)
            threshold = float(metrics["decision_threshold"])
        if args.stage in {"all", "infer"}:
            infer(config, con, model, threshold, metrics)
            config.metadata_path.write_text(
                json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8"
            )
    finally:
        con.close()
    log(f"Done. Outputs: {config.output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
