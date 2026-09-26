from __future__ import annotations

import duckdb


def _safe_ratio(a: str, b: str) -> str:
    return f"CASE WHEN greatest(length({a}), length({b})) = 0 THEN 1.0 ELSE 1.0 - levenshtein({a}, {b})::DOUBLE / greatest(length({a}), length({b})) END"


def _token_jaccard(a: str, b: str) -> str:
    inter = f"len(list_intersect({a}, {b}))"
    union = f"len(list_distinct(list_concat({a}, {b})))"
    return f"CASE WHEN {union} = 0 THEN 1.0 ELSE {inter}::DOUBLE / {union} END"


def create_feature_table(
    con: duckdb.DuckDBPyConnection,
    candidates: str,
    query_records: str,
    target_records: str,
    output_table: str,
    *,
    training: bool,
    append: bool,
) -> int:
    label_sql = (
        "CASE WHEN tp.target_id IS NULL THEN 0 ELSE 1 END::UTINYINT AS label,"
        if training
        else ""
    )
    truth_join = (
        "LEFT JOIN truth_pairs tp ON tp.source1_entity_id = q.entity_id AND tp.target_id = t.entity_id"
        if training
        else ""
    )
    select_sql = f"""
        WITH base AS (
            SELECT
                c.query_rid, c.target_rid, c.target_source, c.candidate_rank,
                q.entity_id AS source1_entity_id,
                t.entity_id AS target_entity_id,
                q.country,
                {label_sql}
                c.block_score, c.exact_hits, c.shape_hits,
                (q.name_norm = t.name_norm)::INTEGER AS name_exact,
                (q.core_name <> '' AND q.core_name = t.core_name)::INTEGER AS core_name_exact,
                (q.address_norm <> '' AND q.address_norm = t.address_norm)::INTEGER AS address_exact,
                jaro_winkler_similarity(q.name_norm, t.name_norm) AS name_jaro,
                jaro_winkler_similarity(q.core_name, t.core_name) AS core_name_jaro,
                CASE WHEN q.address_norm = '' OR t.address_norm = '' THEN 0.0
                     ELSE jaro_winkler_similarity(q.address_norm, t.address_norm) END AS address_jaro,
                {_safe_ratio('q.name_norm', 't.name_norm')} AS name_levenshtein,
                {_safe_ratio('q.core_name', 't.core_name')} AS core_name_levenshtein,
                CASE WHEN q.address_norm = '' OR t.address_norm = '' THEN 0.0
                     ELSE {_safe_ratio('q.address_norm', 't.address_norm')} END AS address_levenshtein,
                jaccard(q.name_norm, t.name_norm) AS name_char_jaccard,
                CASE WHEN q.address_norm = '' OR t.address_norm = '' THEN 0.0
                     ELSE jaccard(q.address_norm, t.address_norm) END AS address_char_jaccard,
                {_token_jaccard('q.name_tokens', 't.name_tokens')} AS name_token_jaccard,
                {_token_jaccard('q.address_tokens', 't.address_tokens')} AS address_token_jaccard,
                len(list_intersect(q.name_tokens, t.name_tokens))::INTEGER AS name_token_overlap,
                len(list_intersect(q.address_tokens, t.address_tokens))::INTEGER AS address_token_overlap,
                len(list_intersect(q.digit_tokens, t.digit_tokens))::INTEGER AS digit_overlap,
                {_token_jaccard('q.digit_tokens', 't.digit_tokens')} AS digit_jaccard,
                least(length(q.name_norm), length(t.name_norm))::DOUBLE /
                    greatest(1, greatest(length(q.name_norm), length(t.name_norm))) AS name_length_ratio,
                least(length(q.address_norm), length(t.address_norm))::DOUBLE /
                    greatest(1, greatest(length(q.address_norm), length(t.address_norm))) AS address_length_ratio,
                (t.address_norm = '')::INTEGER AS target_address_missing
            FROM {candidates} c
            JOIN {query_records} q ON q.rid = c.query_rid
            JOIN {target_records} t ON t.rid = c.target_rid
            {truth_join}
        )
        SELECT *,
            greatest(name_jaro, core_name_jaro, address_jaro) AS max_field_jaro,
            name_jaro + address_jaro AS joint_jaro
        FROM base
    """
    if append:
        con.execute(f"INSERT INTO {output_table} BY NAME {select_sql}")
    else:
        con.execute(f"DROP TABLE IF EXISTS {output_table}")
        con.execute(f"CREATE TABLE {output_table} AS {select_sql}")
    return con.execute(f"SELECT count(*) FROM {output_table}").fetchone()[0]
