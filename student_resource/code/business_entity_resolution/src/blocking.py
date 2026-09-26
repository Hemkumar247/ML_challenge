from __future__ import annotations

import math

import duckdb

from config import PipelineConfig


NAME_STOP = {
    "and", "the", "inc", "incorporated", "llc", "llp", "ltd", "limited",
    "private", "pvt", "plc", "corp", "corporation", "company", "co", "gmbh",
    "sa", "sas", "sarl", "services", "service", "group", "enterprises",
    "enterprise", "center", "centre",
}
ADDRESS_STOP = {
    "road", "rd", "street", "st", "avenue", "ave", "lane", "ln", "drive",
    "dr", "court", "ct", "boulevard", "blvd", "floor", "fl", "unit", "suite",
    "block", "near", "box", "po", "india", "usa", "united", "states", "null",
}


def _quoted(items: set[str]) -> str:
    return ",".join(f"'{x}'" for x in sorted(items))


def create_tokens(con: duckdb.DuckDBPyConnection, records: str, tokens: str) -> int:
    """Create exact, composite, and typo-tolerant blocking signatures.

    Individual Faker tokens can occur thousands of times. A pair of tokens that
    survives in two noisy versions is much more discriminative. To keep this
    bounded, only the twelve longest deterministic tokens are paired (at most 66
    composite keys per record).
    """
    con.execute(f"DROP TABLE IF EXISTS {tokens}")
    con.execute(
        f"CREATE TABLE {tokens} (rid BIGINT, country VARCHAR, kind VARCHAR, signature UBIGINT)"
    )
    row_count = con.execute(f"SELECT count(*) FROM {records}").fetchone()[0]
    shards = max(1, min(12, math.ceil(row_count / 600_000)))
    for shard in range(shards):
        con.execute(
            f"""
        INSERT INTO {tokens}
        WITH exact_tokens AS (
            SELECT DISTINCT rid, country, 'ne'::VARCHAR AS kind, tok AS token
            FROM {records}, unnest(name_tokens) AS u(tok)
            WHERE mod(rid, {shards}) = {shard}
              AND length(tok) >= 3 AND tok NOT IN ({_quoted(NAME_STOP)})
            UNION ALL
            SELECT DISTINCT rid, country, 'ae'::VARCHAR AS kind, tok AS token
            FROM {records}, unnest(address_tokens) AS u(tok)
            WHERE mod(rid, {shards}) = {shard}
              AND length(tok) >= 3 AND tok NOT IN ({_quoted(ADDRESS_STOP)})
        ), token_lists AS (
            SELECT rid, country,
                list(kind || ':' || token ORDER BY length(token) DESC, kind, token)[1:12] AS items
            FROM exact_tokens
            GROUP BY rid, country
        ), pairs AS (
            SELECT rid, country, 'pe'::VARCHAR AS kind,
                hash(a_value || '|' || b_value) AS signature
            FROM token_lists,
                 unnest(items) WITH ORDINALITY AS a(a_value, a_pos),
                 unnest(items) WITH ORDINALITY AS b(b_value, b_pos)
            WHERE a_pos < b_pos
        ), shapes AS (
            SELECT DISTINCT rid, country,
                CASE kind WHEN 'ne' THEN 'ns' ELSE 'as' END AS kind,
                hash(CASE kind WHEN 'ne' THEN 'ns:' ELSE 'as:' END ||
                    left(token, 2) || right(token, 2) || ':' ||
                    cast(floor(length(token) / 3) AS VARCHAR)) AS signature
            FROM exact_tokens
            WHERE length(token) >= 6 AND NOT regexp_matches(token, '^[0-9]+$')
        )
        SELECT rid, country, kind, hash(kind || ':' || token) AS signature FROM exact_tokens
        UNION ALL
        SELECT * FROM shapes
        UNION ALL
        SELECT * FROM pairs
        """
        )
    return con.execute(f"SELECT count(*) FROM {tokens}").fetchone()[0]


def create_candidates(
    con: duckdb.DuckDBPyConnection,
    query_records: str,
    query_tokens: str,
    target_records: str,
    target_tokens: str,
    output_table: str,
    source_number: int,
    config: PipelineConfig,
    *,
    query_filter: str = "TRUE",
    append: bool = False,
) -> dict[str, float]:
    """Retrieve, cheaply rerank, and cap candidates per query entity."""
    stats = f"{target_tokens}_stats"
    rare = f"{target_tokens}_rare"
    con.execute(f"DROP TABLE IF EXISTS {stats}")
    con.execute(
        f"""
        CREATE TABLE {stats} AS
        SELECT country, kind, signature, count(*)::INTEGER AS document_frequency
        FROM {target_tokens}
        GROUP BY ALL
        """
    )
    con.execute(f"DROP TABLE IF EXISTS {rare}")
    con.execute(
        f"""
        CREATE TABLE {rare} AS
        SELECT t.rid, t.country, t.kind, t.signature, s.document_frequency,
               ln((n.country_count + 1.0) / (s.document_frequency + 0.5)) *
               CASE t.kind WHEN 'pe' THEN 1.65 WHEN 'ne' THEN 1.20 WHEN 'ae' THEN 1.00
                           WHEN 'ns' THEN 0.42 ELSE 0.32 END AS weight
        FROM {target_tokens} t
        JOIN {stats} s USING (country, kind, signature)
        JOIN (
            SELECT country, count(*) AS country_count
            FROM {target_records} GROUP BY country
        ) n USING (country)
        WHERE (t.kind IN ('ne', 'ae', 'pe') AND s.document_frequency <= {config.max_token_frequency})
           OR (right(t.kind, 1) = 's' AND s.document_frequency <= {config.max_shape_frequency})
        """
    )
    raw = f"{output_table}_raw_{source_number}"
    shortlist = f"{output_table}_short_{source_number}"
    final = f"{output_table}_part_{source_number}"
    for table in (raw, shortlist, final):
        con.execute(f"DROP TABLE IF EXISTS {table}")
    con.execute(
        f"""
        CREATE TABLE {raw} AS
        SELECT q.rid AS query_rid, t.rid AS target_rid,
               sum(t.weight) AS block_score,
               count(*) FILTER (q.kind IN ('ne', 'ae', 'pe'))::INTEGER AS exact_hits,
               count(*) FILTER (right(q.kind, 1) = 's')::INTEGER AS shape_hits
        FROM {query_tokens} q
        JOIN {query_records} qr ON qr.rid = q.rid
        JOIN {rare} t
          ON t.country = q.country AND t.kind = q.kind AND t.signature = q.signature
        WHERE {query_filter}
        GROUP BY q.rid, t.rid
        """
    )
    con.execute(
        f"""
        CREATE TABLE {shortlist} AS
        SELECT * EXCLUDE(pre_rank)
        FROM (
            SELECT *, row_number() OVER (
                PARTITION BY query_rid
                ORDER BY block_score DESC, exact_hits DESC, target_rid
            ) AS pre_rank
            FROM {raw}
        )
        WHERE pre_rank <= {config.prefilter_candidates}
        """
    )
    con.execute(
        f"""
        CREATE TABLE {final} AS
        WITH rescored AS (
            SELECT c.*,
                jaro_winkler_similarity(q.name_norm, t.name_norm) AS name_jaro,
                CASE WHEN q.address_norm = '' OR t.address_norm = '' THEN 0.0
                     ELSE jaro_winkler_similarity(q.address_norm, t.address_norm) END AS address_jaro
            FROM {shortlist} c
            JOIN {query_records} q ON q.rid = c.query_rid
            JOIN {target_records} t ON t.rid = c.target_rid
        ), ranked AS (
            SELECT *,
                row_number() OVER (
                    PARTITION BY query_rid
                    ORDER BY block_score
                         + 7.0 * greatest(name_jaro, address_jaro)
                         + 2.0 * (name_jaro + address_jaro) DESC,
                         exact_hits DESC, target_rid
                ) AS candidate_rank
            FROM rescored
        )
        SELECT query_rid, target_rid, {source_number}::UTINYINT AS target_source,
               block_score, exact_hits, shape_hits, candidate_rank::UTINYINT AS candidate_rank
        FROM ranked
        WHERE candidate_rank <= {config.max_candidates_per_source}
        """
    )
    if not append:
        con.execute(f"DROP TABLE IF EXISTS {output_table}")
        con.execute(f"CREATE TABLE {output_table} AS SELECT * FROM {final}")
    else:
        con.execute(f"INSERT INTO {output_table} SELECT * FROM {final}")

    n_query = con.execute(
        f"SELECT count(*) FROM {query_records} qr WHERE {query_filter}"
    ).fetchone()[0]
    n_candidates, covered = con.execute(
        f"SELECT count(*), count(DISTINCT query_rid) FROM {final}"
    ).fetchone()
    return {
        "queries": int(n_query),
        "candidates": int(n_candidates),
        "queries_with_candidates": int(covered),
        "average_candidates": float(n_candidates / n_query) if n_query else 0.0,
    }


def drop_blocking_intermediates(
    con: duckdb.DuckDBPyConnection, target_tokens: str, output_table: str, source_number: int
) -> None:
    for table in (
        target_tokens,
        f"{target_tokens}_stats",
        f"{target_tokens}_rare",
        f"{output_table}_raw_{source_number}",
        f"{output_table}_short_{source_number}",
        f"{output_table}_part_{source_number}",
    ):
        con.execute(f"DROP TABLE IF EXISTS {table}")
