from __future__ import annotations

import re
from pathlib import Path

import duckdb

from config import PipelineConfig


LEGAL_WORDS = (
    "inc|incorporated|llc|llp|ltd|limited|private|pvt|plc|corp|corporation|"
    "company|co|gmbh|sa|sas|sarl|bv|ag"
)


def sql_path(path: Path) -> str:
    return str(path.resolve()).replace("'", "''").replace("\\", "/")


def connect(config: PipelineConfig) -> duckdb.DuckDBPyConnection:
    config.work_dir.mkdir(parents=True, exist_ok=True)
    temp_dir = config.work_dir / "duckdb_tmp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(config.db_path))
    con.execute(f"SET threads={max(1, config.threads)}")
    con.execute(f"SET memory_limit='{config.memory_limit}'")
    con.execute(f"SET temp_directory='{sql_path(temp_dir)}'")
    con.execute("SET preserve_insertion_order=false")
    return con


def _record_select(path: Path, limit: int | None = None) -> str:
    limit_sql = f"LIMIT {int(limit)}" if limit else ""
    normalized = f"""
        SELECT
            entity_id,
            coalesce(business_name, '') AS business_name,
            coalesce(business_address, '') AS business_address,
            trim(coalesce(country, '')) AS country,
            trim(regexp_replace(
                strip_accents(lower(coalesce(business_name, ''))),
                '[^\\p{{L}}\\p{{N}}]+', ' ', 'g'
            )) AS name_norm,
            trim(regexp_replace(
                strip_accents(lower(coalesce(business_address, ''))),
                '[^\\p{{L}}\\p{{N}}]+', ' ', 'g'
            )) AS address_norm
        FROM read_csv(
            '{sql_path(path)}', delim='\\t', header=true, all_varchar=true,
            quote='', strict_mode=false, null_padding=true
        )
        {limit_sql}
    """
    return f"""
        WITH normalized AS ({normalized}),
        canonical AS (
            SELECT *,
                trim(regexp_replace(regexp_replace(
                    ' ' || name_norm || ' ',
                    ' ({LEGAL_WORDS}) ', ' ', 'g'
                ), ' ({LEGAL_WORDS}) ', ' ', 'g')) AS core_name
            FROM normalized
        ), tokenized AS (
            SELECT *,
                list_distinct(list_filter(str_split(name_norm, ' '), x -> length(x) >= 2)) AS name_tokens,
                list_distinct(list_filter(str_split(address_norm, ' '), x -> length(x) >= 2)) AS address_tokens,
                list_distinct(regexp_extract_all(name_norm || ' ' || address_norm, '[0-9]+')) AS digit_tokens
            FROM canonical
        )
        SELECT
            row_number() OVER () - 1 AS rid,
            entity_id, business_name, business_address, country,
            name_norm, core_name, address_norm,
            name_tokens, address_tokens, digit_tokens,
            array_to_string(list_transform(name_tokens, x -> left(x, 1)), '') AS name_acronym
        FROM tokenized
    """


def load_records(
    con: duckdb.DuckDBPyConnection,
    table: str,
    path: Path,
    *,
    limit: int | None = None,
) -> int:
    if not re.fullmatch(r"[a-z][a-z0-9_]*", table):
        raise ValueError(f"Unsafe table name: {table}")
    con.execute(f"DROP TABLE IF EXISTS {table}")
    con.execute(f"CREATE TABLE {table} AS {_record_select(path, limit)}")
    return con.execute(f"SELECT count(*) FROM {table}").fetchone()[0]


def load_ground_truth(con: duckdb.DuckDBPyConnection, path: Path) -> None:
    con.execute("DROP TABLE IF EXISTS truth_raw")
    con.execute(
        f"""
        CREATE TABLE truth_raw AS
        SELECT
            source1_entity_id,
            coalesce(matched_entity_ids, '') AS matched_entity_ids
        FROM read_csv(
            '{sql_path(path)}', delim='\\t', header=true, all_varchar=true,
            quote='', strict_mode=false, null_padding=true
        )
        """
    )
    con.execute("DROP TABLE IF EXISTS truth_pairs")
    con.execute(
        """
        CREATE TABLE truth_pairs AS
        SELECT source1_entity_id, target_id
        FROM truth_raw,
        unnest(str_split(matched_entity_ids, ',')) AS u(target_id)
        WHERE target_id <> ''
        """
    )
    con.execute("CREATE INDEX IF NOT EXISTS truth_pair_idx ON truth_pairs(source1_entity_id, target_id)")


def table_exists(con: duckdb.DuckDBPyConnection, table: str) -> bool:
    return bool(
        con.execute(
            "SELECT count(*) FROM information_schema.tables WHERE table_name = ?", [table]
        ).fetchone()[0]
    )
