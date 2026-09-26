"""Measure how often cheap blocking signatures survive the injected noise.

This is an offline diagnostic, not part of inference.  It samples reference
entities from the training labels, joins their true S2/S3 records, and reports
the proportion of positives recoverable by exact token and character signatures.
"""

from __future__ import annotations

import argparse
import re
import unicodedata
from collections import Counter
from itertools import combinations
from pathlib import Path

import polars as pl


LEGAL = {
    "and", "the", "inc", "incorporated", "llc", "llp", "ltd", "limited",
    "private", "pvt", "plc", "corp", "corporation", "company", "co",
    "services", "service", "group", "enterprises", "enterprise", "center",
}
ADDRESS = {
    "road", "rd", "street", "st", "avenue", "ave", "lane", "ln", "drive",
    "dr", "floor", "fl", "unit", "suite", "block", "near", "box", "po",
    "india", "usa", "united", "states",
}


def norm(value: str | None) -> str:
    value = unicodedata.normalize("NFKD", value or "").casefold()
    value = "".join(c for c in value if not unicodedata.combining(c))
    value = re.sub(r"https?://|www\.", " ", value)
    return " ".join(re.findall(r"[^\W_]+", value, flags=re.UNICODE))


def tokens(value: str | None, stop: set[str]) -> set[str]:
    return {x for x in norm(value).split() if len(x) >= 3 and x not in stop}


def grams(value: str | None, n: int = 5) -> set[str]:
    text = norm(value).replace(" ", "")
    return {text[i : i + n] for i in range(max(0, len(text) - n + 1))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("student_resource/dataset/train"))
    parser.add_argument("--sample", type=int, default=50_000)
    args = parser.parse_args()

    gt = (
        pl.read_csv(args.root / "train_ground_truth.tsv", separator="\t", quote_char=None)
        .head(args.sample)
        .with_columns(pl.col("matched_entity_ids").fill_null("").str.split(","))
        .explode("matched_entity_ids")
        .filter(pl.col("matched_entity_ids") != "")
        .rename({"source1_entity_id": "s1_id", "matched_entity_ids": "target_id"})
    )
    s1_ids = gt.select("s1_id").unique()
    s1 = (
        pl.scan_csv(args.root / "train_source1.tsv", separator="\t", quote_char=None)
        .join(s1_ids.lazy(), left_on="entity_id", right_on="s1_id")
        .collect()
        .rename({c: f"a_{c}" for c in ("entity_id", "business_name", "business_address", "country")})
    )
    parts = []
    for source in (2, 3):
        ids = gt.filter(pl.col("target_id").str.starts_with(f"S{source}-")).select("target_id")
        part = (
            pl.scan_csv(args.root / f"train_source{source}.tsv", separator="\t", quote_char=None)
            .join(ids.lazy(), left_on="entity_id", right_on="target_id")
            .collect()
        )
        parts.append(part)
    target = pl.concat(parts).rename(
        {c: f"b_{c}" for c in ("entity_id", "business_name", "business_address", "country")}
    )
    pairs = gt.join(s1, left_on="s1_id", right_on="a_entity_id").join(
        target, left_on="target_id", right_on="b_entity_id"
    )

    counts = {k: 0 for k in ["name_token", "address_token", "any_token", "name_5gram", "address_5gram", "any_5gram", "neither_5gram"]}
    by_missing = {"address_missing": [0, 0], "address_present": [0, 0]}
    shared_counts = Counter()
    pair_hits = Counter()
    for row in pairs.iter_rows(named=True):
        nt = bool(tokens(row["a_business_name"], LEGAL) & tokens(row["b_business_name"], LEGAL))
        at = bool(tokens(row["a_business_address"], ADDRESS) & tokens(row["b_business_address"], ADDRESS))
        ng = bool(grams(row["a_business_name"]) & grams(row["b_business_name"]))
        ag = bool(grams(row["a_business_address"]) & grams(row["b_business_address"]))
        counts["name_token"] += nt
        counts["address_token"] += at
        counts["any_token"] += nt or at
        counts["name_5gram"] += ng
        counts["address_5gram"] += ag
        counts["any_5gram"] += ng or ag
        counts["neither_5gram"] += not (ng or ag)
        bucket = "address_missing" if not row["b_business_address"] else "address_present"
        by_missing[bucket][0] += 1
        by_missing[bucket][1] += ng or ag
        shared_counts[len((tokens(row["a_business_name"], LEGAL) | tokens(row["a_business_address"], ADDRESS)) & (tokens(row["b_business_name"], LEGAL) | tokens(row["b_business_address"], ADDRESS)))] += 1
        def typed(record_prefix: str, cap: int) -> set[tuple[str, str]]:
            values = (["ne:" + x for x in tokens(row[f"{record_prefix}_business_name"], LEGAL)] +
                      ["ae:" + x for x in tokens(row[f"{record_prefix}_business_address"], ADDRESS)])
            values = sorted(values, key=lambda x: (-len(x.split(":", 1)[1]), x))[:cap]
            return set(combinations(values, 2))
        for cap in (6, 8, 10, 12):
            pair_hits[cap] += bool(typed("a", cap) & typed("b", cap))
    n = pairs.height
    print(f"positive pairs: {n:,}")
    for key, value in counts.items():
        print(f"{key:18s} {value:9,d}  {value / n:.6%}")
    print(by_missing)
    print("shared token count", dict(sorted(shared_counts.items())))
    print("selected pair hit", {k: (v, v / n) for k, v in sorted(pair_hits.items())})


if __name__ == "__main__":
    main()
