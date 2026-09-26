from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PipelineConfig:
    resource_root: Path
    work_dir: Path
    output_dir: Path
    threads: int = 8
    memory_limit: str = "6GB"
    train_percent: int = 8
    validation_percent: int = 2
    max_token_frequency: int = 96
    max_shape_frequency: int = 24
    prefilter_candidates: int = 40
    max_candidates_per_source: int = 10
    random_seed: int = 247

    @property
    def train_dir(self) -> Path:
        return self.resource_root / "dataset" / "train"

    @property
    def test_dir(self) -> Path:
        return self.resource_root / "dataset" / "test"

    @property
    def db_path(self) -> Path:
        return self.work_dir / "entity_resolution.duckdb"

    @property
    def model_path(self) -> Path:
        return self.work_dir / "xgb_matcher.json"

    @property
    def metadata_path(self) -> Path:
        return self.output_dir / "run_metrics.json"


FEATURE_COLUMNS = [
    "block_score",
    "exact_hits",
    "shape_hits",
    "name_exact",
    "core_name_exact",
    "address_exact",
    "name_jaro",
    "core_name_jaro",
    "address_jaro",
    "name_levenshtein",
    "core_name_levenshtein",
    "address_levenshtein",
    "name_char_jaccard",
    "address_char_jaccard",
    "name_token_jaccard",
    "address_token_jaccard",
    "name_token_overlap",
    "address_token_overlap",
    "digit_overlap",
    "digit_jaccard",
    "name_length_ratio",
    "address_length_ratio",
    "target_address_missing",
    "max_field_jaro",
    "joint_jaro",
]
