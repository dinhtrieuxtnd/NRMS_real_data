from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from src.data.text import PAD_NEWS, PAD_TOKEN, UNK_TOKEN
from src.utils.io import load_pickle, sha256_file


REQUIRED_ARTIFACTS = {
    "train_samples.pkl",
    "validation_samples.pkl",
    "test_samples.pkl",
    "news_title_mapping.pkl",
    "word_dict.pkl",
    "word_embedding_matrix.pkl",
}


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        value = json.load(file)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _validate_manifest_files(data_dir: Path, manifest: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    artifacts = manifest.get("artifacts", {})
    missing_entries = REQUIRED_ARTIFACTS - set(artifacts)
    if missing_entries:
        errors.append(f"manifest is missing artifacts: {sorted(missing_entries)}")
    for filename in REQUIRED_ARTIFACTS & set(artifacts):
        metadata = artifacts[filename]
        path = data_dir / metadata.get("path", "")
        if not path.is_file():
            errors.append(f"artifact is missing: {path}")
            continue
        if path.stat().st_size != metadata.get("size_bytes"):
            errors.append(f"artifact size mismatch: {filename}")
        if sha256_file(path) != metadata.get("sha256"):
            errors.append(f"artifact checksum mismatch: {filename}")
    return errors


def _validate_train_records(
    records: Any, known_news: set[str], max_history_length: int
) -> list[str]:
    errors: list[str] = []
    if not isinstance(records, list):
        return ["train_samples.pkl must contain a list"]
    required = {"impression_id", "user_id", "time", "history", "positive", "negative_pool"}
    for index, record in enumerate(records):
        if not isinstance(record, dict) or not required <= set(record):
            errors.append(f"train record {index} has an invalid schema")
            continue
        history = record["history"]
        negatives = record["negative_pool"]
        if not history or len(history) > max_history_length:
            errors.append(f"train record {index} has an invalid history length")
        if not negatives:
            errors.append(f"train record {index} has an empty negative pool")
        references = set(history) | set(negatives) | {record["positive"]}
        if unknown := references - known_news:
            errors.append(f"train record {index} references unknown news: {sorted(unknown)}")
        if record["positive"] in negatives:
            errors.append(f"train record {index} contains its positive in negative_pool")
        try:
            datetime.fromisoformat(record["time"])
        except (TypeError, ValueError):
            errors.append(f"train record {index} has an invalid time")
        if len(errors) >= 100:
            break
    return errors


def _validate_evaluation_records(
    records: Any, known_news: set[str], max_history_length: int, split: str
) -> list[str]:
    errors: list[str] = []
    if not isinstance(records, list):
        return [f"{split}_samples.pkl must contain a list"]
    required = {"impression_id", "user_id", "time", "history", "candidates", "labels"}
    for index, record in enumerate(records):
        if not isinstance(record, dict) or not required <= set(record):
            errors.append(f"{split} record {index} has an invalid schema")
            continue
        history = record["history"]
        candidates = record["candidates"]
        labels = record["labels"]
        if not history or len(history) > max_history_length:
            errors.append(f"{split} record {index} has an invalid history length")
        if not candidates or len(candidates) != len(labels):
            errors.append(f"{split} record {index} has misaligned candidates and labels")
        if any(label not in {0, 1} for label in labels):
            errors.append(f"{split} record {index} has an invalid label")
        if unknown := (set(history) | set(candidates)) - known_news:
            errors.append(f"{split} record {index} references unknown news: {sorted(unknown)}")
        try:
            datetime.fromisoformat(record["time"])
        except (TypeError, ValueError):
            errors.append(f"{split} record {index} has an invalid time")
        if len(errors) >= 100:
            break
    return errors


def validate_processed(data_dir: str | Path) -> list[str]:
    root = Path(data_dir)
    errors: list[str] = []
    for filename in ("config.yaml", "manifest.json", "run_info.json", "statistics.json"):
        if not (root / filename).is_file():
            errors.append(f"missing metadata file: {filename}")
    if errors:
        return errors

    manifest = _load_json(root / "manifest.json")
    if manifest.get("schema_version") != "1.0":
        errors.append(f"unsupported schema version: {manifest.get('schema_version')!r}")
    errors.extend(_validate_manifest_files(root, manifest))
    if errors:
        return errors

    with (root / "config.yaml").open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    artifacts_dir = root / "artifacts"
    word_dict = load_pickle(artifacts_dir / "word_dict.pkl")
    news_mapping = load_pickle(artifacts_dir / "news_title_mapping.pkl")
    embedding_matrix = load_pickle(artifacts_dir / "word_embedding_matrix.pkl")

    if not isinstance(word_dict, dict):
        errors.append("word_dict.pkl must contain a mapping")
    elif word_dict.get(PAD_TOKEN) != 0 or word_dict.get(UNK_TOKEN) != 1:
        errors.append("word_dict must assign PAD=0 and UNK=1")

    expected_title_length = config["text"]["max_title_length"]
    if not isinstance(news_mapping, dict) or PAD_NEWS not in news_mapping:
        errors.append("news_title_mapping must contain PAD_NEWS")
        known_news: set[str] = set()
    else:
        known_news = set(news_mapping) - {PAD_NEWS}
        for news_id, title in news_mapping.items():
            if not isinstance(title, np.ndarray) or title.dtype != np.int32:
                errors.append(f"title mapping {news_id} must be an int32 ndarray")
                break
            if title.shape != (expected_title_length,):
                errors.append(f"title mapping {news_id} has shape {title.shape}")
                break
        if not np.count_nonzero(news_mapping[PAD_NEWS]) == 0:
            errors.append("PAD_NEWS title must contain only zeros")

    expected_embedding_shape = (len(word_dict), config["embedding"]["dimension"])
    if not isinstance(embedding_matrix, np.ndarray):
        errors.append("word_embedding_matrix.pkl must contain an ndarray")
    else:
        if embedding_matrix.dtype != np.float32:
            errors.append("embedding matrix must use float32")
        if embedding_matrix.shape != expected_embedding_shape:
            errors.append(
                f"embedding matrix shape is {embedding_matrix.shape}, expected {expected_embedding_shape}"
            )
        if np.count_nonzero(embedding_matrix[0]) != 0:
            errors.append("embedding PAD row must contain only zeros")
        if not np.isfinite(embedding_matrix).all():
            errors.append("embedding matrix contains non-finite values")

    max_history_length = config["sequence"]["max_history_length"]
    train_records = load_pickle(artifacts_dir / "train_samples.pkl")
    validation_records = load_pickle(artifacts_dir / "validation_samples.pkl")
    test_records = load_pickle(artifacts_dir / "test_samples.pkl")
    errors.extend(_validate_train_records(train_records, known_news, max_history_length))
    errors.extend(
        _validate_evaluation_records(
            validation_records, known_news, max_history_length, "validation"
        )
    )
    errors.extend(
        _validate_evaluation_records(test_records, known_news, max_history_length, "test")
    )

    validation_times = [datetime.fromisoformat(record["time"]) for record in validation_records]
    test_times = [datetime.fromisoformat(record["time"]) for record in test_records]
    if validation_times and test_times and max(validation_times) > min(test_times):
        errors.append("validation/test chronology overlaps")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate processed NRMS artifacts")
    parser.add_argument("--data-dir", type=Path, required=True)
    args = parser.parse_args()
    errors = validate_processed(args.data_dir)
    if errors:
        print("Processed data validation failed:")
        for error in errors:
            print(f"  - {error}")
        raise SystemExit(1)
    print("Processed data validation passed")


if __name__ == "__main__":
    main()