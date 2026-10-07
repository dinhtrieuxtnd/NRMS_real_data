from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


INPUT_KEYS = (
    "news",
    "train_behaviors",
    "validation_behaviors",
    "test_behaviors",
    "word_vectors",
)
# Legacy key name accepted for backward compatibility with old configs.
_LEGACY_INPUT_ALIASES = {"glove": "word_vectors"}


def load_config(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError(f"Config {path} must contain a YAML mapping")
    inputs = config.get("input")
    if isinstance(inputs, dict):
        for legacy, current in _LEGACY_INPUT_ALIASES.items():
            if legacy in inputs and current not in inputs:
                inputs[current] = inputs.pop(legacy)
    validate_config(config)
    return config


def _mapping(config: dict[str, Any], key: str, errors: list[str]) -> dict[str, Any]:
    value = config.get(key)
    if not isinstance(value, dict):
        errors.append(f"{key} must be a mapping")
        return {}
    return value


def _positive_int(section: dict[str, Any], key: str, prefix: str, errors: list[str]) -> None:
    value = section.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        errors.append(f"{prefix}.{key} must be a positive integer")


def validate_config(config: dict[str, Any]) -> None:
    errors: list[str] = []
    preprocessing = _mapping(config, "preprocessing", errors)
    inputs = _mapping(config, "input", errors)
    output = _mapping(config, "output", errors)
    text = _mapping(config, "text", errors)
    sequence = _mapping(config, "sequence", errors)
    training = _mapping(config, "training", errors)
    embedding = _mapping(config, "embedding", errors)
    logging_config = _mapping(config, "logging", errors)

    if not isinstance(preprocessing.get("name"), str) or not preprocessing.get("name"):
        errors.append("preprocessing.name must be a non-empty string")
    if not isinstance(preprocessing.get("seed"), int) or isinstance(
        preprocessing.get("seed"), bool
    ):
        errors.append("preprocessing.seed must be an integer")

    for key in INPUT_KEYS:
        if not isinstance(inputs.get(key), str) or not inputs.get(key):
            errors.append(f"input.{key} must be a non-empty path string")

    if not isinstance(output.get("directory"), str) or not output.get("directory"):
        errors.append("output.directory must be a non-empty path string")
    if not isinstance(output.get("overwrite"), bool):
        errors.append("output.overwrite must be a boolean")

    if not isinstance(text.get("lowercase"), bool):
        errors.append("text.lowercase must be a boolean")
    _positive_int(text, "min_frequency", "text", errors)
    _positive_int(text, "max_title_length", "text", errors)
    _positive_int(sequence, "max_history_length", "sequence", errors)
    if not isinstance(sequence.get("deduplicate_history", True), bool):
        errors.append("sequence.deduplicate_history must be a boolean")
    if not isinstance(sequence.get("exclude_history_from_candidates", True), bool):
        errors.append("sequence.exclude_history_from_candidates must be a boolean")
    _positive_int(training, "negative_sampling_ratio", "training", errors)
    _positive_int(embedding, "dimension", "embedding", errors)
    if embedding.get("dimension") != 300:
        errors.append("embedding.dimension must be 300 for NRMS word embeddings")

    init_std = embedding.get("unmatched_init_std")
    if not isinstance(init_std, (int, float)) or isinstance(init_std, bool) or init_std <= 0:
        errors.append("embedding.unmatched_init_std must be positive")
    if str(logging_config.get("level", "")).upper() not in {
        "DEBUG",
        "INFO",
        "WARNING",
        "ERROR",
        "CRITICAL",
    }:
        errors.append("logging.level is invalid")

    if errors:
        details = "\n".join(f"  - {error}" for error in errors)
        raise ValueError(f"Invalid preprocessing config:\n{details}")


def resolve_config_paths(
    config: dict[str, Any], project_root: str | Path
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    resolved = deepcopy(config)
    for key in INPUT_KEYS:
        path = Path(resolved["input"][key])
        resolved["input"][key] = str(path if path.is_absolute() else root / path)

    output_path = Path(resolved["output"]["directory"])
    resolved["output"]["directory"] = str(
        output_path if output_path.is_absolute() else root / output_path
    )
    return resolved


def validate_input_paths(input_config: dict[str, str]) -> dict[str, Path]:
    paths = {key: Path(input_config[key]) for key in INPUT_KEYS}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        details = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(f"Missing input files:\n{details}")
    return paths


def save_config(config: dict[str, Any], output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(config, file, sort_keys=False, allow_unicode=True)