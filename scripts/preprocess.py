from __future__ import annotations

import argparse
import importlib.metadata
import logging
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from src.data.embedding import build_embedding_matrix
from src.data.parser import read_behaviors, read_news
from src.data.preprocessing import (
    build_evaluation_samples,
    build_train_samples,
    time_range,
)
from src.data.text import (
    build_news_title_mapping,
    build_word_dict,
    merge_news,
    tokenize,
)
from src.utils.config import (
    load_config,
    resolve_config_paths,
    validate_input_paths
)
from src.utils.io import (
    file_descriptor,
    prepare_output_directory,
    save_json,
    save_pickle,
    save_yaml,
)
from src.utils.logging import close_logger, setup_logger


ARTIFACT_SCHEMA_VERSION = "1.0"


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _describe_lengths(values: list[int]) -> dict[str, float | int]:
    if not values:
        return {
            "count": 0,
            "min": 0,
            "max": 0,
            "mean": 0.0,
            "p95": 0.0
        }
    array = np.asarray(values)
    return {
        "count": len(values),
        "min": int(array.min()),
        "max": int(array.max()),
        "mean": float(array.mean()),
        "p95": float(np.percentile(array, 95)),
    }


def _package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for package in ("numpy", "pandas", "PyYAML", "torch", "tqdm"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


def _artifact_metadata(
    path: Path,
    data: Any,
    output_dir: Path
) -> dict[str, Any]:
    descriptor = file_descriptor(path)
    descriptor["path"] = str(
        path.relative_to(output_dir)
    ).replace("\\", "/")
    if isinstance(data, np.ndarray):
        descriptor.update(
            {
                "kind": "ndarray",
                "shape": list(data.shape),
                "dtype": str(data.dtype)
            }
        )
    elif isinstance(data, dict):
        descriptor.update({"kind": "mapping", "count": len(data)})
        first_value = next(iter(data.values()), None)
        if isinstance(first_value, np.ndarray):
            descriptor.update(
                {
                    "value_shape": list(first_value.shape),
                    "value_dtype": str(first_value.dtype)
                }
            )
    elif isinstance(data, list):
        descriptor.update({"kind": "records", "count": len(data)})
    return descriptor


def _log_skip_counters(
    logger: logging.Logger,
    split: str,
    counters: dict[str, int]
) -> None:
    skipped = {
        key: value
        for key, value in counters.items()
        if key not in {
            "input_impressions", "output_samples"
        } and value
    }
    if skipped:
        logger.warning("%s skipped impressions: %s", split, skipped)


def run(
    config_path: str | Path,
    *,
    project_root: str | Path | None = None,
    overwrite: bool | None = None,
) -> Path:
    root = Path(
        project_root
    ) if project_root else Path(
        __file__
    ).resolve().parents[1]
    source_config = load_config(config_path)
    config = resolve_config_paths(source_config, root)
    if overwrite is not None:
        config["output"]["overwrite"] = overwrite
    output_dir = prepare_output_directory(
        config["output"]["directory"],
        overwrite=config["output"]["overwrite"]
    )
    logger = setup_logger(
        output_dir / "preprocess.log",
        level=config["logging"]["level"]
    )
    save_yaml(config, output_dir / "config.yaml")

    start_time = _utc_now()
    run_info: dict[str, Any] = {
        "preprocess_name": config["preprocessing"]["name"],
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "status": "running",
        "start_time": start_time.isoformat(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "seed": config["preprocessing"]["seed"],
    }
    save_json(run_info, output_dir / "run_info.json")
    logger.info("Starting preprocessing: %s",
                run_info["preprocess_name"])
    logger.info("Config: %s", Path(config_path).resolve())
    logger.info("Output directory: %s", output_dir)

    try:
        input_paths = validate_input_paths(config["input"])
        for name, path in input_paths.items():
            logger.info("Input %s: %s", name, path)

        train_behaviors = read_behaviors(
            input_paths["train_behaviors"], split_name="train"
        )
        validation_behaviors = read_behaviors(
            input_paths["validation_behaviors"], split_name="validation"
        )
        test_behaviors = read_behaviors(
            input_paths["test_behaviors"], split_name="test"
        )
        train_news = read_news(input_paths["train_news"])
        validation_news = read_news(input_paths["validation_news"])
        test_news = read_news(input_paths["test_news"])
        logger.info(
            "Loaded rows: "             "train_behaviors=%d "             "validation_behaviors=%d "             "test_behaviors=%d "             "train_news=%d "             "validation_news=%d "             "test_news=%d",
            len(train_behaviors),
            len(validation_behaviors),
            len(test_behaviors),
            len(train_news),
            len(validation_news),
            len(test_news),
        )

        text_config = config["text"]
        word_dict = build_word_dict(
            train_news["title"],
            lowercase=text_config["lowercase"],
            min_frequency=text_config["min_frequency"],
        )
        combined_news = merge_news(train_news, validation_news, test_news)
        news_title_mapping = build_news_title_mapping(
            combined_news,
            word_dict,
            max_title_length=text_config["max_title_length"],
            lowercase=text_config["lowercase"],
        )
        logger.info(
            "Built vocabulary and title mapping: " \
            "vocabulary=%d " \
            "news=%d",
            len(word_dict),
            len(news_title_mapping) - 1,
        )

        embedding_config = config["embedding"]
        embedding_matrix, embedding_statistics = build_embedding_matrix(
            input_paths["word_vectors"],
            word_dict,
            dimension=embedding_config["dimension"],
            seed=config["preprocessing"]["seed"],
            unmatched_init_std=embedding_config["unmatched_init_std"],
        )
        logger.info(
            "Loaded word vectors (%s): matched=%d (exact=%d, case_fallback=%d) "
            "oov=%d coverage=%.4f",
            embedding_statistics["format"],
            embedding_statistics["matched_tokens"],
            embedding_statistics["exact_matched_tokens"],
            embedding_statistics["case_fallback_matched_tokens"],
            embedding_statistics["oov_tokens"],
            embedding_statistics["coverage"],
        )

        split_metadata = {
            "train": time_range(train_behaviors),
            "validation": time_range(validation_behaviors),
            "test": time_range(test_behaviors),
        }
        logger.info("Split time ranges: %s", split_metadata)

        known_news_ids = set(combined_news["news_id"])
        max_history_length = config["sequence"]["max_history_length"]
        deduplicate_history = config["sequence"].get("deduplicate_history", True)
        train_samples, train_counters = build_train_samples(
            train_behaviors,
            known_news_ids,
            max_history_length=max_history_length,
            deduplicate_history=deduplicate_history,
        )
        validation_samples, validation_counters = build_evaluation_samples(
            validation_behaviors,
            known_news_ids,
            max_history_length=max_history_length,
            split_name="validation",
            deduplicate_history=deduplicate_history,
        )
        test_samples, test_counters = build_evaluation_samples(
            test_behaviors,
            known_news_ids,
            max_history_length=max_history_length,
            split_name="test",
            deduplicate_history=deduplicate_history,
        )
        for name, counters in (
            ("train", train_counters),
            ("validation", validation_counters),
            ("test", test_counters),
        ):
            _log_skip_counters(logger, name, counters)
        logger.info(
            "Built samples: train=%d validation=%d test=%d",
            len(train_samples),
            len(validation_samples),
            len(test_samples),
        )

        artifacts: dict[str, Any] = {
            "train_samples.pkl": train_samples,
            "validation_samples.pkl": validation_samples,
            "test_samples.pkl": test_samples,
            "news_title_mapping.pkl": news_title_mapping,
            "word_dict.pkl": word_dict,
            "word_embedding_matrix.pkl": embedding_matrix,
        }
        artifact_manifest: dict[str, Any] = {}
        for filename, data in artifacts.items():
            artifact_path = output_dir / "artifacts" / filename
            save_pickle(data, artifact_path)
            artifact_manifest[filename] = _artifact_metadata(
                artifact_path, data, output_dir
            )
            logger.info(
                "Saved artifact %s (%d bytes, sha256=%s)",
                filename,
                artifact_manifest[filename]["size_bytes"],
                artifact_manifest[filename]["sha256"],
            )

        statistics = {
            "source_rows": {
                "train_behaviors": len(train_behaviors),
                "validation_behaviors": len(validation_behaviors),
                "test_behaviors": len(test_behaviors),
                "train_news": len(train_news),
                "validation_news": len(validation_news),
                "test_news": len(test_news),
                "combined_news": len(combined_news),
            },
            "split": split_metadata,
            "samples": {
                "train": train_counters,
                "validation": validation_counters,
                "test": test_counters,
            },
            "history_lengths": {
                "train": _describe_lengths([len(value) for value in train_behaviors["history"]]),
                "validation": _describe_lengths(
                    [len(value) for value in validation_behaviors["history"]]
                ),
                "test": _describe_lengths([len(value) for value in test_behaviors["history"]]),
            },
            "title_token_lengths": _describe_lengths(
                [
                    len(tokenize(title, lowercase=text_config["lowercase"]))
                    for title in combined_news["title"]
                ]
            ),
            "vocabulary_size": len(word_dict),
            "embedding": embedding_statistics,
            "negative_sampling_ratio": config["training"]["negative_sampling_ratio"],
        }
        save_json(statistics, output_dir / "statistics.json")

        input_manifest = {
            name: file_descriptor(path)
            for name, path in input_paths.items()
            if name != "word_vectors"
        }
        input_manifest["word_vectors"] = {
            "path": str(input_paths["word_vectors"].resolve()),
            "size_bytes": embedding_statistics["source_size_bytes"],
            "sha256": embedding_statistics["source_sha256"],
        }
        manifest = {
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "created_at": _utc_now().isoformat(),
            "config": file_descriptor(output_dir / "config.yaml"),
            "inputs": input_manifest,
            "artifacts": artifact_manifest,
            "environment": {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "packages": _package_versions(),
            },
        }
        manifest["config"]["path"] = "config.yaml"
        save_json(manifest, output_dir / "manifest.json")
        run_info["status"] = "completed"
        logger.info("Preprocessing completed")
        return output_dir
    except BaseException as error:
        run_info.update(
            {
                "status": "failed",
                "error_type": type(error).__name__,
                "error_message": str(error),
            }
        )
        logger.exception("Preprocessing failed: %s", error)
        raise
    finally:
        end_time = _utc_now()
        run_info["end_time"] = end_time.isoformat()
        run_info["duration_seconds"] = (
            end_time - start_time
        ).total_seconds()
        save_json(run_info, output_dir / "run_info.json")
        close_logger(logger)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preprocess real news data for NRMS")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=None,
        help="Replace managed files in an existing output directory",
    )
    args = parser.parse_args()
    run(args.config, overwrite=args.overwrite)


if __name__ == "__main__":
    main()