from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.utils.data import DataLoader

from src.checkpointing import load_checkpoint
from src.data.dataset import (
    NRMSVectorEvaluationDataset,
    nrms_vector_evaluation_collate,
)
from src.inference import precompute_news_vectors
from src.training import evaluate_with_predictions
from src.utils.io import load_pickle, save_json


PREDICTION_FIELDS = ("impression_id", "news_id", "label", "score", "rank")


def run(
    run_dir: str | Path,
    *,
    device_override: str | None = None,
    max_batches: int | None = None,
) -> dict[str, float | int]:
    run_path = Path(run_dir)
    config = _load_mapping(run_path / "config.yaml", "run config")
    device = _resolve_device(device_override or config.get("device", "auto"))

    data_dir = Path(config["data"]["directory"])
    if not data_dir.is_absolute():
        data_dir = Path(__file__).resolve().parents[1] / data_dir
    artifacts_dir = data_dir / "artifacts"
    test_samples = load_pickle(artifacts_dir / "test_samples.pkl")
    news_title_mapping = load_pickle(artifacts_dir / "news_title_mapping.pkl")
    embedding_matrix = load_pickle(artifacts_dir / "word_embedding_matrix.pkl")

    news_index = {
        news_id: index for index, news_id in enumerate(news_title_mapping)
    }
    dataset = NRMSVectorEvaluationDataset(
        test_samples,
        news_index,
        max_history_length=config["data"]["max_history_length"],
    )
    loader_config = config["loader"]
    num_workers = loader_config["num_workers"]
    data_loader = DataLoader(
        dataset,
        batch_size=loader_config["validation_batch_size"],
        shuffle=False,
        collate_fn=nrms_vector_evaluation_collate,
        num_workers=num_workers,
        pin_memory=loader_config["pin_memory"] and device.type == "cuda",
        persistent_workers=num_workers > 0,
    )

    loaded = load_checkpoint(
        run_path / "checkpoints" / "best.pt",
        embedding_matrix,
        device,
    )
    news_vector_cache = precompute_news_vectors(
        loaded.model,
        news_title_mapping,
        device,
        batch_size=loader_config.get("news_encoding_batch_size", 512),
        show_progress=config.get("logging", {}).get("progress_bar", True),
    )
    metrics, predictions = evaluate_with_predictions(
        loaded.model,
        data_loader,
        device,
        max_batches=max_batches,
        show_progress=config.get("logging", {}).get("progress_bar", True),
        news_vector_cache=news_vector_cache,
    )
    metrics["hr@10"] = _hit_rate_at_k(predictions, k=10)

    save_json(metrics, run_path / "artifacts" / "test_metrics.json")
    _save_predictions(predictions, run_path / "predictions" / "test_predictions.csv")
    run_info = _load_json_mapping(run_path / "run_info.json", "run info")
    run_info["test_metrics"] = metrics
    run_info["evaluated_at"] = datetime.now(timezone.utc).isoformat()
    run_info["news_vector_cache"] = {
        "news_count": len(news_vector_cache.news_ids),
        "vector_dim": news_vector_cache.vectors.shape[1],
    }
    save_json(run_info, run_path / "run_info.json")
    return metrics


def _hit_rate_at_k(predictions: list[dict[str, Any]], *, k: int) -> float:
    if k <= 0:
        raise ValueError("k must be greater than zero")

    hits_by_impression: dict[str, bool] = {}
    for row in predictions:
        impression_id = str(row["impression_id"])
        hits_by_impression.setdefault(impression_id, False)
        if int(row["label"]) > 0 and int(row["rank"]) <= k:
            hits_by_impression[impression_id] = True

    if not hits_by_impression:
        raise ValueError("predictions must contain at least one impression")
    return sum(hits_by_impression.values()) / len(hits_by_impression)


def _load_mapping(path: Path, description: str) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        value = yaml.safe_load(file)
    if not isinstance(value, dict):
        raise ValueError(f"{description} must contain a mapping")
    return value


def _load_json_mapping(path: Path, description: str) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        value = json.load(file)
    if not isinstance(value, dict):
        raise ValueError(f"{description} must contain a mapping")
    return value


def _resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if value == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if value not in {"cpu", "cuda"}:
        raise ValueError("device must be auto, cpu, or cuda")
    return torch.device(value)


def _save_predictions(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=PREDICTION_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate an NRMS run on its test set")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--max-batches", type=int)
    args = parser.parse_args()
    metrics = run(
        args.run_dir,
        device_override=args.device,
        max_batches=args.max_batches,
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()