from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import yaml

from src.checkpointing import load_checkpoint
from src.inference import precompute_news_vectors, recommend
from src.utils.io import load_pickle, save_json


def run(
    run_dir: str | Path,
    history: list[str],
    *,
    top_k: int = 10,
    candidate_ids: list[str] | None = None,
    device_override: str | None = None,
    output_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    run_path = Path(run_dir)
    config = _load_config(run_path / "config.yaml")
    device = _resolve_device(device_override or config.get("device", "auto"))
    data_dir = Path(config["data"]["directory"])
    if not data_dir.is_absolute():
        data_dir = Path(__file__).resolve().parents[1] / data_dir
    artifacts_dir = data_dir / "artifacts"
    news_title_mapping = load_pickle(artifacts_dir / "news_title_mapping.pkl")
    embedding_matrix = load_pickle(artifacts_dir / "word_embedding_matrix.pkl")

    loaded = load_checkpoint(
        run_path / "checkpoints" / "best.pt",
        embedding_matrix,
        device,
    )
    cache = precompute_news_vectors(
        loaded.model,
        news_title_mapping,
        device,
        batch_size=config["loader"].get("news_encoding_batch_size", 512),
        show_progress=config.get("logging", {}).get("progress_bar", True),
    )
    recommendations = recommend(
        loaded.model,
        cache,
        history,
        max_history_length=config["data"]["max_history_length"],
        top_k=top_k,
        candidate_ids=candidate_ids,
    )
    if output_path is not None:
        save_json(recommendations, output_path)
    return recommendations


def _load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        value = yaml.safe_load(file)
    if not isinstance(value, dict):
        raise ValueError("run config must contain a mapping")
    return value


def _resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if value == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if value not in {"cpu", "cuda"}:
        raise ValueError("device must be auto, cpu, or cuda")
    return torch.device(value)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recommend news from a trained NRMS run"
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--history", nargs="*", default=[])
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--candidates", nargs="*")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    recommendations = run(
        args.run_dir,
        args.history,
        top_k=args.top_k,
        candidate_ids=args.candidates,
        device_override=args.device,
        output_path=args.output,
    )
    print(json.dumps(recommendations, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()