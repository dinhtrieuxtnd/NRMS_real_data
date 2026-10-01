from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt

from src.utils.io import save_json


VALIDATION_METRICS = ("auc", "mrr", "ndcg@5", "ndcg@10")


def write_training_artifacts(
    output_dir: str | Path,
    history: Sequence[Mapping[str, Any]],
    *,
    best_epoch: int,
    best_validation_metrics: Mapping[str, float],
    stop_reason: str,
    training_duration_seconds: float,
) -> dict[str, Any]:
    if not history:
        raise ValueError("training history must not be empty")
    if stop_reason not in {"completed", "early_stopping"}:
        raise ValueError("stop_reason must be completed or early_stopping")

    output_path = Path(output_dir)
    summary = {
        "best_epoch": best_epoch,
        "best_validation_metrics": {
            metric: float(best_validation_metrics[metric])
            for metric in VALIDATION_METRICS
        },
        "final_train_loss": float(history[-1]["train_loss"]),
        "epochs_completed": len(history),
        "stop_reason": stop_reason,
        "training_duration_seconds": training_duration_seconds,
    }
    save_json(summary, output_path / "artifacts" / "summary.json")

    epochs = [int(record["epoch"]) for record in history]
    _save_plot(
        output_path / "plots" / "loss.png",
        epochs,
        [("Train loss", [float(record["train_loss"]) for record in history])],
        ylabel="Loss",
    )
    _save_plot(
        output_path / "plots" / "auc.png",
        epochs,
        [("AUC", [float(record["auc"]) for record in history])],
        ylabel="AUC",
    )
    _save_plot(
        output_path / "plots" / "mrr.png",
        epochs,
        [("MRR", [float(record["mrr"]) for record in history])],
        ylabel="MRR",
    )
    _save_plot(
        output_path / "plots" / "ndcg.png",
        epochs,
        [
            ("nDCG@5", [float(record["ndcg@5"]) for record in history]),
            ("nDCG@10", [float(record["ndcg@10"]) for record in history]),
        ],
        ylabel="nDCG",
    )
    return summary


def _save_plot(
    path: Path,
    epochs: Sequence[int],
    series: Sequence[tuple[str, Sequence[float]]],
    *,
    ylabel: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(7, 4.5))
    for label, values in series:
        axis.plot(epochs, values, marker="o", linewidth=1.8, label=label)
    axis.set_xlabel("Epoch")
    axis.set_ylabel(ylabel)
    axis.set_xticks(epochs)
    axis.grid(alpha=0.25)
    if len(series) > 1:
        axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)