from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.models import NRMS, NRMSConfig
from src.training import Scheduler, SchedulerConfig, TrainingConfig


REQUIRED_CHECKPOINT_FIELDS = {
    "epoch",
    "best_metric",
    "epochs_without_improvement",
    "history",
    "model_config",
    "training_config",
    "scheduler_config",
    "metrics",
    "model_state_dict",
    "optimizer_state_dict",
    "scheduler_state_dict",
}


@dataclass(frozen=True)
class LoadedCheckpoint:
    model: NRMS
    model_config: NRMSConfig
    training_config: TrainingConfig
    scheduler_config: SchedulerConfig
    epoch: int
    best_metric: float
    epochs_without_improvement: int
    history: list[dict[str, Any]]
    metrics: dict[str, Any]
    optimizer_state_dict: dict[str, Any]
    scheduler_state_dict: dict[str, Any] | None

    def restore_optimizer(
        self, optimizer: torch.optim.Optimizer
    ) -> torch.optim.Optimizer:
        optimizer.load_state_dict(self.optimizer_state_dict)
        return optimizer

    def restore_scheduler(self, scheduler: Scheduler | None) -> Scheduler | None:
        if self.scheduler_state_dict is None:
            if scheduler is not None:
                raise ValueError("checkpoint does not contain scheduler state")
            return None
        if scheduler is None:
            raise ValueError("a scheduler is required to restore scheduler state")
        scheduler.load_state_dict(self.scheduler_state_dict)
        return scheduler


def load_checkpoint(
    checkpoint_path: str | Path,
    embedding_matrix: np.ndarray | torch.Tensor,
    device: str | torch.device,
) -> LoadedCheckpoint:
    """Restore an NRMS checkpoint and expose its optimizer state for resume."""
    resolved_device = torch.device(device)
    checkpoint = torch.load(
        Path(checkpoint_path),
        map_location=resolved_device,
        weights_only=True,
    )
    if not isinstance(checkpoint, Mapping):
        raise ValueError("checkpoint must contain a mapping")

    missing_fields = REQUIRED_CHECKPOINT_FIELDS - checkpoint.keys()
    if missing_fields:
        missing = ", ".join(sorted(missing_fields))
        raise ValueError(f"checkpoint is missing required fields: {missing}")

    model_config = NRMSConfig(**checkpoint["model_config"])
    embedding_shape = tuple(np.shape(embedding_matrix))
    if len(embedding_shape) != 2:
        raise ValueError("embedding_matrix must be two-dimensional")
    if embedding_shape[1] != model_config.embedding_dim:
        raise ValueError(
            "embedding dimension mismatch: checkpoint expects "
            f"{model_config.embedding_dim}, got {embedding_shape[1]}"
        )

    training_config = TrainingConfig(**checkpoint["training_config"])
    scheduler_config = SchedulerConfig(**checkpoint["scheduler_config"])
    model = NRMS(embedding_matrix, model_config).to(resolved_device)
    try:
        model.load_state_dict(checkpoint["model_state_dict"])
    except RuntimeError as error:
        raise ValueError(f"checkpoint model state is incompatible: {error}") from error

    return LoadedCheckpoint(
        model=model,
        model_config=model_config,
        training_config=training_config,
        scheduler_config=scheduler_config,
        epoch=int(checkpoint["epoch"]),
        best_metric=float(checkpoint["best_metric"]),
        epochs_without_improvement=int(checkpoint["epochs_without_improvement"]),
        history=[dict(record) for record in checkpoint["history"]],
        metrics=dict(checkpoint["metrics"]),
        optimizer_state_dict=dict(checkpoint["optimizer_state_dict"]),
        scheduler_state_dict=(
            dict(checkpoint["scheduler_state_dict"])
            if checkpoint["scheduler_state_dict"] is not None
            else None
        ),
    )