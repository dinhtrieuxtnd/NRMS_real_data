from __future__ import annotations

import logging
import math
import os
import random
import tempfile
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from tqdm.auto import tqdm

from src.inference import NewsVectorCache, precompute_news_vectors
from src.models import NRMS
from src.utils.io import save_json


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 10
    learning_rate: float = 0.0001
    weight_decay: float = 0.0
    gradient_clip_norm: float | None = 5.0
    patience: int = 3
    min_delta: float = 0.0
    monitor: str = "ndcg@10"
    amp: bool = True
    deterministic: bool = False

    def __post_init__(self) -> None:
        if self.epochs <= 0:
            raise ValueError("epochs must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.weight_decay < 0:
            raise ValueError("weight_decay must not be negative")
        if self.gradient_clip_norm is not None and self.gradient_clip_norm <= 0:
            raise ValueError("gradient_clip_norm must be positive or null")
        if self.patience < 0:
            raise ValueError("patience must not be negative")
        if self.min_delta < 0:
            raise ValueError("min_delta must not be negative")
        if self.monitor not in {"auc", "mrr", "ndcg@5", "ndcg@10"}:
            raise ValueError("monitor must be auc, mrr, ndcg@5, or ndcg@10")
        if not isinstance(self.deterministic, bool):
            raise ValueError("deterministic must be a boolean")


@dataclass(frozen=True)
class SchedulerConfig:
    type: str = "none"
    factor: float = 0.5
    patience: int = 1
    min_lr: float = 0.0
    t_max: int | None = None
    eta_min: float = 0.0

    def __post_init__(self) -> None:
        if self.type not in {"none", "reduce_on_plateau", "cosine"}:
            raise ValueError(
                "scheduler.type must be none, reduce_on_plateau, or cosine"
            )
        if not 0 < self.factor < 1:
            raise ValueError("scheduler.factor must be between 0 and 1")
        if self.patience < 0:
            raise ValueError("scheduler.patience must not be negative")
        if self.min_lr < 0 or self.eta_min < 0:
            raise ValueError("scheduler minimum learning rates must not be negative")
        if self.t_max is not None and self.t_max <= 0:
            raise ValueError("scheduler.t_max must be positive or null")


Scheduler = (
    torch.optim.lr_scheduler.LRScheduler | torch.optim.lr_scheduler.ReduceLROnPlateau
)


def create_scheduler(
    optimizer: torch.optim.Optimizer,
    config: SchedulerConfig,
    *,
    total_epochs: int,
) -> Scheduler | None:
    if config.type == "none":
        return None
    if config.type == "reduce_on_plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=config.factor,
            patience=config.patience,
            min_lr=config.min_lr,
        )
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.t_max or total_epochs,
        eta_min=config.eta_min,
    )


def seed_everything(seed: int, *, deterministic: bool = False) -> None:
    if deterministic:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic


def seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def train_one_epoch(
    model: NRMS,
    data_loader: Iterable[Mapping[str, Any]],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    gradient_clip_norm: float | None = None,
    amp: bool = False,
    max_batches: int | None = None,
    show_progress: bool = False,
) -> float:
    model.train()
    amp_enabled = amp and device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=amp_enabled)
    total_loss = 0.0
    total_examples = 0

    progress = tqdm(
        data_loader,
        desc="train",
        unit="batch",
        leave=False,
        disable=not show_progress,
    )
    for batch_index, batch in enumerate(progress):
        if max_batches is not None and batch_index >= max_batches:
            break

        history = batch["history"].to(device)
        history_mask = batch["history_mask"].to(device)
        candidates = batch["candidates"].to(device)
        targets = batch["labels"].argmax(dim=1).to(device)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=amp_enabled):
            logits = model(history, history_mask, candidates)
            loss = nn.functional.cross_entropy(logits, targets)

        scaler.scale(loss).backward()
        if gradient_clip_norm is not None:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
        scaler.step(optimizer)
        scaler.update()

        batch_size = history.shape[0]
        total_loss += loss.detach().item() * batch_size
        total_examples += batch_size
        progress.set_postfix(loss=f"{total_loss / total_examples:.4f}")

    if total_examples == 0:
        raise ValueError("training DataLoader produced no batches")
    return total_loss / total_examples


@torch.no_grad()
def evaluate(
    model: NRMS,
    data_loader: Iterable[Mapping[str, Any]],
    device: torch.device,
    *,
    max_batches: int | None = None,
    show_progress: bool = False,
    news_vector_cache: NewsVectorCache | None = None,
    news_title_mapping: Mapping[str, np.ndarray] | None = None,
    news_encoding_batch_size: int = 512,
) -> dict[str, float | int]:
    if news_vector_cache is None and news_title_mapping is not None:
        news_vector_cache = precompute_news_vectors(
            model,
            news_title_mapping,
            device,
            batch_size=news_encoding_batch_size,
            show_progress=show_progress,
        )
    metrics, _ = _evaluate_batches(
        model,
        data_loader,
        device,
        max_batches=max_batches,
        show_progress=show_progress,
        collect_predictions=False,
        news_vector_cache=news_vector_cache,
    )
    return metrics


@torch.no_grad()
def evaluate_with_predictions(
    model: NRMS,
    data_loader: Iterable[Mapping[str, Any]],
    device: torch.device,
    *,
    max_batches: int | None = None,
    show_progress: bool = False,
    news_vector_cache: NewsVectorCache | None = None,
) -> tuple[dict[str, float | int], list[dict[str, Any]]]:
    return _evaluate_batches(
        model,
        data_loader,
        device,
        max_batches=max_batches,
        show_progress=show_progress,
        collect_predictions=True,
        news_vector_cache=news_vector_cache,
    )


def _evaluate_batches(
    model: NRMS,
    data_loader: Iterable[Mapping[str, Any]],
    device: torch.device,
    *,
    max_batches: int | None,
    show_progress: bool,
    collect_predictions: bool,
    news_vector_cache: NewsVectorCache | None,
) -> tuple[dict[str, float | int], list[dict[str, Any]]]:
    model.eval()
    totals = {"auc": 0.0, "mrr": 0.0, "ndcg@5": 0.0, "ndcg@10": 0.0}
    impressions = 0
    auc_impressions = 0
    predictions: list[dict[str, Any]] = []

    progress = tqdm(
        data_loader,
        desc="validation",
        unit="batch",
        leave=False,
        disable=not show_progress,
    )
    for batch_index, batch in enumerate(progress):
        if max_batches is not None and batch_index >= max_batches:
            break

        candidate_mask = batch.get("candidate_mask")
        device_candidate_mask = (
            candidate_mask.to(device) if candidate_mask is not None else None
        )
        if news_vector_cache is None:
            logits = model(
                batch["history"].to(device),
                batch["history_mask"].to(device),
                batch["candidates"].to(device),
                device_candidate_mask,
            ).cpu()
        else:
            vectors = news_vector_cache.vectors
            logits = model.score_news_vectors(
                vectors[batch["history_indices"].to(vectors.device)],
                batch["history_mask"].to(vectors.device),
                vectors[batch["candidate_indices"].to(vectors.device)],
                (
                    candidate_mask.to(vectors.device)
                    if candidate_mask is not None
                    else None
                ),
            ).cpu()
        labels = batch["labels"].cpu()
        if candidate_mask is None:
            candidate_mask = torch.ones_like(labels, dtype=torch.bool)

        impression_ids = batch["impression_id"]
        candidate_ids = batch.get("candidate_ids")
        if collect_predictions and candidate_ids is None:
            raise ValueError("evaluation batch must contain candidate_ids")

        for row_index, (impression_id, row_logits, row_labels, row_mask) in enumerate(
            zip(impression_ids, logits, labels, candidate_mask, strict=True)
        ):
            valid_logits = row_logits[row_mask]
            valid_labels = row_labels[row_mask]
            metrics = ranking_metrics(valid_labels, valid_logits)
            impressions += 1
            totals["mrr"] += metrics["mrr"]
            totals["ndcg@5"] += metrics["ndcg@5"]
            totals["ndcg@10"] += metrics["ndcg@10"]
            if not math.isnan(metrics["auc"]):
                totals["auc"] += metrics["auc"]
                auc_impressions += 1

            if collect_predictions:
                row_candidate_ids = candidate_ids[row_index]
                if len(row_candidate_ids) != valid_logits.numel():
                    raise ValueError(
                        "candidate_ids length must match the unpadded candidates"
                    )
                order = torch.argsort(valid_logits, descending=True, stable=True)
                ranks = torch.empty_like(order)
                ranks[order] = torch.arange(1, order.numel() + 1)
                predictions.extend(
                    {
                        "impression_id": impression_id,
                        "news_id": news_id,
                        "label": int(label.item()),
                        "score": float(score.item()),
                        "rank": int(rank.item()),
                    }
                    for news_id, label, score, rank in zip(
                        row_candidate_ids,
                        valid_labels,
                        valid_logits,
                        ranks,
                        strict=True,
                    )
                )

    if impressions == 0:
        raise ValueError("evaluation DataLoader produced no batches")
    evaluation_metrics: dict[str, float | int] = {
        "auc": totals["auc"] / auc_impressions if auc_impressions else math.nan,
        "mrr": totals["mrr"] / impressions,
        "ndcg@5": totals["ndcg@5"] / impressions,
        "ndcg@10": totals["ndcg@10"] / impressions,
        "impressions": impressions,
        "auc_impressions": auc_impressions,
    }
    return evaluation_metrics, predictions


def ranking_metrics(
    labels: torch.Tensor,
    scores: torch.Tensor,
) -> dict[str, float]:
    labels = labels.to(torch.bool).flatten()
    scores = scores.to(torch.float64).flatten()
    if labels.shape != scores.shape or labels.numel() == 0:
        raise ValueError("labels and scores must be non-empty vectors of equal length")

    positives = scores[labels]
    negatives = scores[~labels]
    if positives.numel() and negatives.numel():
        comparisons = positives[:, None] - negatives[None, :]
        auc = (
            (comparisons > 0).to(torch.float64).mean()
            + 0.5 * (comparisons == 0).to(torch.float64).mean()
        ).item()
    else:
        auc = math.nan

    ranked_labels = labels[torch.argsort(scores, descending=True, stable=True)]
    relevant_ranks = torch.nonzero(ranked_labels, as_tuple=False).flatten() + 1
    mrr = 1.0 / relevant_ranks[0].item() if relevant_ranks.numel() else 0.0
    return {
        "auc": auc,
        "mrr": mrr,
        "ndcg@5": _ndcg(ranked_labels, 5),
        "ndcg@10": _ndcg(ranked_labels, 10),
    }


def fit(
    model: NRMS,
    train_loader: Iterable[Mapping[str, Any]],
    validation_loader: Iterable[Mapping[str, Any]],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    output_dir: str | Path,
    config: TrainingConfig,
    *,
    logger: logging.Logger | None = None,
    max_train_batches: int | None = None,
    max_validation_batches: int | None = None,
    show_progress: bool = False,
    initial_epoch: int = 0,
    initial_best_metric: float = -math.inf,
    initial_epochs_without_improvement: int = 0,
    initial_history: Iterable[Mapping[str, Any]] = (),
    validation_news_title_mapping: Mapping[str, np.ndarray] | None = None,
    news_encoding_batch_size: int = 512,
    scheduler: Scheduler | None = None,
    scheduler_config: SchedulerConfig | None = None,
) -> list[dict[str, Any]]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    if initial_epoch < 0:
        raise ValueError("initial_epoch must not be negative")
    if initial_epochs_without_improvement < 0:
        raise ValueError("initial_epochs_without_improvement must not be negative")
    history = [dict(record) for record in initial_history]
    best_metric = initial_best_metric
    epochs_without_improvement = initial_epochs_without_improvement
    model.to(device)

    for epoch in range(initial_epoch + 1, config.epochs + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            gradient_clip_norm=config.gradient_clip_norm,
            amp=config.amp,
            max_batches=max_train_batches,
            show_progress=show_progress,
        )
        validation_metrics = evaluate(
            model,
            validation_loader,
            device,
            max_batches=max_validation_batches,
            show_progress=show_progress,
            news_title_mapping=validation_news_title_mapping,
            news_encoding_batch_size=news_encoding_batch_size,
        )
        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            **validation_metrics,
        }
        history.append(record)

        monitored_value = float(validation_metrics[config.monitor])
        improved = (
            not math.isnan(monitored_value)
            and monitored_value > best_metric + config.min_delta
        )
        if improved:
            best_metric = monitored_value
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        effective_scheduler_config = scheduler_config or SchedulerConfig()
        if scheduler is not None:
            if effective_scheduler_config.type == "reduce_on_plateau":
                scheduler.step(monitored_value)
            else:
                scheduler.step()

        checkpoint = _checkpoint(
            model,
            optimizer,
            config,
            epoch=epoch,
            best_metric=best_metric,
            metrics=record,
            epochs_without_improvement=epochs_without_improvement,
            history=history,
            scheduler=scheduler,
            scheduler_config=effective_scheduler_config,
        )
        checkpoint_dir = output_path / "checkpoints"
        _save_checkpoint(checkpoint, checkpoint_dir / "last.pt")
        if improved:
            _save_checkpoint(checkpoint, checkpoint_dir / "best.pt")
        save_json(
            {"monitor": config.monitor, "epochs": history}, output_path / "history.json"
        )

        if logger is not None:
            logger.info(
                "epoch=%d loss=%.6f auc=%.6f mrr=%.6f ndcg@5=%.6f " "ndcg@10=%.6f%s",
                epoch,
                train_loss,
                validation_metrics["auc"],
                validation_metrics["mrr"],
                validation_metrics["ndcg@5"],
                validation_metrics["ndcg@10"],
                " best" if improved else "",
            )

        if config.patience and epochs_without_improvement >= config.patience:
            if logger is not None:
                logger.info("early stopping after epoch %d", epoch)
            break

    return history


def _ndcg(ranked_labels: torch.Tensor, cutoff: int) -> float:
    limit = min(cutoff, ranked_labels.numel())
    if limit == 0:
        return 0.0
    discounts = torch.log2(torch.arange(2, limit + 2, dtype=torch.float64))
    dcg = (ranked_labels[:limit].to(torch.float64) / discounts).sum()
    ideal_relevant = min(int(ranked_labels.sum().item()), limit)
    if ideal_relevant == 0:
        return 0.0
    idcg = (
        torch.ones(ideal_relevant, dtype=torch.float64) / discounts[:ideal_relevant]
    ).sum()
    return (dcg / idcg).item()


def _checkpoint(
    model: NRMS,
    optimizer: torch.optim.Optimizer,
    config: TrainingConfig,
    *,
    epoch: int,
    best_metric: float,
    metrics: Mapping[str, Any],
    epochs_without_improvement: int = 0,
    history: Iterable[Mapping[str, Any]] = (),
    scheduler: Scheduler | None = None,
    scheduler_config: SchedulerConfig | None = None,
) -> dict[str, Any]:
    effective_scheduler_config = scheduler_config or SchedulerConfig()
    return {
        "epoch": epoch,
        "best_metric": best_metric,
        "epochs_without_improvement": epochs_without_improvement,
        "history": [dict(record) for record in history],
        "model_config": asdict(model.config),
        "training_config": asdict(config),
        "scheduler_config": asdict(effective_scheduler_config),
        "metrics": dict(metrics),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": (
            scheduler.state_dict() if scheduler is not None else None
        ),
    }


def _save_checkpoint(checkpoint: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        torch.save(dict(checkpoint), temporary_path)
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)
