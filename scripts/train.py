from __future__ import annotations

import argparse
import json
import platform
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.utils.data import DataLoader

from src.checkpointing import LoadedCheckpoint, load_checkpoint
from src.data.dataset import (
    NRMSTrainDataset,
    NRMSVectorEvaluationDataset,
    nrms_vector_evaluation_collate,
)
from src.models import NRMS, NRMSConfig
from src.reporting import VALIDATION_METRICS, write_training_artifacts
from src.training import (
    SchedulerConfig,
    TrainingConfig,
    create_scheduler,
    fit,
    seed_everything,
    seed_worker,
)
from src.utils.io import load_pickle, save_json, save_yaml
from src.utils.logging import close_logger, setup_logger


RUN_SUBDIRECTORIES = ("checkpoints", "artifacts", "plots", "predictions")


def run(
    config_path: str | Path,
    *,
    project_root: str | Path | None = None,
    device_override: str | None = None,
    max_train_batches: int | None = None,
    max_validation_batches: int | None = None,
    run_timestamp: datetime | None = None,
    resume: str | Path | None = None,
) -> Path:
    root = Path(project_root) if project_root else Path(__file__).resolve().parents[1]
    config = _load_config(config_path, root)
    if device_override is not None:
        config["device"] = device_override

    resume_path = _resolve_resume_path(resume, root) if resume is not None else None
    run_dir = (
        resume_path.parent.parent
        if resume_path is not None
        else _create_run_directory(
            Path(config["output"]["root"]),
            config["experiment"]["name"],
            timestamp=run_timestamp,
        )
    )
    logger = setup_logger(
        run_dir / "train.log",
        level=config["logging"]["level"],
        append=resume_path is not None,
    )
    if resume_path is None:
        save_yaml(config, run_dir / "config.yaml")

    start_time = datetime.now(timezone.utc)
    previous_duration = 0.0
    previous_training_duration = 0.0
    if resume_path is not None:
        run_info = _load_run_info(run_dir / "run_info.json")
        previous_duration = float(run_info.get("duration_seconds", 0.0))
        previous_training_duration = float(
            run_info.get("training_duration_seconds", 0.0)
        )
        run_info.update(
            {
                "status": "running",
                "resumed_at": start_time.isoformat(),
                "resume_checkpoint": str(resume_path),
                "resume_count": int(run_info.get("resume_count", 0)) + 1,
            }
        )
        run_info.pop("error_type", None)
        run_info.pop("error_message", None)
    else:
        run_info = {
            "experiment_name": config["experiment"]["name"],
            "status": "running",
            "start_time": start_time.isoformat(),
            "seed": config["experiment"]["seed"],
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "run_directory": str(run_dir),
            "test_metrics": {},
            "resume_count": 0,
        }
    save_json(run_info, run_dir / "run_info.json")

    try:
        device = _resolve_device(config["device"])
        run_info["device"] = str(device)
        seed_everything(
            config["experiment"]["seed"],
            deterministic=config["training"].get("deterministic", False),
        )
        run_info["deterministic"] = config["training"].get(
            "deterministic", False
        )
        logger.info("Starting training: %s", config["experiment"]["name"])
        logger.info("Device: %s", device)

        artifacts_dir = Path(config["data"]["directory"]) / "artifacts"
        train_samples = load_pickle(artifacts_dir / "train_samples.pkl")
        validation_samples = load_pickle(artifacts_dir / "validation_samples.pkl")
        news_title_mapping = load_pickle(artifacts_dir / "news_title_mapping.pkl")
        embedding_matrix = load_pickle(artifacts_dir / "word_embedding_matrix.pkl")

        train_dataset = NRMSTrainDataset(
            train_samples,
            news_title_mapping,
            max_history_length=config["data"]["max_history_length"],
            num_negatives=config["data"]["negative_sampling_ratio"],
        )
        news_index = {
            news_id: index for index, news_id in enumerate(news_title_mapping)
        }
        validation_dataset = NRMSVectorEvaluationDataset(
            validation_samples,
            news_index,
            max_history_length=config["data"]["max_history_length"],
        )
        generator = torch.Generator().manual_seed(config["experiment"]["seed"])
        loader_config = config["loader"]
        common_loader_args = {
            "num_workers": loader_config["num_workers"],
            "pin_memory": loader_config["pin_memory"] and device.type == "cuda",
            "persistent_workers": loader_config["num_workers"] > 0,
            "worker_init_fn": (
                seed_worker if loader_config["num_workers"] > 0 else None
            ),
        }
        train_loader = DataLoader(
            train_dataset,
            batch_size=loader_config["train_batch_size"],
            shuffle=True,
            generator=generator,
            **common_loader_args,
        )
        validation_loader = DataLoader(
            validation_dataset,
            batch_size=loader_config["validation_batch_size"],
            shuffle=False,
            collate_fn=nrms_vector_evaluation_collate,
            **common_loader_args,
        )

        model_config = NRMSConfig(**config["model"])
        training_config = TrainingConfig(**config["training"])
        scheduler_config = SchedulerConfig(**config["scheduler"])
        loaded_checkpoint: LoadedCheckpoint | None = None
        if resume_path is not None:
            loaded_checkpoint = load_checkpoint(
                resume_path,
                embedding_matrix,
                device,
            )
            _validate_resume_config(
                model_config,
                training_config,
                scheduler_config,
                loaded_checkpoint,
            )
            model = loaded_checkpoint.model
            save_yaml(config, run_dir / "config.yaml")
            logger.info(
                "Resuming from epoch %d: %s",
                loaded_checkpoint.epoch,
                resume_path,
            )
        else:
            model = NRMS(embedding_matrix, model_config)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=training_config.learning_rate,
            weight_decay=training_config.weight_decay,
        )
        if loaded_checkpoint is not None:
            loaded_checkpoint.restore_optimizer(optimizer)
        scheduler = create_scheduler(
            optimizer,
            scheduler_config,
            total_epochs=training_config.epochs,
        )
        if loaded_checkpoint is not None:
            loaded_checkpoint.restore_scheduler(scheduler)
        total_parameters = sum(parameter.numel() for parameter in model.parameters())
        trainable_parameters = sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        )
        logger.info(
            "Samples: train=%d validation=%d",
            len(train_dataset),
            len(validation_dataset),
        )
        logger.info(
            "Model parameters: total=%d trainable=%d",
            total_parameters,
            trainable_parameters,
        )

        training_started = time.perf_counter()
        history = fit(
            model,
            train_loader,
            validation_loader,
            optimizer,
            device,
            run_dir,
            training_config,
            logger=logger,
            max_train_batches=max_train_batches,
            max_validation_batches=max_validation_batches,
            show_progress=config["logging"].get("progress_bar", True),
            initial_epoch=loaded_checkpoint.epoch if loaded_checkpoint else 0,
            initial_best_metric=(
                loaded_checkpoint.best_metric if loaded_checkpoint else -float("inf")
            ),
            initial_epochs_without_improvement=(
                loaded_checkpoint.epochs_without_improvement
                if loaded_checkpoint
                else 0
            ),
            initial_history=loaded_checkpoint.history if loaded_checkpoint else (),
            validation_news_title_mapping=news_title_mapping,
            news_encoding_batch_size=loader_config["news_encoding_batch_size"],
            scheduler=scheduler,
            scheduler_config=scheduler_config,
        )
        training_duration = previous_training_duration + (
            time.perf_counter() - training_started
        )
        last_checkpoint = load_checkpoint(
            run_dir / "checkpoints" / "last.pt",
            embedding_matrix,
            device,
        )
        stop_reason = (
            "early_stopping"
            if training_config.patience
            and last_checkpoint.epochs_without_improvement
            >= training_config.patience
            else "completed"
        )
        best_checkpoint = load_checkpoint(
            run_dir / "checkpoints" / "best.pt",
            embedding_matrix,
            device,
        )
        best_validation_metrics = {
            metric: float(best_checkpoint.metrics[metric])
            for metric in VALIDATION_METRICS
        }
        write_training_artifacts(
            run_dir,
            history,
            best_epoch=best_checkpoint.epoch,
            best_validation_metrics=best_validation_metrics,
            stop_reason=stop_reason,
            training_duration_seconds=training_duration,
        )
        run_info.update(
            {
                "status": "completed",
                "epochs_completed": len(history),
                "best_epoch": best_checkpoint.epoch,
                "best_validation_metrics": best_validation_metrics,
                "monitor": training_config.monitor,
                "stop_reason": stop_reason,
                "training_duration_seconds": training_duration,
                "scheduler": asdict(scheduler_config),
                "final_learning_rate": float(optimizer.param_groups[0]["lr"]),
            }
        )
        logger.info("Training completed")
        return run_dir
    except BaseException as error:
        run_info.update(
            {
                "status": "failed",
                "error_type": type(error).__name__,
                "error_message": str(error),
            }
        )
        logger.exception("Training failed: %s", error)
        raise
    finally:
        end_time = datetime.now(timezone.utc)
        run_info["end_time"] = end_time.isoformat()
        run_info["duration_seconds"] = previous_duration + (
            end_time - start_time
        ).total_seconds()
        save_json(run_info, run_dir / "run_info.json")
        close_logger(logger)


def _load_config(config_path: str | Path, root: Path) -> dict[str, Any]:
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError("training config must contain a YAML mapping")

    required_sections = {
        "experiment",
        "data",
        "loader",
        "model",
        "training",
        "output",
        "logging",
    }
    missing = required_sections - config.keys()
    if missing:
        raise ValueError(f"training config is missing sections: {', '.join(sorted(missing))}")

    output_root_value = config["output"].get("root")
    if not isinstance(output_root_value, str) or not output_root_value:
        raise ValueError("output.root must be a non-empty path string")
    data_dir = Path(config["data"]["directory"])
    output_root = Path(output_root_value)
    config["data"]["directory"] = str(data_dir if data_dir.is_absolute() else root / data_dir)
    config["output"]["root"] = str(
        output_root if output_root.is_absolute() else root / output_root
    )
    config.setdefault("device", "auto")
    config.setdefault("scheduler", {"type": "none"})
    config["training"].setdefault("deterministic", False)
    config["loader"].setdefault("news_encoding_batch_size", 512)
    _validate_config(config)
    return config


def _validate_config(config: dict[str, Any]) -> None:
    if not config["experiment"].get("name"):
        raise ValueError("experiment.name must be non-empty")
    if not isinstance(config["experiment"].get("seed"), int):
        raise ValueError("experiment.seed must be an integer")
    for key in ("max_history_length", "negative_sampling_ratio"):
        if not isinstance(config["data"].get(key), int) or config["data"][key] <= 0:
            raise ValueError(f"data.{key} must be a positive integer")
    for key in (
        "train_batch_size",
        "validation_batch_size",
        "news_encoding_batch_size",
    ):
        if not isinstance(config["loader"].get(key), int) or config["loader"][key] <= 0:
            raise ValueError(f"loader.{key} must be a positive integer")
    if not isinstance(config["loader"].get("num_workers"), int) or config["loader"]["num_workers"] < 0:
        raise ValueError("loader.num_workers must not be negative")
    if not isinstance(config["loader"].get("pin_memory"), bool):
        raise ValueError("loader.pin_memory must be a boolean")
    if not isinstance(config["output"].get("root"), str) or not config["output"]["root"]:
        raise ValueError("output.root must be a non-empty path string")
    if "progress_bar" in config["logging"] and not isinstance(
        config["logging"]["progress_bar"], bool
    ):
        raise ValueError("logging.progress_bar must be a boolean")
    if config["device"] not in {"auto", "cpu", "cuda"}:
        raise ValueError("device must be auto, cpu, or cuda")
    NRMSConfig(**config["model"])
    TrainingConfig(**config["training"])
    SchedulerConfig(**config["scheduler"])


def _create_run_directory(
    output_root: Path,
    experiment_name: str,
    *,
    timestamp: datetime | None = None,
) -> Path:
    timestamp_name = (timestamp or datetime.now()).strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = output_root / experiment_name / timestamp_name
    run_dir.mkdir(parents=True, exist_ok=False)
    for directory_name in RUN_SUBDIRECTORIES:
        (run_dir / directory_name).mkdir()
    return run_dir


def _resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if value == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(value)


def _resolve_resume_path(value: str | Path, root: Path) -> Path:
    path = Path(value)
    path = path if path.is_absolute() else root / path
    if not path.is_file():
        raise FileNotFoundError(f"resume checkpoint does not exist: {path}")
    if path.parent.name != "checkpoints":
        raise ValueError("resume checkpoint must be inside a checkpoints directory")
    return path


def _load_run_info(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        value = json.load(file)
    if not isinstance(value, dict):
        raise ValueError("run_info.json must contain a mapping")
    return value


def _validate_resume_config(
    model_config: NRMSConfig,
    training_config: TrainingConfig,
    scheduler_config: SchedulerConfig,
    loaded: LoadedCheckpoint,
) -> None:
    if model_config != loaded.model_config:
        raise ValueError("resume model config does not match the checkpoint")
    current_training = asdict(training_config)
    checkpoint_training = asdict(loaded.training_config)
    mismatches = [
        key
        for key, value in checkpoint_training.items()
        if key != "epochs" and current_training[key] != value
    ]
    if mismatches:
        raise ValueError(
            "resume training config does not match the checkpoint: "
            + ", ".join(sorted(mismatches))
        )
    if training_config.epochs <= loaded.epoch:
        raise ValueError(
            "training.epochs must be greater than the checkpoint epoch "
            f"({loaded.epoch})"
        )
    if scheduler_config != loaded.scheduler_config:
        raise ValueError("resume scheduler config does not match the checkpoint")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train NRMS on processed real news data")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--max-validation-batches", type=int)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    run(
        args.config,
        device_override=args.device,
        max_train_batches=args.max_train_batches,
        max_validation_batches=args.max_validation_batches,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()