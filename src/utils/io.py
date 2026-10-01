from __future__ import annotations

import hashlib
import json
import os
import pickle
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import yaml


MANAGED_ROOT_FILES = {
    "config.yaml",
    "manifest.json",
    "preprocess.log",
    "run_info.json",
    "statistics.json",
}
MANAGED_ARTIFACT_FILES = {
    "train_samples.pkl",
    "validation_samples.pkl",
    "test_samples.pkl",
    "news_title_mapping.pkl",
    "word_dict.pkl",
    "word_embedding_matrix.pkl",
}


def prepare_output_directory(path: str | Path, *, overwrite: bool) -> Path:
    output_dir = Path(path)
    if output_dir.exists() and not overwrite:
        raise FileExistsError(
            f"Output directory {output_dir} already exists; choose a new directory "
            "or set output.overwrite=true"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir = output_dir / "artifacts"
    artifacts_dir.mkdir(exist_ok=True)
    if overwrite:
        for filename in MANAGED_ROOT_FILES:
            (output_dir / filename).unlink(missing_ok=True)
        for filename in MANAGED_ARTIFACT_FILES:
            (artifacts_dir / filename).unlink(missing_ok=True)
    return output_dir


def _atomic_write(path: Path, writer: Callable[[Path], None]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        writer(temporary_path)
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def save_pickle(data: Any, path: str | Path) -> None:
    target = Path(path)

    def write(temporary_path: Path) -> None:
        with temporary_path.open("wb") as file:
            pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)

    _atomic_write(target, write)


def load_pickle(path: str | Path) -> Any:
    with Path(path).open("rb") as file:
        return pickle.load(file)


def save_json(data: Any, path: str | Path) -> None:
    target = Path(path)

    def write(temporary_path: Path) -> None:
        with temporary_path.open("w", encoding="utf-8") as file:
            json.dump(data, file, ensure_ascii=False, indent=2)
            file.write("\n")

    _atomic_write(target, write)


def save_yaml(data: Any, path: str | Path) -> None:
    target = Path(path)

    def write(temporary_path: Path) -> None:
        with temporary_path.open("w", encoding="utf-8") as file:
            yaml.safe_dump(data, file, sort_keys=False, allow_unicode=True)

    _atomic_write(target, write)


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        while chunk := file.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def file_descriptor(path: str | Path) -> dict[str, Any]:
    file_path = Path(path)
    return {
        "path": str(file_path.resolve()),
        "size_bytes": file_path.stat().st_size,
        "sha256": sha256_file(file_path),
    }