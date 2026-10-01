from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np

from src.data.text import PAD_TOKEN, UNK_TOKEN


def build_embedding_matrix(
    glove_path: str | Path,
    word_dict: dict[str, int],
    *,
    dimension: int,
    seed: int,
    unmatched_init_std: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    if dimension <= 0:
        raise ValueError("dimension must be positive")
    path = Path(glove_path)
    random = np.random.default_rng(seed)
    matrix = random.normal(
        loc=0.0,
        scale=unmatched_init_std,
        size=(len(word_dict), dimension),
    ).astype(np.float32)
    matrix[word_dict[PAD_TOKEN]] = 0.0

    reserved = {PAD_TOKEN, UNK_TOKEN}
    wanted = set(word_dict) - reserved
    matched: set[str] = set()
    source_digest = hashlib.sha256()
    with path.open("rb") as file:
        for line_number, raw_line in enumerate(file, start=1):
            source_digest.update(raw_line)
            try:
                line = raw_line.decode("utf-8")
            except UnicodeDecodeError as error:
                raise ValueError(f"{path}:{line_number}: embedding is not valid UTF-8") from error
            parts = line.rstrip("\r\n").rsplit(" ", maxsplit=dimension)
            if len(parts) != dimension + 1:
                raise ValueError(
                    f"{path}:{line_number}: expected token plus {dimension} values, "
                    f"found {len(parts)} fields"
                )
            token = parts[0]
            if token not in wanted or token in matched:
                continue
            try:
                vector = np.asarray(parts[1:], dtype=np.float32)
            except ValueError as error:
                raise ValueError(
                    f"{path}:{line_number}: embedding contains a non-numeric value"
                ) from error
            if not np.isfinite(vector).all():
                raise ValueError(f"{path}:{line_number}: embedding contains a non-finite value")
            matrix[word_dict[token]] = vector
            matched.add(token)

    vocabulary_tokens = len(wanted)
    matched_tokens = len(matched)
    statistics = {
        "dimension": dimension,
        "vocabulary_tokens": vocabulary_tokens,
        "matched_tokens": matched_tokens,
        "oov_tokens": vocabulary_tokens - matched_tokens,
        "coverage": matched_tokens / vocabulary_tokens if vocabulary_tokens else 1.0,
        "source_size_bytes": path.stat().st_size,
        "source_sha256": source_digest.hexdigest(),
    }
    return matrix, statistics