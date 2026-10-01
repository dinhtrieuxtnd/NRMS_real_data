from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np

from src.data.text import PAD_TOKEN, UNK_TOKEN


def _is_header(parts: list[str]) -> bool:
    """fastText .vec files start with a `<vocab_size> <dimension>` header line."""
    return len(parts) == 2 and all(part.isdigit() for part in parts)

def build_embedding_matrix(
    vectors_path: str | Path,
    word_dict: dict[str, int],
    *,
    dimension: int,
    seed: int,
    unmatched_init_std: float,
    case_insensitive_fallback: bool = True,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Build an embedding matrix from a text word-vector file.

    Supports both GloVe format (no header) and fastText `.vec` format
    (first line `<vocab_size> <dimension>`). When `case_insensitive_fallback`
    is enabled, a vector for e.g. "Hà" is used for vocabulary token "hà" if no
    exact-case vector exists; exact matches always take precedence.
    """
    if dimension <= 0:
        raise ValueError("dimension must be positive")
    path = Path(vectors_path)
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
    exact_matched: set[str] = set()
    header: dict[str, int] | None = None
    source_digest = hashlib.sha256()
    with path.open("rb") as file:
        for line_number, raw_line in enumerate(file, start=1):
            source_digest.update(raw_line)
            try:
                line = raw_line.decode("utf-8")
            except UnicodeDecodeError as error:
                raise ValueError(f"{path}:{line_number}: embedding is not valid UTF-8") from error
            stripped = line.rstrip("\r\n").rstrip(" ")
            if line_number == 1:
                header_parts = stripped.split()
                if _is_header(header_parts):
                    header = {
                        "vocab_size": int(header_parts[0]),
                        "dimension": int(header_parts[1]),
                    }
                    if header["dimension"] != dimension:
                        raise ValueError(
                            f"{path}: file dimension {header['dimension']} "
                            f"does not match configured dimension {dimension}"
                        )
                    continue
            parts = stripped.rsplit(" ", maxsplit=dimension)
            if len(parts) != dimension + 1:
                raise ValueError(
                    f"{path}:{line_number}: expected token plus {dimension} values, "
                    f"found {len(parts)} fields"
                )
            raw_token = parts[0]
            if raw_token in wanted:
                if raw_token in exact_matched:
                    continue
                token = raw_token
                is_exact = True
            elif case_insensitive_fallback:
                token = raw_token.lower()
                if token not in wanted or token in matched:
                    continue
                is_exact = False
            else:
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
            if is_exact:
                exact_matched.add(token)

    vocabulary_tokens = len(wanted)
    matched_tokens = len(matched)
    statistics = {
        "format": "fasttext_vec" if header is not None else "glove_txt",
        "header": header,
        "dimension": dimension,
        "vocabulary_tokens": vocabulary_tokens,
        "matched_tokens": matched_tokens,
        "exact_matched_tokens": len(exact_matched),
        "case_fallback_matched_tokens": matched_tokens - len(exact_matched),
        "oov_tokens": vocabulary_tokens - matched_tokens,
        "coverage": matched_tokens / vocabulary_tokens if vocabulary_tokens else 1.0,
        "source_size_bytes": path.stat().st_size,
        "source_sha256": source_digest.hexdigest(),
    }
    return matrix, statistics