from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


PAD_NEWS_ID = "<PAD_NEWS>"


class NRMSTrainDataset(Dataset):
    """Build NRMS training examples with fresh negative sampling per access."""

    def __init__(
        self,
        samples: Sequence[Mapping[str, Any]],
        news_title_mapping: Mapping[str, np.ndarray],
        *,
        max_history_length: int = 50,
        num_negatives: int = 4,
    ) -> None:
        _validate_common_arguments(
            news_title_mapping,
            max_history_length=max_history_length,
        )
        if num_negatives <= 0:
            raise ValueError("num_negatives must be positive")

        self.samples = samples
        self.news_title_mapping = news_title_mapping
        self.max_history_length = max_history_length
        self.num_negatives = num_negatives

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        negative_pool = sample["negative_pool"]
        if not negative_pool:
            raise ValueError(f"training sample at index {index} has an empty negative_pool")

        if len(negative_pool) >= self.num_negatives:
            negatives = random.sample(negative_pool, self.num_negatives)
        else:
            negatives = random.choices(negative_pool, k=self.num_negatives)

        candidate_ids = [sample["positive"], *negatives]
        history, history_mask = _build_history(
            sample["history"],
            self.news_title_mapping,
            self.max_history_length,
        )
        return {
            "impression_id": sample["impression_id"],
            "history": history,
            "history_mask": history_mask,
            "candidates": _stack_titles(candidate_ids, self.news_title_mapping),
            "labels": torch.tensor(
                [1, *([0] * self.num_negatives)], dtype=torch.float32
            ),
        }


class NRMSEvaluationDataset(Dataset):
    """Build NRMS evaluation examples without changing candidate order."""

    def __init__(
        self,
        samples: Sequence[Mapping[str, Any]],
        news_title_mapping: Mapping[str, np.ndarray],
        *,
        max_history_length: int = 50,
    ) -> None:
        _validate_common_arguments(
            news_title_mapping,
            max_history_length=max_history_length,
        )
        self.samples = samples
        self.news_title_mapping = news_title_mapping
        self.max_history_length = max_history_length

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        history, history_mask = _build_history(
            sample["history"],
            self.news_title_mapping,
            self.max_history_length,
        )
        return {
            "impression_id": sample["impression_id"],
            "candidate_ids": list(sample["candidates"]),
            "history": history,
            "history_mask": history_mask,
            "candidates": _stack_titles(
                sample["candidates"], self.news_title_mapping
            ),
            "labels": torch.tensor(sample["labels"], dtype=torch.float32),
        }


class NRMSVectorEvaluationDataset(Dataset):
    """Build evaluation examples using indexes into a shared news-vector cache."""

    def __init__(
        self,
        samples: Sequence[Mapping[str, Any]],
        news_index: Mapping[str, int],
        *,
        max_history_length: int = 50,
    ) -> None:
        if max_history_length <= 0:
            raise ValueError("max_history_length must be positive")
        if PAD_NEWS_ID not in news_index:
            raise ValueError(f"news_index must contain {PAD_NEWS_ID!r}")
        self.samples = samples
        self.news_index = news_index
        self.max_history_length = max_history_length

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        history_ids = list(sample["history"][-self.max_history_length :])
        history_size = len(history_ids)
        padded_history = history_ids + [PAD_NEWS_ID] * (
            self.max_history_length - history_size
        )
        history_mask = torch.zeros(self.max_history_length, dtype=torch.bool)
        history_mask[:history_size] = True
        return {
            "impression_id": sample["impression_id"],
            "candidate_ids": list(sample["candidates"]),
            "history_indices": _news_indices(padded_history, self.news_index),
            "history_mask": history_mask,
            "candidate_indices": _news_indices(
                sample["candidates"], self.news_index
            ),
            "labels": torch.tensor(sample["labels"], dtype=torch.float32),
        }


def nrms_evaluation_collate(
    batch: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if not batch:
        raise ValueError("evaluation batch must not be empty")

    candidate_counts = torch.tensor(
        [item["candidates"].shape[0] for item in batch], dtype=torch.long
    )
    max_candidates = int(candidate_counts.max().item())
    candidate_mask = (
        torch.arange(max_candidates).unsqueeze(0) < candidate_counts.unsqueeze(1)
    )
    return {
        "impression_id": [item["impression_id"] for item in batch],
        "candidate_ids": [list(item["candidate_ids"]) for item in batch],
        "history": torch.stack([item["history"] for item in batch]),
        "history_mask": torch.stack([item["history_mask"] for item in batch]),
        "candidates": pad_sequence(
            [item["candidates"] for item in batch],
            batch_first=True,
            padding_value=0,
        ),
        "candidate_mask": candidate_mask,
        "labels": pad_sequence(
            [item["labels"] for item in batch],
            batch_first=True,
            padding_value=0.0,
        ),
    }


def nrms_vector_evaluation_collate(
    batch: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if not batch:
        raise ValueError("evaluation batch must not be empty")
    candidate_counts = torch.tensor(
        [item["candidate_indices"].shape[0] for item in batch], dtype=torch.long
    )
    max_candidates = int(candidate_counts.max().item())
    return {
        "impression_id": [item["impression_id"] for item in batch],
        "candidate_ids": [list(item["candidate_ids"]) for item in batch],
        "history_indices": torch.stack(
            [item["history_indices"] for item in batch]
        ),
        "history_mask": torch.stack([item["history_mask"] for item in batch]),
        "candidate_indices": pad_sequence(
            [item["candidate_indices"] for item in batch],
            batch_first=True,
            padding_value=0,
        ),
        "candidate_mask": (
            torch.arange(max_candidates).unsqueeze(0)
            < candidate_counts.unsqueeze(1)
        ),
        "labels": pad_sequence(
            [item["labels"] for item in batch],
            batch_first=True,
            padding_value=0.0,
        ),
    }


def _validate_common_arguments(
    news_title_mapping: Mapping[str, np.ndarray],
    *,
    max_history_length: int,
) -> None:
    if max_history_length <= 0:
        raise ValueError("max_history_length must be positive")
    if PAD_NEWS_ID not in news_title_mapping:
        raise ValueError(f"news_title_mapping must contain {PAD_NEWS_ID!r}")


def _build_history(
    history_ids: Sequence[str],
    news_title_mapping: Mapping[str, np.ndarray],
    max_history_length: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    truncated_history = list(history_ids[-max_history_length:])
    history_size = len(truncated_history)
    padded_history = truncated_history + [PAD_NEWS_ID] * (
        max_history_length - history_size
    )
    history_mask = torch.zeros(max_history_length, dtype=torch.bool)
    history_mask[:history_size] = True
    return _stack_titles(padded_history, news_title_mapping), history_mask


def _stack_titles(
    news_ids: Sequence[str],
    news_title_mapping: Mapping[str, np.ndarray],
) -> torch.Tensor:
    try:
        titles = [news_title_mapping[news_id] for news_id in news_ids]
    except KeyError as error:
        raise KeyError(f"unknown news ID: {error.args[0]}") from error
    return torch.as_tensor(np.stack(titles), dtype=torch.long)


def _news_indices(
    news_ids: Sequence[str],
    news_index: Mapping[str, int],
) -> torch.Tensor:
    try:
        indices = [news_index[news_id] for news_id in news_ids]
    except KeyError as error:
        raise KeyError(f"unknown news ID: {error.args[0]}") from error
    return torch.tensor(indices, dtype=torch.long)