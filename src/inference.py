from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from tqdm.auto import tqdm

from src.data.dataset import PAD_NEWS_ID
from src.models import NRMS


@dataclass(frozen=True)
class NewsVectorCache:
    news_ids: tuple[str, ...]
    index_by_id: dict[str, int]
    vectors: torch.Tensor

    def indices(self, news_ids: Sequence[str]) -> torch.Tensor:
        try:
            values = [self.index_by_id[news_id] for news_id in news_ids]
        except KeyError as error:
            raise KeyError(f"unknown news ID: {error.args[0]}") from error
        return torch.tensor(values, dtype=torch.long, device=self.vectors.device)


@torch.no_grad()
def precompute_news_vectors(
    model: NRMS,
    news_title_mapping: Mapping[str, np.ndarray],
    device: torch.device,
    *,
    batch_size: int = 512,
    show_progress: bool = False,
) -> NewsVectorCache:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if PAD_NEWS_ID not in news_title_mapping:
        raise ValueError(f"news_title_mapping must contain {PAD_NEWS_ID!r}")

    news_ids = tuple(news_title_mapping)
    index_by_id = {news_id: index for index, news_id in enumerate(news_ids)}
    model.eval()
    vectors: list[torch.Tensor] = []
    starts = range(0, len(news_ids), batch_size)
    for start in tqdm(
        starts,
        desc="encode news",
        unit="batch",
        leave=False,
        disable=not show_progress,
    ):
        batch_ids = news_ids[start : start + batch_size]
        titles = torch.as_tensor(
            np.stack([news_title_mapping[news_id] for news_id in batch_ids]),
            dtype=torch.long,
            device=device,
        )
        vectors.append(model.news_encoder(titles))
    return NewsVectorCache(
        news_ids=news_ids,
        index_by_id=index_by_id,
        vectors=torch.cat(vectors),
    )


@torch.no_grad()
def recommend(
    model: NRMS,
    cache: NewsVectorCache,
    history_ids: Sequence[str],
    *,
    max_history_length: int,
    top_k: int,
    candidate_ids: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    if max_history_length <= 0:
        raise ValueError("max_history_length must be positive")
    if top_k <= 0:
        raise ValueError("top_k must be positive")

    history = list(history_ids[-max_history_length:])
    cache.indices(history)
    excluded = set(history)
    source_candidates = candidate_ids if candidate_ids is not None else cache.news_ids
    filtered_candidates: list[str] = []
    seen: set[str] = set()
    for news_id in source_candidates:
        if news_id == PAD_NEWS_ID or news_id in excluded or news_id in seen:
            continue
        if news_id not in cache.index_by_id:
            raise KeyError(f"unknown news ID: {news_id}")
        seen.add(news_id)
        filtered_candidates.append(news_id)
    if not filtered_candidates:
        return []

    if history:
        history_indices = cache.indices(history)
        clicked_news = cache.vectors[history_indices].unsqueeze(0)
        history_mask = torch.ones(
            (1, len(history)), dtype=torch.bool, device=cache.vectors.device
        )
    else:
        clicked_news = torch.zeros(
            (1, 1, model.config.news_dim),
            dtype=cache.vectors.dtype,
            device=cache.vectors.device,
        )
        history_mask = torch.zeros((1, 1), dtype=torch.bool, device=cache.vectors.device)

    user_vector = model.user_encoder(clicked_news, history_mask).squeeze(0)
    candidate_indices = cache.indices(filtered_candidates)
    scores = torch.mv(cache.vectors[candidate_indices], user_vector)
    order = torch.argsort(scores, descending=True, stable=True)[:top_k]
    return [
        {
            "news_id": filtered_candidates[index],
            "score": float(scores[index].item()),
            "rank": rank,
        }
        for rank, index in enumerate(order.tolist(), start=1)
    ]