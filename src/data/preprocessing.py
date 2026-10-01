from __future__ import annotations

from collections import Counter
from typing import Any

import pandas as pd


def time_range(frame: pd.DataFrame) -> dict[str, str | None]:
    if frame.empty:
        return {"start": None, "end": None}
    return {
        "start": frame["time"].min().isoformat(),
        "end": frame["time"].max().isoformat(),
    }


def validate_news_references(
    behaviors: pd.DataFrame, known_news_ids: set[str], *, split_name: str
) -> None:
    unknown: set[str] = set()
    for row in behaviors.itertuples(index=False):
        unknown.update(news_id for news_id in row.history if news_id not in known_news_ids)
        unknown.update(
            news_id for news_id, _ in row.candidates if news_id not in known_news_ids
        )
    if unknown:
        preview = ", ".join(sorted(unknown)[:10])
        raise ValueError(f"{split_name} references unknown news IDs: {preview}")


def _unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))


def _prepare_history(
    history: list[str], *, max_history_length: int, deduplicate: bool
) -> list[str]:
    if deduplicate:
        # Keep the latest occurrence of each news so the order stays chronological.
        history = list(reversed(_unique(list(reversed(history)))))
    return history[-max_history_length:]


def _merge_candidates(candidates: list[tuple[str, int]]) -> list[tuple[str, int]]:
    """Real logs repeat the same news in one impression; keep one entry per
    news and mark it clicked if any of its occurrences was clicked."""
    labels: dict[str, int] = {}
    for news_id, label in candidates:
        labels[news_id] = max(labels.get(news_id, 0), label)
    return list(labels.items())


def build_train_samples(
    behaviors: pd.DataFrame,
    known_news_ids: set[str],
    *,
    max_history_length: int,
    deduplicate_history: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    validate_news_references(behaviors, known_news_ids, split_name="train")
    samples: list[dict[str, Any]] = []
    counters: Counter[str] = Counter()
    for row in behaviors.itertuples(index=False):
        history = _prepare_history(
            row.history,
            max_history_length=max_history_length,
            deduplicate=deduplicate_history,
        )
        if not history:
            counters["empty_history"] += 1
            continue
        candidates = _merge_candidates(row.candidates)
        positives = [news_id for news_id, label in candidates if label == 1]
        negatives = [news_id for news_id, label in candidates if label == 0]
        if not positives:
            counters["no_positive"] += 1
            continue
        if not negatives:
            counters["no_negative_pool"] += 1
            continue

        common = {
            "impression_id": row.impression_id,
            "user_id": row.user_id,
            "time": row.time.isoformat(),
            "history": history,
            "negative_pool": negatives,
        }
        for positive in positives:
            samples.append({**common, "positive": positive})
    counters["input_impressions"] = len(behaviors)
    counters["output_samples"] = len(samples)
    return samples, dict(counters)


def build_evaluation_samples(
    behaviors: pd.DataFrame,
    known_news_ids: set[str],
    *,
    max_history_length: int,
    split_name: str,
    deduplicate_history: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    validate_news_references(behaviors, known_news_ids, split_name=split_name)
    samples: list[dict[str, Any]] = []
    counters: Counter[str] = Counter()
    for row in behaviors.itertuples(index=False):
        history = _prepare_history(
            row.history,
            max_history_length=max_history_length,
            deduplicate=deduplicate_history,
        )
        if not history:
            counters["empty_history"] += 1
            continue
        candidates = _merge_candidates(row.candidates)
        labels = [label for _, label in candidates]
        # AUC/MRR/nDCG are undefined without both a click and a non-click.
        if 1 not in labels:
            counters["no_positive"] += 1
            continue
        if 0 not in labels:
            counters["no_negative"] += 1
            continue
        samples.append(
            {
                "impression_id": row.impression_id,
                "user_id": row.user_id,
                "time": row.time.isoformat(),
                "history": history,
                "candidates": [news_id for news_id, _ in candidates],
                "labels": labels,
            }
        )
    counters["input_impressions"] = len(behaviors)
    counters["output_samples"] = len(samples)
    return samples, dict(counters)
