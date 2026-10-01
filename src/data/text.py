from __future__ import annotations

import re
import unicodedata
from collections import Counter
from collections.abc import Iterable

import numpy as np
import pandas as pd


PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
PAD_NEWS = "<PAD_NEWS>"
TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", flags=re.UNICODE)


def tokenize(text: str, *, lowercase: bool = True) -> list[str]:
    normalized = unicodedata.normalize("NFKC", str(text))
    normalized = " ".join(normalized.split())
    if lowercase:
        normalized = normalized.lower()
    return TOKEN_PATTERN.findall(normalized)


def build_word_dict(
    titles: Iterable[str],
    *,
    lowercase: bool = True,
    min_frequency: int = 1
) -> dict[str, int]:
    if min_frequency <= 0:
        raise ValueError("min_frequency must be positive")
    counts = Counter(
        token for title in titles for token in tokenize(
            title, lowercase=lowercase)
    )
    tokens = sorted(
        (
            token for token, count in counts.items()
            if count >= min_frequency
        ),
        key=lambda token: (-counts[token], token),
    )
    return {
        PAD_TOKEN: 0,
        UNK_TOKEN: 1,
        **{token: index + 2
            for index, token in enumerate(tokens)}
    }


def merge_news(*news_frames: pd.DataFrame) -> pd.DataFrame:
    if not news_frames:
        raise ValueError("merge_news requires at least one news frame")
    titles: dict[str, str] = {}
    conflicts: set[str] = set()
    for frame in news_frames:
        for news_id, title in zip(frame["news_id"], frame["title"], strict=True):
            if news_id in titles and titles[news_id] != title:
                conflicts.add(news_id)
            titles.setdefault(news_id, title)
    if conflicts:
        preview = ", ".join(sorted(conflicts)[:5])
        raise ValueError(
            "Conflicting titles for duplicate news IDs: "
            + f"{preview}")

    combined = pd.concat(news_frames, ignore_index=True)
    return combined.drop_duplicates(
        subset="news_id",
        keep="first"
    ).reset_index(drop=True)


def encode_title(
    title: str,
    word_dict: dict[str, int],
    *,
    max_title_length: int,
    lowercase: bool = True,
) -> np.ndarray:
    if max_title_length <= 0:
        raise ValueError("max_title_length must be positive")
    unknown_id = word_dict[UNK_TOKEN]
    token_ids = [
        word_dict.get(token, unknown_id)
        for token in tokenize(
            title,
            lowercase=lowercase
        )[:max_title_length]
    ]
    encoded = np.zeros(max_title_length, dtype=np.int32)
    encoded[: len(token_ids)] = token_ids
    return encoded


def build_news_title_mapping(
    news: pd.DataFrame,
    word_dict: dict[str, int],
    *,
    max_title_length: int,
    lowercase: bool = True,
) -> dict[str, np.ndarray]:
    mapping = {PAD_NEWS: np.zeros(max_title_length, dtype=np.int32)}
    for row in news.itertuples(index=False):
        mapping[row.news_id] = encode_title(
            row.title,
            word_dict,
            max_title_length=max_title_length,
            lowercase=lowercase,
        )
    return mapping