from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd


BEHAVIOR_COLUMNS = (
    "impression_id",
    "user_id",
    "time",
    "history",
    "candidates",
)
NEWS_COLUMNS = (
    "news_id",
    "title",
)
# Raw CSV layouts:
#   behaviors (no header): [impression_id,]user_id,time,history,impressions
#   news (optional header "newsId,title"): news_id,title
RAW_BEHAVIOR_FIELDS = (4, 5)
RAW_NEWS_FIELDS = (2,)
NEWS_HEADER_IDS = {"newsid", "news_id"}
TIME_FORMAT = "%Y-%m-%d %H:%M:%S"

def _read_csv_rows(
    file_path: Path, expected_columns: tuple[int, ...]
) -> list[tuple[int, list[str]]]:
    rows: list[tuple[int, list[str]]] = []
    expected = " or ".join(str(count) for count in expected_columns)
    with file_path.open("r", encoding="utf-8-sig", newline="") as file:
        for line_number, row in enumerate(csv.reader(file), start=1):
            if not row or all(not value.strip() for value in row):
                continue
            if len(row) not in expected_columns:
                raise ValueError(
                    f"{file_path}:{line_number}: expected {expected} columns, "
                    f"found {len(row)}"
                )
            rows.append((line_number, row))
    return rows


def parse_candidates(value: str, *, file_path: Path, line_number: int) -> list[tuple[str, int]]:
    candidates: list[tuple[str, int]] = []
    for item in value.split():
        try:
            news_id, raw_label = item.rsplit("-", 1)
        except ValueError as error:
            raise ValueError(
                f"{file_path}:{line_number}: invalid candidate {item!r}"
            ) from error

        if not news_id or raw_label not in {"0", "1"}:
            raise ValueError(
                f"{file_path}:{line_number}: invalid candidate {item!r}; "
                "expected NEWS_ID-0 or NEWS_ID-1"
            )
        candidates.append((news_id, int(raw_label)))

    if not candidates:
        raise ValueError(f"{file_path}:{line_number}: impressions contain no candidates")
    return candidates


def read_news(file_path: str | Path) -> pd.DataFrame:
    path = Path(file_path)
    records: list[dict[str, str]] = []
    for index, (line_number, row) in enumerate(_read_csv_rows(path, RAW_NEWS_FIELDS)):
        news_id, title = (value.strip() for value in row)
        if index == 0 and news_id.lower() in NEWS_HEADER_IDS:
            continue
        if not news_id:
            raise ValueError(f"{path}:{line_number}: news_id is empty")
        records.append({"news_id": news_id, "title": title})
    return pd.DataFrame.from_records(records, columns=NEWS_COLUMNS)


def read_behaviors(file_path: str | Path, *, split_name: str) -> pd.DataFrame:
    """Read a behaviors CSV. Impression ids are prefixed with the split name
    (e.g. ``train-12``); when the raw row has no id, the line number is used."""
    path = Path(file_path)
    records: list[dict[str, object]] = []
    for line_number, row in _read_csv_rows(path, RAW_BEHAVIOR_FIELDS):
        values = [value.strip() for value in row]
        raw_impression_id = values.pop(0) if len(values) == 5 else str(line_number)
        if not raw_impression_id:
            raise ValueError(f"{path}:{line_number}: impression_id is empty")
        user_id, raw_time, raw_history, raw_candidates = values
        if not user_id:
            raise ValueError(f"{path}:{line_number}: user_id is empty")
        try:
            timestamp = pd.to_datetime(raw_time, format=TIME_FORMAT, errors="raise")
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"{path}:{line_number}: invalid timestamp {raw_time!r}"
            ) from error

        records.append(
            {
                "impression_id": f"{split_name}-{raw_impression_id}",
                "user_id": user_id,
                "time": timestamp,
                "history": raw_history.split(),
                "candidates": parse_candidates(
                    raw_candidates,
                    file_path=path,
                    line_number=line_number,
                ),
            }
        )
    return pd.DataFrame.from_records(records, columns=BEHAVIOR_COLUMNS)
