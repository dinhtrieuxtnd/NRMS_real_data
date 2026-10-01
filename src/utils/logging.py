from __future__ import annotations

import logging
from pathlib import Path


def setup_logger(
    log_path: str | Path,
    *,
    level: str = "INFO",
    append: bool = False,
) -> logging.Logger:
    logger = logging.getLogger("nrms.preprocess")
    logger.setLevel(getattr(logging, level.upper()))
    logger.propagate = False

    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler = logging.FileHandler(
        log_path,
        mode="a" if append else "w",
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def close_logger(logger: logging.Logger) -> None:
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()