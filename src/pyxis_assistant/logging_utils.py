from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler

from pyxis_assistant.config import PROJECT_ROOT

LOG_DIR = PROJECT_ROOT / ".pyxis" / "logs"
LOG_FILE = LOG_DIR / "pyxis.log"


def configure_logging() -> None:
    logger = logging.getLogger("pyxis")
    if logger.handlers:
        return

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        "%Y-%m-%d %H:%M:%S",
    )

    file_handler = RotatingFileHandler(
        LOG_FILE,
        maxBytes=1_000_000,
        backupCount=3,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)


def get_logger(name: str) -> logging.Logger:
    configure_logging()
    return logging.getLogger(name)
