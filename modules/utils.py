"""
modules/utils.py

Common utility functions for MedAssist AI.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Iterable


def setup_logger(name: str = "MedAssistAI",
                 log_file: str = "logs/medassist.log") -> logging.Logger:
    """
    Create and return a configured logger.
    """
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s"
    )

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


LOGGER = setup_logger()


def ensure_directories(directories: Iterable[str]) -> None:
    """
    Create directories if they don't already exist.
    """
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)


def timestamp() -> str:
    """
    Current timestamp suitable for filenames.
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def validate_pdf(file_path: str) -> bool:
    """
    Returns True if file exists and has a .pdf extension.
    """
    path = Path(file_path)
    return path.exists() and path.suffix.lower() == ".pdf"


def save_uploaded_file(uploaded_file, destination_folder: str) -> Path:
    """
    Save a Streamlit uploaded file.
    """
    destination = Path(destination_folder)
    destination.mkdir(parents=True, exist_ok=True)

    output_file = destination / uploaded_file.name

    with open(output_file, "wb") as f:
        f.write(uploaded_file.getbuffer())

    LOGGER.info("Saved uploaded file: %s", output_file)

    return output_file


def banner(title: str) -> None:
    """
    Print a formatted console banner.
    """
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


if __name__ == "__main__":
    banner("MedAssist AI Utilities")
    ensure_directories(
        [
            "../documents",
            "../uploads",
            "../images",
            "../vector_store",
            "../logs",
        ]
    )
    LOGGER.info("Utilities module initialized successfully.")
    print("Timestamp:", timestamp())
