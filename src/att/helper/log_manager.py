import logging
import sys
from pathlib import Path


def setup_logger(log_level=logging.INFO, log_dir: Path = None):
    """Centralized logging configuration."""
    handlers = [logging.StreamHandler(sys.stdout)]
    # if log dir does not exist, create the log dir
    if not log_dir.exists():
        log_dir.mkdir(parents=True, exist_ok=True)
    err_log = log_dir / "error.log"
    out_log = log_dir / "output.log"

    # Define Formatters
    # Detailed for files (with timestamps), clean for console
    file_fmt = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_fmt = logging.Formatter('%(levelname)s: %(message)s')

    # --- 1. Console Handler ---
    console_h = logging.StreamHandler(sys.stdout)
    console_h.setLevel(logging.INFO)
    console_h.setFormatter(console_fmt)

    # --- 2. Full Output File Handler ---
    output_h = logging.FileHandler(out_log, encoding='utf-8')
    output_h.setLevel(logging.INFO)
    output_h.setFormatter(file_fmt)

    # --- 3. Error Only File Handler ---
    error_h = logging.FileHandler(err_log, encoding='utf-8')
    error_h.setLevel(logging.ERROR)
    error_h.setFormatter(file_fmt)

    # --- Root Logger Setup ---
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # Remove existing handlers (prevents duplicate logs)
    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    root_logger.addHandler(console_h)
    root_logger.addHandler(output_h)
    root_logger.addHandler(error_h)

    logging.info(f"Logging initialized. Errors -> {err_log.name}, All -> {out_log.name}")