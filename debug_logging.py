# debug_logging.py
from __future__ import annotations
import logging
from contextlib import contextmanager
from time import perf_counter
import pandas as pd

def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("[%(levelname)s] %(name)s: %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger

@contextmanager
def log_time(logger: logging.Logger, message: str):
    t0 = perf_counter()
    try:
        yield
    finally:
        dt = perf_counter() - t0
        logger.info(f"{message} took {dt:.3f}s")

def log_df(logger: logging.Logger, name: str, df: pd.DataFrame):
    try:
        logger.info(f"{name}: rows={len(df) if df is not None else 0}, cols={list(df.columns) if df is not None else []}")
    except Exception:
        logger.info(f"{name}: <unprintable df>")
