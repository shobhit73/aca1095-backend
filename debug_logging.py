# debug_logging.py
import logging, os

def get_logger(name: str):
    level = os.getenv("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=getattr(logging, level, logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )
    return logging.getLogger(name)
