# debug_logging.py
from __future__ import annotations

import json
import logging
import os
import sys
import time
import uuid
from contextvars import ContextVar
from functools import wraps
from typing import Any, Callable, Dict, Optional

try:
    import pandas as pd
except Exception:  # pandas may not be available in every context
    pd = None  # type: ignore

# --------- Context (request/job scoped) ----------
request_id_ctx: ContextVar[str] = ContextVar("request_id", default="-")
route_ctx: ContextVar[str] = ContextVar("route", default="-")

def set_request_context(request_id: Optional[str] = None, route: Optional[str] = None) -> str:
    rid = request_id or str(uuid.uuid4())
    request_id_ctx.set(rid)
    if route:
        route_ctx.set(route)
    return rid

def clear_request_context():
    request_id_ctx.set("-")
    route_ctx.set("-")


# --------- Formatters ----------
class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        base: Dict[str, Any] = {
            "level": record.levelname,
            "ts": int(record.created * 1000),
            "msg": record.getMessage(),
            "logger": record.name,
            "request_id": request_id_ctx.get("-"),
            "route": route_ctx.get("-"),
            "module": record.module,
            "func": record.funcName,
            "line": record.lineno,
        }
        if record.exc_info:
            base["exc_info"] = self.formatException(record.exc_info)
        # include extra dict if present
        if hasattr(record, "extra_data") and isinstance(record.extra_data, dict):
            base.update(record.extra_data)
        return json.dumps(base, ensure_ascii=False)


class PrettyFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        rid = request_id_ctx.get("-")
        route = route_ctx.get("-")
        prefix = f"[{record.levelname}] {record.name} rid={rid} route={route}"
        base = f"{prefix} :: {record.getMessage()}"
        if record.exc_info:
            base += "\n" + self.formatException(record.exc_info)
        return base


# --------- Setup ----------
_default_logger: Optional[logging.Logger] = None

def setup_logging(
    *,
    level: str | int = None,
    json_logs: Optional[bool] = None,
    logfile: Optional[str] = None,
) -> logging.Logger:
    """
    Call once at process boot. Safe to call multiple times (idempotent).
    """
    global _default_logger
    if _default_logger:
        return _default_logger

    lvl = (level or os.getenv("LOG_LEVEL", "INFO")).upper()
    use_json = bool(int(os.getenv("LOG_JSON", "0"))) if json_logs is None else json_logs
    logfile = logfile or os.getenv("LOG_FILE", "")

    root = logging.getLogger()
    root.setLevel(lvl)

    # Remove existing handlers to avoid dupes on hot-reload
    for h in list(root.handlers):
        root.removeHandler(h)

    fmt = JsonFormatter() if use_json else PrettyFormatter()

    # Console
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(lvl)
    sh.setFormatter(fmt)
    root.addHandler(sh)

    # Optional rotating file
    if logfile:
        try:
            from logging.handlers import RotatingFileHandler
            fh = RotatingFileHandler(logfile, maxBytes=5_000_000, backupCount=3)
            fh.setLevel(lvl)
            fh.setFormatter(fmt)
            root.addHandler(fh)
        except Exception:
            root.warning("Failed to attach file handler; continuing with console only.")

    _default_logger = logging.getLogger("aca")
    _default_logger.info("Logging initialized", extra={"extra_data": {"json": use_json, "level": lvl, "logfile": logfile}})
    return _default_logger


def get_logger(name: str | None = None) -> logging.Logger:
    return logging.getLogger(name or "aca")


# --------- Helpers ----------
def df_info(df, name: str) -> Dict[str, Any]:
    if pd is None or df is None:
        return {"df": name, "present": df is not None}
    try:
        rows, cols = (df.shape if hasattr(df, "shape") else (None, None))
        return {"df": name, "rows": rows, "cols": cols, "empty": getattr(df, "empty", None)}
    except Exception:
        return {"df": name, "info": "unavailable"}

def log_df(logger: logging.Logger, df, name: str, level: int = logging.DEBUG):
    logger.log(level, f"DataFrame[{name}]", extra={"extra_data": df_info(df, name)})

def log_time(logger: logging.Logger, label: str):
    """
    Usage:
        with log_time(logger, "build_interim"):
            ... work ...
    """
    class _T:
        def __enter__(self_s):
            self_s.t0 = time.perf_counter()
            logger.debug(f"{label} START")
        def __exit__(self_s, exc_type, exc, tb):
            dt = time.perf_counter() - self_s.t0
            logger.info(f"{label} END ({dt:.3f}s)", extra={"extra_data": {"elapsed_sec": round(dt, 3), "label": label}})
    return _T()

def log_call(logger: logging.Logger):
    """
    Decorator: log function call with key argument shapes/sizes automatically.
    """
    def _decor(fn: Callable):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            try:
                # Build a small args summary
                arg_summ = {}
                for i, a in enumerate(args[:6]):  # limit
                    if pd is not None and hasattr(a, "shape"):
                        arg_summ[f"arg{i}"] = {"rows": a.shape[0], "cols": a.shape[1]}
                    elif isinstance(a, (str, int, float, bool, type(None))):
                        arg_summ[f"arg{i}"] = a
                    else:
                        arg_summ[f"arg{i}"] = type(a).__name__
                for k, v in list(kwargs.items())[:10]:
                    if pd is not None and hasattr(v, "shape"):
                        arg_summ[k] = {"rows": v.shape[0], "cols": v.shape[1]}
                    elif isinstance(v, (str, int, float, bool, type(None))):
                        arg_summ[k] = v
                    else:
                        arg_summ[k] = type(v).__name__
                logger.debug(f"CALL {fn.__name__}", extra={"extra_data": arg_summ})
            except Exception:
                logger.debug(f"CALL {fn.__name__}")
            return fn(*args, **kwargs)
        return wrapper
    return _decor
