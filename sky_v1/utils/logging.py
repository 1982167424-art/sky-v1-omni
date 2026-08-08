"""Structured, human-friendly logger for sky-v1.

- Console: ISO timestamp + level + name + message + KV extras + traceback
- Optional: file handler appended (no rotation)
"""
from __future__ import annotations

import json
import logging
import sys
from datetime import datetime
from typing import Any, Dict, Optional

_ROOT_LOGGER_SETUP = False


class SkyFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        ts = datetime.fromtimestamp(record.created).isoformat(timespec="milliseconds")
        extras: Dict[str, Any] = {}
        for k, v in record.__dict__.items():
            if k in ("args", "msg", "message", "exc_info", "exc_text",
                     "stack_info", "name", "levelname", "levelno", "pathname",
                     "filename", "module", "exc_info", "funcName", "lineno",
                     "thread", "threadName", "process", "processName",
                     "created", "msecs", "relativeCreated", "taskName"):
                continue
            if k.startswith("_"):
                continue
            extras[k] = v
        base = f"[{ts}] [{record.levelname:<7}] [{record.name}] {record.getMessage()}"
        if extras:
            try:
                base += " | " + json.dumps(extras, ensure_ascii=False, default=str)
            except Exception:
                base += " | " + str(extras)
        if record.exc_info:
            base += "\n" + self.formatException(record.exc_info)
        return base


def setup_root_logger(level: str = "INFO", log_file: Optional[str] = None) -> None:
    global _ROOT_LOGGER_SETUP
    if _ROOT_LOGGER_SETUP:
        return
    log_level = getattr(logging, str(level).upper(), logging.INFO)
    root = logging.getLogger("sky_v1")
    root.setLevel(log_level)
    root.propagate = False
    # Clear any default handlers from previous import-time attempts
    for h in list(root.handlers):
        root.removeHandler(h)

    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(log_level)
    ch.setFormatter(SkyFormatter())
    root.addHandler(ch)

    if log_file:
        try:
            fh = logging.FileHandler(log_file, encoding="utf-8")
            fh.setLevel(log_level)
            fh.setFormatter(SkyFormatter())
            root.addHandler(fh)
        except Exception:
            # Never crash logging setup; fall back to console only
            try:
                root.warning(
                    "Failed to attach file handler",
                    extra={"log_file": log_file},
                )
            except Exception:
                pass

    _ROOT_LOGGER_SETUP = True


class _KwargLoggerAdapter(logging.LoggerAdapter):
    """Adapter that accepts arbitrary keyword arguments and forwards them as `extra`.

    Usage compatibility:
        log.warning("msg", foo=42, bar="x")   ->   extra={"foo": 42, "bar": "x"}

    Respects the real Logger signature for reserved kwargs (exc_info, stack_info,
    stacklevel, extra) and merges them properly.
    """

    _RESERVED_KWARGS = frozenset(
        {"exc_info", "stack_info", "stacklevel", "extra"}
    )

    def log(self, level: int, msg: object, *args: Any, **kwargs: Any) -> None:  # type: ignore[override]
        if self.isEnabledFor(level):
            reserved: Dict[str, Any] = {}
            extras: Dict[str, Any] = {}
            for k, v in kwargs.items():
                if k in self._RESERVED_KWARGS:
                    reserved[k] = v
                else:
                    extras[k] = v
            base_extra = reserved.get("extra") or {}
            if isinstance(base_extra, dict) and extras:
                merged = dict(base_extra)
                merged.update(extras)
                reserved["extra"] = merged
            elif extras and not base_extra:
                reserved["extra"] = extras
            self.logger.log(level, msg, *args, **reserved)

    def debug(self, msg: Any, *args: Any, **kwargs: Any) -> None:  # type: ignore[override]
        self.log(logging.DEBUG, msg, *args, **kwargs)

    def info(self, msg: Any, *args: Any, **kwargs: Any) -> None:  # type: ignore[override]
        self.log(logging.INFO, msg, *args, **kwargs)

    def warning(self, msg: Any, *args: Any, **kwargs: Any) -> None:  # type: ignore[override]
        self.log(logging.WARNING, msg, *args, **kwargs)

    warn = warning

    def error(self, msg: Any, *args: Any, **kwargs: Any) -> None:  # type: ignore[override]
        self.log(logging.ERROR, msg, *args, **kwargs)

    def critical(self, msg: Any, *args: Any, **kwargs: Any) -> None:  # type: ignore[override]
        self.log(logging.CRITICAL, msg, *args, **kwargs)

    fatal = critical

    def exception(self, msg: Any, *args: Any, **kwargs: Any) -> None:  # type: ignore[override]
        kwargs.setdefault("exc_info", 1)
        self.log(logging.ERROR, msg, *args, **kwargs)

    def isEnabledFor(self, level: int) -> bool:  # type: ignore[override]
        return self.logger.isEnabledFor(level)

    def setLevel(self, level: int | str) -> None:  # type: ignore[override]
        self.logger.setLevel(level)

    @property
    def level(self) -> int:  # type: ignore[override]
        return self.logger.level

    @property
    def name(self) -> str:
        return self.logger.name

    @property
    def handlers(self) -> list:
        return self.logger.handlers

    def addHandler(self, hdlr: Any) -> None:
        self.logger.addHandler(hdlr)

    def removeHandler(self, hdlr: Any) -> None:
        self.logger.removeHandler(hdlr)

    def hasHandlers(self) -> bool:  # type: ignore[override]
        return self.logger.hasHandlers()

    @property
    def propagate(self) -> bool:
        return self.logger.propagate

    @propagate.setter
    def propagate(self, value: bool) -> None:
        self.logger.propagate = value


def get_logger(name: str) -> Any:
    if not _ROOT_LOGGER_SETUP:
        setup_root_logger()
    if not name.startswith("sky_v1.") and name != "sky_v1":
        name = f"sky_v1.{name}"
    inner = logging.getLogger(name)
    return _KwargLoggerAdapter(inner, {})
