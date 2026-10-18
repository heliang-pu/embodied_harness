"""Logging setup: a readable console stream plus one JSON Lines file per episode (design doc §12)."""

from __future__ import annotations

import faulthandler
import json
import logging
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path
from types import TracebackType

from .context import CONTEXT_FIELDS, current_context

ROOT_LOGGER = "embodied_harness"


class ContextFilter(logging.Filter):
    """Copies the current log context (episode, state, call ids, t) onto every record."""

    def filter(self, record: logging.LogRecord) -> bool:
        context = current_context()
        for name in CONTEXT_FIELDS:
            if not hasattr(record, name):
                setattr(record, name, context.get(name))
        return True


def _code_text(record: logging.LogRecord) -> str | None:
    code = getattr(record, "code", None)
    return None if code is None else str(getattr(code, "value", code))


class JsonFormatter(logging.Formatter):
    """One JSON object per line, so a log file can be filtered with grep or jq."""

    def format(self, record: logging.LogRecord) -> str:
        entry: dict[str, object] = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
            "thread": record.threadName,
        }
        for name in CONTEXT_FIELDS:
            value = getattr(record, name, None)
            if value is not None:
                entry[name] = value
        code = _code_text(record)
        if code is not None:
            entry["code"] = code
        if record.exc_info:
            entry["exc"] = self.formatException(record.exc_info)
        return json.dumps(entry, ensure_ascii=False, default=str)


class ConsoleFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        t = getattr(record, "t", None)
        clock = f"{t:8.2f}s" if isinstance(t, (int, float)) else " " * 9
        episode = getattr(record, "episode_id", None) or "-"
        state = getattr(record, "state", None) or "-"
        call = getattr(record, "skill_call_id", None) or "-"
        code = _code_text(record)
        suffix = f" [{code}]" if code else ""
        name = record.name.removeprefix(ROOT_LOGGER + ".")
        line = (
            f"{episode:<7} {clock} {record.levelname:<7} {state:<7} {call:<7} "
            f"{name}: {record.getMessage()}{suffix}"
        )
        if record.exc_info:
            line += "\n" + self.formatException(record.exc_info)
        return line


def _package_logger() -> logging.Logger:
    logger = logging.getLogger(ROOT_LOGGER)
    if logger.level == logging.NOTSET or logger.level > logging.DEBUG:
        logger.setLevel(logging.DEBUG)
    return logger


def configure_logging(console_level: str | int = "INFO") -> logging.Logger:
    """Attach the console handler once; calling again only changes its level."""
    logger = _package_logger()
    logger.propagate = False
    console = next((h for h in logger.handlers if getattr(h, "embodied_harness_console", False)), None)
    if console is None:
        console = logging.StreamHandler(sys.stderr)
        console.embodied_harness_console = True  # type: ignore[attr-defined]
        console.addFilter(ContextFilter())
        console.setFormatter(ConsoleFormatter())
        logger.addHandler(console)
    console.setLevel(console_level)
    return logger


def open_file_log(path: Path, level: str | int = "DEBUG") -> logging.Handler:
    path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setLevel(level)
    handler.addFilter(ContextFilter())
    handler.setFormatter(JsonFormatter())
    _package_logger().addHandler(handler)
    return handler


def close_file_log(handler: logging.Handler) -> None:
    logging.getLogger(ROOT_LOGGER).removeHandler(handler)
    handler.close()


def install_crash_handlers() -> None:
    """Send uncaught exceptions from any thread to the log, and let a hang dump every stack."""
    logger = logging.getLogger(ROOT_LOGGER)

    def log_uncaught(exc_type: type[BaseException], exc: BaseException, tb: TracebackType | None) -> None:
        logger.critical("uncaught exception", exc_info=(exc_type, exc, tb))

    def log_thread(args: threading.ExceptHookArgs) -> None:
        name = args.thread.name if args.thread else "?"
        exc_info = (args.exc_type, args.exc_value, args.exc_traceback)
        logger.critical("uncaught exception in thread %s", name, exc_info=exc_info)  # type: ignore[arg-type]

    sys.excepthook = log_uncaught
    threading.excepthook = log_thread
    if not faulthandler.is_enabled():
        faulthandler.enable()
