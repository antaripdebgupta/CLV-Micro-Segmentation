"""
Central logging for the CLV pipeline.
- Colour-coded terminal output via colorlog
- Rotating file handler → logs/pipeline.log (5 MB × 10 backups)
- Single logger instance per name (no duplicate handlers on Streamlit reruns)
- Safe for multiprocess environments
"""

import logging
import os
from logging.handlers import RotatingFileHandler

LOG_DIR  = "logs"
LOG_FILE = os.path.join(LOG_DIR, "pipeline.log")

# Colour map for terminal
_COLOUR_FORMATS = {
    "DEBUG":    "%(log_color)s%(levelname)-8s%(reset)s %(blue)s%(name)s%(reset)s — %(message)s",
    "INFO":     "%(log_color)s%(levelname)-8s%(reset)s %(name)s — %(message)s",
    "WARNING":  "%(log_color)s%(levelname)-8s%(reset)s %(name)s — %(message)s",
    "ERROR":    "%(log_color)s%(levelname)-8s%(reset)s %(name)s — %(message)s",
    "CRITICAL": "%(log_color)s%(levelname)-8s%(reset)s %(name)s — %(message)s",
}

_LOG_COLORS = {
    "DEBUG":    "cyan",
    "INFO":     "green",
    "WARNING":  "yellow",
    "ERROR":    "red",
    "CRITICAL": "bold_red",
}

_FILE_FORMAT  = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT  = "%Y-%m-%d %H:%M:%S"

# Registry to avoid adding duplicate handlers when Streamlit reruns the module
_registry: dict[str, logging.Logger] = {}


def _ensure_log_dir() -> None:
    """Create logs/ directory if it doesn't exist."""
    try:
        os.makedirs(LOG_DIR, exist_ok=True)
    except OSError:
        pass  # read-only filesystem — file logging will be skipped


def _make_file_handler() -> logging.Handler | None:
    """Return a RotatingFileHandler, or None if the log directory is not writable."""
    _ensure_log_dir()
    try:
        handler = RotatingFileHandler(
            LOG_FILE,
            maxBytes=5 * 1024 * 1024,  # 5 MB
            backupCount=10,
            encoding="utf-8",
        )
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(logging.Formatter(_FILE_FORMAT, datefmt=_DATE_FORMAT))
        return handler
    except (OSError, PermissionError):
        return None


def _make_console_handler() -> logging.Handler:
    """Return a colour-coded StreamHandler."""
    try:
        import colorlog
        formatter = colorlog.ColoredFormatter(
            "%(log_color)s%(levelname)-8s%(reset)s %(name)s — %(message)s",
            datefmt=_DATE_FORMAT,
            log_colors=_LOG_COLORS,
            reset=True,
            style="%",
        )
    except ImportError:
        # colorlog not installed — fall back to plain formatter
        formatter = logging.Formatter(
            "%(levelname)-8s %(name)s — %(message)s",
            datefmt=_DATE_FORMAT,
        )

    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(formatter)
    return handler


def get_logger(name: str, level: int = logging.DEBUG) -> logging.Logger:
    """
    Return a named logger with console + file handlers.
    Safe to call multiple times with the same name — handlers are only added once.

    Args:
        name:  Logger name, typically __name__ of the calling module.
        level: Minimum log level (default DEBUG).

    Returns:
        logging.Logger instance.
    """
    if name in _registry:
        return _registry[name]

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False  # prevent double-logging to root logger

    if not logger.handlers:
        logger.addHandler(_make_console_handler())
        fh = _make_file_handler()
        if fh:
            logger.addHandler(fh)

    _registry[name] = logger
    return logger


# Module-level convenience logger
log = get_logger("clv_pipeline")


# Quick smoke-test
if __name__ == "__main__":
    _log = get_logger("smoke_test")
    _log.debug("DEBUG message — detailed tracing")
    _log.info("INFO message — pipeline step completed")
    _log.warning("WARNING message — something looks off")
    _log.error("ERROR message — something failed but we continued")
    _log.critical("CRITICAL message — pipeline aborted")
    print(f"\nLog file written to: {os.path.abspath(LOG_FILE)}")