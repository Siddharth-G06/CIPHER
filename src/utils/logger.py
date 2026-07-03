"""
utils/logger.py
---------------
Centralised logging setup for the CIPHER fraud-detection pipeline.

Every module obtains its logger via::

    from src.utils.logger import get_logger
    _logger = get_logger(__name__)

Log format
----------
    YYYY-MM-DDTHH:MM:SS | LEVEL    | package.module | message
"""

import logging
import sys


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Return a named, pre-configured logger for the CIPHER pipeline.

    If the logger already has handlers attached (e.g. the same module calls
    this function twice), no duplicate handler is added — making the function
    safe to call at module import time.

    Args:
        name:  Logger name; pass ``__name__`` from the calling module so log
               lines display the full ``src.feature_layer.preprocessing``-style
               module path.
        level: Minimum logging level (default: ``logging.INFO``).

    Returns:
        A :class:`logging.Logger` instance with a stdout StreamHandler and an
        ISO-8601 timestamp formatter already attached.
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(level)
        # Prevent messages from also propagating to the root logger,
        # which would cause duplicate output if the root has its own handler.
        logger.propagate = False

    return logger
