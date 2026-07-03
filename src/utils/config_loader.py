"""
utils/config_loader.py
----------------------
Thin PyYAML wrapper that loads config/config.yaml and returns a nested dict.

Usage::

    from src.utils.config_loader import load_config
    cfg = load_config()                        # uses default path
    cfg = load_config("config/config.yaml")   # explicit path
"""

from pathlib import Path
from typing import Any

import yaml

from src.utils.logger import get_logger

_logger = get_logger(__name__)


def load_config(path: str = "config/config.yaml") -> dict[str, Any]:
    """
    Load a YAML configuration file and return it as a nested dict.

    Args:
        path: Path to the YAML file, interpreted relative to the current
              working directory (default: ``"config/config.yaml"``).
              When running from the project root this resolves to
              ``<project_root>/config/config.yaml``.

    Returns:
        Nested :class:`dict` mirroring the YAML structure.  Top-level keys
        are ``data``, ``preprocessing``, ``graph_features``, and
        ``feature_selection``.

    Raises:
        FileNotFoundError: If the file does not exist at *path*.
        yaml.YAMLError:    If the file contains invalid YAML syntax.
    """
    config_path = Path(path)
    if not config_path.exists():
        _logger.error(f"Config file not found: {config_path.resolve()}")
        raise FileNotFoundError(
            f"Config file not found: {config_path.resolve()}\n"
            "Ensure you are running from the project root and that "
            "config/config.yaml exists."
        )

    _logger.info(f"Loading config: {config_path.resolve()}")
    with open(config_path, "r", encoding="utf-8") as fh:
        cfg: dict[str, Any] = yaml.safe_load(fh)

    _logger.info(
        f"Config loaded — sections: {list(cfg.keys())}"
    )
    return cfg
