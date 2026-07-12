"""
utils/config_loader.py
----------------------
PyYAML wrapper that loads ``config/config.yaml`` and applies environment-variable
overrides for critical runtime values.

Priority (highest → lowest):
    1. Environment variable (``os.environ``)
    2. Value in ``config/config.yaml``

Environment-variable mapping
-----------------------------
The following env vars are checked and, if set, override their YAML counterparts:

+-----------------------------+-----------------------------------------+
| Environment Variable        | YAML path                               |
+-----------------------------+-----------------------------------------+
| KAFKA_BOOTSTRAP_SERVERS     | kafka.bootstrap_servers                 |
| MLFLOW_TRACKING_URI         | mlflow.tracking_uri                     |
| CIPHER_ENV                  | (top-level key ``cipher_env``)          |
| LOG_LEVEL                   | (top-level key ``log_level``)           |
+-----------------------------+-----------------------------------------+

Each resolved value is logged with its source (``env`` or ``config``).

Usage::

    from src.utils.config_loader import load_config
    cfg = load_config()                        # default path
    cfg = load_config("config/config.yaml")   # explicit path

    # Access works identically whether the value came from env or YAML:
    bootstrap = cfg["kafka"]["bootstrap_servers"]
    tracking  = cfg["mlflow"]["tracking_uri"]
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

from src.utils.logger import get_logger

_logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Mapping: (yaml_section, yaml_key) → ENV_VAR_NAME
# The tuple key mirrors the nested dict path in config.yaml.
# ---------------------------------------------------------------------------
_ENV_OVERRIDES: dict[tuple[str, str], str] = {
    ("kafka",  "bootstrap_servers"): "KAFKA_BOOTSTRAP_SERVERS",
    ("mlflow", "tracking_uri"):      "MLFLOW_TRACKING_URI",
}

# Top-level keys injected directly (not nested under a section)
_ENV_TOP_LEVEL: dict[str, str] = {
    "cipher_env": "CIPHER_ENV",
    "log_level":  "LOG_LEVEL",
}


def _apply_env_overrides(cfg: dict[str, Any]) -> dict[str, Any]:
    """Apply ``os.environ`` overrides to the loaded YAML config dict.

    Args:
        cfg: Mutable config dict loaded from YAML.

    Returns:
        The same dict, mutated in-place with env-var values where set.
    """
    # --- Nested section overrides -------------------------------------------
    for (section, key), env_var in _ENV_OVERRIDES.items():
        env_val = os.environ.get(env_var)
        if env_val is not None:
            section_dict = cfg.setdefault(section, {})
            yaml_val = section_dict.get(key, "<not set>")
            section_dict[key] = env_val
            _logger.info(
                "Config override — %s.%s: '%s' (source: env var %s)",
                section, key, env_val, env_var,
            )
        else:
            yaml_val = cfg.get(section, {}).get(key, "<not set>")
            _logger.debug(
                "Config value   — %s.%s: '%s' (source: config.yaml)",
                section, key, yaml_val,
            )

    # --- Top-level overrides -------------------------------------------------
    for cfg_key, env_var in _ENV_TOP_LEVEL.items():
        env_val = os.environ.get(env_var)
        if env_val is not None:
            cfg[cfg_key] = env_val
            _logger.info(
                "Config override — %s: '%s' (source: env var %s)",
                cfg_key, env_val, env_var,
            )
        else:
            _logger.debug(
                "Config value   — %s: '%s' (source: config.yaml or default)",
                cfg_key, cfg.get(cfg_key, "<not set>"),
            )

    return cfg


def load_config(path: str = "config/config.yaml") -> dict[str, Any]:
    """Load YAML configuration and apply environment-variable overrides.

    Environment variables listed in ``_ENV_OVERRIDES`` and ``_ENV_TOP_LEVEL``
    take precedence over values in the YAML file.  Each resolved value is
    logged with its source (``env`` vs ``config``).

    Args:
        path: Path to the YAML file, relative to the current working directory
              (default: ``"config/config.yaml"``).

    Returns:
        Nested :class:`dict` mirroring the YAML structure, with any env-var
        overrides applied.  Top-level sections include ``data``,
        ``preprocessing``, ``graph_features``, ``kafka``, ``mlflow``, etc.

    Raises:
        FileNotFoundError: If the file does not exist at *path*.
        yaml.YAMLError:    If the file contains invalid YAML syntax.
    """
    config_path = Path(path)
    if not config_path.exists():
        _logger.error("Config file not found: %s", config_path.resolve())
        raise FileNotFoundError(
            f"Config file not found: {config_path.resolve()}\n"
            "Ensure you are running from the project root and that "
            "config/config.yaml exists."
        )

    _logger.info("Loading config from: %s", config_path.resolve())
    with open(config_path, "r", encoding="utf-8") as fh:
        cfg: dict[str, Any] = yaml.safe_load(fh)

    _logger.info("Config loaded — sections: %s", list(cfg.keys()))

    # Apply environment-variable overrides
    cfg = _apply_env_overrides(cfg)

    return cfg
