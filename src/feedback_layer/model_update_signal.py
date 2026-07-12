"""
src/feedback_layer/model_update_signal.py
------------------------------------------
Lightweight file-based signalling between the retraining pipeline and the
live Kafka consumer.

Design
------
When :class:`~src.feedback_layer.retrain_pipeline.RetrainingPipeline`
promotes a challenger model it writes a small JSON file (the *signal*).
The :class:`~src.serving_layer.kafka_consumer.FraudDetectionConsumer` checks
for this file on every *N*-th poll (default N=100) via :meth:`check_signal`
and hot-reloads the promoted model.  After reload it calls
:meth:`clear_signal` to remove the file, preventing duplicate reloads.

Three-operation contract:

* :meth:`write_signal` — atomic write (temp-file + rename) to avoid partial
  reads by the consumer.
* :meth:`check_signal` — read-only; returns the new model version or ``None``.
* :meth:`clear_signal` — delete the signal file; idempotent.

Usage::

    # In the retraining pipeline (after promoting challenger):
    sig = ModelUpdateSignal()
    sig.write_signal("v2")

    # In the Kafka consumer poll loop (every N messages):
    sig = ModelUpdateSignal()
    new_version = sig.check_signal()
    if new_version:
        reload_model(new_version)
        sig.clear_signal()
"""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from src.utils.config_loader import load_config
from src.utils.logger import get_logger

_logger = get_logger(__name__)


class ModelUpdateSignal:
    """File-based signal channel between the retraining pipeline and the consumer.

    The signal file is a JSON object with two keys:

    .. code-block:: json

        {
            "new_model_version": "<version string>",
            "written_at": "<ISO-8601 UTC timestamp>"
        }

    Attributes:
        _signal_path: :class:`~pathlib.Path` to the JSON signal file.
    """

    def __init__(
        self,
        signal_path: Optional[str] = None,
        config_path: str = "config/config.yaml",
    ) -> None:
        """Initialise with the signal file path from config.

        Args:
            signal_path: Override path to the JSON signal file.  Defaults to
                ``feedback.model_updated_signal_path`` from
                ``config/config.yaml``.
            config_path: Path to the YAML configuration file.
        """
        if signal_path is None:
            cfg = load_config(config_path)
            signal_path = cfg["feedback"]["model_updated_signal_path"]

        self._signal_path = Path(signal_path)
        self._signal_path.parent.mkdir(parents=True, exist_ok=True)
        _logger.info(
            "ModelUpdateSignal initialised | path=%s",
            self._signal_path.resolve(),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def write_signal(self, new_model_version: str) -> None:
        """Atomically write the model-updated signal file.

        Uses a temp file + rename to guarantee the consumer never observes
        a partially-written signal.

        Args:
            new_model_version: The version string of the newly promoted model
                (e.g. ``"v2"`` or an MLflow model version number).
        """
        payload = {
            "new_model_version": new_model_version,
            "written_at": datetime.now(timezone.utc).isoformat(),
        }
        # Write to a sibling temp file then atomically rename.
        tmp_fd, tmp_path = tempfile.mkstemp(
            dir=self._signal_path.parent, suffix=".tmp"
        )
        try:
            with os.fdopen(tmp_fd, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, indent=2)
            os.replace(tmp_path, self._signal_path)
        except Exception:
            # Clean up the temp file if rename failed.
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

        _logger.info(
            "ModelUpdateSignal.write_signal — signal written | version=%s, path=%s",
            new_model_version,
            self._signal_path,
        )

    def check_signal(self) -> Optional[str]:
        """Read the signal file and return the new model version if present.

        Returns:
            The ``new_model_version`` string if the signal file exists and is
            valid JSON; ``None`` otherwise (no pending update).
        """
        if not self._signal_path.exists():
            return None

        try:
            with open(self._signal_path, "r", encoding="utf-8") as fh:
                payload: dict = json.load(fh)
            version: str = payload["new_model_version"]
            _logger.info(
                "ModelUpdateSignal.check_signal — pending update detected | "
                "version=%s, written_at=%s",
                version,
                payload.get("written_at"),
            )
            return version
        except (json.JSONDecodeError, KeyError, OSError) as exc:
            _logger.warning(
                "ModelUpdateSignal.check_signal — could not read signal file "
                "'%s': %s. Ignoring.",
                self._signal_path,
                exc,
            )
            return None

    def clear_signal(self) -> None:
        """Delete the signal file after the consumer has reloaded the model.

        Idempotent — does not raise if the file has already been removed.
        """
        try:
            self._signal_path.unlink()
            _logger.info(
                "ModelUpdateSignal.clear_signal — signal file removed | path=%s",
                self._signal_path,
            )
        except FileNotFoundError:
            _logger.debug(
                "ModelUpdateSignal.clear_signal — signal file already absent, "
                "no-op | path=%s",
                self._signal_path,
            )
        except OSError as exc:
            _logger.error(
                "ModelUpdateSignal.clear_signal — failed to delete '%s': %s",
                self._signal_path,
                exc,
            )
            raise
