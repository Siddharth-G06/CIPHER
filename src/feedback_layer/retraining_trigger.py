"""
src/feedback_layer/retraining_trigger.py
------------------------------------------
Polling trigger that fires the retraining pipeline when any of three
conditions is met.

Trigger conditions
------------------
1. **Feedback volume** — the number of unused feedback records in
   :class:`~src.feedback_layer.feedback_store.FeedbackStore` reaches or
   exceeds ``feedback.retraining_threshold`` (default: 100).
2. **Drift detected** — the drift-events JSON file written by
   :class:`~src.ml_layer.drift_detector.RetrainingTriggerObserver` contains
   at least one event whose ``timestamp`` is newer than
   ``last_retraining_time``.
3. **Scheduled** — ``datetime.now()`` - ``last_retraining_time`` exceeds
   ``feedback.scheduled_retraining_days`` (default: 7 days).

The trigger runs in a background daemon thread (via
:meth:`RetrainingTrigger.start_polling`).  On each tick it calls
:meth:`check_triggers` and, if any trigger fires, invokes
``pipeline.run(trigger_reason=<reason>)``.

Usage::

    from src.feedback_layer.retrain_pipeline import RetrainingPipeline
    from src.feedback_layer.retraining_trigger import RetrainingTrigger

    pipeline = RetrainingPipeline()
    trigger  = RetrainingTrigger()
    trigger.start_polling(pipeline)         # returns immediately; runs as daemon
"""

from __future__ import annotations

import json
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from src.feedback_layer.feedback_store import FeedbackStore
from src.utils.config_loader import load_config
from src.utils.logger import get_logger

if TYPE_CHECKING:
    from src.feedback_layer.retrain_pipeline import RetrainingPipeline

_logger = get_logger(__name__)


class RetrainingTrigger:
    """Watchdog that evaluates three retraining conditions on a regular cadence.

    The trigger is stateless between process restarts — ``last_retraining_time``
    defaults to the current time on initialisation, meaning the *scheduled*
    trigger will not fire immediately after a fresh start.

    Attributes:
        _cfg: The ``feedback`` config sub-dict from ``config.yaml``.
        _store: Reference to the shared :class:`~src.feedback_layer.feedback_store.FeedbackStore`.
        _last_retraining_time: UTC datetime of the last completed retraining run.
        _drift_path: :class:`~pathlib.Path` to the JSON drift-events file written
            by :class:`~src.ml_layer.drift_detector.RetrainingTriggerObserver`.
        _lock: Thread lock guarding ``_last_retraining_time``.
    """

    def __init__(
        self,
        store: Optional[FeedbackStore] = None,
        config_path: str = "config/config.yaml",
    ) -> None:
        """Initialise the trigger from config.

        Args:
            store: Shared :class:`~src.feedback_layer.feedback_store.FeedbackStore`
                instance.  A new store is created if ``None`` (useful in tests).
            config_path: Path to the YAML configuration file.
        """
        cfg = load_config(config_path)
        self._cfg: dict = cfg["feedback"]
        self._drift_path: Path = Path(
            cfg["drift"]["retraining_trigger_path"]
        )

        self._store: FeedbackStore = store or FeedbackStore(
            config_path=config_path
        )
        self._last_retraining_time: datetime = datetime.now(timezone.utc)
        self._lock: threading.Lock = threading.Lock()

        _logger.info(
            "RetrainingTrigger initialised | threshold=%d, scheduled_days=%d, "
            "drift_path=%s",
            self._cfg["retraining_threshold"],
            self._cfg["scheduled_retraining_days"],
            self._drift_path,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check_triggers(self) -> Optional[str]:
        """Evaluate all three trigger conditions and return the first fired.

        Conditions are checked in priority order:
        ``feedback_volume`` → ``drift_detected`` → ``scheduled``.

        Returns:
            A non-empty trigger-reason string if any condition is met, or
            ``None`` if retraining should not run yet.
        """
        # Trigger 1 — feedback volume
        try:
            unused = self._store.get_unused_feedback()
            threshold: int = int(self._cfg["retraining_threshold"])
            if len(unused) >= threshold:
                _logger.info(
                    "RetrainingTrigger — FIRED: feedback_volume_threshold "
                    "| unused=%d >= threshold=%d",
                    len(unused),
                    threshold,
                )
                return "feedback_volume_threshold"
        except Exception as exc:
            _logger.error(
                "RetrainingTrigger.check_triggers — error checking feedback "
                "volume: %s",
                exc,
            )

        # Trigger 2 — drift event newer than last retraining
        try:
            drift_reason = self._check_drift_trigger()
            if drift_reason:
                return drift_reason
        except Exception as exc:
            _logger.error(
                "RetrainingTrigger.check_triggers — error checking drift "
                "events: %s",
                exc,
            )

        # Trigger 3 — scheduled
        try:
            scheduled_days: int = int(self._cfg["scheduled_retraining_days"])
            with self._lock:
                elapsed = datetime.now(timezone.utc) - self._last_retraining_time
            if elapsed >= timedelta(days=scheduled_days):
                _logger.info(
                    "RetrainingTrigger — FIRED: scheduled | elapsed=%s, "
                    "threshold=%d days",
                    elapsed,
                    scheduled_days,
                )
                return "scheduled"
        except Exception as exc:
            _logger.error(
                "RetrainingTrigger.check_triggers — error checking schedule: %s",
                exc,
            )

        return None

    def start_polling(
        self,
        pipeline: "RetrainingPipeline",
        interval_seconds: Optional[int] = None,
    ) -> threading.Thread:
        """Launch the trigger polling loop in a background daemon thread.

        The thread runs indefinitely until the process exits.  If a trigger
        fires, the pipeline is executed synchronously inside the thread
        (blocking the next poll cycle until completion).

        Args:
            pipeline: The :class:`~src.feedback_layer.retrain_pipeline.RetrainingPipeline`
                to invoke when a trigger fires.
            interval_seconds: Poll interval in seconds.  Defaults to
                ``feedback.trigger_poll_interval_seconds`` from config (300 s).

        Returns:
            The started daemon :class:`~threading.Thread`.
        """
        if interval_seconds is None:
            interval_seconds = int(
                self._cfg.get("trigger_poll_interval_seconds", 300)
            )

        def _poll_loop() -> None:
            _logger.info(
                "RetrainingTrigger — polling thread started | interval=%ds",
                interval_seconds,
            )
            while True:
                try:
                    reason = self.check_triggers()
                    if reason:
                        _logger.info(
                            "RetrainingTrigger — running pipeline | trigger=%s",
                            reason,
                        )
                        result = pipeline.run(trigger_reason=reason)
                        _logger.info(
                            "RetrainingTrigger — pipeline completed | "
                            "promoted=%s, run_id=%s",
                            result.get("promoted"),
                            result.get("mlflow_run_id"),
                        )
                        # Update the clock regardless of promotion outcome so
                        # the scheduled trigger resets.
                        with self._lock:
                            self._last_retraining_time = datetime.now(
                                timezone.utc
                            )
                except Exception as exc:
                    _logger.error(
                        "RetrainingTrigger._poll_loop — unexpected error: %s",
                        exc,
                        exc_info=True,
                    )

                time.sleep(interval_seconds)

        thread = threading.Thread(target=_poll_loop, daemon=True, name="cipher-retrain-trigger")
        thread.start()
        _logger.info(
            "RetrainingTrigger.start_polling — daemon thread started | "
            "name=%s, interval=%ds",
            thread.name,
            interval_seconds,
        )
        return thread

    def update_last_retraining_time(
        self, ts: Optional[datetime] = None
    ) -> None:
        """Manually update the last retraining timestamp (useful after manual runs).

        Args:
            ts: UTC datetime to record.  Defaults to ``datetime.now(utc)``
                if not provided.
        """
        with self._lock:
            self._last_retraining_time = ts or datetime.now(timezone.utc)
        _logger.info(
            "RetrainingTrigger — last_retraining_time updated to %s",
            self._last_retraining_time.isoformat(),
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _check_drift_trigger(self) -> Optional[str]:
        """Check whether any drift event post-dates the last retraining.

        Reads the JSON array produced by
        :class:`~src.ml_layer.drift_detector.RetrainingTriggerObserver` and
        checks whether any event's ``timestamp`` is newer than
        ``self._last_retraining_time``.

        Returns:
            ``"drift_detected"`` if a qualifying event is found, else ``None``.
        """
        if not self._drift_path.exists():
            _logger.debug(
                "RetrainingTrigger._check_drift_trigger — drift events file "
                "not found at '%s'; skipping.",
                self._drift_path,
            )
            return None

        try:
            with open(self._drift_path, "r", encoding="utf-8") as fh:
                events: list = json.load(fh)
        except (json.JSONDecodeError, OSError) as exc:
            _logger.warning(
                "RetrainingTrigger._check_drift_trigger — could not read '%s': %s",
                self._drift_path,
                exc,
            )
            return None

        if not isinstance(events, list) or not events:
            return None

        with self._lock:
            cutoff = self._last_retraining_time

        for event in events:
            ts_raw = event.get("timestamp")
            if not ts_raw:
                continue
            try:
                ts = datetime.fromisoformat(str(ts_raw))
                # Make timezone-aware if naive (assume UTC from drift_detector).
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                if ts > cutoff:
                    _logger.info(
                        "RetrainingTrigger — FIRED: drift_detected | "
                        "event_ts=%s, cutoff=%s",
                        ts.isoformat(),
                        cutoff.isoformat(),
                    )
                    return "drift_detected"
            except (ValueError, TypeError) as exc:
                _logger.debug(
                    "RetrainingTrigger._check_drift_trigger — could not parse "
                    "timestamp '%s': %s",
                    ts_raw,
                    exc,
                )
                continue

        return None
