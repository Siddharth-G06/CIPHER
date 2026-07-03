"""
src/ml_layer/drift_detector.py
-------------------------------
Stateful concept-drift detector for the CIPHER fraud-detection pipeline.

Architecture
------------
:class:`DriftDetector` is the *subject* in the Observer pattern.  Any number
of :class:`~src.ml_layer.drift_observer.DriftObserver` instances can be
registered; they are called synchronously whenever drift is detected.

Two independent drift signals are monitored:

1. **ADWIN** (river library) — tracks the rolling prediction error rate.
   Fires when the error distribution shifts significantly.  One scalar
   ``(0 or 1)`` is fed to ADWIN per :meth:`~DriftDetector.update` call.

2. **Population Stability Index (PSI)** — tracks the feature-value
   distribution of a sliding window of recent transactions against a
   baseline computed from training data.  Fires when any monitored feature
   exceeds the *critical* PSI threshold, regardless of ADWIN state.

Concrete observer implementations also live in this module:

* :class:`LoggingObserver` — logs drift events at WARNING level.
* :class:`RetrainingTriggerObserver` — appends drift events to a JSON file
  that the retraining pipeline polls.

Typical usage::

    from src.ml_layer.drift_detector import DriftDetector, LoggingObserver

    detector = DriftDetector()
    detector.set_psi_baseline(X_train, feature_cols=["card_degree_1h", ...])
    detector.register_observer(LoggingObserver())

    for y_true, y_pred, X_row in stream:
        drift_detected = detector.update(y_true, y_pred, X_row)
"""

from __future__ import annotations

import json
import pickle
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from river import drift as river_drift

from src.ml_layer.drift_observer import DriftObserver
from src.utils.config_loader import load_config
from src.utils.logger import get_logger

_logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# PSI helper
# ---------------------------------------------------------------------------

_PSI_EPSILON = 1e-6  # prevents log(0)


def _compute_psi_single(
    reference_counts: np.ndarray,
    current_values: np.ndarray,
    bin_edges: np.ndarray,
) -> float:
    """Compute PSI for a single feature.

    Args:
        reference_counts: Counts per bin from the training (baseline) data.
        current_values:   Raw values from the current window to bin.
        bin_edges:        Bin edges computed from training data.

    Returns:
        PSI scalar.  Values < 0.1 indicate no drift, 0.1–0.2 slight drift,
        > 0.2 significant drift.
    """
    # Bin current values using training bin edges.
    # Values outside the training range fall into the first/last bin via clip.
    current_counts, _ = np.histogram(current_values, bins=bin_edges)

    # Convert to proportions; guard against zero with epsilon.
    ref_prop = reference_counts / (reference_counts.sum() + _PSI_EPSILON)
    cur_prop = current_counts / (current_counts.sum() + _PSI_EPSILON)

    ref_prop = np.clip(ref_prop, _PSI_EPSILON, None)
    cur_prop = np.clip(cur_prop, _PSI_EPSILON, None)

    psi = float(np.sum((cur_prop - ref_prop) * np.log(cur_prop / ref_prop)))
    return psi


# ---------------------------------------------------------------------------
# DriftDetector
# ---------------------------------------------------------------------------


class DriftDetector:
    """Stateful concept-drift detector combining ADWIN and PSI signals.

    The detector maintains an internal ADWIN window that persists across
    calls to :meth:`update`, a rolling buffer of recent prediction errors,
    and a sliding window of recent feature rows for PSI computation.

    Attributes:
        _adwin: :class:`river.drift.ADWIN` instance.  State persists across
            all :meth:`update` calls.
        _observers: List of registered :class:`DriftObserver` instances.
        _error_buffer: :class:`collections.deque` of the last
            ``rolling_error_buffer_size`` error values (0 or 1).
        _psi_window: :class:`collections.deque` of the last
            ``psi_current_window_size`` feature rows as :class:`dict`.
        _psi_baseline: ``None`` until :meth:`set_psi_baseline` is called;
            afterwards a dict mapping feature name → baseline info.
        _drift_history: List of all drift-event dicts emitted so far.
        _cfg: The ``drift`` sub-dict loaded from ``config.yaml``.

    Note:
        :meth:`save_state` / :meth:`load_state` serialise the *entire*
        detector including the ADWIN window via ``pickle``.
    """

    def __init__(self, config_path: str = "config/config.yaml") -> None:
        """Initialise the DriftDetector from config.

        Args:
            config_path: Path to ``config/config.yaml`` relative to the
                project root (default: ``"config/config.yaml"``).
        """
        cfg = load_config(config_path)
        self._cfg: dict[str, Any] = cfg["drift"]

        # ADWIN — strictly online; one scalar per update() call.
        self._adwin = river_drift.ADWIN(delta=float(self._cfg["adwin_delta"]))

        # Rolling error buffer
        self._error_buffer: deque[int] = deque(
            maxlen=int(self._cfg["rolling_error_buffer_size"])
        )

        # PSI sliding window — stores feature rows as dicts
        self._psi_window: deque[dict] = deque(
            maxlen=int(self._cfg["psi_current_window_size"])
        )

        # PSI baseline — None until set_psi_baseline() is called
        self._psi_baseline: dict[str, dict] | None = None

        # Observer list and drift history
        self._observers: list[DriftObserver] = []
        self._drift_history: list[dict] = []

        self._monitored_features: list[str] = list(
            self._cfg.get("monitored_features", [])
        )
        self._psi_critical_threshold: float = float(
            self._cfg["psi_critical_threshold"]
        )
        self._psi_bins: int = int(self._cfg["psi_bins"])

        _logger.info(
            "DriftDetector initialised | adwin_delta=%.4f, "
            "error_buffer_size=%d, psi_window_size=%d, "
            "monitored_features=%s",
            float(self._cfg["adwin_delta"]),
            int(self._cfg["rolling_error_buffer_size"]),
            int(self._cfg["psi_current_window_size"]),
            self._monitored_features,
        )

    # ------------------------------------------------------------------
    # Observer management
    # ------------------------------------------------------------------

    def register_observer(self, observer: DriftObserver) -> None:
        """Register a drift-event observer.

        Args:
            observer: Concrete :class:`DriftObserver` instance to add.
        """
        self._observers.append(observer)
        _logger.info(
            "DriftDetector — registered observer: %s",
            type(observer).__name__,
        )

    def _notify_observers(self, drift_info: dict) -> None:
        """Call every registered observer with the drift event dict.

        ``drift_info`` is always a fully populated dict with all six required
        keys before this method is called.

        Args:
            drift_info: Complete drift-event dictionary.
        """
        for obs in self._observers:
            try:
                obs.on_drift_detected(drift_info)
            except Exception as exc:  # noqa: BLE001
                _logger.error(
                    "DriftDetector — observer %s raised an exception: %s",
                    type(obs).__name__,
                    exc,
                )

    # ------------------------------------------------------------------
    # PSI baseline
    # ------------------------------------------------------------------

    def set_psi_baseline(
        self, X_train: pd.DataFrame, feature_cols: list[str] | None = None
    ) -> None:
        """Compute and store PSI baseline distributions from training data.

        Warning:
            This method must be called with **training data only**, before any
            stream data flows through :meth:`update`.  Calling it after
            streaming has begun will contaminate the baseline with drifted data
            and produce misleading PSI scores.

        Args:
            X_train: Training feature DataFrame.  Must contain all columns
                listed in ``feature_cols`` (or ``monitored_features`` from
                config if ``feature_cols`` is ``None``).
            feature_cols: Optional list of columns to monitor.  Defaults to
                ``drift.monitored_features`` from ``config.yaml``.
        """
        cols = feature_cols if feature_cols is not None else self._monitored_features
        self._psi_baseline = {}

        for col in cols:
            if col not in X_train.columns:
                _logger.warning(
                    "set_psi_baseline — column '%s' not found in X_train; skipping.",
                    col,
                )
                continue

            values = X_train[col].dropna().values.astype(float)
            counts, bin_edges = np.histogram(values, bins=self._psi_bins)
            self._psi_baseline[col] = {
                "bin_edges": bin_edges,
                "ref_counts": counts,
            }

        _logger.info(
            "PSI baseline set for %d features: %s",
            len(self._psi_baseline),
            list(self._psi_baseline.keys()),
        )

    # ------------------------------------------------------------------
    # PSI computation
    # ------------------------------------------------------------------

    def compute_psi(self, X_current: pd.DataFrame) -> dict[str, float]:
        """Compute PSI for each monitored feature against the stored baseline.

        Args:
            X_current: DataFrame of recent transactions (current window).

        Returns:
            Dictionary mapping feature name → PSI score.  Features not in the
            baseline are omitted.  An empty dict is returned if no baseline has
            been set.
        """
        if self._psi_baseline is None:
            return {}

        psi_scores: dict[str, float] = {}
        for col, info in self._psi_baseline.items():
            if col not in X_current.columns:
                _logger.warning(
                    "compute_psi — column '%s' not in current window; skipping.", col
                )
                continue
            current_values = X_current[col].dropna().values.astype(float)
            if len(current_values) == 0:
                psi_scores[col] = 0.0
                continue
            psi_scores[col] = _compute_psi_single(
                reference_counts=info["ref_counts"],
                current_values=current_values,
                bin_edges=info["bin_edges"],
            )

        return psi_scores

    # ------------------------------------------------------------------
    # Core update loop
    # ------------------------------------------------------------------

    def update(
        self,
        y_true: int,
        y_pred: int,
        X_row: pd.DataFrame,
    ) -> bool:
        """Process one prediction and check for drift.

        This is the core streaming method.  Call it once per transaction in
        arrival order.  ADWIN receives exactly one scalar per call — never
        batched.  PSI is computed whenever the current window reaches
        ``psi_current_window_size`` rows and a baseline has been set.

        The two signals (ADWIN and PSI) are **independent**:
        PSI crossing the critical threshold notifies observers even when
        ADWIN has not fired.

        Args:
            y_true: Ground-truth label (0 or 1).
            y_pred: Model prediction (0 or 1).
            X_row:  Single-row DataFrame containing the feature values for
                this transaction.  Used to populate the PSI sliding window.

        Returns:
            ``True`` if drift was detected (by either signal), ``False``
            otherwise.
        """
        # 1. Compute error and push to buffer.
        error = int(y_true != y_pred)
        self._error_buffer.append(error)

        # 2. Feed one scalar to ADWIN — strictly online, never batched.
        self._adwin.update(error)
        adwin_fired: bool = bool(self._adwin.drift_detected)

        # 3. Add feature row to PSI window.
        if isinstance(X_row, pd.DataFrame):
            row_dict = X_row.iloc[0].to_dict() if len(X_row) > 0 else {}
        else:
            row_dict = dict(X_row)
        self._psi_window.append(row_dict)

        # 4. PSI check — independent of ADWIN.
        psi_scores: dict[str, float] = {}
        psi_fired: bool = False
        psi_critical_features: dict[str, float] = {}

        psi_window_size = int(self._cfg["psi_current_window_size"])
        if (
            self._psi_baseline is not None
            and len(self._psi_window) >= psi_window_size
        ):
            X_window = pd.DataFrame(list(self._psi_window))
            psi_scores = self.compute_psi(X_window)
            psi_critical_features = {
                feat: score
                for feat, score in psi_scores.items()
                if score > self._psi_critical_threshold
            }
            psi_fired = bool(psi_critical_features)

        # 5. Build drift_info — ALL keys always populated, regardless of
        #    which signal fired. Observers can rely on this contract.
        if adwin_fired or psi_fired:
            if adwin_fired and psi_fired:
                drift_type = "adwin+psi"
            elif adwin_fired:
                drift_type = "adwin"
            else:
                drift_type = "psi"

            drift_info: dict[str, Any] = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "drift_type": drift_type,
                "adwin_error_rate": float(self._adwin.estimation),
                "psi_scores": psi_scores,   # {} when PSI not available
                "window_size": len(self._psi_window),
                "recommended_action": (
                    "retrain" if adwin_fired or psi_critical_features
                    else "monitor"
                ),
            }

            _logger.warning(
                "DriftDetector — drift detected | type=%s, "
                "adwin_error_rate=%.4f, psi_critical=%s",
                drift_type,
                drift_info["adwin_error_rate"],
                psi_critical_features,
            )

            # 6. Notify all observers synchronously.
            self._notify_observers(drift_info)

            # 7. Record in history.
            self._drift_history.append(drift_info)

            return True

        return False

    # ------------------------------------------------------------------
    # History
    # ------------------------------------------------------------------

    def get_drift_history(self) -> list[dict]:
        """Return a copy of all drift events detected so far.

        Returns:
            List of drift-info dicts, in chronological order.
        """
        return list(self._drift_history)

    # ------------------------------------------------------------------
    # State persistence
    # ------------------------------------------------------------------

    def save_state(self, path: str) -> None:
        """Serialise the entire detector state (including ADWIN window) to disk.

        The whole object is pickled, preserving the ADWIN window, PSI
        baseline, error buffer, PSI window, and drift history.

        Args:
            path: Destination file path.  Parent directories are created
                automatically if they do not exist.
        """
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "wb") as fh:
            pickle.dump(self, fh, protocol=pickle.HIGHEST_PROTOCOL)
        _logger.info("DriftDetector — state saved to '%s'", path)

    @classmethod
    def load_state(cls, path: str) -> "DriftDetector":
        """Deserialise a previously saved detector state from disk.

        Args:
            path: Path to the pickle file written by :meth:`save_state`.

        Returns:
            The deserialised :class:`DriftDetector` with all internal state
            restored, including the ADWIN window, PSI baseline, and history.

        Raises:
            FileNotFoundError: If ``path`` does not exist.
        """
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(
                f"Drift detector state file not found: {p.resolve()}"
            )
        with open(p, "rb") as fh:
            detector: DriftDetector = pickle.load(fh)
        _logger.info("DriftDetector — state loaded from '%s'", path)
        return detector


# ---------------------------------------------------------------------------
# LoggingObserver
# ---------------------------------------------------------------------------


class LoggingObserver(DriftObserver):
    """Observer that logs every drift event at WARNING level.

    This is the simplest concrete observer — useful for development,
    debugging, and audit trails.

    Example::

        detector.register_observer(LoggingObserver())
    """

    def on_drift_detected(self, drift_info: dict) -> None:
        """Log the full drift-info dict at WARNING level.

        Args:
            drift_info: Complete drift-event dictionary from
                :class:`DriftDetector`.
        """
        _logger.warning(
            "LoggingObserver — DRIFT EVENT | timestamp=%s, type=%s, "
            "adwin_error_rate=%.4f, window_size=%d, action=%s | "
            "psi_scores=%s",
            drift_info["timestamp"],
            drift_info["drift_type"],
            drift_info["adwin_error_rate"],
            drift_info["window_size"],
            drift_info["recommended_action"],
            drift_info["psi_scores"],
        )


# ---------------------------------------------------------------------------
# RetrainingTriggerObserver
# ---------------------------------------------------------------------------


class RetrainingTriggerObserver(DriftObserver):
    """Observer that appends drift events to a JSON file for pipeline polling.

    The retraining pipeline polls the file; this observer decouples drift
    detection from retraining without requiring a message queue.

    The file stores a **JSON array** of drift-event dicts.  Each new event is
    *appended* to the existing list — the file is never overwritten.

    Example::

        observer = RetrainingTriggerObserver(
            trigger_path="logs/drift_events.json"
        )
        detector.register_observer(observer)

    Attributes:
        _trigger_path: Absolute :class:`~pathlib.Path` to the trigger file.
    """

    def __init__(
        self,
        trigger_path: str | None = None,
        config_path: str = "config/config.yaml",
    ) -> None:
        """Initialise the observer.

        Args:
            trigger_path: Path to the JSON trigger file.  If ``None``, the
                path is read from ``drift.retraining_trigger_path`` in
                ``config.yaml``.
            config_path: Path to the YAML configuration file.
        """
        if trigger_path is None:
            cfg = load_config(config_path)
            trigger_path = cfg["drift"]["retraining_trigger_path"]

        self._trigger_path = Path(trigger_path)
        self._trigger_path.parent.mkdir(parents=True, exist_ok=True)
        _logger.info(
            "RetrainingTriggerObserver — trigger file: '%s'",
            self._trigger_path,
        )

    def on_drift_detected(self, drift_info: dict) -> None:
        """Append drift event to the JSON trigger file.

        Reads the existing JSON array (or starts a new one if the file does
        not exist), appends the new event, and writes the updated array back.
        The file is **never truncated** — all historical events accumulate.

        Args:
            drift_info: Complete drift-event dictionary from
                :class:`DriftDetector`.
        """
        # Read existing events (or start fresh).
        if self._trigger_path.exists():
            try:
                with open(self._trigger_path, "r", encoding="utf-8") as fh:
                    events: list[dict] = json.load(fh)
                if not isinstance(events, list):
                    _logger.warning(
                        "RetrainingTriggerObserver — existing file is not a "
                        "JSON list; resetting to empty list."
                    )
                    events = []
            except (json.JSONDecodeError, OSError) as exc:
                _logger.error(
                    "RetrainingTriggerObserver — could not read '%s': %s. "
                    "Starting fresh.",
                    self._trigger_path,
                    exc,
                )
                events = []
        else:
            events = []

        # Append new event and write back.
        events.append(drift_info)
        with open(self._trigger_path, "w", encoding="utf-8") as fh:
            json.dump(events, fh, indent=2, default=str)

        _logger.info(
            "RetrainingTriggerObserver — appended event to '%s' "
            "(total events: %d)",
            self._trigger_path,
            len(events),
        )
