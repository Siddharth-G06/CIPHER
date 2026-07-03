"""
src/ml_layer/drift_observer.py
-------------------------------
Abstract base class for CIPHER drift-event observers.

Any component that needs to react to detected drift (logging, alerting,
triggering retraining) must subclass :class:`DriftObserver` and implement
:meth:`on_drift_detected`.

The observer is registered with :class:`~src.ml_layer.drift_detector.DriftDetector`
via :meth:`~src.ml_layer.drift_detector.DriftDetector.register_observer`.

Example::

    from src.ml_layer.drift_observer import DriftObserver

    class SlackObserver(DriftObserver):
        def on_drift_detected(self, drift_info: dict) -> None:
            slack_client.post(f"Drift detected: {drift_info['drift_type']}")
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class DriftObserver(ABC):
    """Abstract base class for drift-event observers (Observer pattern).

    Concrete subclasses are registered with :class:`DriftDetector` and are
    called synchronously whenever a drift event is detected.  Each observer
    receives a ``drift_info`` dictionary guaranteed to contain the following
    keys regardless of which signal (ADWIN or PSI) triggered the event:

    .. code-block:: python

        {
            "timestamp":          str,   # ISO-8601 UTC timestamp
            "drift_type":         str,   # "adwin" | "psi" | "adwin+psi"
            "adwin_error_rate":   float, # current ADWIN mean estimate
            "psi_scores":         dict,  # {feature_name: psi_value, ...}
            "window_size":        int,   # number of rows in current PSI window
            "recommended_action": str,   # "retrain" | "monitor"
        }

    Warning:
        Observers are called on the same thread as the caller of
        :meth:`~DriftDetector.update`.  Expensive operations (e.g. model
        retraining) should be delegated to a background process; use the file-
        based :class:`~src.ml_layer.drift_detector.RetrainingTriggerObserver`
        for this decoupling pattern.
    """

    @abstractmethod
    def on_drift_detected(self, drift_info: dict) -> None:
        """Handle a drift detection event.

        Args:
            drift_info: Dictionary with exactly the keys listed in the class
                docstring.  All keys are always present regardless of which
                signal (ADWIN vs PSI) triggered the event.

        Returns:
            None.
        """
