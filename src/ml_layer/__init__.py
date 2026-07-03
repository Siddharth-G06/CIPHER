# CIPHER — src/ml_layer package
"""
ml_layer
--------
Machine-learning layer of the CIPHER fraud-detection pipeline.

Public exports
--------------
BaseDetector               : Abstract base class for all detectors.
LightGBMDetector           : Supervised gradient-boosting detector.
IsolationForestDetector    : Unsupervised anomaly detector.
EnsembleDetector           : Weighted combination of the two above.
DriftObserver              : Abstract base class for drift-event observers.
DriftDetector              : Stateful ADWIN + PSI drift detector (subject).
LoggingObserver            : Logs drift events at WARNING level.
RetrainingTriggerObserver  : Appends drift events to a JSON trigger file.
DriftSimulator             : Injects controlled concept drift into DataFrames.
"""

from src.ml_layer.base import BaseDetector
from src.ml_layer.drift_detector import (
    DriftDetector,
    LoggingObserver,
    RetrainingTriggerObserver,
)
from src.ml_layer.drift_observer import DriftObserver
from src.ml_layer.drift_simulator import DriftSimulator
from src.ml_layer.model import (
    EnsembleDetector,
    IsolationForestDetector,
    LightGBMDetector,
)

__all__ = [
    "BaseDetector",
    "LightGBMDetector",
    "IsolationForestDetector",
    "EnsembleDetector",
    "DriftObserver",
    "DriftDetector",
    "LoggingObserver",
    "RetrainingTriggerObserver",
    "DriftSimulator",
]
