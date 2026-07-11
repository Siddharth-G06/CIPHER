"""
src/feedback_layer
------------------
Human-in-the-loop feedback ingestion and automated model retraining layer
for the CIPHER fraud-detection system.

Public surface
--------------
* :class:`~src.feedback_layer.feedback_store.FeedbackStore` — append-only
  SQLite event store for analyst decisions.
* :class:`~src.feedback_layer.feedback_consumer.FeedbackConsumer` — Kafka
  consumer that persists analyst labels arriving on
  ``cipher.feedback.labels``.
* :class:`~src.feedback_layer.retraining_trigger.RetrainingTrigger` —
  watches feedback volume, drift events, and the schedule clock and fires
  the retraining pipeline when any threshold is crossed.
* :class:`~src.feedback_layer.retrain_pipeline.RetrainingPipeline` — full
  champion/challenger retraining sequence with MLflow tracking and automatic
  model promotion.
* :class:`~src.feedback_layer.model_update_signal.ModelUpdateSignal` — thin
  file-based signalling so the live Kafka consumer can hot-reload a freshly
  promoted model without restart.
"""

from src.feedback_layer.feedback_store import FeedbackStore, FeedbackRecord, AnalystDecision
from src.feedback_layer.model_update_signal import ModelUpdateSignal

__all__ = [
    "FeedbackStore",
    "FeedbackRecord",
    "AnalystDecision",
    "ModelUpdateSignal",
]
