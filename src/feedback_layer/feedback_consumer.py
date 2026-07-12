"""
src/feedback_layer/feedback_consumer.py
-----------------------------------------
Kafka consumer that ingests analyst feedback labels from the
``cipher.feedback.labels`` topic and persists them to the
:class:`~src.feedback_layer.feedback_store.FeedbackStore`.

Designed to run in a **background thread** alongside the main
:class:`~src.serving_layer.kafka_consumer.FraudDetectionConsumer` so that
feedback collection never blocks transaction scoring.

Message format
--------------
Messages on ``cipher.feedback.labels`` must conform to
:class:`~src.serving_layer.schemas.FeedbackMessage` (Pydantic v2 model):

.. code-block:: json

    {
        "transaction_id": "txn-001",
        "analyst_id":     "analyst-42",
        "analyst_decision": "CONFIRM_FRAUD",
        "submitted_at":   "2026-07-11T05:00:00+00:00",
        "confidence":     0.9,
        "model_score":    0.87,
        "true_label":     1
    }

Fields ``confidence``, ``model_score``, and ``true_label`` are optional in
the Kafka message and default to ``0.5``, ``0.0``, ``1`` respectively.

Usage::

    from src.feedback_layer.feedback_consumer import FeedbackConsumer
    import threading

    consumer = FeedbackConsumer()
    t = threading.Thread(target=consumer.run, daemon=True)
    t.start()
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Optional

from confluent_kafka import Consumer, KafkaError

from src.feedback_layer.feedback_store import (
    AnalystDecision,
    FeedbackRecord,
    FeedbackStore,
)
from src.serving_layer.schemas import FeedbackMessage
from src.utils.config_loader import load_config
from src.utils.logger import get_logger

_logger = get_logger(__name__)

# Map from Kafka message analyst_decision strings → AnalystDecision enum.
# Handles both the exact enum values and common aliases sent by analyst UIs.
_DECISION_MAP: dict[str, AnalystDecision] = {
    "CONFIRM_FRAUD": AnalystDecision.CONFIRM_FRAUD,
    "CONFIRMED_FRAUD": AnalystDecision.CONFIRM_FRAUD,
    "FALSE_POSITIVE": AnalystDecision.FALSE_POSITIVE,
    "ESCALATE": AnalystDecision.ESCALATE,
    "ESCALATED": AnalystDecision.ESCALATE,
}


class FeedbackConsumer:
    """Kafka consumer for analyst feedback labels.

    Polls ``cipher.feedback.labels``, deserialises each message as a
    :class:`~src.serving_layer.schemas.FeedbackMessage`, maps it to a
    :class:`~src.feedback_layer.feedback_store.FeedbackRecord`, and persists
    it via :class:`~src.feedback_layer.feedback_store.FeedbackStore`.

    Manual offset commit is used (``enable.auto.commit=False``) so that an
    application crash before writing to SQLite does not lose the message.

    Args:
        config_path: Path to the YAML configuration file.
    """

    def __init__(self, config_path: str = "config/config.yaml") -> None:
        """Initialise Kafka consumer and FeedbackStore from config.

        Args:
            config_path: Path to the YAML configuration file.
        """
        cfg = load_config(config_path)
        kafka_cfg = cfg["kafka"]

        consumer_config = {
            "bootstrap.servers": kafka_cfg["bootstrap_servers"],
            "group.id": "cipher-feedback-consumer",
            "auto.offset.reset": kafka_cfg["consumer"]["auto_offset_reset"],
            "enable.auto.commit": False,
            "max.poll.interval.ms": kafka_cfg["consumer"]["max_poll_interval_ms"],
            "session.timeout.ms": kafka_cfg["consumer"]["session_timeout_ms"],
        }

        self._topic: str = kafka_cfg["topics"]["feedback"]
        self._consumer: Consumer = Consumer(consumer_config)
        self._store: FeedbackStore = FeedbackStore(config_path=config_path)

        _logger.info(
            "FeedbackConsumer initialised | topic=%s, group=cipher-feedback-consumer",
            self._topic,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, max_messages: Optional[int] = None) -> None:
        """Start the poll loop; blocks until interrupted or *max_messages* consumed.

        Intended to be run in a background daemon thread.

        Args:
            max_messages: Stop after processing this many messages.  ``None``
                (default) means run indefinitely.
        """
        self._consumer.subscribe([self._topic])
        _logger.info(
            "FeedbackConsumer.run — subscribed to '%s'", self._topic
        )

        processed = 0
        try:
            while True:
                if max_messages is not None and processed >= max_messages:
                    _logger.info(
                        "FeedbackConsumer.run — reached max_messages=%d, stopping.",
                        max_messages,
                    )
                    break

                msg = self._consumer.poll(timeout=1.0)

                if msg is None:
                    continue
                if msg.error():
                    if msg.error().code() == KafkaError._PARTITION_EOF:
                        continue
                    _logger.error(
                        "FeedbackConsumer — Kafka error: %s", msg.error()
                    )
                    break

                self._handle_message(msg)

                # Commit only after successful SQLite write.
                self._consumer.commit(asynchronous=False)
                processed += 1

        except KeyboardInterrupt:
            _logger.info("FeedbackConsumer.run — interrupted by user.")
        finally:
            self._consumer.close()
            _logger.info(
                "FeedbackConsumer.run — consumer closed | total_processed=%d",
                processed,
            )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _handle_message(self, msg) -> None:
        """Deserialise a single Kafka message and persist to FeedbackStore.

        Args:
            msg: A :class:`confluent_kafka.Message` from the poll loop.
        """
        raw_val = ""
        try:
            raw_val = msg.value().decode("utf-8")
            data = json.loads(raw_val)

            # Deserialise with Pydantic (schema validation + extra=allow).
            feedback_msg = FeedbackMessage(**data)

            # Map analyst_decision string → AnalystDecision enum.
            decision_str = feedback_msg.analyst_decision.upper().replace(
                " ", "_"
            )
            decision = _DECISION_MAP.get(decision_str)
            if decision is None:
                _logger.warning(
                    "FeedbackConsumer — unknown analyst_decision='%s' for "
                    "tx=%s; defaulting to ESCALATE.",
                    feedback_msg.analyst_decision,
                    feedback_msg.transaction_id,
                )
                decision = AnalystDecision.ESCALATE

            # Extract optional enrichment fields sent by the analyst UI.
            confidence: float = float(data.get("confidence", 0.5))
            model_score: float = float(data.get("model_score", 0.0))
            true_label: int = int(data.get("true_label", 1))
            model_version: str = str(data.get("model_version", "unknown"))
            drift_active: bool = bool(data.get("drift_active", False))
            notes: Optional[str] = data.get("notes")

            record = FeedbackRecord(
                transaction_id=feedback_msg.transaction_id,
                analyst_id=feedback_msg.analyst_id,
                decision=decision,
                confidence=max(0.0, min(1.0, confidence)),
                model_score=model_score,
                true_label=true_label,
                notes=notes,
                reviewed_at=feedback_msg.submitted_at.isoformat(),
                model_version=model_version,
                drift_active=drift_active,
            )

            self._store.add_feedback(record)

        except Exception as exc:
            _logger.error(
                "FeedbackConsumer._handle_message — failed to process message "
                "(raw='%.200s'): %s",
                raw_val,
                exc,
                exc_info=True,
            )
            # Do NOT commit — let Kafka retry this message.
            # Raise so the caller's commit is skipped.
            raise
