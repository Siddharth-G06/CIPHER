"""
src/feedback_layer/feedback_publisher.py
-----------------------------------------
Kafka producer dedicated to publishing analyst feedback labels to the
``cipher.feedback.labels`` topic.

This is intentionally separate from :class:`~src.serving_layer.kafka_producer.TransactionProducer`
to maintain clean topic ownership: the transaction producer owns
``cipher.transactions.*`` and ``cipher.drift.events``; this publisher
owns ``cipher.feedback.labels``.

Usage::

    from src.feedback_layer.feedback_publisher import FeedbackPublisher
    from src.feedback_layer.feedback_store import FeedbackRecord, AnalystDecision

    pub = FeedbackPublisher()
    pub.publish(record)
    pub.flush()
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from confluent_kafka import Producer

from src.feedback_layer.feedback_store import FeedbackRecord
from src.utils.config_loader import load_config
from src.utils.logger import get_logger

_logger = get_logger(__name__)


class FeedbackPublisher:
    """Confluent Kafka producer for the ``cipher.feedback.labels`` topic.

    Wraps a :class:`confluent_kafka.Producer` with a clean, typed interface
    that accepts :class:`~src.feedback_layer.feedback_store.FeedbackRecord`
    objects and serialises them to JSON.

    Args:
        config_path: Path to the YAML configuration file.

    Attributes:
        _topic: The Kafka topic name for feedback messages.
        _producer: Underlying :class:`confluent_kafka.Producer` instance.
        _published: Count of successfully delivered messages.
        _failed: Count of delivery failures.
    """

    def __init__(self, config_path: str = "config/config.yaml") -> None:
        """Initialise the producer from config.

        Args:
            config_path: Path to the YAML configuration file.
        """
        cfg = load_config(config_path)
        kafka_cfg = cfg["kafka"]

        self._topic: str = kafka_cfg["topics"]["feedback"]

        producer_config: dict[str, Any] = {
            "bootstrap.servers": kafka_cfg["bootstrap_servers"],
            "acks": kafka_cfg["producer"]["acks"],
            "retries": kafka_cfg["producer"]["retries"],
            "linger.ms": kafka_cfg["producer"]["linger_ms"],
            "batch.size": kafka_cfg["producer"]["batch_size"],
            "compression.type": kafka_cfg["producer"]["compression_type"],
        }

        self._producer = Producer(producer_config)
        self._published: int = 0
        self._failed: int = 0

        _logger.info(
            "FeedbackPublisher initialised | topic=%s, bootstrap=%s",
            self._topic,
            kafka_cfg["bootstrap_servers"],
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def publish(self, record: FeedbackRecord) -> None:
        """Serialise and publish a feedback record to Kafka.

        The message key is the ``transaction_id`` encoded as UTF-8, which
        ensures all feedback for the same transaction lands on the same
        partition (order-preserving for per-transaction event replay).

        Args:
            record: The :class:`~src.feedback_layer.feedback_store.FeedbackRecord`
                to publish.
        """
        payload = {
            "transaction_id": record.transaction_id,
            "analyst_id": record.analyst_id,
            "analyst_decision": record.decision.value
            if hasattr(record.decision, "value")
            else str(record.decision),
            "submitted_at": record.reviewed_at,
            "confidence": record.confidence,
            "model_score": record.model_score,
            "true_label": record.true_label,
            "notes": record.notes,
            "model_version": record.model_version,
            "drift_active": record.drift_active,
        }

        raw = json.dumps(payload, default=str).encode("utf-8")
        key = record.transaction_id.encode("utf-8")

        self._producer.produce(
            topic=self._topic,
            key=key,
            value=raw,
            callback=self._delivery_callback,
        )
        self._producer.poll(0)
        _logger.info(
            "FeedbackPublisher.publish — queued | tx=%s, decision=%s",
            record.transaction_id,
            payload["analyst_decision"],
        )

    def flush(self, timeout: float = 5.0) -> None:
        """Flush all pending messages and wait for delivery confirmations.

        Args:
            timeout: Maximum seconds to wait for flush completion.
        """
        remaining = self._producer.flush(timeout=timeout)
        if remaining > 0:
            _logger.warning(
                "FeedbackPublisher.flush — %d messages not delivered within %.1fs",
                remaining,
                timeout,
            )
        else:
            _logger.debug("FeedbackPublisher.flush — all messages delivered.")

    def get_stats(self) -> dict:
        """Return delivery statistics.

        Returns:
            Dictionary with keys ``published`` and ``failed``.
        """
        return {"published": self._published, "failed": self._failed}

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _delivery_callback(self, err: Any, msg: Any) -> None:
        """Kafka delivery report callback.

        Args:
            err: Delivery error, or ``None`` on success.
            msg: The delivered message object.
        """
        if err is not None:
            self._failed += 1
            _logger.error(
                "FeedbackPublisher — delivery failed | topic=%s, err=%s",
                self._topic,
                err,
            )
        else:
            self._published += 1
            _logger.debug(
                "FeedbackPublisher — delivered | topic=%s, partition=%d, offset=%d",
                msg.topic(),
                msg.partition(),
                msg.offset(),
            )
