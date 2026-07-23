"""
src/serving_layer/kafka_producer.py
-----------------------------------
Kafka producer for the CIPHER streaming layer.
Simulates a real-time payment processor feed by streaming historical data,
and handles publishing out-of-band events (like drift alerts and flags).
"""

from __future__ import annotations

import json
import pickle
import time
from typing import Any
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from confluent_kafka import Producer
from confluent_kafka.admin import AdminClient, NewTopic

from src.serving_layer.report_generator import FlaggedTransaction
from src.utils.config_loader import load_config
from src.utils.logger import get_logger
from src.serving_layer.schemas import TransactionMessage, FlaggedMessage, DriftEventMessage

_logger = get_logger(__name__)


def _json_serializer(obj: Any) -> Any:
    """Helper to serialize datetime objects for json.dumps."""
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Type {type(obj)} not serializable")


class TransactionProducer:
    """Kafka producer for CIPHER transactions and events."""

    def __init__(self, config_path: str = "config/config.yaml") -> None:
        """Initialize the producer with settings from config."""
        full_cfg = load_config(config_path)
        self._cfg = full_cfg["kafka"]
        self._topic_cfg = self._cfg["topic_config"]

        producer_config = {
            "bootstrap.servers": self._cfg["bootstrap_servers"],
            "acks": self._cfg["producer"]["acks"],
            "retries": self._cfg["producer"]["retries"],
            "linger.ms": self._cfg["producer"]["linger_ms"],
            "batch.size": self._cfg["producer"]["batch_size"],
            "compression.type": self._cfg["producer"]["compression_type"],
        }
        
        self.producer = Producer(producer_config)
        
        # Stats
        self._total_published = 0
        self._total_failed = 0
        self._total_latency_ms = 0.0

        _logger.info("TransactionProducer initialized | bootstrap_servers=%s", self._cfg["bootstrap_servers"])

    def setup_topics(self, topics: list[str]) -> None:
        """Create Kafka topics if they don't exist."""
        admin = AdminClient({"bootstrap.servers": self._cfg["bootstrap_servers"]})
        
        existing_topics = admin.list_topics(timeout=10).topics
        new_topics = []

        for topic in topics:
            if topic in existing_topics:
                _logger.debug("Topic '%s' already exists.", topic)
                continue
            
            partitions = self._topic_cfg["raw_partitions"] if topic == self._cfg["topics"]["raw"] else self._topic_cfg["default_partitions"]
            
            # Retention mapping
            if topic == self._cfg["topics"]["raw"]:
                retention = self._topic_cfg["raw_retention_ms"]
            elif topic == self._cfg["topics"]["flagged"]:
                retention = self._topic_cfg["flagged_retention_ms"]
            elif topic == self._cfg["topics"]["drift"]:
                retention = self._topic_cfg["drift_retention_ms"]
            else:
                retention = self._topic_cfg["raw_retention_ms"] # fallback

            new_topics.append(
                NewTopic(
                    topic,
                    num_partitions=partitions,
                    replication_factor=1,
                    config={"retention.ms": str(retention)}
                )
            )

        if new_topics:
            fs = admin.create_topics(new_topics)
            for topic, f in fs.items():
                try:
                    f.result()  # The result itself is None
                    _logger.info("Topic '%s' created.", topic)
                except Exception as e:
                    _logger.error("Failed to create topic '%s': %s", topic, e)

    def delivery_callback(self, err: Any, msg: Any) -> None:
        """Callback executed when message delivery succeeds or fails."""
        if err is not None:
            self._total_failed += 1
            _logger.error("Message delivery failed: %s", err)
        else:
            self._total_published += 1
            # Note: real latency calculation would require embedding timestamp in payload
            # For simulation we just increment count
            pass

    def publish_transaction(self, transaction: dict) -> None:
        """Serialize and publish a raw transaction."""
        topic = self._cfg["topics"]["raw"]
        
        # Build Pydantic model for validation
        tx_msg = TransactionMessage(
            transaction_id=str(transaction.get("TransactionID") or transaction.get("transaction_id") or "UNKNOWN"),
            card1=int(transaction.get("card1") or 0),
            timestamp=transaction.get("TransactionDT") or datetime.now(timezone.utc),
            amount=float(transaction.get("TransactionAmt") or transaction.get("amount") or 0.0),
            features=transaction
        )
        
        payload = tx_msg.model_dump_json(serialize_as_any=True).encode("utf-8")
        key = str(tx_msg.card1).encode("utf-8")
        
        self.producer.produce(
            topic=topic,
            key=key,
            value=payload,
            callback=self.delivery_callback
        )
        self.producer.poll(0)

    def publish_flagged(self, flagged_transaction: FlaggedTransaction) -> None:
        """Serialize and publish a flagged transaction."""
        topic = self._cfg["topics"]["flagged"]
        
        summary = None
        if flagged_transaction.explanation is not None:
            summary = getattr(flagged_transaction.explanation, "plain_english_summary", None)

        msg = FlaggedMessage(
            transaction_id=flagged_transaction.transaction_id,
            timestamp=flagged_transaction.timestamp,
            ensemble_score=flagged_transaction.ensemble_score,
            lgbm_score=flagged_transaction.lgbm_score,
            iso_score=flagged_transaction.iso_score,
            raw_features=flagged_transaction.raw_features,
            graph_features=flagged_transaction.graph_features,
            explanation_summary=summary,
            drift_active=flagged_transaction.drift_active,
            drift_info=flagged_transaction.drift_info,
            model_version=flagged_transaction.model_version
        )
        
        payload = msg.model_dump_json(serialize_as_any=True).encode("utf-8")
        self.producer.produce(topic=topic, value=payload, callback=self.delivery_callback)
        self.producer.poll(0)

    def publish_drift_event(self, drift_info: dict) -> None:
        """Serialize and publish a drift event."""
        topic = self._cfg["topics"]["drift"]
        
        msg = DriftEventMessage(
            feature=drift_info.get("feature", "unknown"),
            psi=drift_info.get("psi", 0.0),
            adwin_window_size=drift_info.get("adwin_window_size"),
            alert_type=drift_info.get("alert_type", "psi_critical"),
            detected_at=datetime.now(timezone.utc)
        )
        
        payload = msg.model_dump_json(serialize_as_any=True).encode("utf-8")
        self.producer.produce(topic=topic, value=payload, callback=self.delivery_callback)
        self.producer.poll(0)

    def stream_from_csv(self, csv_path: str, delay_ms: int = 100) -> None:
        """Stream a CSV dataset row by row to simulate real-time traffic."""
        _logger.info("Starting stream from %s with delay %dms", csv_path, delay_ms)
        try:
            df = pd.read_csv(csv_path)
            if "TransactionDT" in df.columns:
                df = df.sort_values("TransactionDT")
                
            log_every = int(self._cfg["stream"].get("log_every_n", 1000))
            count = 0
            start_time = time.time()
            
            for _, row in df.iterrows():
                # Convert row to dict, replacing NaNs with None
                row_dict = {k: (None if pd.isna(v) else v) for k, v in row.items()}
                
                # Assume TransactionDT is seconds from some epoch if it's numeric
                dt = row_dict.get("TransactionDT")
                if isinstance(dt, (int, float)):
                    row_dict["TransactionDT"] = datetime.fromtimestamp(dt, tz=timezone.utc)
                    
                self.publish_transaction(row_dict)
                time.sleep(delay_ms / 1000.0)
                
                count += 1
                if count % log_every == 0:
                    elapsed = time.time() - start_time
                    tps = log_every / elapsed if elapsed > 0 else 0
                    _logger.info("Streamed %d messages | Throughput: %.2f msg/sec", count, tps)
                    start_time = time.time()
                    
        except KeyboardInterrupt:
            _logger.info("Streaming interrupted by user.")
        except Exception as e:
            import traceback
            traceback.print_exc()
            _logger.error("Streaming failed: %s", e)
            
        self.close()

    def stream_from_pickle(
        self,
        pickle_path: str = "data/processed/train_featured.pkl",
        delay_ms: int = 100,
        include_labels: bool = True,
    ) -> None:
        """Stream pre-processed rows from train_featured.pkl.

        This is the preferred streaming method because the model was trained
        on positional numpy arrays whose feature names are Column_0..Column_N.
        Sending rows from the pickle ensures the consumer sees features in
        exactly the right positional order with no column-name mismatch.

        Args:
            pickle_path: Path to the 5-tuple pickle produced by prepare_data.py.
            delay_ms: Inter-message delay in milliseconds.
            include_labels: If True, include 'isFraud' in the features dict
                so the consumer can use it as ground-truth for drift tracking.
        """
        _logger.info("Starting stream from pickle %s with delay %dms", pickle_path, delay_ms)
        try:
            with open(pickle_path, "rb") as fh:
                payload = pickle.load(fh)

            if isinstance(payload, dict):
                X = np.asarray(payload["X_train"])
                y = np.asarray(payload["y_train"])
                feature_names = list(payload["feature_names"])
            else:
                X, X_test, y, y_test, feature_names = payload
                X = np.asarray(X)
                y = np.asarray(y)

            _logger.info(
                "Loaded %d rows, %d features from pickle | fraud rate=%.2f%%",
                len(X), X.shape[1], y.mean() * 100,
            )

            log_every = int(self._cfg["stream"].get("log_every_n", 1000))
            start_time = time.time()

            for i, (row, label) in enumerate(zip(X, y)):
                # Build named dict: Column_0 .. Column_N  (matches model feature names)
                row_dict: dict[str, Any] = {
                    f"Column_{j}": float(v) for j, v in enumerate(row)
                }
                if include_labels:
                    row_dict["isFraud"] = int(label)

                # Also embed human-readable feature names so the dashboard can display them
                for fname, val in zip(feature_names, row):
                    row_dict.setdefault(fname, float(val))

                tx_id = f"pkl-{i:07d}"
                row_dict["transaction_id"] = tx_id
                row_dict["TransactionID"] = tx_id
                row_dict["amount"] = float(row_dict.get("TransactionAmt", row[0]))
                row_dict["TransactionDT"] = datetime.now(timezone.utc).isoformat()

                self.publish_transaction(row_dict)
                time.sleep(delay_ms / 1000.0)

                if (i + 1) % log_every == 0:
                    elapsed = time.time() - start_time
                    tps = log_every / elapsed if elapsed > 0 else 0
                    _logger.info("Streamed %d messages | %.2f msg/sec", i + 1, tps)
                    start_time = time.time()

        except KeyboardInterrupt:
            _logger.info("Pickle streaming interrupted by user.")
        except Exception as e:
            import traceback
            traceback.print_exc()
            _logger.error("Pickle streaming failed: %s", e)

        self.close()

    def get_stats(self) -> dict:
        """Return producer stats."""
        avg_latency = self._total_latency_ms / max(1, self._total_published)
        return {
            "total_published": self._total_published,
            "total_failed": self._total_failed,
            "avg_latency_ms": avg_latency
        }

    def close(self) -> None:
        """Flush and close producer."""
        _logger.info("Flushing producer... (Total published: %d, Failed: %d)", self._total_published, self._total_failed)
        self.producer.flush(timeout=5)


if __name__ == "__main__":
    producer = TransactionProducer()
    
    cfg = load_config()
    all_topics = [v for k, v in cfg["kafka"]["topics"].items()]
    producer.setup_topics(all_topics)
    
    # Try to find a valid CSV path for streaming demo
    csv_path = cfg["data"]["transaction_path"]
    delay_ms = cfg["kafka"]["stream"]["delay_ms"]
    
    import os
    if os.path.exists(csv_path):
        producer.stream_from_csv(csv_path, delay_ms=50)
    else:
        _logger.warning("CSV path %s not found. Skipping stream_from_csv demo.", csv_path)
