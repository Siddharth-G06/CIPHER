"""
src/serving_layer/kafka_consumer.py
-----------------------------------
Kafka consumer for the CIPHER streaming layer.
Runs the full ML pipeline on every arriving transaction.
"""

from __future__ import annotations

import collections
import json
import time
from datetime import datetime, timezone
from typing import Optional

import pandas as pd
from confluent_kafka import Consumer, KafkaError, KafkaException

from src.ml_layer.model import EnsembleDetector
from src.ml_layer.drift_detector import DriftDetector
from src.ml_layer.explainer import SHAPExplainer
from src.serving_layer.report_generator import ReportGenerator, FlaggedTransaction
from src.serving_layer.kafka_producer import TransactionProducer
from src.serving_layer.schemas import TransactionMessage
from src.utils.config_loader import load_config
from src.utils.logger import get_logger

_logger = get_logger(__name__)


class FraudDetectionConsumer:
    """Consumes raw transactions and runs the CIPHER pipeline."""

    def __init__(self, config_path: str = "config/config.yaml") -> None:
        """Initialize consumer, load models and tools."""
        full_cfg = load_config(config_path)
        self._cfg = full_cfg["kafka"]
        
        consumer_config = {
            "bootstrap.servers": self._cfg["bootstrap_servers"],
            "group.id": self._cfg["consumer"]["group_id"],
            "auto.offset.reset": self._cfg["consumer"]["auto_offset_reset"],
            "enable.auto.commit": False,
            "max.poll.interval.ms": self._cfg["consumer"]["max_poll_interval_ms"],
            "session.timeout.ms": self._cfg["consumer"]["session_timeout_ms"],
        }
        
        self.consumer = Consumer(consumer_config)
        self.producer = TransactionProducer(config_path) # For DLQ, flags, drift
        
        # Review Queue for UI (max 500)
        self.review_queue: collections.deque = collections.deque(maxlen=500)
        
        # Load ML Components
        _logger.info("Loading ML models...")
        self.ensemble = EnsembleDetector()
        model_path = full_cfg["model"]["artifacts"]["local_model_path"]
        try:
            self.ensemble = self.ensemble.load(model_path)
            _logger.info("Ensemble loaded successfully.")
        except Exception as e:
            _logger.warning("Failed to load ensemble from %s. Using untrained model for tests. Error: %s", model_path, e)
            
        lgbm_path = full_cfg["explainer"].get("lgbm_model_path", "models/lgbm_model.pkl")
        feature_names = []
        if getattr(self.ensemble, "lgbm_", None) and getattr(self.ensemble.lgbm_, "model_", None):
            feature_names = list(self.ensemble.lgbm_.model_.feature_name_)
            
        try:
            self.explainer = SHAPExplainer(
                model_path=lgbm_path,
                feature_names=feature_names,
                config_path=config_path
            )
        except Exception as e:
            _logger.warning("Failed to initialize SHAPExplainer. Explanations will be skipped. Error: %s", e)
            self.explainer = None
        self.report_generator = ReportGenerator(config_path)
        
        self.drift_detector = DriftDetector(config_path)
        drift_state_path = full_cfg["drift"]["detector_state_path"]
        import os
        if os.path.exists(drift_state_path):
            try:
                self.drift_detector.load_state(drift_state_path)
                _logger.info("Drift detector state loaded.")
            except Exception as e:
                _logger.warning("Failed to load drift state: %s", e)
                
        # Register a local observer that forwards drift events to Kafka
        class KafkaDriftObserver:
            def __init__(self, consumer: FraudDetectionConsumer):
                self.consumer = consumer
            def on_drift_detected(self, drift_info: dict) -> None:
                self.consumer._handle_drift_event(drift_info)
                
        self.drift_detector.register_observer(KafkaDriftObserver(self))
        
        # Stats
        self._messages_processed = 0
        self._messages_flagged = 0
        self._messages_failed = 0
        self._total_processing_ms = 0.0

    def get_drift_detector(self) -> DriftDetector:
        """Return this consumer's live drift detector instance."""
        return self.drift_detector

    def get_review_queue(self) -> collections.deque:
        """Return the live review queue."""
        return self.review_queue

    def subscribe(self) -> None:
        """Subscribe to the raw transactions topic."""
        topic = self._cfg["topics"]["raw"]
        self.consumer.subscribe([topic])
        _logger.info("Subscribed to topic: %s", topic)

    def _publish_to_dlq(self, raw_msg: str, error: str) -> None:
        """Send unprocessable message to DLQ."""
        topic = self._cfg["topics"]["dlq"]
        payload = json.dumps({
            "original": raw_msg,
            "error": error,
            "failed_at": datetime.now(timezone.utc).isoformat()
        }).encode("utf-8")
        self.producer.producer.produce(topic=topic, value=payload)
        self.producer.producer.poll(0)
        _logger.warning("Sent message to DLQ: %s", error)

    def _handle_drift_event(self, drift_info: dict) -> None:
        """Handle drift event by publishing to Kafka."""
        self.producer.publish_drift_event(drift_info)
        _logger.warning("Drift event published: %s", drift_info.get("feature"))

    def process_message(self, msg) -> Optional[FlaggedTransaction]:
        """Process a single Kafka message."""
        start_time = time.time()
        raw_val = ""
        try:
            raw_val = msg.value().decode("utf-8")
            data = json.loads(raw_val)
            
            # 1. & 2. Deserialize and Validate
            tx_msg = TransactionMessage(**data)
        except Exception as e:
            self._messages_failed += 1
            self._publish_to_dlq(raw_val, f"Validation error: {e}")
            return None
            
        try:
            features = tx_msg.features
            
            # 3. Build feature vector aligned to model's expected columns in one shot
            # (avoids DataFrame fragmentation from inserting columns one-by-one)
            numeric_features = {k: v for k, v in features.items() if isinstance(v, (int, float))}

            if self.ensemble.lgbm_ is not None and self.ensemble.lgbm_.model_ is not None:
                expected_cols = self.ensemble.lgbm_.model_.feature_name_
                # Build aligned dict: use incoming value if present, else sentinel -999
                aligned = {col: numeric_features.get(col, -999.0) for col in expected_cols}
                X_row_df = pd.DataFrame([aligned], columns=expected_cols)
            else:
                X_row_df = pd.DataFrame([numeric_features])

            X_row_np = X_row_df.values
            
            # 4. Predict
            # Since predict_proba requires 2D array, and we have 1 row
            ensemble_score = float(self.ensemble.predict_proba(X_row_np)[0])
            
            # We need sub-scores. In reality, EnsembleDetector abstracts this, but FlaggedTransaction requires it.
            # If not easily accessible via public API, we'll try to extract them, or default to ensemble_score.
            lgbm_score = float(self.ensemble.lgbm_.predict_proba(X_row_np)[0]) if hasattr(self.ensemble, 'lgbm_') and self.ensemble.lgbm_.model_ else ensemble_score
            iso_score = float(self.ensemble.iso_.predict_proba(X_row_np)[0]) if hasattr(self.ensemble, 'iso_') and self.ensemble.iso_.model_ else ensemble_score
            
            threshold = self.ensemble.threshold_
            is_fraud = ensemble_score >= threshold
            y_pred = int(is_fraud)

            # 5. Drift detector update — use isFraud label if producer injected it
            y_true = int(features.get("isFraud", y_pred))
            drift_detected = self.drift_detector.update(y_true, y_pred, X_row_df)
            drift_info = None
            if drift_detected and self.drift_detector._drift_history:
                drift_info = self.drift_detector._drift_history[-1]

            # 6. If flagged
            if is_fraud:
                self._messages_flagged += 1
                
                flagged_tx = FlaggedTransaction(
                    transaction_id=tx_msg.transaction_id,
                    timestamp=tx_msg.timestamp,
                    ensemble_score=ensemble_score,
                    lgbm_score=lgbm_score,
                    iso_score=iso_score,
                    raw_features=features, # Just using all features for simplicity
                    graph_features={},     # Assume graph features are mixed in 'features'
                    explanation=None,      # Placeholder, SHAP will be ready later
                    drift_active=drift_detected,
                    drift_info=drift_info,
                    model_version="v1.0"
                )
                
                # Trigger explanation asynchronously
                if self.explainer is not None:
                    # Callback for explainer to trigger report generator when done
                    def on_explain_done(tx_id, result):
                        flagged_tx.explanation = result
                        self.report_generator.generate_report_async(flagged_tx, lambda path: _logger.info("Report done: %s", path))
                        
                    self.explainer.explain_async(tx_msg.transaction_id, X_row_df.iloc[0], callback=on_explain_done)
                
                self.producer.publish_flagged(flagged_tx)
                
                # Or simply pass the flagged_tx to review_queue and let it be updated.
                self.review_queue.appendleft(flagged_tx)
                
                # 7. Return
                return flagged_tx
                
        except Exception as e:
            self._messages_failed += 1
            self._publish_to_dlq(raw_val, f"Processing error: {e}")
            _logger.error("Error processing message: %s", e, exc_info=True)
            return None
        finally:
            self._messages_processed += 1
            self._total_processing_ms += (time.time() - start_time) * 1000

        return None

    def run(self, max_messages: Optional[int] = None) -> None:
        """Main poll loop."""
        try:
            count = 0
            while True:
                if max_messages and count >= max_messages:
                    break
                    
                msg = self.consumer.poll(timeout=1.0)
                if msg is None:
                    continue
                if msg.error():
                    if msg.error().code() == KafkaError._PARTITION_EOF:
                        continue
                    else:
                        _logger.error("Consumer error: %s", msg.error())
                        break
                        
                # Process
                self.process_message(msg)
                
                # Commit offset manually after successful processing/DLQ publish
                self.consumer.commit(asynchronous=False)
                count += 1
                
        except KeyboardInterrupt:
            _logger.info("Consumer interrupted by user.")
        finally:
            self._save_state()
            self.consumer.close()
            self.producer.close()
            _logger.info("Final Consumer Stats: %s", self.get_stats())

    def get_review_queue(self) -> collections.deque:
        return self.review_queue

    def get_stats(self) -> dict:
        avg_ms = self._total_processing_ms / max(1, self._messages_processed)
        # Consumer lag requires admin client or manual offset calculation, returning 0 for simplicity
        return {
            "messages_processed": self._messages_processed,
            "messages_flagged": self._messages_flagged,
            "messages_failed": self._messages_failed,
            "avg_processing_ms": avg_ms,
            "consumer_lag": 0 
        }

    def _save_state(self) -> None:
        cfg = load_config()
        path = cfg["drift"]["detector_state_path"]
        try:
            self.drift_detector.save_state(path)
            _logger.info("Saved drift detector state to %s", path)
        except Exception as e:
            _logger.warning("Could not save drift detector state: %s", e)


if __name__ == "__main__":
    consumer = FraudDetectionConsumer()
    consumer.subscribe()
    consumer.run(max_messages=10000)
