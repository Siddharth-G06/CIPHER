"""
tests/test_kafka.py
-------------------
Unit tests for the CIPHER Kafka streaming layer.
Uses unittest.mock to simulate Kafka Producer and Consumer.
"""

import json
from unittest.mock import MagicMock, patch

import pytest
from confluent_kafka import KafkaError

from src.serving_layer.kafka_consumer import FraudDetectionConsumer
from src.serving_layer.kafka_producer import TransactionProducer


@pytest.fixture
def mock_producer():
    with patch("src.serving_layer.kafka_producer.Producer") as mock:
        yield mock


@pytest.fixture
def mock_consumer():
    with patch("src.serving_layer.kafka_consumer.Consumer") as mock:
        yield mock


def test_producer_serializes_transaction_correctly(mock_producer):
    """Verify publish_transaction serializes required fields to valid JSON."""
    producer = TransactionProducer()
    
    test_tx = {
        "TransactionID": "TEST1234",
        "card1": 12345,
        "TransactionAmt": 99.99,
        "TransactionDT": 1718461800,  # some unix timestamp
        "extra_field": "hello"
    }
    
    producer.publish_transaction(test_tx)
    
    # Check that produce was called
    producer_instance = mock_producer.return_value
    assert producer_instance.produce.called
    
    # Get kwargs passed to produce
    _, kwargs = producer_instance.produce.call_args
    payload = kwargs["value"].decode("utf-8")
    key = kwargs["key"].decode("utf-8")
    
    assert key == "12345"
    
    parsed = json.loads(payload)
    assert parsed["transaction_id"] == "TEST1234"
    assert parsed["card1"] == 12345
    assert parsed["amount"] == 99.99
    assert "extra_field" in parsed["features"]


def test_dlq_publish_on_validation_error(mock_consumer, mock_producer):
    """Verify _publish_to_dlq called on malformed message."""
    # We patch Producer in kafka_consumer because it creates a TransactionProducer
    with patch("src.serving_layer.kafka_producer.Producer"):
        consumer = FraudDetectionConsumer()
        
        # Mock the process_message input
        mock_msg = MagicMock()
        # Invalid JSON
        mock_msg.value.return_value = b"{malformed_json_here"
        
        # Ensure _publish_to_dlq is called
        with patch.object(consumer, "_publish_to_dlq") as mock_dlq:
            consumer.process_message(mock_msg)
            mock_dlq.assert_called_once()
            args, _ = mock_dlq.call_args
            assert args[0] == "{malformed_json_here"
            assert "Validation error" in args[1]


def test_review_queue_maxlen(mock_consumer, mock_producer):
    """Verify review_queue never exceeds 500."""
    with patch("src.serving_layer.kafka_producer.Producer"):
        consumer = FraudDetectionConsumer()
        
        # Mock predict to always flag
        consumer.ensemble.predict_proba = MagicMock(return_value=[0.9])
        consumer.ensemble.threshold_ = 0.5
        consumer.explainer = MagicMock()
        consumer.explainer.explain_async = MagicMock()
        
        mock_msg = MagicMock()
        valid_json = json.dumps({
            "transaction_id": "TEST",
            "card1": 123,
            "timestamp": "2024-06-15T14:30:00Z",
            "amount": 100.0,
            "features": {"f1": 1}
        }).encode("utf-8")
        mock_msg.value.return_value = valid_json
        
        # Process 600 messages
        for _ in range(600):
            consumer.process_message(mock_msg)
            
        assert len(consumer.review_queue) == 500


def test_stats_increment_correctly(mock_consumer, mock_producer):
    """Verify messages_processed=10 in get_stats()."""
    with patch("src.serving_layer.kafka_producer.Producer"):
        consumer = FraudDetectionConsumer()
        
        consumer.ensemble.predict_proba = MagicMock(return_value=[0.1])
        consumer.ensemble.threshold_ = 0.5
        
        mock_msg = MagicMock()
        valid_json = json.dumps({
            "transaction_id": "TEST",
            "card1": 123,
            "timestamp": "2024-06-15T14:30:00Z",
            "amount": 100.0,
            "features": {"f1": 1}
        }).encode("utf-8")
        mock_msg.value.return_value = valid_json
        
        for _ in range(10):
            consumer.process_message(mock_msg)
            
        stats = consumer.get_stats()
        assert stats["messages_processed"] == 10
        assert stats["messages_failed"] == 0


def test_flagged_transaction_published_above_threshold(mock_consumer, mock_producer):
    """Verify publish_flagged called when score > threshold."""
    with patch("src.serving_layer.kafka_producer.Producer"):
        consumer = FraudDetectionConsumer()
        
        consumer.ensemble.predict_proba = MagicMock(return_value=[0.8])
        consumer.ensemble.threshold_ = 0.5
        consumer.explainer = MagicMock()
        consumer.explainer.explain_async = MagicMock()
        
        with patch.object(consumer.producer, "publish_flagged") as mock_pub:
            mock_msg = MagicMock()
            valid_json = json.dumps({
                "transaction_id": "TEST_FLAG",
                "card1": 123,
                "timestamp": "2024-06-15T14:30:00Z",
                "amount": 100.0,
                "features": {"f1": 1}
            }).encode("utf-8")
            mock_msg.value.return_value = valid_json
            
            consumer.process_message(mock_msg)
            mock_pub.assert_called_once()
            
            # Check argument
            flagged_tx = mock_pub.call_args[0][0]
            assert flagged_tx.transaction_id == "TEST_FLAG"
            assert flagged_tx.ensemble_score == 0.8


def test_no_flag_below_threshold(mock_consumer, mock_producer):
    """Verify publish_flagged NOT called when score < threshold."""
    with patch("src.serving_layer.kafka_producer.Producer"):
        consumer = FraudDetectionConsumer()
        
        consumer.ensemble.predict_proba = MagicMock(return_value=[0.2])
        consumer.ensemble.threshold_ = 0.5
        
        with patch.object(consumer.producer, "publish_flagged") as mock_pub:
            mock_msg = MagicMock()
            valid_json = json.dumps({
                "transaction_id": "TEST_NO_FLAG",
                "card1": 123,
                "timestamp": "2024-06-15T14:30:00Z",
                "amount": 100.0,
                "features": {"f1": 1}
            }).encode("utf-8")
            mock_msg.value.return_value = valid_json
            
            consumer.process_message(mock_msg)
            mock_pub.assert_not_called()
