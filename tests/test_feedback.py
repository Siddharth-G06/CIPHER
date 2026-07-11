"""
tests/test_feedback.py
------------------------
Unit tests for the CIPHER human-in-the-loop feedback and retraining layer.

Test coverage
-------------
* FeedbackStore append-only semantics and duplicate rejection.
* mark_as_used / get_unused_feedback filtering.
* Sample weight computation in _merge_with_feedback.
* Champion/Challenger promotion and rejection decision logic.
* ModelUpdateSignal write → check → clear lifecycle.
* RetrainingTrigger feedback-volume trigger.
* FeedbackStore stats aggregation.

All tests use in-memory SQLite (``db_path=":memory:"``) and mock out
MLflow / EnsembleDetector to avoid requiring trained models or a running
MLflow server.
"""

from __future__ import annotations

import json
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import List
from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np
import pytest

from src.feedback_layer.feedback_store import (
    AnalystDecision,
    FeedbackRecord,
    FeedbackStore,
)
from src.feedback_layer.model_update_signal import ModelUpdateSignal
from src.feedback_layer.retraining_trigger import RetrainingTrigger


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_record(
    tx_id: str = "txn-001",
    analyst_id: str = "analyst-1",
    decision: AnalystDecision = AnalystDecision.CONFIRM_FRAUD,
    confidence: float = 0.8,
    model_score: float = 0.75,
    true_label: int = 1,
) -> FeedbackRecord:
    """Create a minimal :class:`FeedbackRecord` for use in tests."""
    return FeedbackRecord(
        transaction_id=tx_id,
        analyst_id=analyst_id,
        decision=decision,
        confidence=confidence,
        model_score=model_score,
        true_label=true_label,
    )


def _in_memory_store() -> FeedbackStore:
    """Return a :class:`FeedbackStore` backed by an in-memory SQLite database."""
    return FeedbackStore(db_path=":memory:")


# ---------------------------------------------------------------------------
# FeedbackStore — append-only / idempotent insert
# ---------------------------------------------------------------------------

class TestFeedbackStoreAppendOnly:
    """FeedbackStore must silently ignore duplicate transaction_ids."""

    def test_feedback_store_append_only(self) -> None:
        """Adding two records with the same transaction_id keeps only one."""
        store = _in_memory_store()
        record = _make_record(tx_id="txn-dup-001")

        store.add_feedback(record)
        store.add_feedback(record)   # duplicate — should be ignored

        all_records: List[FeedbackRecord] = store.get_feedback_history(limit=10)
        assert len(all_records) == 1, (
            "FeedbackStore must be append-only: a duplicate transaction_id "
            "must not create a second row."
        )
        assert all_records[0].transaction_id == "txn-dup-001"

    def test_different_transaction_ids_both_stored(self) -> None:
        """Two distinct transaction_ids should each be stored."""
        store = _in_memory_store()
        store.add_feedback(_make_record(tx_id="txn-A"))
        store.add_feedback(_make_record(tx_id="txn-B"))

        all_records = store.get_feedback_history(limit=10)
        assert len(all_records) == 2


# ---------------------------------------------------------------------------
# FeedbackStore — mark_as_used
# ---------------------------------------------------------------------------

class TestMarkAsUsed:
    """mark_as_used must flip the flag only for specified IDs."""

    def test_mark_as_used_updates_flag(self) -> None:
        """add 3 records, mark 2, get_unused_feedback returns only 1."""
        store = _in_memory_store()
        ids = ["txn-mark-001", "txn-mark-002", "txn-mark-003"]
        for tx_id in ids:
            store.add_feedback(_make_record(tx_id=tx_id))

        store.mark_as_used(["txn-mark-001", "txn-mark-002"], run_id="run-abc")

        unused = store.get_unused_feedback()
        assert len(unused) == 1
        assert unused[0].transaction_id == "txn-mark-003"

    def test_marked_records_carry_run_id(self) -> None:
        """Records marked as used must store the retraining run_id."""
        store = _in_memory_store()
        store.add_feedback(_make_record(tx_id="txn-run-001"))
        store.mark_as_used(["txn-run-001"], run_id="run-xyz-999")

        history = store.get_feedback_history(limit=1)
        assert history[0].used_in_retraining is True
        assert history[0].retraining_run_id == "run-xyz-999"

    def test_mark_empty_list_is_noop(self) -> None:
        """Calling mark_as_used with an empty list must not raise."""
        store = _in_memory_store()
        store.add_feedback(_make_record(tx_id="txn-noop-001"))
        store.mark_as_used([], run_id="run-noop")  # should be a no-op

        unused = store.get_unused_feedback()
        assert len(unused) == 1  # still unused


# ---------------------------------------------------------------------------
# Sample weight computation
# ---------------------------------------------------------------------------

class TestSampleWeightComputation:
    """_merge_with_feedback must compute weights as specified."""

    def _pipeline_with_store(self, store: FeedbackStore):
        """Return a RetrainingPipeline whose _store is replaced by *store*."""
        from src.feedback_layer.retrain_pipeline import RetrainingPipeline

        with patch("src.feedback_layer.retrain_pipeline.FeedbackStore"):
            with patch("src.feedback_layer.retrain_pipeline.mlflow"):
                with patch("src.feedback_layer.retrain_pipeline.MlflowClient"):
                    pipeline = RetrainingPipeline.__new__(RetrainingPipeline)
                    pipeline._cfg = {
                        "feedback": {
                            "retraining_threshold": 100,
                            "scheduled_retraining_days": 7,
                            "min_improvement_auc_pr": 0.005,
                            "base_data_pickle_path": "data/processed/train_featured.pkl",
                            "model_updated_signal_path": "models/model_updated.json",
                            "trigger_poll_interval_seconds": 300,
                            "feedback_base_weight_multiplier": 1.0,
                        },
                        "mlflow": {
                            "experiment_name": "test-exp",
                            "tracking_uri": "./mlruns",
                            "model_registry_name": "TestModel",
                        },
                        "model": {
                            "lgbm": {
                                "n_estimators": 10,
                                "learning_rate": 0.1,
                                "num_leaves": 31,
                                "max_depth": -1,
                                "subsample": 0.8,
                                "colsample_bytree": 0.8,
                                "min_child_samples": 5,
                                "reg_alpha": 0.0,
                                "reg_lambda": 0.0,
                            },
                            "artifacts": {"local_model_path": "models/test.pkl"},
                        },
                        "explainer": {"plots_dir": "plots/shap"},
                    }
                    pipeline._feedback_cfg = pipeline._cfg["feedback"]
                    pipeline._mlflow_cfg = pipeline._cfg["mlflow"]
                    pipeline._store = store
                    pipeline._signal = MagicMock()
                    pipeline._client = MagicMock()
                    pipeline._config_path = "config/config.yaml"
                    # New instance attrs added in the updated __init__.
                    pipeline._X_test = None
                    pipeline._y_test = None
                    pipeline._last_run_id = ""
        return pipeline

    def test_sample_weight_computation(self) -> None:
        """Feedback records get weight = (base_size/feedback_size) * confidence;
        base records get weight 1.0.
        """
        store = _in_memory_store()
        pipeline = self._pipeline_with_store(store)

        n_base = 100
        n_feedback = 5
        X_base = pd.DataFrame(np.zeros((n_base, 3)), columns=["f0", "f1", "f2"])
        y_base = pd.Series(np.zeros(n_base, dtype=int))

        feedback_records = [
            _make_record(
                tx_id=f"txn-fb-{i}",
                decision=AnalystDecision.CONFIRM_FRAUD,
                confidence=0.9,
            )
            for i in range(n_feedback)
        ]

        X_merged, y_merged, weights = pipeline._merge_with_feedback(
            X_base, y_base, feedback_records
        )

        # Base weights must all be 1.0.
        base_weights = weights[:n_base]
        np.testing.assert_array_almost_equal(base_weights, np.ones(n_base))

        # Feedback weights = (base_size / feedback_size) * confidence.
        expected_fb_weight = (n_base / n_feedback) * 0.9
        feedback_weights = weights[n_base:]
        np.testing.assert_array_almost_equal(
            feedback_weights,
            np.full(n_feedback, expected_fb_weight),
            decimal=6,
        )

    def test_escalate_records_excluded_from_merge(self) -> None:
        """ESCALATE records must not appear in the merged training set."""
        store = _in_memory_store()
        pipeline = self._pipeline_with_store(store)

        X_base = pd.DataFrame(np.zeros((10, 2)), columns=["f0", "f1"])
        y_base = pd.Series(np.zeros(10, dtype=int))

        escalate_record = _make_record(
            tx_id="txn-esc-001", decision=AnalystDecision.ESCALATE
        )

        X_merged, y_merged, weights = pipeline._merge_with_feedback(
            X_base, y_base, [escalate_record]
        )

        # Only base rows should be present.
        assert len(X_merged) == 10
        assert len(weights) == 10


# ---------------------------------------------------------------------------
# Champion / Challenger promotion logic
# ---------------------------------------------------------------------------

class TestChampionChallengerPromotion:
    """_should_promote must apply AUC-PR + F1 guard-rail rules correctly."""

    def _pipeline(self):
        from src.feedback_layer.retrain_pipeline import RetrainingPipeline

        with patch("src.feedback_layer.retrain_pipeline.FeedbackStore"):
            with patch("src.feedback_layer.retrain_pipeline.mlflow"):
                with patch("src.feedback_layer.retrain_pipeline.MlflowClient"):
                    pipeline = RetrainingPipeline.__new__(RetrainingPipeline)
                    pipeline._feedback_cfg = {
                        "min_improvement_auc_pr": 0.005,
                        "feedback_base_weight_multiplier": 1.0,
                    }
                    pipeline._cfg = {"explainer": {"plots_dir": "plots/shap"}}
                    pipeline._signal = MagicMock()
                    pipeline._client = MagicMock()
                    # New instance attrs added in the updated __init__.
                    pipeline._X_test = None
                    pipeline._y_test = None
                    pipeline._last_run_id = ""
        return pipeline

    def test_champion_challenger_promotes_better_model(self) -> None:
        """Challenger with auc_pr=0.86 > champion 0.85+0.005 should be promoted."""
        pipeline = self._pipeline()
        champion_metrics = {"auc_pr": 0.85, "f1": 0.70}
        challenger_metrics = {"auc_pr": 0.86, "f1": 0.70}

        assert pipeline._should_promote(
            challenger_metrics, champion_metrics
        ), "Challenger that clearly beats champion should be promoted."

    def test_champion_challenger_rejects_worse_model(self) -> None:
        """Challenger with auc_pr=0.84 < champion 0.85 should be rejected."""
        pipeline = self._pipeline()
        champion_metrics = {"auc_pr": 0.85, "f1": 0.70}
        challenger_metrics = {"auc_pr": 0.84, "f1": 0.70}

        assert not pipeline._should_promote(
            challenger_metrics, champion_metrics
        ), "Challenger worse than champion should be rejected."

    def test_champion_challenger_rejects_when_improvement_insufficient(self) -> None:
        """Challenger with delta < min_improvement (0.003 < 0.005) should be rejected."""
        pipeline = self._pipeline()
        champion_metrics = {"auc_pr": 0.85, "f1": 0.70}
        challenger_metrics = {"auc_pr": 0.853, "f1": 0.70}  # delta = 0.003

        assert not pipeline._should_promote(
            challenger_metrics, champion_metrics
        ), "Insufficient AUC-PR improvement must not trigger promotion."

    def test_champion_challenger_rejects_when_f1_drops_too_much(self) -> None:
        """Challenger must not be promoted if F1 drops below 99% of champion."""
        pipeline = self._pipeline()
        champion_metrics = {"auc_pr": 0.85, "f1": 0.80}
        # F1 guard-rail: 0.80 * 0.99 = 0.792; challenger at 0.78 is below.
        challenger_metrics = {"auc_pr": 0.86, "f1": 0.78}

        assert not pipeline._should_promote(
            challenger_metrics, champion_metrics
        ), "F1 guard-rail must prevent promotion if F1 drops more than 1%."

    def test_auto_promotes_when_no_champion(self) -> None:
        """With no champion (empty dict) the challenger must always be promoted."""
        pipeline = self._pipeline()
        challenger_metrics = {"auc_pr": 0.60, "f1": 0.50}

        assert pipeline._should_promote(
            challenger_metrics, {}
        ), "Challenger must auto-promote when no champion is registered."


# ---------------------------------------------------------------------------
# ModelUpdateSignal
# ---------------------------------------------------------------------------

class TestModelUpdateSignal:
    """ModelUpdateSignal write → check → clear cycle must work atomically."""

    def test_model_update_signal_write_read_clear(self, tmp_path: Path) -> None:
        """Write signal, check_signal returns version, clear_signal removes it."""
        signal_path = str(tmp_path / "model_updated.json")
        signal = ModelUpdateSignal(signal_path=signal_path)

        # Initially no signal.
        assert signal.check_signal() is None

        # After write, check_signal returns the version.
        signal.write_signal("v2")
        version = signal.check_signal()
        assert version == "v2", f"Expected 'v2', got '{version}'"

        # After clear, check_signal returns None again.
        signal.clear_signal()
        assert signal.check_signal() is None

    def test_clear_signal_is_idempotent(self, tmp_path: Path) -> None:
        """Calling clear_signal when file is absent must not raise."""
        signal_path = str(tmp_path / "missing_signal.json")
        signal = ModelUpdateSignal(signal_path=signal_path)
        signal.clear_signal()   # file doesn't exist — should be a no-op

    def test_signal_file_has_timestamp(self, tmp_path: Path) -> None:
        """The written signal file must include a 'written_at' timestamp."""
        signal_path = str(tmp_path / "sig.json")
        signal = ModelUpdateSignal(signal_path=signal_path)
        signal.write_signal("v3")

        with open(signal_path) as fh:
            payload = json.load(fh)

        assert "written_at" in payload
        assert payload["new_model_version"] == "v3"


# ---------------------------------------------------------------------------
# RetrainingTrigger — feedback volume trigger
# ---------------------------------------------------------------------------

class TestRetrainingTriggerFeedbackVolume:
    """RetrainingTrigger must fire when unused feedback exceeds threshold."""

    def test_retraining_trigger_feedback_volume(self) -> None:
        """check_triggers returns 'feedback_volume_threshold' with 101 unused records."""
        # Build a mock store that returns 101 unused records.
        mock_store = MagicMock(spec=FeedbackStore)
        mock_store.get_unused_feedback.return_value = [
            _make_record(tx_id=f"txn-vol-{i}") for i in range(101)
        ]

        trigger = RetrainingTrigger.__new__(RetrainingTrigger)
        trigger._cfg = {
            "retraining_threshold": 100,
            "scheduled_retraining_days": 7,
            "trigger_poll_interval_seconds": 300,
        }
        trigger._store = mock_store
        trigger._drift_path = Path("nonexistent_drift_events.json")
        trigger._last_retraining_time = datetime.now(timezone.utc)
        trigger._lock = threading.Lock()

        reason = trigger.check_triggers()
        assert reason == "feedback_volume_threshold", (
            f"Expected 'feedback_volume_threshold', got '{reason}'"
        )

    def test_retraining_trigger_no_fire_below_threshold(self) -> None:
        """check_triggers returns None when unused feedback is below threshold."""
        mock_store = MagicMock(spec=FeedbackStore)
        mock_store.get_unused_feedback.return_value = [
            _make_record(tx_id=f"txn-low-{i}") for i in range(50)
        ]

        trigger = RetrainingTrigger.__new__(RetrainingTrigger)
        trigger._cfg = {
            "retraining_threshold": 100,
            "scheduled_retraining_days": 7,
            "trigger_poll_interval_seconds": 300,
        }
        trigger._store = mock_store
        trigger._drift_path = Path("nonexistent_drift_events.json")
        trigger._last_retraining_time = datetime.now(timezone.utc)
        trigger._lock = threading.Lock()

        reason = trigger.check_triggers()
        assert reason is None, (
            f"Expected None when below threshold, got '{reason}'"
        )


# ---------------------------------------------------------------------------
# FeedbackStore — stats
# ---------------------------------------------------------------------------

class TestFeedbackStatsCorrectCounts:
    """get_feedback_stats must return accurate counts for each decision type."""

    def test_feedback_stats_correct_counts(self) -> None:
        """3 CONFIRM_FRAUD + 2 FALSE_POSITIVE → verify exact counts in stats dict."""
        store = _in_memory_store()

        for i in range(3):
            store.add_feedback(
                _make_record(
                    tx_id=f"txn-cf-{i}",
                    decision=AnalystDecision.CONFIRM_FRAUD,
                    confidence=0.9,
                )
            )
        for i in range(2):
            store.add_feedback(
                _make_record(
                    tx_id=f"txn-fp-{i}",
                    decision=AnalystDecision.FALSE_POSITIVE,
                    confidence=0.7,
                )
            )

        stats = store.get_feedback_stats()

        assert stats["total_records"] == 5, (
            f"Expected 5 total records, got {stats['total_records']}"
        )
        assert stats["confirm_fraud_count"] == 3, (
            f"Expected 3 CONFIRM_FRAUD, got {stats['confirm_fraud_count']}"
        )
        assert stats["false_positive_count"] == 2, (
            f"Expected 2 FALSE_POSITIVE, got {stats['false_positive_count']}"
        )
        assert stats["escalate_count"] == 0, (
            f"Expected 0 ESCALATE, got {stats['escalate_count']}"
        )
        assert stats["unused_count"] == 5, (
            f"Expected 5 unused, got {stats['unused_count']}"
        )
        # avg confidence: (3*0.9 + 2*0.7) / 5 = (2.7 + 1.4) / 5 = 0.82
        assert abs(stats["avg_confidence"] - 0.82) < 0.01, (
            f"avg_confidence mismatch: expected 0.82, got {stats['avg_confidence']}"
        )

    def test_stats_after_mark_as_used(self) -> None:
        """unused_count must decrease after mark_as_used."""
        store = _in_memory_store()
        for i in range(4):
            store.add_feedback(_make_record(tx_id=f"txn-su-{i}"))

        store.mark_as_used(["txn-su-0", "txn-su-1"], run_id="run-test")
        stats = store.get_feedback_stats()

        assert stats["unused_count"] == 2
        assert stats["total_records"] == 4  # event log is append-only


# ---------------------------------------------------------------------------
# Imports needed inside test bodies (delayed to avoid top-level import issues)
# ---------------------------------------------------------------------------

import pandas as pd  # noqa: E402  (kept at bottom so tests above can reference it)
