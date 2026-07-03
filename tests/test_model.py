"""
tests/test_model.py
-------------------
Unit tests for the CIPHER ML layer: LightGBMDetector, IsolationForestDetector,
and EnsembleDetector.

All tests use small synthetic datasets so they run quickly in CI without
requiring real transaction data.  The IsolationForest score-inversion test
uses deliberately constructed anomalous samples (extreme outliers) and normal
samples to verify the direction of the risk score.

Tests
-----
* test_lgbm_predict_proba_range          — all outputs in [0, 1]
* test_isolation_forest_score_inverted   — anomalies score higher than normals
* test_ensemble_weights_sum_to_one       — lgbm_w + iso_w ≈ 1.0 (float-safe)
* test_evaluate_returns_all_metrics      — evaluate() dict has required keys
* test_model_save_load_consistency       — round-trip predictions are identical
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path (handled by conftest.py in practice,
# but added here defensively so the file can be executed directly too).
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent.parent))


# ---------------------------------------------------------------------------
# Helpers — synthetic datasets
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(seed=0)

REQUIRED_METRIC_KEYS = {"f1", "auc_roc", "auc_pr", "precision", "recall"}


def _make_classification_data(
    n_samples: int = 400,
    n_features: int = 20,
    fraud_ratio: float = 0.1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (X_train, X_test, y_train, y_test) for binary classification.

    Fraud class (1) is drawn from a slightly shifted Gaussian so the signal is
    learnable even with tiny datasets.

    Args:
        n_samples: Total number of samples (train + test combined).
        n_features: Number of features.
        fraud_ratio: Fraction of samples labelled fraud.

    Returns:
        (X_train, X_test, y_train, y_test) with an 80/20 split.
    """
    n_fraud = max(int(n_samples * fraud_ratio), 2)
    n_legit = n_samples - n_fraud

    X_legit = RNG.normal(loc=0.0, scale=1.0, size=(n_legit, n_features))
    X_fraud = RNG.normal(loc=3.0, scale=1.0, size=(n_fraud, n_features))

    X = np.vstack([X_legit, X_fraud])
    y = np.concatenate(
        [np.zeros(n_legit, dtype=int), np.ones(n_fraud, dtype=int)]
    )

    split = int(n_samples * 0.8)
    shuffle_idx = RNG.permutation(n_samples)
    X_shuf, y_shuf = X[shuffle_idx], y[shuffle_idx]

    return X_shuf[:split], X_shuf[split:], y_shuf[:split], y_shuf[split:]


def _make_anomaly_data(
    n_normal: int = 200,
    n_anomaly: int = 20,
    n_features: int = 10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (X_normal, X_anomaly, X_all, y_all) for anomaly detection tests.

    Normal samples come from N(0, 1); anomalous samples come from N(50, 1)
    — extreme outliers that any Isolation Forest should detect easily.

    Args:
        n_normal:   Number of normal (legitimate) samples.
        n_anomaly:  Number of anomalous samples.
        n_features: Number of features.

    Returns:
        (X_normal, X_anomaly, X_all, y_all) where y_all=0 for normal and
        y_all=1 for anomalous samples.
    """
    X_normal = RNG.normal(loc=0.0, scale=1.0, size=(n_normal, n_features))
    X_anomaly = RNG.normal(loc=50.0, scale=1.0, size=(n_anomaly, n_features))
    X_all = np.vstack([X_normal, X_anomaly])
    y_all = np.concatenate(
        [np.zeros(n_normal, dtype=int), np.ones(n_anomaly, dtype=int)]
    )
    return X_normal, X_anomaly, X_all, y_all


# ---------------------------------------------------------------------------
# Minimal config fixture — patches load_config so no file I/O is needed.
# ---------------------------------------------------------------------------

_MINIMAL_CONFIG: dict = {
    "model": {
        "lgbm": {
            "n_estimators": 50,          # fast for tests
            "learning_rate": 0.1,
            "num_leaves": 15,
            "max_depth": 4,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_samples": 5,
            "reg_alpha": 0.0,
            "reg_lambda": 0.0,
        },
        "isolation_forest": {
            "n_estimators": 50,
            "max_samples": 100,
            "contamination": 0.1,
            "random_state": 42,
        },
        "ensemble": {
            "lgbm_weight": 0.7,
            "iso_weight": 0.3,
            "threshold": 0.5,
        },
        "artifacts": {
            "local_model_path": "models/cipher_ensemble.pkl",
        },
    },
    "mlflow": {
        "experiment_name": "test",
        "tracking_uri": "./mlruns",
        "model_registry_name": "TestModel",
    },
}


@pytest.fixture
def patched_config():
    """Patch load_config across the entire ml_layer package for all tests."""
    with patch(
        "src.ml_layer.model.load_config", return_value=_MINIMAL_CONFIG
    ):
        yield


# ---------------------------------------------------------------------------
# Test 1 — LightGBMDetector predict_proba outputs must be in [0, 1]
# ---------------------------------------------------------------------------


def test_lgbm_predict_proba_range(patched_config: None) -> None:
    """All predict_proba() outputs from LightGBMDetector must lie in [0, 1].

    Tests that the class-1 probability column retrieved from
    ``LGBMClassifier.predict_proba`` is bounded and not, e.g., affected by
    a log-odds or raw score that could exceed the unit interval.
    """
    from src.ml_layer.model import LightGBMDetector

    X_train, X_test, y_train, _ = _make_classification_data()
    detector = LightGBMDetector()
    detector.train(X_train, y_train)

    proba = detector.predict_proba(X_test)

    assert proba.ndim == 1, "predict_proba() must return a 1-D array"
    assert len(proba) == len(X_test), "Output length must equal n_test_samples"
    assert np.all(proba >= 0.0), f"Found negative probabilities: min={proba.min()}"
    assert np.all(proba <= 1.0), f"Found probabilities > 1: max={proba.max()}"


# ---------------------------------------------------------------------------
# Test 2 — IsolationForest: anomalous samples must score higher than normals
# ---------------------------------------------------------------------------


def test_isolation_forest_score_inverted(patched_config: None) -> None:
    """High-anomaly samples must receive higher risk scores than normal samples.

    Verifies the score inversion formula::

        score = 1 - (raw - raw.min()) / (raw.max() - raw.min())

    by fitting on normal data and evaluating on extreme outliers (N(50,1))
    alongside in-distribution normals.  The mean score for anomalies must
    exceed the mean score for normals by a meaningful margin.
    """
    from src.ml_layer.model import IsolationForestDetector

    X_normal, X_anomaly, X_all, _ = _make_anomaly_data(
        n_normal=300, n_anomaly=30, n_features=10
    )

    detector = IsolationForestDetector()
    # Train on normal data only (unsupervised; y is ignored).
    detector.train(X_normal, np.zeros(len(X_normal), dtype=int))

    scores = detector.predict_proba(X_all)
    normal_scores = scores[: len(X_normal)]
    anomaly_scores = scores[len(X_normal) :]

    mean_normal = float(np.mean(normal_scores))
    mean_anomaly = float(np.mean(anomaly_scores))

    assert mean_anomaly > mean_normal, (
        f"Anomaly scores (mean={mean_anomaly:.4f}) must exceed normal scores "
        f"(mean={mean_normal:.4f}).  Score inversion may be incorrect."
    )
    # Expect a large gap with such extreme outliers.
    assert (mean_anomaly - mean_normal) > 0.3, (
        f"Expected a substantial score gap (>0.3) between anomalies and "
        f"normals, got {mean_anomaly - mean_normal:.4f}."
    )


# ---------------------------------------------------------------------------
# Test 3 — Ensemble weights must sum to 1.0 (within floating-point tolerance)
# ---------------------------------------------------------------------------


def test_ensemble_weights_sum_to_one(patched_config: None) -> None:
    """lgbm_weight + iso_weight must equal 1.0 (within 1e-6 tolerance).

    Uses a tolerance rather than exact equality because YAML parsing and
    Python float arithmetic can introduce tiny rounding errors.
    """
    from src.ml_layer.model import EnsembleDetector

    detector = EnsembleDetector()
    weight_sum = detector.lgbm_weight_ + detector.iso_weight_

    assert abs(weight_sum - 1.0) < 1e-6, (
        f"lgbm_weight ({detector.lgbm_weight_}) + "
        f"iso_weight ({detector.iso_weight_}) = {weight_sum}, "
        f"expected 1.0 (tolerance 1e-6)."
    )


# ---------------------------------------------------------------------------
# Test 4 — evaluate() must return a dict with all five required keys
# ---------------------------------------------------------------------------


def test_evaluate_returns_all_metrics(patched_config: None) -> None:
    """evaluate() on EnsembleDetector must return all five required metric keys.

    The serving layer and MLflow logger depend on the presence of
    ``{f1, auc_roc, auc_pr, precision, recall}`` in the returned dict.
    """
    from src.ml_layer.model import EnsembleDetector

    X_train, X_test, y_train, y_test = _make_classification_data()
    detector = EnsembleDetector()
    detector.train(X_train, y_train)

    metrics = detector.evaluate(X_test, y_test)

    assert isinstance(metrics, dict), "evaluate() must return a dict"
    missing = REQUIRED_METRIC_KEYS - set(metrics.keys())
    assert not missing, (
        f"evaluate() dict is missing required keys: {missing}"
    )

    # All values must be finite floats in [0, 1] (they are ratios / AUCs).
    for key in REQUIRED_METRIC_KEYS:
        val = metrics[key]
        assert isinstance(val, float), f"metrics['{key}'] must be a float"
        assert 0.0 <= val <= 1.0, (
            f"metrics['{key}'] = {val} is outside [0, 1]"
        )


# ---------------------------------------------------------------------------
# Test 5 — Save/load round-trip produces identical predictions
# ---------------------------------------------------------------------------


def test_model_save_load_consistency(patched_config: None) -> None:
    """Predictions must be identical before and after a joblib save/load.

    Trains an EnsembleDetector, captures ``predict_proba`` on the test set,
    serialises the ensemble to a temporary file, loads it back, and verifies
    that the loaded model produces bit-for-bit identical predictions.
    """
    from src.ml_layer.model import EnsembleDetector

    X_train, X_test, y_train, _ = _make_classification_data()
    detector = EnsembleDetector()
    detector.train(X_train, y_train)

    proba_before = detector.predict_proba(X_test)

    with tempfile.TemporaryDirectory() as tmp_dir:
        save_path = str(Path(tmp_dir) / "ensemble_test.pkl")
        detector.save(save_path)

        # Load back into a fresh EnsembleDetector shell.
        loader = EnsembleDetector()
        loaded_detector = loader.load(save_path)

    proba_after = loaded_detector.predict_proba(X_test)

    assert np.array_equal(proba_before, proba_after), (
        "predict_proba() outputs differ after save/load round-trip.\n"
        f"Max absolute diff: {np.abs(proba_before - proba_after).max():.2e}"
    )
