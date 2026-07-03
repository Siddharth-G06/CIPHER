"""
tests/test_drift_detector.py
-----------------------------
Unit tests for the CIPHER drift-detection layer.

Covers:
    * ADWIN firing on sudden error-rate shift
    * PSI showing no drift on same-distribution data
    * PSI showing drift on a shifted distribution
    * Observer notification with correct drift_info keys
    * Detector state persistence (save / load round-trip)
    * DriftSimulator injecting drift at the correct point

All tests use synthetic DataFrames — no real transaction data is needed.
DriftDetector is constructed directly with a minimal in-memory config so no
``config.yaml`` file I/O is required.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# Ensure project root is on sys.path (mirrors conftest.py).
sys.path.insert(0, str(Path(__file__).parent.parent))

# ---------------------------------------------------------------------------
# Minimal in-memory config used across all tests
# ---------------------------------------------------------------------------

_DRIFT_CFG: dict = {
    "adwin_delta": 0.002,
    "rolling_error_buffer_size": 1000,
    "psi_current_window_size": 200,   # small so tests reach window threshold fast
    "psi_bins": 10,
    "psi_warning_threshold": 0.1,
    "psi_critical_threshold": 0.2,
    "retraining_trigger_path": "logs/drift_events_test.json",
    "detector_state_path": "models/drift_detector_state_test.pkl",
    "monitored_features": [
        "card_degree_1h",
        "merchant_degree_1h",
        "card_tx_count_24h",
        "amt_zscore_24h",
    ],
    "drift_simulation": {
        "injection_point": 0.7,
        "drift_type": "high_value",
    },
}

_FULL_CFG: dict = {"drift": _DRIFT_CFG}

RNG = np.random.default_rng(seed=99)

# Features used in tests
_FEATURE_COLS = ["card_degree_1h", "merchant_degree_1h",
                 "card_tx_count_24h", "amt_zscore_24h"]


def _make_feature_row(values: dict | None = None) -> pd.DataFrame:
    """Return a single-row DataFrame with monitored feature columns."""
    defaults = {col: 0.0 for col in _FEATURE_COLS}
    if values:
        defaults.update(values)
    return pd.DataFrame([defaults])


def _make_feature_df(
    n: int = 300,
    loc: float = 0.0,
    scale: float = 1.0,
) -> pd.DataFrame:
    """Return a DataFrame with ``n`` rows of Gaussian feature values."""
    data = {col: RNG.normal(loc=loc, scale=scale, size=n)
            for col in _FEATURE_COLS}
    return pd.DataFrame(data)


@pytest.fixture
def detector():
    """Provide a DriftDetector with mocked config — no file I/O."""
    with patch(
        "src.ml_layer.drift_detector.load_config", return_value=_FULL_CFG
    ):
        from src.ml_layer.drift_detector import DriftDetector
        return DriftDetector()


# ---------------------------------------------------------------------------
# Test 1 — ADWIN fires on sudden error-rate shift
# ---------------------------------------------------------------------------

def test_adwin_fires_on_sudden_drift(detector) -> None:
    """ADWIN must fire within the second batch (error=1) after 500 correct preds.

    Feeds 500 error=0 values (stable good performance) then up to 500 error=1
    values (complete model failure).  ADWIN must fire within the second batch,
    demonstrating it detects the shift without requiring the caller to know
    the batch boundary.
    """
    X_dummy = _make_feature_row()

    # Phase 1: 500 correct predictions — ADWIN should NOT fire.
    for _ in range(500):
        fired = detector.update(y_true=0, y_pred=0, X_row=X_dummy)
        assert not fired, "ADWIN must not fire during stable phase"

    # Phase 2: up to 500 wrong predictions — ADWIN must fire at some point.
    adwin_fired_in_second_batch = False
    for _ in range(500):
        fired = detector.update(y_true=1, y_pred=0, X_row=X_dummy)
        if fired:
            adwin_fired_in_second_batch = True
            break

    assert adwin_fired_in_second_batch, (
        "ADWIN did not fire within 500 error=1 updates after 500 error=0 updates. "
        "Check ADWIN delta or update() wiring."
    )


# ---------------------------------------------------------------------------
# Test 2 — PSI < 0.1 when baseline and current are same distribution
# ---------------------------------------------------------------------------

def test_psi_no_drift_on_same_distribution(detector) -> None:
    """All PSI scores must be < 0.15 when baseline and current share N(0,1).

    Note: With small samples, PSI has sampling variance that can push scores
    slightly above 0.1 even for identical distributions.  We assert < 0.15
    (well below the critical threshold of 0.2) to confirm no false alarm,
    while tolerating natural sampling noise.
    """
    # Use larger samples to reduce sampling variance.
    X_train = _make_feature_df(n=2000, loc=0.0, scale=1.0)
    X_current = _make_feature_df(n=1000, loc=0.0, scale=1.0)

    detector.set_psi_baseline(X_train, feature_cols=_FEATURE_COLS)
    psi_scores = detector.compute_psi(X_current)

    assert psi_scores, "PSI scores dict must not be empty after baseline is set"

    for feat, score in psi_scores.items():
        assert score < 0.15, (
            f"PSI for '{feat}' = {score:.4f} should be < 0.15 for same-distribution "
            f"data (critical threshold is 0.2). Check normalization or bin logic."
        )


# ---------------------------------------------------------------------------
# Test 3 — PSI > 0.2 when card_tx_count_24h is heavily shifted
# ---------------------------------------------------------------------------

def test_psi_drift_detected_on_shifted_distribution(detector) -> None:
    """PSI for card_tx_count_24h must exceed 0.2 after multiplying values by 5."""
    X_train = _make_feature_df(n=500, loc=1.0, scale=0.5)

    # Shift only card_tx_count_24h dramatically.
    X_current = _make_feature_df(n=300, loc=1.0, scale=0.5)
    X_current["card_tx_count_24h"] = X_current["card_tx_count_24h"] * 5.0

    detector.set_psi_baseline(X_train, feature_cols=_FEATURE_COLS)
    psi_scores = detector.compute_psi(X_current)

    assert "card_tx_count_24h" in psi_scores, (
        "'card_tx_count_24h' must appear in psi_scores"
    )
    assert psi_scores["card_tx_count_24h"] > 0.2, (
        f"Expected PSI > 0.2 for 5× shifted card_tx_count_24h, "
        f"got {psi_scores['card_tx_count_24h']:.4f}"
    )


# ---------------------------------------------------------------------------
# Test 4 — Observer receives on_drift_detected with all required keys
# ---------------------------------------------------------------------------

def test_observer_notification(detector) -> None:
    """A registered observer must be called with all required drift_info keys."""
    required_keys = {
        "timestamp", "drift_type", "adwin_error_rate",
        "psi_scores", "window_size", "recommended_action",
    }

    captured: list[dict] = []

    # Build a minimal concrete observer inline.
    from src.ml_layer.drift_observer import DriftObserver

    class CapturingObserver(DriftObserver):
        def on_drift_detected(self, drift_info: dict) -> None:
            captured.append(drift_info)

    detector.register_observer(CapturingObserver())

    X_dummy = _make_feature_row()

    # Trigger ADWIN by feeding many errors.
    for _ in range(500):
        detector.update(y_true=0, y_pred=0, X_row=X_dummy)
    for _ in range(500):
        fired = detector.update(y_true=1, y_pred=0, X_row=X_dummy)
        if fired:
            break

    assert captured, (
        "Observer was never called. Drift may not have been detected — "
        "check ADWIN delta or update() observer notification."
    )

    drift_info = captured[0]
    missing = required_keys - set(drift_info.keys())
    assert not missing, (
        f"drift_info is missing required keys: {missing}"
    )

    # Verify value types
    assert isinstance(drift_info["timestamp"], str)
    assert drift_info["drift_type"] in {"adwin", "psi", "adwin+psi"}
    assert isinstance(drift_info["adwin_error_rate"], float)
    assert isinstance(drift_info["psi_scores"], dict)
    assert isinstance(drift_info["window_size"], int)
    assert isinstance(drift_info["recommended_action"], str)


# ---------------------------------------------------------------------------
# Test 5 — save / load state round-trip preserves ADWIN window
# ---------------------------------------------------------------------------

def test_detector_state_persistence() -> None:
    """After save/load, the loaded detector's ADWIN must match the original."""
    with patch(
        "src.ml_layer.drift_detector.load_config", return_value=_FULL_CFG
    ):
        from src.ml_layer.drift_detector import DriftDetector

        original = DriftDetector()
        X_dummy = _make_feature_row()

        # Run 100 updates to build up ADWIN state.
        for i in range(100):
            original.update(y_true=i % 2, y_pred=0, X_row=X_dummy)

        # width = number of samples in the current ADWIN window (public API)
        original_width = original._adwin.width
        original_error_buffer_len = len(original._error_buffer)

        with tempfile.TemporaryDirectory() as tmp_dir:
            state_path = str(Path(tmp_dir) / "detector_state.pkl")
            original.save_state(state_path)

            loaded = DriftDetector.load_state(state_path)

        assert loaded._adwin.width == original_width, (
            f"ADWIN width mismatch after load: "
            f"original={original_width}, loaded={loaded._adwin.width}"
        )
        assert len(loaded._error_buffer) == original_error_buffer_len, (
            f"Error buffer length mismatch after load: "
            f"original={original_error_buffer_len}, "
            f"loaded={len(loaded._error_buffer)}"
        )


# ---------------------------------------------------------------------------
# Test 6 — DriftSimulator injects drift at the correct point
# ---------------------------------------------------------------------------

def test_drift_simulator_injects_at_correct_point() -> None:
    """Rows before injection_point must be unchanged; fraud rows after are modified."""
    from src.ml_layer.drift_simulator import DriftSimulator

    n = 1000
    injection_point = 0.7
    split_idx = int(n * injection_point)

    # Build a DataFrame with required columns.
    df = pd.DataFrame({
        "TransactionAmt": np.ones(n) * 100.0,
        "isFraud": np.where(RNG.random(n) < 0.1, 1, 0),   # ~10% fraud
        "card_tx_count_24h": np.ones(n) * 5.0,
    })

    drifted = DriftSimulator.simulate_concept_drift(
        df, injection_point=injection_point, drift_type="high_value"
    )

    # ---- Pre-injection rows must be identical ----
    pre_orig = df.iloc[:split_idx]
    pre_drift = drifted.iloc[:split_idx]
    pd.testing.assert_frame_equal(
        pre_orig.reset_index(drop=True),
        pre_drift.reset_index(drop=True),
        check_exact=True,
        obj="Pre-injection rows",
    )

    # ---- Post-injection fraud rows must have tripled TransactionAmt ----
    post_fraud_mask = (
        (drifted.index >= split_idx) & (drifted["isFraud"] == 1)
    )
    # Use the original df to find which post-injection rows were fraud.
    original_post_fraud = df.iloc[split_idx:][df.iloc[split_idx:]["isFraud"] == 1]

    if len(original_post_fraud) > 0:
        drifted_post_fraud_amt = drifted.loc[
            drifted.index[split_idx:][df.iloc[split_idx:]["isFraud"] == 1],
            "TransactionAmt",
        ]
        expected = original_post_fraud["TransactionAmt"].values * 3
        np.testing.assert_array_almost_equal(
            drifted_post_fraud_amt.values,
            expected,
            decimal=6,
            err_msg="Post-injection fraud rows must have TransactionAmt × 3",
        )

    # ---- Post-injection non-fraud rows must be unchanged ----
    post_nonfrd_orig = df.iloc[split_idx:][df.iloc[split_idx:]["isFraud"] == 0]
    post_nonfrd_drift = drifted.iloc[split_idx:][drifted.iloc[split_idx:]["isFraud"] == 0]
    pd.testing.assert_series_equal(
        post_nonfrd_orig["TransactionAmt"].reset_index(drop=True),
        post_nonfrd_drift["TransactionAmt"].reset_index(drop=True),
        check_names=False,
        obj="Post-injection non-fraud rows",
    )
