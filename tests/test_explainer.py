"""
tests/test_explainer.py
------------------------
Unit tests for the CIPHER SHAP explainability layer.

All tests train a tiny LightGBM model (n_estimators=20) on synthetic data
so they run fast in CI without real transaction data.  SHAPExplainer is
constructed directly — no disk model I/O for the core tests (a tempfile
is used for the save/load test).

Tests
-----
* test_shap_values_sum_to_prediction      — efficiency axiom in log-odds space
* test_top_features_sorted_by_abs_shap    — descending |SHAP| order
* test_direction_labels_correct           — increases/decreases_risk labels
* test_cache_hit_faster_than_miss         — second call ≥ 10× faster
* test_cache_invalidation                 — recomputes after invalidate_cache()
* test_plain_english_mentions_top_features— summary mentions top-3 feature names
* test_global_importance_returns_correct_columns — DataFrame columns check
"""

from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
import pytest
from lightgbm import LGBMClassifier

# Ensure project root is on path (mirrors conftest.py).
sys.path.insert(0, str(Path(__file__).parent.parent))

# ---------------------------------------------------------------------------
# Synthetic dataset + trained LightGBM fixture
# ---------------------------------------------------------------------------

_RNG = np.random.default_rng(seed=7)
_N_TRAIN = 400
_N_TEST = 100
_N_FEAT = 15

_FEATURE_NAMES = [f"feat_{i:02d}" for i in range(_N_FEAT)]

_MINIMAL_CFG = {
    "explainer": {
        "model_path": "models/lgbm_model.pkl",
        "lgbm_model_path": "models/lgbm_model.pkl",
        "plots_dir": "plots/shap_test",
        "max_cache_size": 100,
        "threadpool_workers": 2,
        "top_features_count": 10,
        "interaction_top_n": 5,
        "global_sample_size": 50,
    }
}


def _make_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (X_train, X_test, y_train, y_test) as numpy arrays."""
    X_legit = _RNG.normal(0.0, 1.0, (_N_TRAIN, _N_FEAT))
    X_fraud = _RNG.normal(3.0, 1.0, (int(_N_TRAIN * 0.1), _N_FEAT))
    X_tr = np.vstack([X_legit, X_fraud])
    y_tr = np.concatenate(
        [np.zeros(len(X_legit), dtype=int), np.ones(len(X_fraud), dtype=int)]
    )
    X_te = np.vstack([
        _RNG.normal(0.0, 1.0, (int(_N_TEST * 0.9), _N_FEAT)),
        _RNG.normal(3.0, 1.0, (int(_N_TEST * 0.1), _N_FEAT)),
    ])
    y_te = np.concatenate([
        np.zeros(int(_N_TEST * 0.9), dtype=int),
        np.ones(int(_N_TEST * 0.1), dtype=int),
    ])
    return X_tr, X_te, y_tr, y_te


@pytest.fixture(scope="module")
def trained_model_and_data():
    """Train a small LightGBM once for all tests in this module."""
    X_tr, X_te, y_tr, y_te = _make_data()
    model = LGBMClassifier(
        n_estimators=20,
        num_leaves=8,
        learning_rate=0.1,
        verbose=-1,
        random_state=0,
    )
    model.fit(
        pd.DataFrame(X_tr, columns=_FEATURE_NAMES),
        y_tr,
    )
    return model, X_tr, X_te, y_tr, y_te


@pytest.fixture
def explainer_with_tmpdir(trained_model_and_data, tmp_path):
    """Provide a SHAPExplainer backed by a temp-dir model file."""
    model, *_ = trained_model_and_data

    model_path = str(tmp_path / "lgbm_test.pkl")
    joblib.dump(model, model_path)

    # Override plots_dir to a temp path so we don't pollute the repo.
    cfg = {
        "explainer": {
            **_MINIMAL_CFG["explainer"],
            "plots_dir": str(tmp_path / "shap_plots"),
        }
    }

    import src.ml_layer.explainer
    with patch("src.ml_layer.explainer.load_config", return_value=cfg):
        from src.ml_layer.explainer import SHAPExplainer
        exp = SHAPExplainer(
            model_path=model_path,
            feature_names=_FEATURE_NAMES,
        )
    return exp


def _make_X_row(explainer_with_tmpdir) -> pd.DataFrame:
    """Return a single fraud-like row as a DataFrame."""
    # Use a fraud-like sample (high values).
    row = _RNG.normal(3.0, 0.5, _N_FEAT)
    return pd.DataFrame([row], columns=_FEATURE_NAMES)


# ---------------------------------------------------------------------------
# Test 1 — Efficiency axiom: sum(shap) + base_value ≈ raw_score
# ---------------------------------------------------------------------------

def test_shap_values_sum_to_prediction(
    explainer_with_tmpdir, trained_model_and_data
) -> None:
    """sum(shap_values) + base_value must ≈ raw log-odds score (within 1e-5).

    The efficiency (completeness) axiom of SHAP states that the sum of all
    SHAP values plus the expected value equals the model's raw output.
    For LightGBM binary classification with model_output='raw', the raw
    output is in log-odds space.  We verify the axiom against the booster's
    direct prediction, NOT against predict_proba (which is in probability
    space after sigmoid).
    """
    from src.ml_layer.explainer import SHAPExplainer

    exp = explainer_with_tmpdir
    X_row = _make_X_row(exp)
    result = exp.explain("TX_axiom", X_row)

    # Raw score from the booster in log-odds space.
    raw_score = float(
        exp._lgbm_model.booster_.predict(X_row.values.astype(float), raw_score=True)[0]
    )

    shap_sum_plus_base = float(result.shap_values.sum() + result.base_value)

    assert abs(shap_sum_plus_base - raw_score) < 1e-5, (
        f"Efficiency axiom violated: "
        f"sum(shap)+base={shap_sum_plus_base:.8f}, "
        f"raw_score={raw_score:.8f}, "
        f"diff={abs(shap_sum_plus_base - raw_score):.2e}"
    )


# ---------------------------------------------------------------------------
# Test 2 — Top features must be sorted by |SHAP| descending
# ---------------------------------------------------------------------------

def test_top_features_sorted_by_abs_shap(explainer_with_tmpdir) -> None:
    """top_features list must be sorted by |shap_value| descending."""
    exp = explainer_with_tmpdir
    X_row = _make_X_row(exp)
    result = exp.explain("TX_sorted", X_row)

    abs_vals = [abs(f["shap_value"]) for f in result.top_features]
    for i in range(len(abs_vals) - 1):
        assert abs_vals[i] >= abs_vals[i + 1], (
            f"top_features[{i}] |shap|={abs_vals[i]:.4f} < "
            f"top_features[{i+1}] |shap|={abs_vals[i+1]:.4f}. "
            "List is not sorted descending by absolute SHAP value."
        )


# ---------------------------------------------------------------------------
# Test 3 — Direction labels: positive SHAP → increases_risk
# ---------------------------------------------------------------------------

def test_direction_labels_correct(explainer_with_tmpdir) -> None:
    """Features with SHAP > 0 must have direction='increases_risk', else 'decreases_risk'."""
    exp = explainer_with_tmpdir
    X_row = _make_X_row(exp)
    result = exp.explain("TX_direction", X_row)

    for feat in result.top_features:
        sv = feat["shap_value"]
        direction = feat["direction"]
        if sv > 0:
            assert direction == "increases_risk", (
                f"Feature '{feat['feature_name']}' has positive SHAP ({sv:.4f}) "
                f"but direction='{direction}'"
            )
        elif sv < 0:
            assert direction == "decreases_risk", (
                f"Feature '{feat['feature_name']}' has negative SHAP ({sv:.4f}) "
                f"but direction='{direction}'"
            )
        # sv == 0 is a degenerate case; no assertion needed.


# ---------------------------------------------------------------------------
# Test 4 — Cache hit must be significantly faster than cache miss
# ---------------------------------------------------------------------------

def test_cache_hit_faster_than_miss(explainer_with_tmpdir) -> None:
    """Second explain() call (cache hit) must be ≥ 10× faster than the first."""
    exp = explainer_with_tmpdir
    X_row = _make_X_row(exp)
    tx_id = "TX_cache_speed"

    t0 = time.perf_counter()
    exp.explain(tx_id, X_row)
    miss_time = time.perf_counter() - t0

    t1 = time.perf_counter()
    exp.explain(tx_id, X_row)   # cache hit
    hit_time = time.perf_counter() - t1

    speedup = miss_time / max(hit_time, 1e-9)
    assert speedup >= 10.0, (
        f"Cache hit speedup = {speedup:.1f}× (expected ≥ 10×). "
        f"miss={miss_time * 1000:.1f}ms, hit={hit_time * 1000:.3f}ms."
    )


# ---------------------------------------------------------------------------
# Test 5 — Cache invalidation forces recomputation
# ---------------------------------------------------------------------------

def test_cache_invalidation(explainer_with_tmpdir) -> None:
    """After invalidate_cache(), explain() must recompute (non-zero computation time)."""
    exp = explainer_with_tmpdir
    X_row = _make_X_row(exp)
    tx_id = "TX_invalidate"

    # First call — prime the cache.
    exp.explain(tx_id, X_row)
    assert tx_id in exp._cache

    # Invalidate — cache should be empty.
    exp.invalidate_cache()
    assert tx_id not in exp._cache, (
        "Cache still contains the entry after invalidate_cache()"
    )

    # Second call — must be a fresh computation, not a cache hit.
    result = exp.explain(tx_id, X_row)
    assert result.computation_time_ms > 0.001, (
        f"computation_time_ms={result.computation_time_ms:.4f}ms after cache "
        "invalidation — expected a real computation (> 0.001ms)"
    )


# ---------------------------------------------------------------------------
# Test 6 — Plain-English summary mentions the top-3 feature names
# ---------------------------------------------------------------------------

def test_plain_english_mentions_top_features(explainer_with_tmpdir) -> None:
    """plain_english_summary must contain the names of the top 3 features."""
    exp = explainer_with_tmpdir
    X_row = _make_X_row(exp)
    result = exp.explain("TX_summary_text", X_row)

    summary = result.plain_english_summary
    top3_names = [f["feature_name"] for f in result.top_features[:3]]

    for name in top3_names:
        assert name in summary, (
            f"Feature '{name}' (one of the top-3) is not mentioned in "
            f"plain_english_summary:\n  {summary}"
        )


# ---------------------------------------------------------------------------
# Test 7 — get_global_importance() returns DataFrame with correct columns
# ---------------------------------------------------------------------------

def test_global_importance_returns_correct_columns(
    explainer_with_tmpdir, trained_model_and_data
) -> None:
    """get_global_importance() must return a DataFrame with the two required columns."""
    exp = explainer_with_tmpdir
    _, X_te, _, _ = trained_model_and_data[1], trained_model_and_data[1], \
                     trained_model_and_data[2], trained_model_and_data[3]

    # Build a small test-set sample.
    _, X_te_raw, _, _ = _make_data()
    X_sample = pd.DataFrame(X_te_raw[:30], columns=_FEATURE_NAMES)

    df = exp.get_global_importance(X_sample, n_features=10)

    assert isinstance(df, pd.DataFrame), (
        f"get_global_importance() must return a DataFrame, got {type(df)}"
    )
    required_cols = {"feature_name", "mean_abs_shap"}
    missing = required_cols - set(df.columns)
    assert not missing, (
        f"DataFrame is missing required columns: {missing}. "
        f"Got: {list(df.columns)}"
    )
    assert len(df) <= 10, (
        f"Expected at most 10 rows, got {len(df)}"
    )
    # Verify sorted descending by mean_abs_shap
    vals = df["mean_abs_shap"].tolist()
    for i in range(len(vals) - 1):
        assert vals[i] >= vals[i + 1], (
            f"DataFrame is not sorted descending at index {i}: "
            f"{vals[i]:.4f} < {vals[i+1]:.4f}"
        )
    # All mean_abs_shap values must be non-negative
    assert all(v >= 0 for v in vals), "mean_abs_shap values must be non-negative"
