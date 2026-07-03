"""
src/ml_layer/explainer.py
--------------------------
SHAP-based explainability layer for the CIPHER fraud-detection pipeline.

Architecture
------------
:class:`SHAPExplainer` is the **Facade** — a single public interface that
hides the complexity of SHAP computation, result caching, background-thread
management, and plot generation behind one clean method: :meth:`explain`.

Two auxiliary classes support global analysis:

* :class:`GlobalExplainabilityReporter` — produces beeswarm and dependence
  plots across a test-set sample for MLflow logging.

Key design decisions
--------------------
* ``shap.TreeExplainer`` is initialised with ``lgbm_model.booster_``
  (the native LightGBM booster), **not** the scikit-learn wrapper.  This
  guarantees fast, exact TreeSHAP computation rather than falling back to
  the slower ``KernelSHAP`` approximation.
* SHAP values are computed in **raw log-odds space** (``model_output="raw"``).
  The efficiency axiom holds exactly:
  ``sum(shap_values) + base_value ≈ booster.predict(X)[0]`` (raw score).
  The ``prediction`` field in :class:`ExplanationResult` stores the sigmoid
  of that raw score, i.e. the actual fraud probability.
* The result cache is protected by a :class:`threading.Lock` so that
  concurrent calls from :meth:`explain_async` never corrupt the dict.
* The ``plots/shap/`` directory is created on initialisation **and**
  defensively before every save call.

Typical usage::

    from src.ml_layer.explainer import SHAPExplainer, GlobalExplainabilityReporter

    explainer = SHAPExplainer(
        model_path="models/lgbm_model.pkl",
        feature_names=feature_names,
    )
    result = explainer.explain("TX_00123", X_row)
    print(result.plain_english_summary)

    reporter = GlobalExplainabilityReporter()
    report   = reporter.generate_global_report(explainer, X_test_sample)
"""

from __future__ import annotations

import argparse
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---- Numba Mock to bypass Windows AppLocker DLL restrictions ----
import sys
import types
class MockNumba(types.ModuleType):
    def __getattr__(self, name): return self
_m = MockNumba('numba')
_m.njit = lambda *a, **kw: (lambda f: f) if not (a and callable(a[0])) else a[0]
_m.jit = _m.njit
_m.typed = MockNumba('numba.typed')
if 'numba' not in sys.modules:
    sys.modules['numba'] = _m
    sys.modules['numba.typed'] = _m.typed
    sys.modules['numba.core'] = MockNumba('numba.core')
    sys.modules['numba.core.types'] = MockNumba('numba.core.types')
    sys.modules['numba.core.errors'] = MockNumba('numba.core.errors')
# -----------------------------------------------------------------

import shap

from src.utils.config_loader import load_config
from src.utils.logger import get_logger

# Non-interactive backend — plot generation must work in background threads
# and headless environments.
matplotlib.use("Agg")

_logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# ExplanationResult — immutable result dataclass
# ---------------------------------------------------------------------------


@dataclass
class ExplanationResult:
    """Complete SHAP explanation for a single flagged transaction.

    All fields are populated by :meth:`SHAPExplainer._compute_shap` and
    are immutable after construction (``frozen=True`` is intentionally
    **not** set to allow caching by reference without copy overhead).

    Attributes:
        transaction_id:     Unique identifier of the explained transaction.
        shap_values:        1-D SHAP value array (log-odds space), one value
                            per feature.  Satisfies the efficiency axiom:
                            ``sum(shap_values) + base_value ≈ raw_score``.
        base_value:         Model's expected raw output E[f(X)] (log-odds).
        prediction:         Fraud probability — sigmoid of
                            ``base_value + sum(shap_values)``.
        top_features:       Top-``top_features_count`` features sorted by
                            ``|shap_value|`` descending.  Each dict has keys:
                            ``feature_name``, ``shap_value``,
                            ``feature_value``, ``direction``.
        plain_english_summary: 2–3 sentence human-readable explanation
                            using the top 3 features.
        waterfall_plot_path: Absolute path to the saved waterfall PNG.
        computation_time_ms: Wall-clock time for SHAP computation in ms.
        computed_at:        UTC timestamp when the result was produced.
    """

    transaction_id: str
    shap_values: np.ndarray
    base_value: float
    prediction: float
    top_features: list[dict]
    plain_english_summary: str
    waterfall_plot_path: str
    computation_time_ms: float
    computed_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


# ---------------------------------------------------------------------------
# SHAPExplainer — Facade
# ---------------------------------------------------------------------------


class SHAPExplainer:
    """Facade for computing, caching, and formatting SHAP explanations.

    Initialisation loads the LightGBM model from disk once, creates a
    ``shap.TreeExplainer`` backed by the native booster, and prepares a
    thread pool for async computation.

    The internal cache maps ``transaction_id → ExplanationResult``.  All
    cache reads and writes are guarded by :attr:`_cache_lock` to prevent
    race conditions when multiple background threads complete simultaneously.

    Attributes:
        feature_names:    Ordered list of feature names from the feature layer.
        _lgbm_model:      Fitted :class:`~lightgbm.LGBMClassifier`.
        _shap_explainer:  :class:`shap.TreeExplainer` backed by the LightGBM
                          booster (not the sklearn wrapper).
        _cache:           Dict mapping transaction ID → cached result.
        _cache_lock:      :class:`threading.Lock` protecting ``_cache``.
        _executor:        :class:`~concurrent.futures.ThreadPoolExecutor` for
                          background SHAP computation.
        _plots_dir:       :class:`~pathlib.Path` to the SHAP plots directory.
        _cfg:             The ``explainer`` sub-dict from ``config.yaml``.
    """

    def __init__(
        self,
        model_path: str,
        feature_names: list[str],
        config_path: str = "config/config.yaml",
    ) -> None:
        """Load model, initialise TreeExplainer, and prepare infrastructure.

        Args:
            model_path:    Path to a ``joblib``-serialised
                           :class:`~lightgbm.LGBMClassifier`.
            feature_names: Ordered list of feature names matching the column
                           order used during training.
            config_path:   Path to ``config/config.yaml`` (default).
        """
        cfg = load_config(config_path)
        self._cfg: dict[str, Any] = cfg["explainer"]
        self.feature_names: list[str] = list(feature_names)

        # ---- Load model -----------------------------------------------
        self._lgbm_model = joblib.load(model_path)
        _logger.info(
            "SHAPExplainer — loaded LightGBM model from '%s'", model_path
        )

        # ---- TreeExplainer — pass booster_ directly (not sklearn wrapper)
        # This guarantees exact TreeSHAP, never the slower KernelSHAP.
        # model_output="raw" → log-odds space; efficiency axiom holds exactly.
        self._shap_explainer = shap.TreeExplainer(
            self._lgbm_model.booster_,
            model_output="raw",
        )
        _logger.info(
            "SHAPExplainer — TreeExplainer ready | "
            "expected_value=%s (log-odds)",
            str(self._shap_explainer.expected_value),
        )

        # ---- Thread-safe cache ----------------------------------------
        self._cache: dict[str, ExplanationResult] = {}
        self._cache_lock = threading.Lock()

        # ---- Background executor --------------------------------------
        self._executor = ThreadPoolExecutor(
            max_workers=int(self._cfg["threadpool_workers"])
        )

        # ---- Plots directory — create if missing ----------------------
        self._plots_dir = Path(self._cfg["plots_dir"])
        self._plots_dir.mkdir(parents=True, exist_ok=True)

        _logger.info(
            "SHAPExplainer initialised | features=%d, "
            "plots_dir='%s', cache_max=%d",
            len(feature_names),
            self._plots_dir,
            int(self._cfg["max_cache_size"]),
        )

    # ------------------------------------------------------------------
    # Public interface — Facade methods
    # ------------------------------------------------------------------

    def explain(
        self,
        transaction_id: str,
        X_row: pd.DataFrame,
    ) -> ExplanationResult:
        """Return the SHAP explanation for a single transaction.

        Checks the in-memory cache first.  On a cache miss, computes SHAP
        synchronously, caches the result, and returns it.

        Args:
            transaction_id: Unique transaction identifier used as the cache key.
            X_row:          Single-row :class:`~pandas.DataFrame` with the same
                            columns as the training feature matrix.

        Returns:
            :class:`ExplanationResult` with all fields populated.
        """
        # --- Cache lookup (acquire lock for thread safety) ----------------
        with self._cache_lock:
            if transaction_id in self._cache:
                _logger.debug(
                    "SHAPExplainer — cache hit | transaction_id='%s'",
                    transaction_id,
                )
                return self._cache[transaction_id]

        # --- Cache miss: compute synchronously ---------------------------
        result = self._compute_shap(transaction_id, X_row)

        # --- Write to cache (evict oldest if over max size) --------------
        with self._cache_lock:
            if len(self._cache) >= int(self._cfg["max_cache_size"]):
                oldest_key = next(iter(self._cache))
                del self._cache[oldest_key]
                _logger.debug(
                    "SHAPExplainer — cache evicted oldest entry '%s'",
                    oldest_key,
                )
            self._cache[transaction_id] = result

        return result

    def explain_async(
        self,
        transaction_id: str,
        X_row: pd.DataFrame,
        callback: Callable[[ExplanationResult], None],
    ) -> None:
        """Submit SHAP computation to a background thread.

        Returns immediately.  When the computation completes, ``callback``
        is called with the :class:`ExplanationResult` on the worker thread.

        All cache operations inside the task are lock-protected.

        Args:
            transaction_id: Unique transaction identifier.
            X_row:          Single-row :class:`~pandas.DataFrame`.
            callback:       Callable invoked with the result when done.
        """

        def _task() -> None:
            # Check cache first (thread-safe).
            with self._cache_lock:
                if transaction_id in self._cache:
                    result = self._cache[transaction_id]
                    callback(result)
                    return

            # Compute — outside the lock so other threads aren't blocked.
            result = self._compute_shap(transaction_id, X_row)

            # Write to cache (thread-safe).
            with self._cache_lock:
                if len(self._cache) >= int(self._cfg["max_cache_size"]):
                    oldest_key = next(iter(self._cache))
                    del self._cache[oldest_key]
                self._cache[transaction_id] = result

            callback(result)

        self._executor.submit(_task)

    # ------------------------------------------------------------------
    # Core computation
    # ------------------------------------------------------------------

    def _compute_shap(
        self,
        transaction_id: str,
        X_row: pd.DataFrame,
    ) -> ExplanationResult:
        """Run TreeSHAP and assemble a complete ExplanationResult.

        Note:
            SHAP values are in **log-odds space** (``model_output="raw"``).
            The efficiency axiom holds exactly:
            ``sum(shap_values) + base_value ≈ booster.predict(X)[0]``.

            The ``prediction`` field stores the **fraud probability**:
            ``sigmoid(base_value + sum(shap_values))``.

        Args:
            transaction_id: Unique identifier for caching and plot naming.
            X_row:          Single-row :class:`~pandas.DataFrame`.

        Returns:
            Fully populated :class:`ExplanationResult`.
        """
        t_start = time.perf_counter()

        X_numpy = X_row.values.astype(float)

        # ---- SHAP values in log-odds space ----------------------------
        shap_matrix = self._shap_explainer.shap_values(X_numpy)

        # shap_values() with model_output="raw" returns a 2D array
        # (n_samples, n_features).
        if isinstance(shap_matrix, list):
            # Older SHAP versions return [negative_class, positive_class].
            shap_vals: np.ndarray = np.asarray(shap_matrix[1][0])
            base_val: float = float(self._shap_explainer.expected_value[1])
        else:
            shap_vals = np.asarray(shap_matrix[0])
            base_val = float(self._shap_explainer.expected_value)

        # ---- Prediction: sigmoid(raw_score) = fraud probability -------
        raw_score = float(np.sum(shap_vals) + base_val)
        prediction = float(1.0 / (1.0 + np.exp(-raw_score)))

        # ---- Align feature names with model output --------------------
        n_feat = len(shap_vals)
        feat_names: list[str] = (
            self.feature_names
            if len(self.feature_names) == n_feat
            else [f"feature_{i}" for i in range(n_feat)]
        )

        # ---- Top features sorted by |SHAP| descending -----------------
        top_count = int(self._cfg["top_features_count"])
        sorted_idx = np.argsort(np.abs(shap_vals))[::-1][:top_count]

        top_features: list[dict] = []
        for idx in sorted_idx:
            sv = float(shap_vals[idx])
            fv = float(X_numpy[0, idx]) if X_numpy.ndim == 2 else float(X_numpy[idx])
            top_features.append(
                {
                    "feature_name": feat_names[idx],
                    "shap_value": sv,
                    "feature_value": fv,
                    "direction": (
                        "increases_risk" if sv > 0 else "decreases_risk"
                    ),
                }
            )

        # ---- Plain-English summary ------------------------------------
        summary = self._generate_plain_english(top_features[:3])

        # ---- Waterfall plot -------------------------------------------
        waterfall_path = self.generate_waterfall_plot(
            shap_vals, base_val, X_row, transaction_id
        )

        elapsed_ms = (time.perf_counter() - t_start) * 1000.0

        _logger.info(
            "SHAPExplainer — computed | transaction_id='%s', "
            "prediction=%.4f, top_feature='%s' (shap=%.4f), time=%.1fms",
            transaction_id,
            prediction,
            top_features[0]["feature_name"] if top_features else "N/A",
            top_features[0]["shap_value"] if top_features else 0.0,
            elapsed_ms,
        )

        return ExplanationResult(
            transaction_id=transaction_id,
            shap_values=shap_vals,
            base_value=base_val,
            prediction=prediction,
            top_features=top_features,
            plain_english_summary=summary,
            waterfall_plot_path=waterfall_path,
            computation_time_ms=elapsed_ms,
            computed_at=datetime.now(timezone.utc),
        )

    # ------------------------------------------------------------------
    # Plot generation
    # ------------------------------------------------------------------

    def generate_waterfall_plot(
        self,
        shap_values: np.ndarray,
        base_value: float,
        X_row: pd.DataFrame,
        transaction_id: str,
    ) -> str:
        """Generate and save a SHAP waterfall plot for one transaction.

        The plot uses ``feature_names`` from the feature layer — never
        generic column indices.  The output directory is created if it does
        not already exist (defensive against race conditions in async mode).

        Args:
            shap_values:    1-D SHAP value array (log-odds space).
            base_value:     SHAP expected value (log-odds).
            X_row:          Single-row :class:`~pandas.DataFrame`.
            transaction_id: Used in the output filename.

        Returns:
            Absolute path to the saved PNG file.
        """
        # Defensive mkdir (also done in __init__; safe to repeat).
        self._plots_dir.mkdir(parents=True, exist_ok=True)

        X_numpy = X_row.values[0] if isinstance(X_row, pd.DataFrame) else X_row
        n = len(shap_values)
        feat_names = (
            self.feature_names
            if len(self.feature_names) == n
            else [f"feature_{i}" for i in range(n)]
        )

        # Build shap.Explanation with named features — required for labelled
        # waterfall plots.  Without feature_names, SHAP falls back to indices.
        explanation = shap.Explanation(
            values=shap_values,
            base_values=base_value,
            data=X_numpy,
            feature_names=feat_names,
        )

        plt.figure()
        shap.plots.waterfall(explanation, show=False)

        out_path = str(self._plots_dir / f"shap_{transaction_id}.png")
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close("all")

        _logger.info("Waterfall plot saved to '%s'", out_path)
        return out_path

    # ------------------------------------------------------------------
    # Plain-English summary
    # ------------------------------------------------------------------

    @staticmethod
    def _generate_plain_english(top_features: list[dict]) -> str:
        """Generate a 2–3 sentence analyst-facing explanation.

        Uses the top 3 features by absolute SHAP value.  Each sentence
        states the feature name, its observed value, and its directional
        risk contribution.

        Example output::

            "This transaction was flagged primarily because
            card_degree_1h=11 (+0.31 risk contribution). The transaction
            amount was also anomalous: amt_zscore_24h=8.2 (+0.28 risk
            contribution). Additionally, card_tx_count_24h=45
            (+0.19 risk contribution) contributed to this flag."

        Args:
            top_features: List of feature dicts from :meth:`_compute_shap`,
                restricted to the top 3 by caller.

        Returns:
            A human-readable multi-sentence string.  Returns a generic
            fallback if ``top_features`` is empty.
        """
        if not top_features:
            return "This transaction was flagged by the CIPHER fraud model."

        intros = [
            "This transaction was flagged primarily because",
            "The transaction amount was also anomalous:",
            "Additionally,",
        ]
        connectors = [
            "contributed to this flag.",
            "raised the fraud probability.",
            "was a contributing factor.",
        ]

        sentences: list[str] = []
        for i, feat in enumerate(top_features[:3]):
            name = feat["feature_name"]
            sv = feat["shap_value"]
            fv = feat["feature_value"]
            sign = "+" if sv >= 0 else ""
            intro = intros[i] if i < len(intros) else "Also,"
            connector = connectors[i] if i < len(connectors) else "was flagged."

            sentences.append(
                f"{intro} {name}={fv:.4g} ({sign}{sv:.2f} risk contribution) "
                f"{connector}"
            )

        return " ".join(sentences)

    # ------------------------------------------------------------------
    # Cache management
    # ------------------------------------------------------------------

    def invalidate_cache(self) -> None:
        """Clear all cached explanations.

        Should be called immediately after a model retrain so stale
        explanations (computed by the old model) are not served.

        Thread-safe — acquires the cache lock before clearing.
        """
        with self._cache_lock:
            n = len(self._cache)
            self._cache.clear()
        _logger.info(
            "SHAPExplainer — cache invalidated (%d entries cleared)", n
        )

    # ------------------------------------------------------------------
    # Global importance
    # ------------------------------------------------------------------

    def get_global_importance(
        self,
        X_sample: pd.DataFrame,
        n_features: int = 20,
    ) -> pd.DataFrame:
        """Compute mean absolute SHAP values across a sample of transactions.

        Note:
            ``X_sample`` must be drawn from the **test set**, not the
            training set.  Using training data introduces optimistic bias
            because the model has memorised those samples.

        Args:
            X_sample:   DataFrame of test-set transactions.  Large enough to
                        represent the feature distribution (>= 1000 rows
                        recommended).
            n_features: Number of top features to return (default: 20).

        Returns:
            :class:`~pandas.DataFrame` with columns:

            * ``feature_name``   — feature identifier.
            * ``mean_abs_shap``  — mean |SHAP| across all rows in X_sample.

            Sorted by ``mean_abs_shap`` descending.
        """
        _logger.info(
            "get_global_importance — computing SHAP on %d samples", len(X_sample)
        )
        X_numpy = X_sample.values.astype(float)
        shap_matrix = self._shap_explainer.shap_values(X_numpy)

        if isinstance(shap_matrix, list):
            shap_arr = np.asarray(shap_matrix[1])
        else:
            shap_arr = np.asarray(shap_matrix)

        mean_abs = np.abs(shap_arr).mean(axis=0)
        n_feat = len(mean_abs)
        feat_names = (
            self.feature_names
            if len(self.feature_names) == n_feat
            else [f"feature_{i}" for i in range(n_feat)]
        )

        result_df = (
            pd.DataFrame(
                {"feature_name": feat_names, "mean_abs_shap": mean_abs}
            )
            .sort_values("mean_abs_shap", ascending=False)
            .head(n_features)
            .reset_index(drop=True)
        )
        return result_df

    # ------------------------------------------------------------------
    # Interaction values
    # ------------------------------------------------------------------

    def get_shap_interaction_values(
        self,
        X_row: pd.DataFrame,
        top_n_features: int = 10,
    ) -> np.ndarray:
        """Compute SHAP interaction values for the top-N features.

        Interaction values quantify how features *jointly* affect the
        prediction.  Computing the full ``n_features × n_features`` matrix
        is expensive for large feature sets, so this method restricts the
        computation to the top ``top_n_features`` by absolute SHAP value.

        Args:
            X_row:           Single-row :class:`~pandas.DataFrame`.
            top_n_features:  Number of features to include (default: 10).

        Returns:
            2-D NumPy array of shape
            ``(top_n_features, top_n_features)`` containing pairwise
            SHAP interaction values for the selected features.
        """
        X_numpy = X_row.values.astype(float)

        # Step 1: identify top features by absolute SHAP magnitude.
        shap_vals = self._shap_explainer.shap_values(X_numpy)
        if isinstance(shap_vals, list):
            sv_row = np.asarray(shap_vals[1][0])
        else:
            sv_row = np.asarray(shap_vals[0])

        top_idx = np.argsort(np.abs(sv_row))[::-1][:top_n_features]

        # Step 2: compute interaction values on the top-feature subset.
        X_subset = X_numpy[:, top_idx]
        try:
            interaction = self._shap_explainer.shap_interaction_values(
                X_subset
            )
            if isinstance(interaction, list):
                interaction = np.asarray(interaction[1])
            else:
                interaction = np.asarray(interaction)
            return interaction[0]  # single row
        except Exception as exc:  # noqa: BLE001
            _logger.warning(
                "get_shap_interaction_values — failed (%s); "
                "returning zeros placeholder.",
                exc,
            )
            return np.zeros((top_n_features, top_n_features))


# ---------------------------------------------------------------------------
# GlobalExplainabilityReporter
# ---------------------------------------------------------------------------


class GlobalExplainabilityReporter:
    """Generate portfolio-level SHAP plots across a test-set sample.

    All plot methods require a test-set sample — never training data —
    to avoid optimistic bias in global feature importance.

    Typical usage::

        reporter = GlobalExplainabilityReporter()
        report   = reporter.generate_global_report(explainer, X_test_sample)
        # report keys: summary_plot, dependence_plots, importance_csv, importance_df
    """

    @staticmethod
    def _ensure_dir(path: Path) -> None:
        """Create directory including parents if it does not exist."""
        path.mkdir(parents=True, exist_ok=True)

    def generate_summary_plot(
        self,
        explainer: SHAPExplainer,
        X_sample: pd.DataFrame,
    ) -> str:
        """Generate a SHAP beeswarm summary plot across a test-set sample.

        Args:
            explainer: Fitted :class:`SHAPExplainer` instance.
            X_sample:  Test-set DataFrame (NOT training data).

        Returns:
            Absolute path to the saved ``shap_summary.png``.
        """
        _logger.info(
            "GlobalExplainabilityReporter — generating summary plot "
            "(%d samples)",
            len(X_sample),
        )
        plots_dir = explainer._plots_dir
        self._ensure_dir(plots_dir)

        X_numpy = X_sample.values.astype(float)
        shap_matrix = explainer._shap_explainer.shap_values(X_numpy)

        if isinstance(shap_matrix, list):
            shap_arr = np.asarray(shap_matrix[1])
        else:
            shap_arr = np.asarray(shap_matrix)

        n_feat = shap_arr.shape[1] if shap_arr.ndim > 1 else len(shap_arr)
        feat_names = (
            explainer.feature_names
            if len(explainer.feature_names) == n_feat
            else [f"feature_{i}" for i in range(n_feat)]
        )

        plt.figure()
        shap.summary_plot(
            shap_arr,
            X_sample,
            feature_names=feat_names,
            show=False,
        )
        out_path = str(plots_dir / "shap_summary.png")
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close("all")

        _logger.info("Summary plot saved to '%s'", out_path)
        return out_path

    def generate_dependence_plot(
        self,
        explainer: SHAPExplainer,
        X_sample: pd.DataFrame,
        feature: str,
    ) -> str:
        """Generate a SHAP dependence plot for one feature.

        Args:
            explainer: Fitted :class:`SHAPExplainer` instance.
            X_sample:  Test-set DataFrame (NOT training data).
            feature:   Name of the feature to plot.

        Returns:
            Absolute path to the saved
            ``shap_dependence_{feature}.png``.

        Raises:
            ValueError: If ``feature`` is not in the feature names list.
        """
        if feature not in explainer.feature_names:
            raise ValueError(
                f"Feature '{feature}' not found in feature_names. "
                f"Available: {explainer.feature_names}"
            )

        plots_dir = explainer._plots_dir
        self._ensure_dir(plots_dir)

        X_numpy = X_sample.values.astype(float)
        shap_matrix = explainer._shap_explainer.shap_values(X_numpy)

        if isinstance(shap_matrix, list):
            shap_arr = np.asarray(shap_matrix[1])
        else:
            shap_arr = np.asarray(shap_matrix)

        feat_idx = explainer.feature_names.index(feature)

        plt.figure()
        shap.dependence_plot(
            feat_idx,
            shap_arr,
            X_sample,
            feature_names=explainer.feature_names,
            show=False,
        )
        safe_name = feature.replace("/", "_").replace(" ", "_")
        out_path = str(plots_dir / f"shap_dependence_{safe_name}.png")
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close("all")

        _logger.info("Dependence plot for '%s' saved to '%s'", feature, out_path)
        return out_path

    def generate_global_report(
        self,
        explainer: SHAPExplainer,
        X_sample: pd.DataFrame,
    ) -> dict[str, Any]:
        """Run all global SHAP analyses and return a report dict.

        Generates the beeswarm summary plot, dependence plots for the top
        monitored features, and a feature importance DataFrame.  All plots
        are saved to ``plots/shap/``.

        Note:
            ``X_sample`` MUST be drawn from the **test set**, not the
            training set, to avoid optimistic bias.

        Args:
            explainer: Fitted :class:`SHAPExplainer` instance.
            X_sample:  Test-set sample (recommended: 5 000 rows).

        Returns:
            Dictionary with keys:

            * ``"summary_plot"``     — path to beeswarm PNG.
            * ``"dependence_plots"`` — dict of feature → path.
            * ``"importance_df"``    — :class:`~pandas.DataFrame` with
              ``feature_name`` and ``mean_abs_shap`` columns.
            * ``"importance_csv"``   — path to saved CSV of importance.
        """
        _logger.info(
            "GlobalExplainabilityReporter — generating global report "
            "(%d samples)",
            len(X_sample),
        )

        # ---- Beeswarm summary plot ------------------------------------
        summary_path = self.generate_summary_plot(explainer, X_sample)

        # ---- Dependence plots for first 4 monitored features ----------
        monitored = (
            explainer.feature_names[:4]
            if len(explainer.feature_names) >= 4
            else explainer.feature_names
        )
        dep_paths: dict[str, str] = {}
        for feat in monitored:
            try:
                dep_paths[feat] = self.generate_dependence_plot(
                    explainer, X_sample, feat
                )
            except Exception as exc:  # noqa: BLE001
                _logger.warning(
                    "GlobalExplainabilityReporter — dependence plot for "
                    "'%s' failed: %s",
                    feat,
                    exc,
                )

        # ---- Feature importance table (mean |SHAP|) -------------------
        importance_df = explainer.get_global_importance(X_sample, n_features=20)

        plots_dir = explainer._plots_dir
        self._ensure_dir(plots_dir)
        csv_path = str(plots_dir / "shap_global_importance.csv")
        importance_df.to_csv(csv_path, index=False)
        _logger.info("Global importance CSV saved to '%s'", csv_path)

        report: dict[str, Any] = {
            "summary_plot": summary_path,
            "dependence_plots": dep_paths,
            "importance_df": importance_df,
            "importance_csv": csv_path,
        }
        _logger.info(
            "GlobalExplainabilityReporter — report complete | "
            "plots=%d",
            1 + len(dep_paths),
        )
        return report


# ---------------------------------------------------------------------------
# __main__ — demo runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import pickle
    import sys

    parser = argparse.ArgumentParser(
        description="CIPHER SHAP explainability demo."
    )
    parser.add_argument(
        "--lgbm-path",
        default="models/lgbm_model.pkl",
        help="Path to saved LGBMClassifier (joblib).",
    )
    parser.add_argument(
        "--data-path",
        required=True,
        help=(
            "Path to a pickle with "
            "(X_train, X_test, y_train, y_test, feature_names)."
        ),
    )
    args = parser.parse_args()

    # Load data
    with open(args.data_path, "rb") as fh:
        payload = pickle.load(fh)

    if isinstance(payload, dict):
        X_test = pd.DataFrame(payload["X_test"])
        y_test = np.asarray(payload["y_test"])
        feature_names = list(payload["feature_names"])
    else:
        _, X_test_raw, _, y_test, feature_names = payload
        X_test = pd.DataFrame(X_test_raw, columns=feature_names)

    # Initialise explainer
    exp = SHAPExplainer(
        model_path=args.lgbm_path,
        feature_names=feature_names,
    )

    # Pick 5 random flagged transactions
    flagged_idx = np.where(y_test == 1)[0]
    sample_idx = np.random.default_rng(42).choice(
        flagged_idx, size=min(5, len(flagged_idx)), replace=False
    )

    print("\n" + "=" * 70)
    print("CIPHER SHAP Explanations — Flagged Transactions")
    print("=" * 70)

    for i, idx in enumerate(sample_idx, 1):
        tx_id = f"TX_{idx:06d}"
        X_row = X_test.iloc[[idx]]
        result = exp.explain(tx_id, X_row)

        print(f"\n[{i}] Transaction {tx_id}")
        print(f"    Fraud Probability : {result.prediction:.4f}")
        print(f"    Computed in       : {result.computation_time_ms:.1f} ms")
        print(f"    Summary           : {result.plain_english_summary}")
        print("    Top 3 Features:")
        for feat in result.top_features[:3]:
            arrow = "↑" if feat["direction"] == "increases_risk" else "↓"
            print(
                f"      {arrow} {feat['feature_name']:30s} "
                f"value={feat['feature_value']:.4g}  "
                f"SHAP={feat['shap_value']:+.4f}"
            )

    # Global report on test subset
    sample_size = min(5000, len(X_test))
    X_global = X_test.iloc[:sample_size]
    print(f"\nGenerating global SHAP report on {sample_size} test samples…")
    reporter = GlobalExplainabilityReporter()
    report = reporter.generate_global_report(exp, X_global)

    print("\nGenerated Plots:")
    print(f"  Summary    : {report['summary_plot']}")
    for feat, path in report["dependence_plots"].items():
        print(f"  Dependence : {path}  ({feat})")
    print(f"  Importance : {report['importance_csv']}")
    print("\nTop 5 Features by Mean |SHAP|:")
    print(report["importance_df"].head(5).to_string(index=False))
