"""
src/utils/mlflow_client.py
---------------------------
Cached MLflow query helpers for the CIPHER Streamlit dashboard.

All query functions use ``@st.cache_data(ttl=60)`` so that the MLflow
tracking server is not hammered on every Streamlit rerun.  Functions are
**module-level** (not instance methods) so Streamlit's cache can hash
their arguments correctly.

Design notes
------------
* Functions accept ``tracking_uri`` and ``registry_name`` as explicit
  parameters — this makes them pure functions that Streamlit's hash-based
  cache can reason about (instance methods with ``self`` are not cacheable
  without ``hash_funcs`` overrides).
* ``@st.cache_data`` is applied at import time so the first import from
  ``app.py`` registers the cache entries.
* All functions fail gracefully: MLflow exceptions produce an empty result
  rather than crashing the dashboard.

Usage::

    from src.utils.mlflow_client import (
        get_model_versions,
        get_experiment_runs,
        get_champion_metrics,
        get_challenger_metrics,
    )

    versions = get_model_versions(tracking_uri, registry_name)
"""

from __future__ import annotations

import streamlit as st
from typing import Any

import mlflow
from mlflow.tracking import MlflowClient

from src.utils.logger import get_logger

_logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Module-level cached functions (Streamlit-cache-compatible)
# ---------------------------------------------------------------------------


@st.cache_data(ttl=60, show_spinner=False)
def get_model_versions(
    tracking_uri: str,
    registry_name: str,
) -> list[dict]:
    """Return all registered model versions sorted by version number DESC.

    Args:
        tracking_uri: MLflow tracking server URI.
        registry_name: Name of the registered model in the MLflow Registry.

    Returns:
        List of dicts, each with keys: ``version``, ``stage``, ``run_id``,
        ``created_at``, ``auc_pr``, ``f1``, ``precision``, ``recall``.
        Empty list if the model is not registered or MLflow is unreachable.
    """
    try:
        mlflow.set_tracking_uri(tracking_uri)
        client = MlflowClient()
        mvs = client.search_model_versions(
            f"name='{registry_name}'",
            order_by=["version_number DESC"],
        )
        rows: list[dict] = []
        for mv in mvs:
            run_metrics: dict = {}
            try:
                run_data = client.get_run(mv.run_id).data
                m = run_data.metrics

                def _pick(m: dict, *keys: str, default=None):
                    for k in keys:
                        if k in m:
                            return m[k]
                    return default

                run_metrics = {
                    "auc_pr":    _pick(m, "challenger_auc_pr",    "ensemble.auc_pr"),
                    "f1":        _pick(m, "challenger_f1",        "ensemble.f1"),
                    "precision": _pick(m, "challenger_precision", "ensemble.precision"),
                    "recall":    _pick(m, "challenger_recall",    "ensemble.recall"),
                    "trigger_reason": run_data.tags.get("trigger_reason", ""),
                    "outcome":        run_data.tags.get("outcome", ""),
                }
            except Exception:  # noqa: BLE001
                pass
            rows.append(
                {
                    "version": mv.version,
                    "stage": mv.current_stage,
                    "run_id": mv.run_id,
                    "created_at": mv.creation_timestamp,
                    **run_metrics,
                }
            )
        _logger.debug(
            "get_model_versions — fetched %d versions for '%s'",
            len(rows),
            registry_name,
        )
        return rows
    except Exception as exc:  # noqa: BLE001
        _logger.warning("get_model_versions — MLflow error: %s", exc)
        return []


@st.cache_data(ttl=60, show_spinner=False)
def get_experiment_runs(
    tracking_uri: str,
    experiment_name: str,
    n: int = 10,
) -> list[dict]:
    """Return the last *n* runs from an MLflow experiment.

    Args:
        tracking_uri: MLflow tracking server URI.
        experiment_name: Name of the MLflow experiment.
        n: Number of most-recent runs to return (default: 10).

    Returns:
        List of run dicts with keys: ``run_id``, ``status``, ``start_time``,
        ``trigger_reason``, ``outcome``, ``challenger_auc_pr``,
        ``champion_auc_pr``, ``promoted``.
    """
    try:
        mlflow.set_tracking_uri(tracking_uri)
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            return []
        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["attributes.start_time DESC"],
            max_results=n,
        )
        rows: list[dict] = []
        for _, row in runs.iterrows():
            rows.append(
                {
                    "run_id": row.get("run_id", ""),
                    "status": row.get("status", ""),
                    "start_time": row.get("start_time", None),
                    "trigger_reason": row.get("tags.trigger_reason", ""),
                    "outcome": row.get("tags.outcome", ""),
                    "challenger_auc_pr": row.get("metrics.challenger_auc_pr", None),
                    "champion_auc_pr": row.get("metrics.champion_auc_pr", None),
                    "promoted": row.get("tags.outcome", "") == "promoted",
                }
            )
        _logger.debug(
            "get_experiment_runs — fetched %d runs from '%s'",
            len(rows),
            experiment_name,
        )
        return rows
    except Exception as exc:  # noqa: BLE001
        _logger.warning("get_experiment_runs — MLflow error: %s", exc)
        return []


@st.cache_data(ttl=60, show_spinner=False)
def get_champion_metrics(
    tracking_uri: str,
    registry_name: str,
) -> dict:
    """Return metrics of the current Production / champion model version.

    Tries the ``champion`` alias first (MLflow ≥2.9); falls back to the
    ``Production`` stage for older deployments.

    Args:
        tracking_uri: MLflow tracking server URI.
        registry_name: Name of the registered model.

    Returns:
        Dict with keys ``version``, ``auc_pr``, ``f1``, ``precision``,
        ``recall``, ``auc_roc``.  Empty dict if no champion is registered.
    """
    try:
        mlflow.set_tracking_uri(tracking_uri)
        client = MlflowClient()

        # Try alias first, then Production stage.
        try:
            mv = client.get_model_version_by_alias(registry_name, "champion")
        except Exception:  # noqa: BLE001
            prod_versions = client.get_latest_versions(
                registry_name, stages=["Production"]
            )
            if not prod_versions:
                # fall back to any version
                staging_versions = client.get_latest_versions(
                    registry_name, stages=["Staging"]
                )
                if not staging_versions:
                    return {}
                mv = staging_versions[0]
            else:
                mv = prod_versions[0]

        run_data = client.get_run(mv.run_id).data
        m = run_data.metrics

        def _pick(m: dict, *keys: str, default: float = 0.0) -> float:
            """Return first key found in m, fallback to default."""
            for k in keys:
                if k in m:
                    return m[k]
            return default

        return {
            "version": mv.version,
            "auc_pr":    _pick(m, "challenger_auc_pr",    "ensemble.auc_pr",    "auc_pr"),
            "f1":        _pick(m, "challenger_f1",        "ensemble.f1",        "f1"),
            "precision": _pick(m, "challenger_precision", "ensemble.precision", "precision"),
            "recall":    _pick(m, "challenger_recall",    "ensemble.recall",    "recall"),
            "auc_roc":   _pick(m, "challenger_auc_roc",   "ensemble.auc_roc",  "auc_roc"),
        }
    except Exception as exc:  # noqa: BLE001
        _logger.warning("get_champion_metrics — MLflow error: %s", exc)
        return {}


@st.cache_data(ttl=60, show_spinner=False)
def get_challenger_metrics(
    tracking_uri: str,
    registry_name: str,
) -> dict:
    """Return metrics of the current Staging / challenger model version.

    Args:
        tracking_uri: MLflow tracking server URI.
        registry_name: Name of the registered model.

    Returns:
        Dict with keys ``version``, ``auc_pr``, ``f1``, ``precision``,
        ``recall``, ``auc_roc``.  Empty dict if no challenger is in Staging.
    """
    try:
        mlflow.set_tracking_uri(tracking_uri)
        client = MlflowClient()

        # Try alias first, then Staging stage.
        try:
            mv = client.get_model_version_by_alias(registry_name, "challenger")
        except Exception:  # noqa: BLE001
            staging_versions = client.get_latest_versions(
                registry_name, stages=["Staging"]
            )
            if not staging_versions:
                return {}
            mv = staging_versions[0]

        run_data = client.get_run(mv.run_id).data
        m = run_data.metrics

        def _pick(m: dict, *keys: str, default: float = 0.0) -> float:
            for k in keys:
                if k in m:
                    return m[k]
            return default

        return {
            "version": mv.version,
            "auc_pr":    _pick(m, "challenger_auc_pr",    "ensemble.auc_pr",    "auc_pr"),
            "f1":        _pick(m, "challenger_f1",        "ensemble.f1",        "f1"),
            "precision": _pick(m, "challenger_precision", "ensemble.precision", "precision"),
            "recall":    _pick(m, "challenger_recall",    "ensemble.recall",    "recall"),
            "auc_roc":   _pick(m, "challenger_auc_roc",   "ensemble.auc_roc",  "auc_roc"),
        }
    except Exception as exc:  # noqa: BLE001
        _logger.warning("get_challenger_metrics — MLflow error: %s", exc)
        return {}


# ---------------------------------------------------------------------------
# MLflowQueryClient — thin class wrapper for convenient import in tests
# ---------------------------------------------------------------------------


class MLflowQueryClient:
    """Thin façade that delegates to module-level cached functions.

    Accepts ``tracking_uri`` and ``registry_name`` once in ``__init__`` so
    callers don't need to pass them on every method call.  The underlying
    module-level functions carry the ``@st.cache_data`` cache, so this class
    adds zero overhead.

    Args:
        tracking_uri: MLflow tracking server URI.
        registry_name: Registered model name in the MLflow Model Registry.
        experiment_name: Name of the MLflow experiment.
    """

    def __init__(
        self,
        tracking_uri: str,
        registry_name: str,
        experiment_name: str,
    ) -> None:
        self._tracking_uri = tracking_uri
        self._registry_name = registry_name
        self._experiment_name = experiment_name

    def get_model_versions(self) -> list[dict]:
        """See :func:`get_model_versions`."""
        return get_model_versions(self._tracking_uri, self._registry_name)

    def get_experiment_runs(self, n: int = 10) -> list[dict]:
        """See :func:`get_experiment_runs`."""
        return get_experiment_runs(self._tracking_uri, self._experiment_name, n)

    def get_champion_metrics(self) -> dict:
        """See :func:`get_champion_metrics`."""
        return get_champion_metrics(self._tracking_uri, self._registry_name)

    def get_challenger_metrics(self) -> dict:
        """See :func:`get_challenger_metrics`."""
        return get_challenger_metrics(self._tracking_uri, self._registry_name)
