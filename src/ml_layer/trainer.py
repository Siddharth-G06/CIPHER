"""
src/ml_layer/trainer.py
-----------------------
MLflow-integrated training pipeline for the CIPHER fraud-detection ensemble.

The single public entry point is :func:`train_and_log`, which:

1. Loads the master configuration from ``config/config.yaml``.
2. Loads pre-processed, graph-featured data from a pickle file.
3. Trains :class:`~src.ml_layer.model.EnsembleDetector`.
4. Opens an **MLflow run** that wraps the entire train → evaluate → artifact
   generation → model registration sequence (so no artifacts are lost if a
   step fails early).
5. Logs all hyperparameters, metrics, a confusion-matrix plot, and a
   top-20 LightGBM feature importance plot as MLflow artifacts.
6. Registers the fitted ensemble in the MLflow Model Registry under the name
   defined in ``config.yaml`` (``mlflow.model_registry_name``).
7. Persists a local joblib copy to the path defined in
   ``config.yaml`` (``model.artifacts.local_model_path``).

Usage::

    python -m src.ml_layer.trainer \\
        --data-path data/processed/featured_data.pkl

or programmatically::

    from src.ml_layer.trainer import train_and_log
    train_and_log(data_path="data/processed/featured_data.pkl")
"""

from __future__ import annotations

import argparse
import pickle
import tempfile
from pathlib import Path
from typing import Any

import joblib
import matplotlib
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

from src.ml_layer.model import EnsembleDetector
from src.utils.config_loader import load_config
from src.utils.logger import get_logger

# Use non-interactive Agg backend so plots can be generated without a display.
matplotlib.use("Agg")

_logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _load_data(
    data_path: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Load pre-processed data from a pickle file.

    The pickle is expected to contain either:

    * A :class:`dict` with keys ``X_train``, ``X_test``, ``y_train``,
      ``y_test``, and ``feature_names``, **or**
    * A 5-tuple ``(X_train, X_test, y_train, y_test, feature_names)``.

    Args:
        data_path: Path to the pickle file produced by the feature layer.

    Returns:
        Tuple of ``(X_train, X_test, y_train, y_test, feature_names)``.

    Raises:
        ValueError: If the pickle contains an unsupported data structure.
        FileNotFoundError: If ``data_path`` does not exist.
    """
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path.resolve()}")

    _logger.info("Loading data from '%s'", path.resolve())
    with open(path, "rb") as fh:
        payload = pickle.load(fh)

    if isinstance(payload, dict):
        X_train: np.ndarray = np.asarray(payload["X_train"])
        X_test: np.ndarray = np.asarray(payload["X_test"])
        y_train: np.ndarray = np.asarray(payload["y_train"])
        y_test: np.ndarray = np.asarray(payload["y_test"])
        feature_names: list[str] = list(payload["feature_names"])
    elif isinstance(payload, (tuple, list)) and len(payload) == 5:
        X_train, X_test, y_train, y_test, feature_names = payload
        X_train = np.asarray(X_train)
        X_test = np.asarray(X_test)
        y_train = np.asarray(y_train)
        y_test = np.asarray(y_test)
        feature_names = list(feature_names)
    else:
        raise ValueError(
            f"Unsupported pickle payload type: {type(payload)}. "
            "Expected a dict with keys (X_train, X_test, y_train, y_test, "
            "feature_names) or a 5-tuple."
        )

    _logger.info(
        "Data loaded | X_train=%s, X_test=%s, features=%d",
        X_train.shape,
        X_test.shape,
        len(feature_names),
    )
    return X_train, X_test, y_train, y_test, feature_names


def _flatten_params(cfg: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    """Recursively flatten a nested config dict for MLflow param logging.

    Args:
        cfg: Nested configuration dictionary.
        prefix: Key prefix accumulated during recursion.

    Returns:
        Flat dict mapping dotted-path keys to scalar values.
        Example: ``{"model.lgbm.n_estimators": 500}``.
    """
    flat: dict[str, Any] = {}
    for k, v in cfg.items():
        full_key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            flat.update(_flatten_params(v, prefix=full_key))
        else:
            flat[full_key] = v
    return flat


def _log_confusion_matrix(
    y_test: np.ndarray,
    y_pred: np.ndarray,
    tmp_dir: str,
) -> str:
    """Generate and save a confusion-matrix plot; return the file path.

    Args:
        y_test: True binary labels.
        y_pred: Predicted binary labels.
        tmp_dir: Directory in which to write the PNG file.

    Returns:
        Absolute path to the saved PNG file.
    """
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Legitimate", "Fraud"],
    )
    disp.plot(ax=ax, colorbar=True, cmap="Blues")
    ax.set_title("CIPHER Ensemble — Confusion Matrix", fontsize=13)
    fig.tight_layout()

    out_path = str(Path(tmp_dir) / "confusion_matrix.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    _logger.info("Confusion matrix saved to '%s'", out_path)
    return out_path


def _log_feature_importance(
    ensemble: EnsembleDetector,
    feature_names: list[str],
    tmp_dir: str,
    top_n: int = 20,
) -> str:
    """Generate and save a top-N LightGBM feature importance bar chart.

    Uses ``feature_importances_`` (split-based gain importance from the fitted
    LGBMClassifier) together with ``feature_names`` from the feature layer
    pickle — **not** generic column indices — so the plot is interpretable.

    Args:
        ensemble: Trained :class:`~src.ml_layer.model.EnsembleDetector`.
        feature_names: Ordered list of feature names matching the column
            order in ``X_train``.
        tmp_dir: Directory in which to write the PNG file.
        top_n: Number of top features to display (default: 20).

    Returns:
        Absolute path to the saved PNG file.
    """
    lgbm_model = ensemble.lgbm_.model_
    importances: np.ndarray = lgbm_model.feature_importances_

    # Guard against mismatch between model and feature_names length.
    if len(importances) != len(feature_names):
        _logger.warning(
            "feature_importances_ length (%d) != feature_names length (%d). "
            "Falling back to index labels.",
            len(importances),
            len(feature_names),
        )
        names = [f"feature_{i}" for i in range(len(importances))]
    else:
        names = feature_names

    # Select top N by importance.
    indices = np.argsort(importances)[::-1][:top_n]
    top_importances = importances[indices]
    top_names = [names[i] for i in indices]

    fig, ax = plt.subplots(figsize=(10, 7))
    y_pos = np.arange(len(top_names))
    ax.barh(y_pos, top_importances[::-1], align="center", color="#4C72B0")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_names[::-1], fontsize=9)
    ax.set_xlabel("Feature Importance (split)", fontsize=11)
    ax.set_title(
        f"CIPHER LightGBM — Top {top_n} Feature Importances", fontsize=13
    )
    fig.tight_layout()

    out_path = str(Path(tmp_dir) / "feature_importance_top20.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    _logger.info("Feature importance plot saved to '%s'", out_path)
    return out_path


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def train_and_log(
    data_path: str,
    config_path: str = "config/config.yaml",
) -> EnsembleDetector:
    """Train the CIPHER ensemble and log everything to MLflow.

    The entire training, evaluation, and artifact generation pipeline runs
    inside a **single** ``mlflow.start_run()`` context so that all artifacts
    are associated with the same run and no records are lost on partial
    failure.

    Steps performed:

    1. Load config and data.
    2. Train :class:`~src.ml_layer.model.EnsembleDetector`.
    3. Open MLflow run (wrapping steps 4–9).
    4. Log all hyperparameters (flat dotted-key format).
    5. Log all metrics from ``ensemble.compile_describe()``.
    6. Log confusion-matrix PNG as MLflow artifact.
    7. Log top-20 feature importance PNG as MLflow artifact.
    8. Register the fitted ensemble in the MLflow Model Registry.
    9. Save a local joblib copy.

    Args:
        data_path: Path to the pickle file containing
            ``(X_train, X_test, y_train, y_test, feature_names)``.
        config_path: Path to the YAML configuration file
            (default: ``"config/config.yaml"``).

    Returns:
        The trained :class:`~src.ml_layer.model.EnsembleDetector` instance.
    """
    # ------------------------------------------------------------------
    # 1. Load config and data
    # ------------------------------------------------------------------
    cfg = load_config(config_path)
    mlflow_cfg: dict[str, Any] = cfg["mlflow"]
    artifact_cfg: dict[str, Any] = cfg["model"]["artifacts"]

    X_train, X_test, y_train, y_test, feature_names = _load_data(data_path)

    # ------------------------------------------------------------------
    # 2. Train ensemble
    # ------------------------------------------------------------------
    _logger.info("Initialising EnsembleDetector")
    ensemble = EnsembleDetector(config_path)

    _logger.info("Starting ensemble training")
    ensemble.train(X_train, y_train)
    _logger.info("Ensemble training complete")

    # ------------------------------------------------------------------
    # 3–9. MLflow run — wraps the full eval + artifact + registry flow
    # ------------------------------------------------------------------
    mlflow.set_tracking_uri(mlflow_cfg["tracking_uri"])
    mlflow.set_experiment(mlflow_cfg["experiment_name"])

    _logger.info(
        "Opening MLflow run under experiment '%s'",
        mlflow_cfg["experiment_name"],
    )

    with mlflow.start_run() as run:
        run_id = run.info.run_id
        _logger.info("MLflow run started | run_id=%s", run_id)

        # ---- 4. Log hyperparameters -----------------------------------
        flat_params = _flatten_params(cfg["model"])
        flat_params.update(_flatten_params(cfg.get("mlflow", {}), prefix="mlflow"))
        # MLflow param values must be strings; cast everything.
        mlflow.log_params({k: str(v) for k, v in flat_params.items()})
        _logger.info("MLflow — logged %d hyperparameters", len(flat_params))

        # ---- 5. Log metrics ------------------------------------------
        summary = ensemble.compile_describe(X_test, y_test)

        metrics_to_log: dict[str, float] = {}
        for model_key, metrics_dict in [
            ("lgbm", summary["lgbm_metrics"]),
            ("iso", summary["iso_metrics"]),
            ("ensemble", summary["ensemble_metrics"]),
        ]:
            for metric_name, metric_value in metrics_dict.items():
                metrics_to_log[f"{model_key}.{metric_name}"] = metric_value

        mlflow.log_metrics(metrics_to_log)
        _logger.info("MLflow — logged %d metrics", len(metrics_to_log))

        # ---- 6–7. Artifact plots -------------------------------------
        y_pred = ensemble.predict(X_test)

        with tempfile.TemporaryDirectory() as tmp_dir:
            cm_path = _log_confusion_matrix(y_test, y_pred, tmp_dir)
            fi_path = _log_feature_importance(
                ensemble, feature_names, tmp_dir, top_n=20
            )
            mlflow.log_artifact(cm_path, artifact_path="plots")
            mlflow.log_artifact(fi_path, artifact_path="plots")
            _logger.info("MLflow — confusion matrix and feature importance plots logged")

        # ---- 8. Register model ---------------------------------------
        registry_name: str = mlflow_cfg["model_registry_name"]
        model_uri = f"runs:/{run_id}/model"

        _logger.info(
            "Logging ensemble to MLflow model store (uri='%s')", model_uri
        )
        # Log the whole ensemble object using sklearn flavour.
        mlflow.sklearn.log_model(
            sk_model=ensemble,
            artifact_path="model",
            registered_model_name=registry_name,
        )
        _logger.info(
            "MLflow — model registered as '%s' → Staging", registry_name
        )

        # Transition the latest version to Staging.
        client = mlflow.tracking.MlflowClient()
        latest_versions = client.get_latest_versions(
            registry_name, stages=["None"]
        )
        if latest_versions:
            latest_version = latest_versions[-1].version
            client.transition_model_version_stage(
                name=registry_name,
                version=latest_version,
                stage="Staging",
                archive_existing_versions=False,
            )
            _logger.info(
                "MLflow — '%s' v%s transitioned to Staging",
                registry_name,
                latest_version,
            )

        # ---- 9. Local joblib copy ------------------------------------
        local_path: str = artifact_cfg["local_model_path"]
        ensemble.save(local_path)
        _logger.info(
            "Local ensemble copy saved to '%s'", local_path
        )

    _logger.info("MLflow run %s closed successfully", run_id)
    return ensemble


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train CIPHER EnsembleDetector and log to MLflow."
    )
    parser.add_argument(
        "--data-path",
        required=True,
        help=(
            "Path to a pickle file containing "
            "(X_train, X_test, y_train, y_test, feature_names)."
        ),
    )
    parser.add_argument(
        "--config-path",
        default="config/config.yaml",
        help="Path to config/config.yaml (default: config/config.yaml).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    train_and_log(data_path=args.data_path, config_path=args.config_path)
