"""
src/ml_layer/model.py
---------------------
Concrete detector implementations for the CIPHER fraud-detection pipeline.

Three classes are provided:

* :class:`LightGBMDetector`  — supervised gradient-boosting classifier.
* :class:`IsolationForestDetector` — unsupervised anomaly detector.
* :class:`EnsembleDetector`  — weighted combination of the two above.

All hyperparameters are read from ``config/config.yaml`` via
:func:`src.utils.config_loader.load_config`.  No values are hardcoded.
All log output is routed through :func:`src.utils.logger.get_logger`.

Typical usage::

    from src.ml_layer.model import EnsembleDetector

    detector = EnsembleDetector()
    detector.train(X_train, y_train)
    proba   = detector.predict_proba(X_test)
    labels  = detector.predict(X_test, threshold=0.5)
    metrics = detector.evaluate(X_test, y_test)
    summary = detector.describe()
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.ml_layer.base import BaseDetector
from src.utils.config_loader import load_config
from src.utils.logger import get_logger

_logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# LightGBMDetector
# ---------------------------------------------------------------------------


class LightGBMDetector(BaseDetector):
    """Supervised gradient-boosting fraud detector backed by LightGBM.

    Hyperparameters are loaded from the ``model.lgbm`` section of
    ``config/config.yaml``.  ``scale_pos_weight`` is **not** read from config;
    it is auto-computed from the training labels as
    ``n_negative / n_positive`` to account for the class imbalance present in
    real-world fraud datasets.

    Attributes:
        model_: Fitted :class:`~lightgbm.LGBMClassifier` instance.  ``None``
            until :meth:`train` is called.
        cfg_: The ``model.lgbm`` sub-dict from ``config.yaml``.
    """

    def __init__(self, config_path: str = "config/config.yaml") -> None:
        """Initialise the detector and load hyperparameters from config.

        Args:
            config_path: Path to the YAML configuration file, relative to the
                project root (default: ``"config/config.yaml"``).
        """
        super().__init__()
        cfg = load_config(config_path)
        self.cfg_: dict[str, Any] = cfg["model"]["lgbm"]
        _logger.info(
            "LightGBMDetector initialised — hyperparams: %s", self.cfg_
        )

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Fit the LightGBM classifier on labelled training data.

        ``scale_pos_weight`` is computed automatically as
        ``count(y==0) / count(y==1)`` to handle class imbalance.

        Args:
            X_train: Feature matrix of shape ``(n_samples, n_features)``.
            y_train: Binary target vector (0 = legitimate, 1 = fraud).
        """
        n_neg = int(np.sum(y_train == 0))
        n_pos = int(np.sum(y_train == 1))
        scale_pos_weight = n_neg / max(n_pos, 1)

        _logger.info(
            "LightGBMDetector — training start | samples=%d, features=%d, "
            "pos=%d, neg=%d, scale_pos_weight=%.4f",
            X_train.shape[0],
            X_train.shape[1],
            n_pos,
            n_neg,
            scale_pos_weight,
        )

        self.model_ = LGBMClassifier(
            n_estimators=self.cfg_["n_estimators"],
            learning_rate=self.cfg_["learning_rate"],
            num_leaves=self.cfg_["num_leaves"],
            max_depth=self.cfg_["max_depth"],
            subsample=self.cfg_["subsample"],
            colsample_bytree=self.cfg_["colsample_bytree"],
            min_child_samples=self.cfg_["min_child_samples"],
            reg_alpha=self.cfg_["reg_alpha"],
            reg_lambda=self.cfg_["reg_lambda"],
            scale_pos_weight=scale_pos_weight,
            objective="binary",
            n_jobs=-1,
            random_state=42,
            verbose=-1,
        )
        self.model_.fit(X_train, y_train)

        _logger.info(
            "LightGBMDetector — training complete | n_features_used=%d",
            X_train.shape[1],
        )

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return class-1 (fraud) probability for each sample.

        Args:
            X: Feature matrix of shape ``(n_samples, n_features)``.

        Returns:
            1-D array of shape ``(n_samples,)`` with fraud probabilities in
            ``[0, 1]``.

        Raises:
            RuntimeError: If :meth:`train` has not been called yet.
        """
        if self.model_ is None:
            raise RuntimeError(
                "LightGBMDetector.train() must be called before predict_proba()."
            )
        return self.model_.predict_proba(X)[:, 1]

    def predict(self, X: np.ndarray, threshold: float) -> np.ndarray:
        """Return binary fraud predictions by thresholding fraud probabilities.

        Args:
            X: Feature matrix of shape ``(n_samples, n_features)``.
            threshold: Decision boundary in ``(0, 1)``.

        Returns:
            Integer array of shape ``(n_samples,)`` with values ``0`` or ``1``.
        """
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self, X_test: np.ndarray, y_test: np.ndarray
    ) -> dict[str, float]:
        """Compute classification metrics on a held-out test set.

        The decision threshold used for F1, precision, and recall is fixed at
        ``0.5``.  AUC metrics are threshold-agnostic.

        Args:
            X_test: Feature matrix of shape ``(n_samples, n_features)``.
            y_test: True binary labels of shape ``(n_samples,)``.

        Returns:
            Dictionary with keys:

            * ``"f1"``       — F1 score (harmonic mean of precision/recall).
            * ``"auc_roc"``  — Area under the ROC curve.
            * ``"auc_pr"``   — Area under the Precision-Recall curve
              (computed via :func:`~sklearn.metrics.average_precision_score`).
            * ``"precision"``— Precision at threshold 0.5.
            * ``"recall"``   — Recall at threshold 0.5.
        """
        proba = self.predict_proba(X_test)
        preds = (proba >= 0.5).astype(int)

        metrics: dict[str, float] = {
            "f1": f1_score(y_test, preds, zero_division=0),
            "auc_roc": roc_auc_score(y_test, proba),
            "auc_pr": average_precision_score(y_test, proba),
            "precision": precision_score(y_test, preds, zero_division=0),
            "recall": recall_score(y_test, preds, zero_division=0),
        }

        _logger.info(
            "LightGBMDetector — evaluation | f1=%.4f, auc_roc=%.4f, "
            "auc_pr=%.4f, precision=%.4f, recall=%.4f",
            metrics["f1"],
            metrics["auc_roc"],
            metrics["auc_pr"],
            metrics["precision"],
            metrics["recall"],
        )
        return metrics

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Serialise the fitted model to disk using :func:`joblib.dump`.

        Args:
            path: Destination file path.  Parent directories are created
                automatically if they do not exist.
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model_, path)
        _logger.info("LightGBMDetector — model saved to '%s'", path)

    def load(self, path: str) -> "LightGBMDetector":
        """Deserialise a previously saved LightGBM model from disk.

        Args:
            path: Path to the ``.pkl`` file written by :meth:`save`.

        Returns:
            ``self`` with ``model_`` populated from disk.
        """
        self.model_ = joblib.load(path)
        _logger.info("LightGBMDetector — model loaded from '%s'", path)
        return self


# ---------------------------------------------------------------------------
# IsolationForestDetector
# ---------------------------------------------------------------------------


class IsolationForestDetector(BaseDetector):
    """Unsupervised anomaly detector backed by sklearn's Isolation Forest.

    Hyperparameters are loaded from the ``model.isolation_forest`` section of
    ``config/config.yaml``.

    Score normalisation
    -------------------
    Isolation Forest's :meth:`~sklearn.ensemble.IsolationForest.decision_function`
    returns more **negative** values for anomalous samples.  To obtain an
    intuitive risk score (high value = high anomaly risk) the raw scores are
    min-max normalised and then **inverted**::

        raw   = model.decision_function(X)           # anomalies near −1
        score = 1 − (raw − min(raw)) / (max(raw) − min(raw))
        # anomalies → score ≈ 1.0 ✓

    Attributes:
        model_: Fitted :class:`~sklearn.ensemble.IsolationForest` instance.
        cfg_: The ``model.isolation_forest`` sub-dict from ``config.yaml``.
    """

    def __init__(self, config_path: str = "config/config.yaml") -> None:
        """Initialise the detector and load hyperparameters from config.

        Args:
            config_path: Path to the YAML configuration file.
        """
        super().__init__()
        cfg = load_config(config_path)
        self.cfg_: dict[str, Any] = cfg["model"]["isolation_forest"]
        _logger.info(
            "IsolationForestDetector initialised — hyperparams: %s", self.cfg_
        )

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Fit the Isolation Forest on the training feature matrix.

        Note:
            Isolation Forest is an *unsupervised* algorithm; ``y_train`` is
            accepted for interface compatibility but is **not** used during
            fitting.

        Args:
            X_train: Feature matrix of shape ``(n_samples, n_features)``.
            y_train: Ignored.  Present for :class:`BaseDetector` compatibility.
        """
        # max_samples may be the string "auto" from YAML; pass it as-is.
        max_samples: Any = self.cfg_["max_samples"]

        _logger.info(
            "IsolationForestDetector — training start | samples=%d, features=%d",
            X_train.shape[0],
            X_train.shape[1],
        )

        self.model_ = IsolationForest(
            n_estimators=self.cfg_["n_estimators"],
            max_samples=max_samples,
            contamination=self.cfg_["contamination"],
            random_state=self.cfg_["random_state"],
            n_jobs=-1,
        )
        self.model_.fit(X_train)

        _logger.info("IsolationForestDetector — training complete")

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return normalised anomaly risk scores for each sample.

        The raw ``decision_function`` output (where anomalies score near −1)
        is min-max normalised and then inverted so that high output values
        correspond to high anomaly risk::

            raw   = decision_function(X)
            score = 1 − (raw − min) / (max − min)

        Args:
            X: Feature matrix of shape ``(n_samples, n_features)``.

        Returns:
            1-D array of shape ``(n_samples,)`` with scores in ``[0, 1]``.
            Values close to ``1`` indicate high anomaly / fraud risk.

        Raises:
            RuntimeError: If :meth:`train` has not been called yet.
        """
        if self.model_ is None:
            raise RuntimeError(
                "IsolationForestDetector.train() must be called before "
                "predict_proba()."
            )
        raw: np.ndarray = self.model_.decision_function(X)

        raw_min = raw.min()
        raw_max = raw.max()
        denominator = raw_max - raw_min

        if denominator == 0.0:
            # All samples receive the same score — return 0.5 for each.
            _logger.warning(
                "IsolationForestDetector — decision_function has zero range; "
                "returning uniform scores of 0.5."
            )
            return np.full(len(raw), 0.5)

        # Invert: anomalies (low raw) → high score
        score: np.ndarray = 1.0 - (raw - raw_min) / denominator
        return score

    def predict(self, X: np.ndarray, threshold: float) -> np.ndarray:
        """Return binary anomaly predictions by thresholding risk scores.

        Args:
            X: Feature matrix of shape ``(n_samples, n_features)``.
            threshold: Decision boundary in ``(0, 1)``.

        Returns:
            Integer array of shape ``(n_samples,)`` with values ``0`` or ``1``.
        """
        score = self.predict_proba(X)
        return (score >= threshold).astype(int)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self, X_test: np.ndarray, y_test: np.ndarray
    ) -> dict[str, float]:
        """Compute evaluation metrics on a held-out test set.

        Args:
            X_test: Feature matrix of shape ``(n_samples, n_features)``.
            y_test: True binary labels of shape ``(n_samples,)``.

        Returns:
            Dictionary with keys:

            * ``"f1"``      — F1 score at threshold 0.5.
            * ``"auc_roc"`` — Area under the ROC curve.
        """
        score = self.predict_proba(X_test)
        preds = (score >= 0.5).astype(int)

        metrics: dict[str, float] = {
            "f1": f1_score(y_test, preds, zero_division=0),
            "auc_roc": roc_auc_score(y_test, score),
        }

        _logger.info(
            "IsolationForestDetector — evaluation | f1=%.4f, auc_roc=%.4f",
            metrics["f1"],
            metrics["auc_roc"],
        )
        return metrics

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Serialise the fitted model to disk using :func:`joblib.dump`.

        Args:
            path: Destination file path.  Parent directories are created
                automatically if they do not exist.
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model_, path)
        _logger.info("IsolationForestDetector — model saved to '%s'", path)

    def load(self, path: str) -> "IsolationForestDetector":
        """Deserialise a previously saved Isolation Forest model from disk.

        Args:
            path: Path to the ``.pkl`` file written by :meth:`save`.

        Returns:
            ``self`` with ``model_`` populated from disk.
        """
        self.model_ = joblib.load(path)
        _logger.info(
            "IsolationForestDetector — model loaded from '%s'", path
        )
        return self


# ---------------------------------------------------------------------------
# EnsembleDetector
# ---------------------------------------------------------------------------


class EnsembleDetector(BaseDetector):
    """Weighted ensemble of a LightGBM and an Isolation Forest detector.

    Combines the fraud-probability scores of both sub-models as a weighted
    average::

        ensemble_score = lgbm_weight × lgbm_score + iso_weight × iso_score

    Weights and decision threshold are loaded from the ``model.ensemble``
    section of ``config/config.yaml``.

    Attributes:
        lgbm_: Underlying :class:`LightGBMDetector` instance.
        iso_:  Underlying :class:`IsolationForestDetector` instance.
        lgbm_weight_: Weight applied to the LightGBM score.
        iso_weight_:  Weight applied to the Isolation Forest score.
        threshold_:   Decision boundary used by :meth:`predict`.
        model_:       Always ``None`` for the ensemble (sub-models hold the
            actual fitted objects); present for interface compatibility.
    """

    def __init__(self, config_path: str = "config/config.yaml") -> None:
        """Initialise both sub-detectors and load ensemble hyperparameters.

        Args:
            config_path: Path to the YAML configuration file.
        """
        super().__init__()
        cfg = load_config(config_path)
        ensemble_cfg: dict[str, Any] = cfg["model"]["ensemble"]

        self.lgbm_weight_: float = float(ensemble_cfg["lgbm_weight"])
        self.iso_weight_: float = float(ensemble_cfg["iso_weight"])
        self.threshold_: float = float(ensemble_cfg["threshold"])

        self.lgbm_: LightGBMDetector = LightGBMDetector(config_path)
        self.iso_: IsolationForestDetector = IsolationForestDetector(
            config_path
        )

        _logger.info(
            "EnsembleDetector initialised | lgbm_weight=%.2f, "
            "iso_weight=%.2f, threshold=%.2f",
            self.lgbm_weight_,
            self.iso_weight_,
            self.threshold_,
        )

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Train both sub-detectors sequentially on the same training data.

        Args:
            X_train: Feature matrix of shape ``(n_samples, n_features)``.
            y_train: Binary target vector (0 = legitimate, 1 = fraud).
        """
        _logger.info("EnsembleDetector — starting sub-model training")
        self.lgbm_.train(X_train, y_train)
        self.iso_.train(X_train, y_train)
        _logger.info("EnsembleDetector — both sub-models trained successfully")

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return weighted ensemble fraud-risk scores.

        Computes::

            score = lgbm_weight × lgbm.predict_proba(X)
                  + iso_weight  × iso.predict_proba(X)

        Args:
            X: Feature matrix of shape ``(n_samples, n_features)``.

        Returns:
            1-D array of shape ``(n_samples,)`` with combined scores in
            ``[0, 1]``.
        """
        lgbm_score = self.lgbm_.predict_proba(X)
        iso_score = self.iso_.predict_proba(X)
        combined: np.ndarray = (
            self.lgbm_weight_ * lgbm_score + self.iso_weight_ * iso_score
        )
        return combined

    def predict(self, X: np.ndarray, threshold: float | None = None) -> np.ndarray:
        """Return binary fraud predictions using the ensemble score.

        Args:
            X: Feature matrix of shape ``(n_samples, n_features)``.
            threshold: Decision boundary.  If ``None``, the value from
                ``config.yaml`` (``model.ensemble.threshold``) is used.

        Returns:
            Integer array of shape ``(n_samples,)`` with values ``0`` or ``1``.
        """
        t = threshold if threshold is not None else self.threshold_
        score = self.predict_proba(X)
        return (score >= t).astype(int)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self, X_test: np.ndarray, y_test: np.ndarray
    ) -> dict[str, float]:
        """Compute full classification metrics on the ensemble score.

        Args:
            X_test: Feature matrix of shape ``(n_samples, n_features)``.
            y_test: True binary labels of shape ``(n_samples,)``.

        Returns:
            Dictionary with keys: ``"f1"``, ``"auc_roc"``, ``"auc_pr"``,
            ``"precision"``, ``"recall"``.
        """
        score = self.predict_proba(X_test)
        preds = (score >= self.threshold_).astype(int)

        metrics: dict[str, float] = {
            "f1": f1_score(y_test, preds, zero_division=0),
            "auc_roc": roc_auc_score(y_test, score),
            "auc_pr": average_precision_score(y_test, score),
            "precision": precision_score(y_test, preds, zero_division=0),
            "recall": recall_score(y_test, preds, zero_division=0),
        }

        _logger.info(
            "EnsembleDetector — evaluation | f1=%.4f, auc_roc=%.4f, "
            "auc_pr=%.4f, precision=%.4f, recall=%.4f",
            metrics["f1"],
            metrics["auc_roc"],
            metrics["auc_pr"],
            metrics["precision"],
            metrics["recall"],
        )
        return metrics

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def describe(self) -> dict[str, Any]:
        """Return a structured summary of sub-model and ensemble performance.

        Intended to be called after :meth:`train` and :meth:`evaluate` so
        that all component metrics are available.  The returned dict is
        logged to MLflow as a flat set of metrics in
        :mod:`src.ml_layer.trainer`.

        Returns:
            Dictionary with the following structure::

                {
                    "weights": {
                        "lgbm_weight": float,
                        "iso_weight":  float,
                        "threshold":   float,
                    },
                    "lgbm_metrics":     dict[str, float],   # from LightGBMDetector.evaluate()
                    "iso_metrics":      dict[str, float],   # from IsolationForestDetector.evaluate()
                    "ensemble_metrics": dict[str, float],   # from EnsembleDetector.evaluate()
                }

        Note:
            This method does **not** call ``evaluate()`` internally; the
            caller is responsible for providing evaluation data to each
            sub-detector and the ensemble separately and storing the results.
            Use :meth:`compile_describe` when you have evaluation data ready.
        """
        return {
            "weights": {
                "lgbm_weight": self.lgbm_weight_,
                "iso_weight": self.iso_weight_,
                "threshold": self.threshold_,
            },
            "lgbm_metrics": getattr(self, "_lgbm_metrics_", {}),
            "iso_metrics": getattr(self, "_iso_metrics_", {}),
            "ensemble_metrics": getattr(self, "_ensemble_metrics_", {}),
        }

    def compile_describe(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> dict[str, Any]:
        """Evaluate all sub-models and ensemble, then return the full summary dict.

        This is the primary method used by the trainer to generate the MLflow
        summary.  It calls :meth:`~LightGBMDetector.evaluate` on the LightGBM
        sub-model, :meth:`~IsolationForestDetector.evaluate` on the Isolation
        Forest sub-model, and :meth:`evaluate` on the ensemble itself, caching
        the results as instance attributes before returning them via
        :meth:`describe`.

        Args:
            X_test: Feature matrix of shape ``(n_samples, n_features)``.
            y_test: True binary labels of shape ``(n_samples,)``.

        Returns:
            The same nested dict as :meth:`describe`.
        """
        _logger.info(
            "EnsembleDetector — compiling full describe() summary"
        )
        self._lgbm_metrics_: dict[str, float] = self.lgbm_.evaluate(
            X_test, y_test
        )
        self._iso_metrics_: dict[str, float] = self.iso_.evaluate(
            X_test, y_test
        )
        self._ensemble_metrics_: dict[str, float] = self.evaluate(
            X_test, y_test
        )
        return self.describe()

    # ------------------------------------------------------------------
    # Persistence — delegate to sub-models; ensemble is a composition
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Serialise the entire ensemble (both sub-models) to a single file.

        Args:
            path: Destination file path.  Parent directories are created
                automatically if they do not exist.
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        _logger.info("EnsembleDetector — ensemble saved to '%s'", path)

    def load(self, path: str) -> "EnsembleDetector":
        """Deserialise a previously saved ensemble from disk.

        Note:
            Returns a **new** :class:`EnsembleDetector` loaded from disk,
            not ``self``.  Assign the return value::

                detector = EnsembleDetector()
                detector = detector.load("models/cipher_ensemble.pkl")

        Args:
            path: Path to the ``.pkl`` file written by :meth:`save`.

        Returns:
            The deserialised :class:`EnsembleDetector` instance.
        """
        loaded: EnsembleDetector = joblib.load(path)
        _logger.info("EnsembleDetector — ensemble loaded from '%s'", path)
        return loaded
