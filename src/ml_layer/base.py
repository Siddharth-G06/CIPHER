"""
src/ml_layer/base.py
--------------------
Abstract base class for all CIPHER fraud-detection models.

Every concrete detector in the ML layer must subclass :class:`BaseDetector`
and implement its six abstract methods.  This contract ensures that
:class:`EnsembleDetector`, the trainer, and the serving layer can all
interact with any detector through a uniform interface.

Example::

    from src.ml_layer.base import BaseDetector

    class MyDetector(BaseDetector):
        def train(self, X_train, y_train): ...
        def predict_proba(self, X): ...
        def predict(self, X, threshold): ...
        def evaluate(self, X_test, y_test): ...
        def save(self, path): ...
        def load(self, path): ...
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class BaseDetector(ABC):
    """Abstract base class for CIPHER fraud-detection models.

    All concrete detectors must implement every abstract method defined here.
    The interface is intentionally minimal so that the ensemble and serving
    layers do not depend on implementation details of individual models.

    Attributes:
        model_: The fitted underlying model object. Set to ``None`` until
            :meth:`train` is called.  Concrete subclasses are expected to
            populate this attribute during training.
    """

    def __init__(self) -> None:
        self.model_: Any = None

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    @abstractmethod
    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Fit the model on labelled training data.

        Args:
            X_train: Feature matrix of shape ``(n_samples, n_features)``.
            y_train: Binary target vector of shape ``(n_samples,)``.
                Values must be ``0`` (legitimate) or ``1`` (fraud).

        Returns:
            None.  The fitted model is stored in ``self.model_``.
        """

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return fraud-probability scores for each sample.

        Args:
            X: Feature matrix of shape ``(n_samples, n_features)``.

        Returns:
            1-D array of shape ``(n_samples,)`` with values in ``[0, 1]``
            where values close to 1 indicate high fraud / anomaly risk.
        """

    @abstractmethod
    def predict(self, X: np.ndarray, threshold: float) -> np.ndarray:
        """Return binary fraud predictions by thresholding fraud-probability scores.

        Args:
            X: Feature matrix of shape ``(n_samples, n_features)``.
            threshold: Decision boundary in ``(0, 1)``.  Samples with a
                fraud-probability score ``>= threshold`` are flagged as fraud
                (class ``1``).

        Returns:
            Binary prediction array of shape ``(n_samples,)`` with dtype
            ``int``.
        """

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    @abstractmethod
    def evaluate(
        self, X_test: np.ndarray, y_test: np.ndarray
    ) -> dict[str, float]:
        """Compute evaluation metrics on a held-out test set.

        Args:
            X_test: Feature matrix of shape ``(n_samples, n_features)``.
            y_test: True binary labels of shape ``(n_samples,)``.

        Returns:
            Dictionary mapping metric name to metric value.  At minimum the
            returned dict must contain the key ``"f1"``.
        """

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    @abstractmethod
    def save(self, path: str) -> None:
        """Serialise the fitted model to disk.

        Args:
            path: Destination file path (e.g. ``"models/lgbm.pkl"``).
                Parent directories must already exist.

        Returns:
            None.
        """

    @abstractmethod
    def load(self, path: str) -> "BaseDetector":
        """Deserialise a previously saved model from disk.

        Args:
            path: Path to the serialised model file written by :meth:`save`.

        Returns:
            The detector instance with ``model_`` populated from disk.  Most
            implementations return ``self`` after updating ``self.model_``.
        """
