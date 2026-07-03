"""
src/ml_layer/drift_simulator.py
--------------------------------
Concept-drift injection utilities for the CIPHER fraud-detection pipeline.

:class:`DriftSimulator` provides a single class-method,
:meth:`~DriftSimulator.simulate_concept_drift`, which modifies a copy of a
transaction DataFrame to introduce controlled drift after a configurable
injection point.  The original DataFrame is **never mutated**.

Three drift modes are supported:

* ``"high_value"``    — Fraud transaction amounts suddenly triple
  (card-present skimming shifts to high-value CNP fraud).
* ``"low_velocity"``  — Fraudsters slow card usage to evade velocity rules.
* ``"label_flip"``    — 20 % of *fraud* labels are silently flipped to
  legitimate (simulates mislabelling or fraud pattern change).

Typical usage::

    from src.ml_layer.drift_simulator import DriftSimulator

    df_drifted = DriftSimulator.simulate_concept_drift(
        df,
        injection_point=0.7,
        drift_type="high_value",
    )
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

_logger = get_logger(__name__)


class DriftSimulator:
    """Utility class for injecting concept drift into transaction DataFrames.

    All methods are class-methods; the class is never instantiated directly.

    Drift is injected only into rows **after** the ``injection_point`` fraction
    of the dataset.  Rows before the injection point are guaranteed to be
    byte-for-byte identical to the original DataFrame.
    """

    @classmethod
    def simulate_concept_drift(
        cls,
        df: pd.DataFrame,
        injection_point: float = 0.7,
        drift_type: str = "high_value",
    ) -> pd.DataFrame:
        """Inject concept drift into a copy of a transaction DataFrame.

        Only rows **after** ``injection_point`` fraction of the dataset are
        modified.  The pre-injection portion is returned unchanged.

        Args:
            df: Input transaction DataFrame.  Must contain ``isFraud`` column.
                For ``"high_value"`` drift, must also contain
                ``TransactionAmt``.  For ``"low_velocity"`` drift, must also
                contain ``card_tx_count_24h``.
            injection_point: Fraction of dataset where drift starts.  Must be
                in ``(0, 1)``.  E.g. ``0.7`` means drift affects the last 30 %
                of rows.
            drift_type: One of ``"high_value"``, ``"low_velocity"``, or
                ``"label_flip"``.

        Returns:
            A **copy** of ``df`` with drift injected after the injection point.
            The original DataFrame is not modified.

        Raises:
            ValueError: If ``drift_type`` is not one of the supported values,
                or if required columns are missing from ``df``.
        """
        supported = {"high_value", "low_velocity", "label_flip"}
        if drift_type not in supported:
            raise ValueError(
                f"drift_type '{drift_type}' not recognised. "
                f"Supported: {supported}"
            )
        if "isFraud" not in df.columns:
            raise ValueError("DataFrame must contain an 'isFraud' column.")

        split_idx = int(len(df) * injection_point)
        drifted = df.copy()

        # Boolean mask: rows that are post-injection AND are fraud
        post_injection_mask = pd.Series(False, index=drifted.index)
        post_injection_mask.iloc[split_idx:] = True
        fraud_mask = drifted["isFraud"] == 1
        post_fraud_mask = post_injection_mask & fraud_mask

        n_post_fraud = int(post_fraud_mask.sum())
        _logger.info(
            "DriftSimulator — injection_point=%.2f (row %d), drift_type='%s', "
            "post-injection fraud rows=%d",
            injection_point,
            split_idx,
            drift_type,
            n_post_fraud,
        )

        if drift_type == "high_value":
            cls._inject_high_value(drifted, post_fraud_mask)
        elif drift_type == "low_velocity":
            cls._inject_low_velocity(drifted, post_fraud_mask)
        elif drift_type == "label_flip":
            cls._inject_label_flip(drifted, post_injection_mask, fraud_mask)

        return drifted

    # ------------------------------------------------------------------
    # Private injection helpers
    # ------------------------------------------------------------------

    @classmethod
    def _inject_high_value(
        cls, df: pd.DataFrame, post_fraud_mask: pd.Series
    ) -> None:
        """Triple TransactionAmt for fraud rows after the injection point.

        Args:
            df: DataFrame to modify **in-place** (already a copy).
            post_fraud_mask: Boolean mask — True for post-injection fraud rows.
        """
        if "TransactionAmt" not in df.columns:
            raise ValueError(
                "'high_value' drift requires a 'TransactionAmt' column."
            )
        df.loc[post_fraud_mask, "TransactionAmt"] *= 3
        _logger.info(
            "DriftSimulator — 'high_value': multiplied TransactionAmt × 3 "
            "for %d fraud rows",
            int(post_fraud_mask.sum()),
        )

    @classmethod
    def _inject_low_velocity(
        cls, df: pd.DataFrame, post_fraud_mask: pd.Series
    ) -> None:
        """Set card_tx_count_24h = 1 for fraud rows after injection point.

        Simulates fraudsters deliberately slowing transaction velocity to avoid
        velocity-based detection rules.

        Args:
            df: DataFrame to modify **in-place** (already a copy).
            post_fraud_mask: Boolean mask — True for post-injection fraud rows.
        """
        if "card_tx_count_24h" not in df.columns:
            raise ValueError(
                "'low_velocity' drift requires a 'card_tx_count_24h' column."
            )
        df.loc[post_fraud_mask, "card_tx_count_24h"] = 1
        _logger.info(
            "DriftSimulator — 'low_velocity': set card_tx_count_24h=1 "
            "for %d fraud rows",
            int(post_fraud_mask.sum()),
        )

    @classmethod
    def _inject_label_flip(
        cls,
        df: pd.DataFrame,
        post_injection_mask: pd.Series,
        fraud_mask: pd.Series,
    ) -> None:
        """Flip 20 % of fraud-only labels to 0 after the injection point.

        Only fraud rows (``isFraud == 1``) in the post-injection window are
        eligible for flipping.  Non-fraud rows are never touched.  This
        simulates either a sudden drop in true fraud rate, or systematic
        mislabelling of a new fraud pattern.

        Args:
            df: DataFrame to modify **in-place** (already a copy).
            post_injection_mask: Boolean mask — True for all post-injection rows.
            fraud_mask: Boolean mask — True for all fraud rows (whole DataFrame).
        """
        # Eligible: post-injection AND fraud
        eligible_mask = post_injection_mask & fraud_mask
        eligible_indices = df.index[eligible_mask].tolist()

        n_flip = max(1, int(len(eligible_indices) * 0.20))
        rng = np.random.default_rng(seed=42)
        flip_indices = rng.choice(eligible_indices, size=n_flip, replace=False)

        df.loc[flip_indices, "isFraud"] = 0
        _logger.info(
            "DriftSimulator — 'label_flip': flipped %d / %d post-injection "
            "fraud labels to 0 (20%% of fraud rows only)",
            n_flip,
            len(eligible_indices),
        )
