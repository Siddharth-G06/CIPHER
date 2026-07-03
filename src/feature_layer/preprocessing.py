"""
feature_layer/preprocessing.py
-------------------------------
Cleaning, encoding, and temporal-split pipeline for the IEEE-CIS dataset.

The original monolithic ``preprocess()`` function (data_preprocessing.py at the
project root) is decomposed here into two composable, config-driven steps:

1. :func:`clean_and_encode` — drop sparse columns, add missing-value
   indicators, fill sentinels, sort temporally, and label-encode categoricals.
   Encoder is **fit only on the first train_frac rows** to prevent leakage.

2. :func:`temporal_split` — slice the sorted DataFrame into train / test
   feature matrices and compute ``scale_pos_weight`` from the train portion.

Core logic is **identical** to the original file.  Only the structure,
config-driven parameters, and print → logger replacements differ.

Typical usage
-------------
::

    from src.utils.config_loader import load_config
    from src.data_layer.loader import load_raw
    from src.feature_layer.preprocessing import clean_and_encode, temporal_split

    cfg    = load_config()
    df_raw = load_raw(cfg["data"]["transaction_path"],
                      cfg["data"]["identity_path"])
    df, encoders = clean_and_encode(df_raw, cfg)
    X_train, X_test, y_train, y_test, features, spw = temporal_split(df, cfg)
"""

from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from src.utils.logger import get_logger

_logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Step 1 — Cleaning and encoding
# ---------------------------------------------------------------------------

def clean_and_encode(
    df: pd.DataFrame,
    cfg: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, LabelEncoder]]:
    """
    Drop sparse columns, add missing-value indicators, fill sentinels,
    sort temporally, and label-encode categorical columns.

    Processing steps (order is significant):

    1. Drop any column whose null rate exceeds
       ``cfg["preprocessing"]["missing_threshold"]``.
    2. For every remaining column that still has nulls, create a binary
       ``<col>_was_missing`` indicator column.
    3. Fill the original null values with
       ``cfg["preprocessing"]["sentinel_fill_value"]`` (-999 by default).
       LightGBM treats this as a distinct split point, so the indicator and
       the sentinel together carry the same signal as a two-branch split.
    4. Sort the DataFrame by ``TransactionDT`` (required so the encoder is
       fit on the chronologically correct training slice in step 5).
    5. Label-encode all ``object``-dtype columns, **fitting only on the first**
       ``train_frac * n`` rows to prevent data leakage from the test set.
       Any test-set category unseen during training is mapped to
       ``cfg["preprocessing"]["unseen_label_value"]`` (-1).

    Args:
        df:  Raw merged DataFrame from
             :func:`~src.data_layer.loader.load_raw`.
             Must contain a ``TransactionDT`` column for temporal ordering.
        cfg: Config dict from
             :func:`~src.utils.config_loader.load_config`.
             Required keys (all under ``"preprocessing"``):
               - ``missing_threshold``
               - ``train_frac``
               - ``sentinel_fill_value``
               - ``unseen_label_value``

    Returns:
        A tuple ``(df_clean, encoders)`` where:

        - ``df_clean`` is the cleaned, encoded, temporally-sorted DataFrame
          with index reset to ``0 … n-1``.
        - ``encoders`` maps each encoded column name to its fitted
          :class:`~sklearn.preprocessing.LabelEncoder` (trained on train
          rows only).

    Raises:
        KeyError: If a required config key is absent from *cfg*.
    """
    missing_threshold: float = cfg["preprocessing"]["missing_threshold"]
    sentinel:          int   = cfg["preprocessing"]["sentinel_fill_value"]
    unseen:            int   = cfg["preprocessing"]["unseen_label_value"]
    train_frac:        float = cfg["preprocessing"]["train_frac"]

    # ------------------------------------------------------------------
    # 1. Drop columns whose missing rate exceeds the threshold
    # ------------------------------------------------------------------
    _logger.info(
        f"[1/5] Dropping columns with >{missing_threshold * 100:.0f}% missing ..."
    )
    missing_rate = df.isnull().mean()
    cols_to_drop = missing_rate[missing_rate > missing_threshold].index.tolist()
    df = df.drop(columns=cols_to_drop)

    if cols_to_drop:
        _logger.info(
            f"      Dropped {len(cols_to_drop)} column(s): {cols_to_drop}"
        )
    else:
        _logger.info("      No columns exceeded the missing threshold.")
    _logger.info(f"      Remaining columns: {df.shape[1]}")

    # ------------------------------------------------------------------
    # 2. Missing-value indicator columns
    # ------------------------------------------------------------------
    _logger.info("[2/5] Adding _was_missing indicator columns ...")
    cols_with_nulls: list[str] = [c for c in df.columns if df[c].isnull().any()]
    indicator_df = pd.DataFrame(
        {f"{c}_was_missing": df[c].isnull().astype(np.int8) for c in cols_with_nulls},
        index=df.index,
    )
    df = pd.concat([df, indicator_df], axis=1)
    _logger.info(
        f"      Added {len(cols_with_nulls)} indicator column(s) "
        f"for: {cols_with_nulls[:5]}{'...' if len(cols_with_nulls) > 5 else ''}"
    )

    # ------------------------------------------------------------------
    # 3. Sentinel fill
    # ------------------------------------------------------------------
    _logger.info(f"[3/5] Filling remaining nulls with sentinel {sentinel} ...")
    df[cols_with_nulls] = df[cols_with_nulls].fillna(sentinel)

    # ------------------------------------------------------------------
    # 4. Sort by TransactionDT
    # ------------------------------------------------------------------
    _logger.info("[4/5] Sorting by TransactionDT ...")
    df = df.sort_values("TransactionDT").reset_index(drop=True)

    train_cutoff: int = int(len(df) * train_frac)
    _logger.info(
        f"      Train boundary: first {train_cutoff:,} rows "
        f"({train_frac * 100:.0f}% of {len(df):,})"
    )

    # ------------------------------------------------------------------
    # 5. Label-encode categorical (object) columns
    #    Fit only on the training slice to prevent leakage.
    # ------------------------------------------------------------------
    cat_cols: list[str] = [
        c for c in df.select_dtypes(include=["object", "string"]).columns
        if c not in ("isFraud",)
    ]
    _logger.info(
        f"[5/5] Label-encoding {len(cat_cols)} categorical column(s) "
        "(fit on train rows only) ..."
    )
    encoders: dict[str, LabelEncoder] = {}

    for col in cat_cols:
        le = LabelEncoder()
        train_vals = df[col].iloc[:train_cutoff].astype(str)
        le.fit(train_vals)
        encoders[col] = le

        full_vals = df[col].astype(str)
        label_map = {label: idx for idx, label in enumerate(le.classes_)}
        df[col]   = full_vals.map(label_map).fillna(unseen).astype(int)

    _logger.info(
        f"clean_and_encode complete — shape: {df.shape[0]:,} rows x {df.shape[1]} columns"
    )
    return df, encoders


# ---------------------------------------------------------------------------
# Step 2 — Temporal split
# ---------------------------------------------------------------------------

def temporal_split(
    df: pd.DataFrame,
    cfg: dict[str, Any],
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
    pd.Series,
    list[str],
    float,
]:
    """
    Temporally split the encoded DataFrame into train and test feature matrices.

    The split is **positional** (no shuffle) on a DataFrame that must already
    be sorted by ``TransactionDT`` (as produced by :func:`clean_and_encode`).
    ``scale_pos_weight`` is computed from the **train portion only** to avoid
    leaking test-set class distribution into training hyper-parameters.

    Args:
        df:  Cleaned, encoded, temporally-sorted DataFrame (output of
             :func:`clean_and_encode`).
             Must contain ``isFraud``, ``TransactionDT``, and ``TransactionID``.
        cfg: Config dict from
             :func:`~src.utils.config_loader.load_config`.
             Required key: ``preprocessing.train_frac``.

    Returns:
        A tuple of six items:

        - ``X_train`` — Training feature :class:`~pandas.DataFrame` (index reset).
        - ``X_test``  — Test feature :class:`~pandas.DataFrame` (index reset).
        - ``y_train`` — Training labels :class:`~pandas.Series` (index reset).
        - ``y_test``  — Test labels :class:`~pandas.Series` (index reset).
        - ``feature_names`` — :class:`list` of feature column names
          (excludes ``TransactionID``, ``TransactionDT``, ``isFraud``).
        - ``scale_pos_weight`` — ``n_neg / n_pos`` from the training set,
          for use as LightGBM's ``scale_pos_weight`` parameter.

    Raises:
        KeyError: If ``isFraud`` column is absent from *df*.
    """
    train_frac:    float = cfg["preprocessing"]["train_frac"]
    train_cutoff:  int   = int(len(df) * train_frac)

    train_df = df.iloc[:train_cutoff].copy()
    test_df  = df.iloc[train_cutoff:].copy()

    _logger.info("[6/7] Temporal split:")
    _logger.info(f"      Train rows : {len(train_df):,}")
    _logger.info(f"      Test  rows : {len(test_df):,}")
    _logger.info(
        f"      Train TransactionDT : "
        f"{train_df['TransactionDT'].min()} → {train_df['TransactionDT'].max()}"
    )
    _logger.info(
        f"      Test  TransactionDT : "
        f"{test_df['TransactionDT'].min()} → {test_df['TransactionDT'].max()}"
    )

    # scale_pos_weight — train portion only
    n_neg: int   = int((train_df["isFraud"] == 0).sum())
    n_pos: int   = int((train_df["isFraud"] == 1).sum())
    scale_pos_weight: float = n_neg / n_pos

    _logger.info("[6/7] scale_pos_weight:")
    _logger.info(f"      Train non-fraud : {n_neg:,}")
    _logger.info(f"      Train fraud     : {n_pos:,}")
    _logger.info(f"      scale_pos_weight = {scale_pos_weight:.4f}")

    cols_to_exclude: set[str] = {"TransactionID", "TransactionDT", "isFraud"}
    feature_names: list[str]  = [
        c for c in train_df.columns if c not in cols_to_exclude
    ]
    _logger.info(f"[7/7] Feature count: {len(feature_names)}")

    X_train = train_df[feature_names].reset_index(drop=True)
    X_test  = test_df[feature_names].reset_index(drop=True)
    y_train = train_df["isFraud"].reset_index(drop=True)
    y_test  = test_df["isFraud"].reset_index(drop=True)

    _logger.info("temporal_split complete.")
    return X_train, X_test, y_train, y_test, feature_names, scale_pos_weight


# ---------------------------------------------------------------------------
# Quick-run block
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os

    from src.utils.config_loader import load_config
    from src.data_layer.loader import load_raw

    _cfg = load_config()
    _df_raw = load_raw(
        _cfg["data"]["transaction_path"],
        _cfg["data"]["identity_path"],
    )
    _df, _encoders = clean_and_encode(_df_raw, _cfg)
    _X_tr, _X_te, _y_tr, _y_te, _feats, _spw = temporal_split(_df, _cfg)

    _logger.info("-- Output shapes --")
    _logger.info(f"  X_train : {_X_tr.shape}")
    _logger.info(f"  X_test  : {_X_te.shape}")
    _logger.info(f"  Features: {len(_feats)}")
    _logger.info(f"  scale_pos_weight: {_spw:.4f}")
