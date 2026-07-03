"""
tests/test_preprocessing.py
----------------------------
Unit tests for src/feature_layer/preprocessing.py

Tests target three potential bugs in the cleaning / encoding / split logic:

1. test_no_temporal_leakage_in_split
   Verifies that the temporal split is clean: max(train DT) <= min(test DT).
   Uses <= because tied timestamps at the boundary are valid behaviour.

2. test_label_encoder_fit_only_on_train
   Injects an unseen category "UNSEEN_Z" exclusively into test rows, then
   verifies it maps to the unseen_label_value sentinel (-1) and is NOT
   present among the encoder's training classes.

3. test_missing_indicator_columns_exist
   Verifies that _was_missing indicator columns are created for columns
   that had nulls (and were kept because their missing rate is < 0.90),
   and that fully-populated columns do NOT receive an indicator.

All tests use small synthetic DataFrames — no real CSV files are needed.
"""

import numpy as np
import pandas as pd
import pytest

from src.feature_layer.preprocessing import clean_and_encode, temporal_split

# ---------------------------------------------------------------------------
# Shared config matching config/config.yaml defaults
# ---------------------------------------------------------------------------

DEFAULT_CFG: dict = {
    "preprocessing": {
        "missing_threshold":    0.90,
        "train_frac":           0.80,
        "sentinel_fill_value": -999,
        "unseen_label_value":   -1,
    }
}


# ---------------------------------------------------------------------------
# Synthetic data factory
# ---------------------------------------------------------------------------

def _make_df(n: int = 100, seed: int = 42) -> pd.DataFrame:
    """
    Build a minimal synthetic transaction DataFrame.

    TransactionDT increases monotonically so any sort preserves the original
    row order, making expected train/test boundaries predictable in tests.
    """
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "TransactionID": np.arange(n),
            "TransactionDT": np.arange(n) * 100 + 1_000,  # 1000, 1100, …
            "TransactionAmt": rng.uniform(10.0, 1_000.0, n),
            "isFraud":        (rng.random(n) > 0.96).astype(int),
            "ProductCD":      rng.choice(["W", "H", "C", "S", "R"], n).tolist(),
            "card1":          rng.integers(1_000, 9_999, n),
        }
    )


# ---------------------------------------------------------------------------
# Test 1 — No temporal leakage in split
# ---------------------------------------------------------------------------

def test_no_temporal_leakage_in_split() -> None:
    """
    max(TransactionDT in train) must be <= min(TransactionDT in test).

    Note: <= rather than < because the current implementation slices by
    positional index (iloc), so tied DT values at the boundary are split
    across partitions without introducing leakage — the train portion
    simply cannot see rows that come after the cutoff position.
    """
    df = _make_df(100)
    df_clean, _ = clean_and_encode(df.copy(), DEFAULT_CFG)

    train_cutoff = int(len(df_clean) * DEFAULT_CFG["preprocessing"]["train_frac"])

    train_dt_max = df_clean["TransactionDT"].iloc[:train_cutoff].max()
    test_dt_min  = df_clean["TransactionDT"].iloc[train_cutoff:].min()

    assert train_dt_max <= test_dt_min, (
        f"Temporal leakage detected!\n"
        f"  max(train TransactionDT) = {train_dt_max}\n"
        f"  min(test  TransactionDT) = {test_dt_min}\n"
        "  Train timestamps leaked past the split boundary."
    )


# ---------------------------------------------------------------------------
# Test 2 — Label encoder fit only on train
# ---------------------------------------------------------------------------

def test_label_encoder_fit_only_on_train() -> None:
    """
    A category seen only in the test portion must map to unseen_label_value (-1).

    We inject "UNSEEN_Z" into exactly the test rows (rows >= train_cutoff in
    the sorted DataFrame), then verify:
      (a) "UNSEEN_Z" is absent from the fitted LabelEncoder's classes_.
      (b) Every test row's encoded ProductCD value equals -1.
    """
    df = _make_df(100)

    # Sort first so we can precisely target the test slice before calling
    # clean_and_encode (which re-sorts internally, but the DTs are already
    # monotonically increasing so the sort is a no-op).
    df = df.sort_values("TransactionDT").reset_index(drop=True)
    test_start = int(100 * DEFAULT_CFG["preprocessing"]["train_frac"])  # = 80
    df.loc[test_start:, "ProductCD"] = "UNSEEN_Z"

    df_clean, encoders = clean_and_encode(df.copy(), DEFAULT_CFG)

    le           = encoders["ProductCD"]
    unseen_val   = DEFAULT_CFG["preprocessing"]["unseen_label_value"]
    train_cutoff = int(len(df_clean) * DEFAULT_CFG["preprocessing"]["train_frac"])
    test_encoded = df_clean["ProductCD"].iloc[train_cutoff:]

    # (a) The encoder must not have seen "UNSEEN_Z" during fit
    assert "UNSEEN_Z" not in le.classes_, (
        f"'UNSEEN_Z' was found in training classes: {le.classes_}\n"
        "The encoder was fit on test data — data leakage!"
    )

    # (b) All test rows that had "UNSEEN_Z" must be the unseen sentinel
    assert (test_encoded == unseen_val).all(), (
        f"Expected all test rows to encode to {unseen_val}.\n"
        f"Actual unique values found: {sorted(test_encoded.unique())}"
    )


# ---------------------------------------------------------------------------
# Test 3 — Missing indicator columns exist
# ---------------------------------------------------------------------------

def test_missing_indicator_columns_exist() -> None:
    """
    _was_missing indicators must be created for retained null columns;
    fully-populated columns must NOT receive an indicator.

    Column setup
    ------------
    colA : 50 % null  → missing rate 0.50 < 0.90  → kept, gets indicator
    colB : 80 % null  → missing rate 0.80 < 0.90  → kept, gets indicator
    colC :  0 % null  → no nulls                   → kept, no indicator

    (A column at 100 % null would be dropped before indicators are added;
    that case is not tested here because the intent is to verify indicator
    creation for retained columns.)
    """
    df = _make_df(100)

    # colA: first 50 rows null (50 % missing rate)
    df["colA"] = np.where(df.index < 50, np.nan, 1.0)
    # colB: first 80 rows null (80 % missing rate)
    df["colB"] = np.where(df.index < 80, np.nan, 2.0)
    # colC: fully populated
    df["colC"] = 42.0

    df_clean, _ = clean_and_encode(df.copy(), DEFAULT_CFG)

    assert "colA_was_missing" in df_clean.columns, (
        "colA_was_missing not found. "
        "Expected an indicator for a 50%-null column below the drop threshold."
    )
    assert "colB_was_missing" in df_clean.columns, (
        "colB_was_missing not found. "
        "Expected an indicator for an 80%-null column below the drop threshold."
    )
    assert "colC_was_missing" not in df_clean.columns, (
        "colC_was_missing should NOT exist for a fully-populated column."
    )

    # Sanity check: indicator values match original null positions.
    # (colA had 50 nulls; sort by DT does not change the count.)
    assert df_clean["colA_was_missing"].sum() == 50, (
        f"colA_was_missing sum should be 50, got {df_clean['colA_was_missing'].sum()}"
    )
    assert df_clean["colB_was_missing"].sum() == 80, (
        f"colB_was_missing sum should be 80, got {df_clean['colB_was_missing'].sum()}"
    )
