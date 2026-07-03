"""
tests/test_graph_features.py
-----------------------------
Unit tests for src/feature_layer/graph_features.py

Tests target two high-risk properties:

1. test_graph_features_no_future_leakage  ← HIGHEST RISK
   Verifies that card_degree_1h counts only transactions with
   TransactionDT STRICTLY LESS THAN the current row's TransactionDT.
   Hand-computed expected values are compared row by row; any mismatch
   prints a "LEAKAGE DETECTED" message with the row index and values
   before the assertion fails.

2. test_amt_zscore_handles_single_transaction
   Verifies that amt_zscore_24h returns exactly 0.0 — never NaN or inf —
   when the card's past window contains one transaction (std = 0) or
   multiple transactions with identical amounts (std = 0).

All tests use small deterministic DataFrames — no real CSV files needed.
"""

import numpy as np
import pandas as pd
import pytest

from src.feature_layer.graph_features import add_graph_features

# ---------------------------------------------------------------------------
# Shared test config
# ---------------------------------------------------------------------------

_CFG: dict = {
    "graph_features": {
        "short_window_seconds":  3_600,
        "long_window_seconds":  86_400,
        "graph_sample_size":     1_000,
    }
}


# ---------------------------------------------------------------------------
# Test 1 — No future leakage in card_degree_1h
# ---------------------------------------------------------------------------

def test_graph_features_no_future_leakage() -> None:
    """
    card_degree_1h must reflect ONLY past transactions (dt < current_dt).

    Synthetic setup
    ---------------
    One card (card1=42), five transactions at known timestamps, visiting
    three distinct merchants in sequence:

        idx  dt    merchant
        0    1000  W
        1    2000  H
        2    3000  W
        3    4000  C
        4    5000  W

    Window = 3 600 s.

    Hand-computed expected card_degree_1h
    -------------------------------------
    Row 0 (dt=1000):
        cutoff = 1000-3600 = -2600
        past entries with dt > -2600 AND dt < 1000 : none
        unique merchants = {}  → 0

    Row 1 (dt=2000):
        cutoff = 2000-3600 = -1600
        past: [(1000,'W')]  (1000 > -1600 ✓, 1000 < 2000 ✓)
        unique = {'W'}  → 1

    Row 2 (dt=3000):
        cutoff = 3000-3600 = -600
        past: [(1000,'W'), (2000,'H')]
        unique = {'W','H'}  → 2

    Row 3 (dt=4000):
        cutoff = 4000-3600 = 400
        past: [(1000,'W'), (2000,'H'), (3000,'W')]  (1000 > 400 ✓)
        unique = {'W','H'}  → 2

    Row 4 (dt=5000):
        cutoff = 5000-3600 = 1400
        evict: 1000 <= 1400  → evicted
        past: [(2000,'H'), (3000,'W'), (4000,'C')]
        unique = {'H','W','C'}  → 3

    Expected: [0, 1, 2, 2, 3]

    Implementation correctness note
    --------------------------------
    The deque eviction condition is ``dq[0][0] <= cutoff`` which removes
    entries at exactly ``dt - window``.  The current row is appended AFTER
    reading.  Together these guarantee the window is the open interval
    ``(dt - window, dt)`` — strictly past, strictly before current.
    """
    df = pd.DataFrame(
        {
            "TransactionID": [1, 2, 3, 4, 5],
            "TransactionDT": [1_000, 2_000, 3_000, 4_000, 5_000],
            "TransactionAmt": [100.0, 200.0, 150.0, 300.0, 250.0],
            "card1":    [42, 42, 42, 42, 42],
            "ProductCD": ["W", "H", "W", "C", "W"],
        }
    )

    expected_degree: list[int] = [0, 1, 2, 2, 3]

    result = add_graph_features(df.copy(), _CFG)
    computed: list[int] = result["card_degree_1h"].tolist()

    leakage_found: bool = False
    for i, (exp, got) in enumerate(zip(expected_degree, computed)):
        if exp != got:
            leakage_found = True
            print(
                f"\nLEAKAGE DETECTED at row {i}: "
                f"expected={exp}, got={got} "
                f"| TransactionDT={result['TransactionDT'].iloc[i]} "
                f"| window={_CFG['graph_features']['short_window_seconds']}s"
            )

    assert not leakage_found, (
        f"card_degree_1h values do not match hand-computed past-only values.\n"
        f"Expected : {expected_degree}\n"
        f"Got      : {computed}\n"
        "See 'LEAKAGE DETECTED' messages above for per-row details."
    )

    # Explicit boundary check: row 0 must have no history
    first_val = result["card_degree_1h"].iloc[0]
    assert first_val == 0, (
        f"Row 0 (first transaction) card_degree_1h must be 0, got {first_val}.\n"
        "This indicates the current transaction is being counted in its own window."
    )

    # Verify row 4 eviction: t=1000 must NOT be counted (1000 <= cutoff=1400)
    last_val = result["card_degree_1h"].iloc[4]
    assert last_val == 3, (
        f"Row 4 card_degree_1h should be 3 (eviction of t=1000 expected), got {last_val}.\n"
        "t=1000 should have been evicted because 1000 <= (5000 - 3600) = 1400."
    )


# ---------------------------------------------------------------------------
# Test 2 — amt_zscore_24h handles std == 0 safely
# ---------------------------------------------------------------------------

def test_amt_zscore_handles_single_transaction() -> None:
    """
    amt_zscore_24h must return exactly 0.0 — never NaN, never inf — when
    the card's past-24h window has std == 0.

    Three rows for card 99 within the 24-hour window:

        idx  dt    amt     past window when computing
        0    1000  100.0   empty → zscore = 0  (no history)
        1    2000  100.0   [100.0]  → std=0 → zscore = 0  (single past tx)
        2    3000  200.0   [100.0, 100.0] → std=0 → zscore = 0  (identical past)

    In all three cases dividing by std would be a division by zero.
    The implementation must guard this with an explicit ``if std > 0`` check.
    """
    df = pd.DataFrame(
        {
            "TransactionID": [1, 2, 3],
            "TransactionDT": [1_000, 2_000, 3_000],
            "TransactionAmt": [100.0, 100.0, 200.0],
            "card1":    [99, 99, 99],
            "ProductCD": ["W", "W", "W"],
        }
    )

    result = add_graph_features(df.copy(), _CFG)
    zscores: list[float] = result["amt_zscore_24h"].tolist()

    # ---- NaN / inf guard (applies to all three rows) -------------------
    for i, val in enumerate(zscores):
        assert not np.isnan(val), (
            f"Row {i}: amt_zscore_24h is NaN — "
            "std=0 division guard not implemented correctly."
        )
        assert not np.isinf(val), (
            f"Row {i}: amt_zscore_24h is inf — "
            "std=0 division guard not implemented correctly."
        )

    # ---- Row-level correctness -----------------------------------------
    assert zscores[0] == 0.0, (
        f"Row 0 (empty history): expected 0.0, got {zscores[0]}"
    )
    assert zscores[1] == 0.0, (
        f"Row 1 (one past tx, std=0): expected 0.0, got {zscores[1]}\n"
        "With a single past transaction std is 0 — result must be 0, not NaN/inf."
    )
    assert zscores[2] == 0.0, (
        f"Row 2 (two identical past amounts, std=0): expected 0.0, got {zscores[2]}"
    )
