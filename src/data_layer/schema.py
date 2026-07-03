"""
data_layer/schema.py
--------------------
Pydantic schema for validating raw IEEE-CIS transaction rows.

Validation is always non-blocking: violations are logged at WARNING level
and returned to the caller as a list.  The pipeline is never interrupted
by a schema failure — the intent is observability, not hard enforcement.

Usage::

    from src.data_layer.schema import validate_sample
    violations = validate_sample(df)   # logs and returns violation list
"""

import random
from typing import Annotated, Any

import pandas as pd
from pydantic import BaseModel, Field, ValidationError

from src.utils.logger import get_logger

_logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Schema definition
# ---------------------------------------------------------------------------

class TransactionSchema(BaseModel):
    """
    Pydantic v2 model for a single IEEE-CIS transaction row.

    Attributes:
        TransactionID:  Unique integer row key.
        isFraud:        Binary fraud label — must be exactly 0 or 1.
        TransactionDT:  Positive integer timestamp (seconds since reference).
        TransactionAmt: Positive float (transaction amount in USD).
    """

    TransactionID:  int
    isFraud:        Annotated[int,   Field(ge=0, le=1)]
    TransactionDT:  Annotated[int,   Field(gt=0)]
    TransactionAmt: Annotated[float, Field(gt=0.0)]


# ---------------------------------------------------------------------------
# Validation helper
# ---------------------------------------------------------------------------

def validate_sample(
    df: pd.DataFrame,
    n: int = 1000,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """
    Validate a random sample of rows from *df* against
    :class:`TransactionSchema`.

    The function never raises on schema violations.  Every failing row is
    logged at WARNING level and returned in the output list so callers can
    inspect issues without interrupting the pipeline.

    Args:
        df:   DataFrame to validate.  Must contain the four schema columns:
              ``TransactionID``, ``isFraud``, ``TransactionDT``,
              ``TransactionAmt``.  If any are absent, validation is skipped
              with a WARNING and an empty list is returned.
        n:    Number of rows to sample (default 1 000).  If *df* has fewer
              rows than *n*, the entire DataFrame is validated.
        seed: Random seed for reproducible sampling (default 42).

    Returns:
        List of violation dicts, one per failing row::

            [{"row_index": int, "error": str}, ...]

        Returns an empty list when all sampled rows pass validation.
    """
    required_cols = {"TransactionID", "isFraud", "TransactionDT", "TransactionAmt"}
    missing_cols  = required_cols - set(df.columns)
    if missing_cols:
        _logger.warning(
            f"Schema validation skipped — required columns absent: {missing_cols}"
        )
        return []

    sample_size = min(n, len(df))
    rng         = random.Random(seed)
    indices     = rng.sample(range(len(df)), sample_size)
    sample      = df.iloc[indices]

    _logger.info(
        f"Validating {sample_size:,} sampled rows against TransactionSchema ..."
    )

    violations: list[dict[str, Any]] = []

    for idx, row in sample.iterrows():
        try:
            TransactionSchema(
                TransactionID  = int(row["TransactionID"]),
                isFraud        = int(row["isFraud"]),
                TransactionDT  = int(row["TransactionDT"]),
                TransactionAmt = float(row["TransactionAmt"]),
            )
        except ValidationError as exc:
            first_error = exc.errors()[0]
            violation   = {
                "row_index": idx,
                "field":     str(first_error.get("loc", ["?"])[0]),
                "error":     first_error.get("msg", str(exc)),
                "value":     first_error.get("input", None),
            }
            violations.append(violation)
            _logger.warning(
                f"Schema violation | row {idx} | "
                f"field='{violation['field']}' | "
                f"value={violation['value']!r} | "
                f"{violation['error']}"
            )

    if violations:
        _logger.warning(
            f"Schema validation complete — "
            f"{len(violations)} violation(s) found in {sample_size:,} sampled rows"
        )
    else:
        _logger.info(
            f"Schema validation complete — all {sample_size:,} sampled rows passed"
        )

    return violations
