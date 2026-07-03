"""
data_layer/loader.py
--------------------
Raw data ingestion for the IEEE-CIS Fraud Detection dataset.

Responsibility
--------------
Load the two CSV files and left-join them on ``TransactionID``.
No cleaning, encoding, or feature engineering happens here — this module
only handles I/O so that the rest of the pipeline receives a single
merged DataFrame.
"""

from pathlib import Path
from typing import Optional

import pandas as pd

from src.utils.logger import get_logger

_logger = get_logger(__name__)


def load_raw(
    transaction_path: str,
    identity_path: str,
) -> pd.DataFrame:
    """
    Load ``train_transaction.csv`` and ``train_identity.csv``, then
    left-join on ``TransactionID``.

    The join is a left join so every transaction row is preserved even when
    no corresponding identity record exists (identity columns will be NaN
    for those rows).

    Args:
        transaction_path: Path to ``train_transaction.csv``.
        identity_path:    Path to ``train_identity.csv``.

    Returns:
        Merged :class:`~pandas.DataFrame` containing all transaction columns
        plus identity columns (NaN where no identity record matched).
        Shape: ``(n_transactions, n_transaction_cols + n_identity_cols - 1)``.

    Raises:
        FileNotFoundError: If either CSV does not exist at the given path.
    """
    t_path = Path(transaction_path)
    i_path = Path(identity_path)

    if not t_path.exists():
        _logger.error(f"Transaction file not found: {t_path.resolve()}")
        raise FileNotFoundError(
            f"Transaction file not found: {t_path.resolve()}"
        )
    if not i_path.exists():
        _logger.error(f"Identity file not found: {i_path.resolve()}")
        raise FileNotFoundError(
            f"Identity file not found: {i_path.resolve()}"
        )

    _logger.info(f"Reading transactions : {t_path}")
    transactions = pd.read_csv(transaction_path)
    _logger.info(
        f"Transactions loaded  : {len(transactions):,} rows, "
        f"{transactions.shape[1]} columns"
    )

    _logger.info(f"Reading identity     : {i_path}")
    identity = pd.read_csv(identity_path)
    _logger.info(
        f"Identity loaded      : {len(identity):,} rows, "
        f"{identity.shape[1]} columns"
    )

    df = transactions.merge(identity, on="TransactionID", how="left")
    _logger.info(
        f"Merged (left join)   : {df.shape[0]:,} rows x {df.shape[1]} columns"
    )
    return df
