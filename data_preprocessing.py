"""
data_preprocessing.py
---------------------
Preprocessing pipeline for the IEEE-CIS Fraud Detection dataset.

Pipeline overview:
  1. Load & merge train_transaction + train_identity on TransactionID (left join)
  2. Drop columns whose missing-value rate exceeds 90 %
  3. For every remaining column that still has nulls:
       - add a binary `<col>_was_missing` indicator
       - fill nulls with -999  (LightGBM treats -999 as a sentinel natively)
  4. Label-encode all object / string columns
  5. Temporal train/test split: sort by TransactionDT, first 80 % -> train,
     last 20 % -> test  (no shuffle - preserves time ordering)
  6. Compute scale_pos_weight from the TRAIN portion only
  7. Drop TransactionID and TransactionDT from features
  8. Return X_train, X_test, y_train, y_test, feature_names, scale_pos_weight
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


# ---------------------------------------------------------------------------
# Main preprocessing function
# ---------------------------------------------------------------------------

def preprocess(
    transaction_path: str,
    identity_path: str,
    missing_threshold: float = 0.90,
    train_frac: float = 0.80,
) -> tuple:
    """
    Load, clean, encode, and split the IEEE-CIS fraud dataset.

    Parameters
    ----------
    transaction_path : str
        Path to train_transaction.csv
    identity_path : str
        Path to train_identity.csv
    missing_threshold : float
        Drop columns with a missing rate strictly above this value (default 0.90)
    train_frac : float
        Fraction of rows (sorted by TransactionDT) to use as training data

    Returns
    -------
    X_train, X_test : pd.DataFrame
    y_train, y_test : pd.Series
    feature_names   : list[str]
    scale_pos_weight : float
    """

    # ------------------------------------------------------------------
    # 1. Load CSVs and left-join on TransactionID
    # ------------------------------------------------------------------
    print("[1/7] Loading data ...")
    transactions = pd.read_csv(transaction_path)
    identity     = pd.read_csv(identity_path)

    # Left join so every transaction row is kept even if it has no identity record
    df = transactions.merge(identity, on="TransactionID", how="left")
    print(f"      Merged shape: {df.shape}")

    # ------------------------------------------------------------------
    # 2. Drop columns whose missing rate exceeds the threshold
    # ------------------------------------------------------------------
    print(f"[2/7] Dropping columns with >{missing_threshold*100:.0f}% missing ...")
    missing_rate = df.isnull().mean()          # fraction of NaNs per column
    cols_to_drop = missing_rate[missing_rate > missing_threshold].index.tolist()
    df.drop(columns=cols_to_drop, inplace=True)
    print(f"      Dropped {len(cols_to_drop)} columns  |  Remaining: {df.shape[1]}")

    # ------------------------------------------------------------------
    # 3. Missing-value indicators + sentinel fill
    # ------------------------------------------------------------------
    print("[3/7] Adding _was_missing indicators and filling nulls with -999 ...")
    cols_with_nulls = [c for c in df.columns if df[c].isnull().any()]

    # Build all indicator columns first to avoid modifying df while iterating
    indicator_df = pd.DataFrame(
        {f"{c}_was_missing": df[c].isnull().astype(np.int8) for c in cols_with_nulls},
        index=df.index,
    )
    df = pd.concat([df, indicator_df], axis=1)

    # Now fill the original columns with the sentinel value -999.
    # LightGBM can learn from this sentinel as a distinct split point,
    # allowing it to treat "was missing" as an informative signal.
    df[cols_with_nulls] = df[cols_with_nulls].fillna(-999)
    print(f"      Added {len(cols_with_nulls)} indicator columns")

    # ------------------------------------------------------------------
    # 4. Label-encode object/string columns
    #    We fit ONLY on the training rows (first 80 % after temporal sort)
    #    to avoid data leakage from the test portion.
    # ------------------------------------------------------------------
    print("[4/7] Sorting by TransactionDT to prepare temporal split ...")

    # Sort now so we can correctly define the train boundary before encoding
    df.sort_values("TransactionDT", inplace=True)
    df.reset_index(drop=True, inplace=True)

    train_cutoff = int(len(df) * train_frac)   # index of first test row

    # Identify all categorical (object) columns - exclude the target
    cat_cols = [
        c for c in df.select_dtypes(include="object").columns
        if c not in ("isFraud",)
    ]

    print(f"      Label-encoding {len(cat_cols)} categorical columns ...")
    encoders: dict = {}

    for col in cat_cols:
        le = LabelEncoder()

        # Cast to str so mixed-type columns are handled cleanly;
        # -999 sentinel was already applied to numeric cols, not object cols
        train_vals = df[col].iloc[:train_cutoff].astype(str)
        le.fit(train_vals)
        encoders[col] = le

        # Transform the full column.
        # Any label in the test set that was unseen during training is mapped to -1,
        # which LightGBM can handle safely as an out-of-vocabulary category.
        full_vals = df[col].astype(str)
        label_map = {label: idx for idx, label in enumerate(le.classes_)}
        df[col] = full_vals.map(label_map).fillna(-1).astype(int)

    # ------------------------------------------------------------------
    # 5. Temporal train/test split  (already sorted above in step 4)
    # ------------------------------------------------------------------
    print("[5/7] Splitting into train / test (temporal, no shuffle) ...")
    train_df = df.iloc[:train_cutoff].copy()
    test_df  = df.iloc[train_cutoff:].copy()

    # ------------------------------------------------------------------
    # 6. Compute scale_pos_weight from TRAIN only
    #    Formula: (# negative examples) / (# positive examples)
    #    This compensates for the heavy class imbalance typical in fraud data;
    #    the value is passed directly to LightGBM's `scale_pos_weight` param.
    # ------------------------------------------------------------------
    print("[6/7] Computing scale_pos_weight ...")
    n_neg = (train_df["isFraud"] == 0).sum()   # non-fraud rows in train
    n_pos = (train_df["isFraud"] == 1).sum()   # fraud rows in train
    scale_pos_weight = n_neg / n_pos
    print(f"      Train non-fraud: {n_neg:,}  |  Train fraud: {n_pos:,}")
    print(f"      scale_pos_weight = {scale_pos_weight:.4f}")

    # ------------------------------------------------------------------
    # 7. Drop TransactionID and TransactionDT from features
    #    - TransactionID is a meaningless row key (not a predictive signal)
    #    - TransactionDT was used only for ordering and must not leak into features
    # ------------------------------------------------------------------
    print("[7/7] Dropping TransactionID and TransactionDT from feature set ...")
    cols_to_exclude = {"TransactionID", "TransactionDT", "isFraud"}

    feature_names = [c for c in train_df.columns if c not in cols_to_exclude]

    X_train = train_df[feature_names].reset_index(drop=True)
    X_test  = test_df[feature_names].reset_index(drop=True)
    y_train = train_df["isFraud"].reset_index(drop=True)
    y_test  = test_df["isFraud"].reset_index(drop=True)

    print("\n[OK] Preprocessing complete.")
    return X_train, X_test, y_train, y_test, feature_names, scale_pos_weight


# ---------------------------------------------------------------------------
# Quick-run block for local testing
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os

    # Expects CSV files at <project_root>/data/
    DATA_DIR         = os.path.join(os.path.dirname(__file__), "data")
    TRANSACTION_PATH = os.path.join(DATA_DIR, "train_transaction.csv")
    IDENTITY_PATH    = os.path.join(DATA_DIR, "train_identity.csv")

    X_train, X_test, y_train, y_test, feature_names, scale_pos_weight = preprocess(
        transaction_path=TRANSACTION_PATH,
        identity_path=IDENTITY_PATH,
    )

    print("\n-- Output shapes ------------------------------------------")
    print(f"  X_train : {X_train.shape}")
    print(f"  X_test  : {X_test.shape}")
    print(f"  y_train : {y_train.shape}")
    print(f"  y_test  : {y_test.shape}")
    print(f"\n  Number of features : {len(feature_names)}")
    print(f"  scale_pos_weight   : {scale_pos_weight:.4f}")
    print("-----------------------------------------------------------")
