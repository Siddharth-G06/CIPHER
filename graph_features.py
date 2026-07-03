"""
graph_features.py
-----------------
Graph-based feature engineering for the IEEE-CIS Fraud Detection dataset.

Input
-----
  The merged and preprocessed DataFrame produced by data_preprocessing.py,
  BEFORE the train/test split — i.e. the full DataFrame sorted by TransactionDT
  that still contains the columns:
      TransactionDT   – Unix-like timestamp (seconds)
      TransactionAmt  – raw transaction amount (before any transformation)
      card1           – card identifier (used as-is, even if label-encoded)
      ProductCD       – merchant proxy  (used as-is, even if label-encoded)
      isFraud         – target label (optional; used only for correlation stats)

Output
------
  The same DataFrame with 6 new columns:
      card_degree_1h          – # unique merchants this card hit in past 3600 s
      merchant_degree_1h      – # unique cards  that hit this merchant in past 3600 s
      card_tx_count_24h       – # transactions for this card in past 86400 s
      card_total_amt_24h      – total TransactionAmt for this card in past 86400 s
      amt_zscore_24h          – z-score of current amount vs past 86400 s history
                                (0 when std == 0 / no history)
      card_merchant_tx_count_24h – # times exact card-merchant pair appeared in past 86400 s

Implementation notes
--------------------
  * All features are computed with strict past-only windows (no lookahead).
  * A time-based sliding-window approach is used:
      - For count/sum features we store per-group deque structures keyed on
        (card1) or (card1, ProductCD) or (ProductCD), each entry being a
        (TransactionDT, value) tuple.
      - We iterate once through the sorted DataFrame, so complexity is O(n*k)
        where k is the average window size -- far cheaper than O(n^2).
  * NetworkX is used ONLY for a sample graph visualization (1 000 rows).
    Node prefixes: 'c_' for card nodes, 'm_' for merchant nodes.
  * NaN graph features are filled with 0.
"""

import os
import warnings
from collections import defaultdict, deque

import networkx as nx
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Window constants (seconds)
# ---------------------------------------------------------------------------
_1H  = 3_600      # 1 hour
_24H = 86_400     # 24 hours


# ---------------------------------------------------------------------------
# Core feature-engineering function
# ---------------------------------------------------------------------------

def add_graph_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add 6 graph-inspired, time-windowed features to *df* using only past
    transactions for each row (strict no-lookahead).

    Parameters
    ----------
    df : pd.DataFrame
        Full merged+preprocessed DataFrame (pre-split) that contains at minimum:
            TransactionDT, TransactionAmt, card1, ProductCD

    Returns
    -------
    pd.DataFrame
        Original DataFrame with 6 new feature columns appended.
        The DataFrame index is reset and sorted by TransactionDT.
    """

    # -----------------------------------------------------------------------
    # 0. Enforce sort order and reset index so iloc == positional index
    # -----------------------------------------------------------------------
    df = df.sort_values("TransactionDT").reset_index(drop=True)
    n  = len(df)

    # Pull raw numpy arrays for speed inside the hot loop
    dt_arr   = df["TransactionDT"].to_numpy(dtype=np.int64)
    amt_arr  = df["TransactionAmt"].to_numpy(dtype=np.float64)
    card_arr = df["card1"].to_numpy()          # may be int or str after encoding
    prod_arr = df["ProductCD"].to_numpy()      # same

    # -----------------------------------------------------------------------
    # 1. Pre-allocate output arrays
    # -----------------------------------------------------------------------
    card_degree_1h             = np.zeros(n, dtype=np.int32)
    merchant_degree_1h         = np.zeros(n, dtype=np.int32)
    card_tx_count_24h          = np.zeros(n, dtype=np.int32)
    card_total_amt_24h         = np.zeros(n, dtype=np.float64)
    amt_zscore_24h             = np.zeros(n, dtype=np.float64)
    card_merchant_tx_count_24h = np.zeros(n, dtype=np.int32)

    # -----------------------------------------------------------------------
    # 2. Sliding-window data structures
    #
    #    For each "group key" we keep a deque of (TransactionDT, extra_value)
    #    tuples sorted ascending by TransactionDT (because we process rows in
    #    that order).  Before reading, we pop expired entries from the left.
    #
    #    Structures:
    #      card_1h_merchants[card]     -> deque of (dt, merchant)   -- for card_degree_1h
    #      merchant_1h_cards[merchant] -> deque of (dt, card)       -- for merchant_degree_1h
    #      card_24h_txs[card]          -> deque of (dt, amount)     -- for count, sum, zscore
    #      pair_24h[(card, merchant)]  -> deque of dt values        -- for pair count
    # -----------------------------------------------------------------------
    card_1h_merchants = defaultdict(deque)
    merchant_1h_cards = defaultdict(deque)
    card_24h_txs      = defaultdict(deque)
    pair_24h          = defaultdict(deque)

    # -----------------------------------------------------------------------
    # 3. Main loop -- O(n * average_window_size)
    # -----------------------------------------------------------------------
    for i in range(n):
        dt      = int(dt_arr[i])
        amt     = float(amt_arr[i])
        card    = card_arr[i]
        merch   = prod_arr[i]
        pair    = (card, merch)

        cutoff_1h  = dt - _1H
        cutoff_24h = dt - _24H

        # -------------------------------------------------------------------
        # Feature 1: card_degree_1h
        #   Unique merchants this card visited in (dt-3600, dt)  -- PAST only
        # -------------------------------------------------------------------
        dq = card_1h_merchants[card]
        # Evict expired entries
        while dq and dq[0][0] <= cutoff_1h:
            dq.popleft()
        # Count unique merchants currently in window
        card_degree_1h[i] = len({entry[1] for entry in dq})

        # -------------------------------------------------------------------
        # Feature 2: merchant_degree_1h
        #   Unique cards that hit this merchant in (dt-3600, dt)
        # -------------------------------------------------------------------
        dq2 = merchant_1h_cards[merch]
        while dq2 and dq2[0][0] <= cutoff_1h:
            dq2.popleft()
        merchant_degree_1h[i] = len({entry[1] for entry in dq2})

        # -------------------------------------------------------------------
        # Features 3, 4, 5: card_tx_count_24h / card_total_amt_24h / amt_zscore_24h
        #   All based on card's past 86400 s history
        # -------------------------------------------------------------------
        dq3 = card_24h_txs[card]
        while dq3 and dq3[0][0] <= cutoff_24h:
            dq3.popleft()

        if dq3:
            past_amounts = np.array([e[1] for e in dq3], dtype=np.float64)
            count_24h    = len(past_amounts)
            total_24h    = float(past_amounts.sum())
            mean_24h     = float(past_amounts.mean())
            std_24h      = float(past_amounts.std())

            card_tx_count_24h[i]  = count_24h
            card_total_amt_24h[i] = total_24h

            # z-score: 0 when std==0 (all past amounts identical / only 1 past tx)
            if std_24h > 0.0:
                amt_zscore_24h[i] = (amt - mean_24h) / std_24h
            else:
                amt_zscore_24h[i] = 0.0
        # else: all three stay at 0 (no past history) -- already initialised

        # -------------------------------------------------------------------
        # Feature 6: card_merchant_tx_count_24h
        #   Times this exact (card, merchant) pair appeared in past 86400 s
        # -------------------------------------------------------------------
        dq4 = pair_24h[pair]
        while dq4 and dq4[0] <= cutoff_24h:
            dq4.popleft()
        card_merchant_tx_count_24h[i] = len(dq4)

        # -------------------------------------------------------------------
        # Update all structures AFTER reading (strict past-only)
        # -------------------------------------------------------------------
        card_1h_merchants[card].append((dt, merch))
        merchant_1h_cards[merch].append((dt, card))
        card_24h_txs[card].append((dt, amt))
        pair_24h[pair].append(dt)

    # -----------------------------------------------------------------------
    # 4. Attach new columns to DataFrame
    # -----------------------------------------------------------------------
    df = df.copy()
    df["card_degree_1h"]             = card_degree_1h
    df["merchant_degree_1h"]         = merchant_degree_1h
    df["card_tx_count_24h"]          = card_tx_count_24h
    df["card_total_amt_24h"]         = card_total_amt_24h
    df["amt_zscore_24h"]             = amt_zscore_24h
    df["card_merchant_tx_count_24h"] = card_merchant_tx_count_24h

    # Safety: fill any NaN that might have leaked in
    graph_cols = [
        "card_degree_1h", "merchant_degree_1h",
        "card_tx_count_24h", "card_total_amt_24h",
        "amt_zscore_24h", "card_merchant_tx_count_24h",
    ]
    df[graph_cols] = df[graph_cols].fillna(0)

    print(f"[graph_features] Added {len(graph_cols)} graph features  "
          f"(DataFrame shape: {df.shape})")
    return df


# ---------------------------------------------------------------------------
# NetworkX sample graph builder (visualization / stats only)
# ---------------------------------------------------------------------------

def build_sample_graph(df: pd.DataFrame, n_sample: int = 1000) -> nx.Graph:
    """
    Build a bipartite graph on the first *n_sample* rows of *df*.

    Nodes
    -----
      'c_<card1>'     -- card node     (bipartite=0)
      'm_<ProductCD>' -- merchant node (bipartite=1)

    Edges
    -----
      An undirected edge between card and merchant for each transaction.
      Edge weight = number of times that (card, merchant) pair appears
      in the sample.

    Returns
    -------
    nx.Graph
    """
    sample = df.head(n_sample)

    G = nx.Graph()

    for _, row in sample.iterrows():
        c_node = f"c_{row['card1']}"
        m_node = f"m_{row['ProductCD']}"

        if not G.has_node(c_node):
            G.add_node(c_node, bipartite=0, node_type="card")
        if not G.has_node(m_node):
            G.add_node(m_node, bipartite=1, node_type="merchant")

        if G.has_edge(c_node, m_node):
            G[c_node][m_node]["weight"] += 1
        else:
            G.add_edge(c_node, m_node, weight=1)

    return G


def print_graph_stats(G: nx.Graph) -> None:
    """Print basic structural statistics of the sample graph."""
    card_nodes     = [n for n, d in G.nodes(data=True) if d.get("node_type") == "card"]
    merchant_nodes = [n for n, d in G.nodes(data=True) if d.get("node_type") == "merchant"]
    weights        = [d["weight"] for _, _, d in G.edges(data=True)]

    print("\n-- Sample Graph Statistics -----------------------------------------")
    print(f"  Total nodes      : {G.number_of_nodes():,}")
    print(f"    Card nodes     : {len(card_nodes):,}")
    print(f"    Merchant nodes : {len(merchant_nodes):,}")
    print(f"  Total edges      : {G.number_of_edges():,}")
    print(f"  Graph density    : {nx.density(G):.6f}")
    if weights:
        print(f"  Edge weight      : min={min(weights)}, "
              f"max={max(weights)}, mean={np.mean(weights):.2f}")

    # Connected components
    components = list(nx.connected_components(G))
    print(f"  Connected components  : {len(components)}")
    largest = max(components, key=len)
    print(f"  Largest component size: {len(largest)} nodes")

    # Degree stats
    deg_vals = [d for _, d in G.degree()]
    print(f"  Node degree      : min={min(deg_vals)}, "
          f"max={max(deg_vals)}, mean={np.mean(deg_vals):.2f}")
    print("--------------------------------------------------------------------\n")


# ---------------------------------------------------------------------------
# Quick-run / demo block
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    DATA_DIR         = os.path.join(os.path.dirname(__file__), "data")
    TRANSACTION_PATH = os.path.join(DATA_DIR, "train_transaction.csv")
    IDENTITY_PATH    = os.path.join(DATA_DIR, "train_identity.csv")

    # -----------------------------------------------------------------------
    # Replicate only the merge + minimal cleaning from data_preprocessing.py
    # so we can get the full pre-split DataFrame with TransactionDT intact.
    # card1 and ProductCD are kept in their raw (unencoded) form so they
    # serve as natural node identifiers in the graph.
    # -----------------------------------------------------------------------
    print("Loading raw data ...")
    if not os.path.exists(TRANSACTION_PATH):
        print(f"[ERROR] File not found: {TRANSACTION_PATH}")
        print("        Place train_transaction.csv and train_identity.csv in ./data/")
        sys.exit(1)

    transactions = pd.read_csv(TRANSACTION_PATH)
    identity     = pd.read_csv(IDENTITY_PATH)

    df_raw = transactions.merge(identity, on="TransactionID", how="left")
    print(f"  Merged shape: {df_raw.shape}")

    # Drop extremely sparse columns (>90 % missing) -- mirrors preprocessing.py
    missing_rate = df_raw.isnull().mean()
    cols_to_drop = missing_rate[missing_rate > 0.90].index.tolist()
    # Always keep the four columns required for graph features
    required_cols = {"TransactionDT", "TransactionAmt", "card1", "ProductCD"}
    cols_to_drop  = [c for c in cols_to_drop if c not in required_cols]
    df_raw.drop(columns=cols_to_drop, inplace=True)

    # Sentinel fill for remaining nulls (numeric columns only)
    cols_with_nulls = [
        c for c in df_raw.columns
        if df_raw[c].isnull().any() and c not in required_cols
    ]
    df_raw[cols_with_nulls] = df_raw[cols_with_nulls].fillna(-999)

    # Sort by TransactionDT (add_graph_features will enforce this too)
    df_raw.sort_values("TransactionDT", inplace=True)
    df_raw.reset_index(drop=True, inplace=True)
    print(f"  Working shape after minimal cleaning: {df_raw.shape}\n")

    # -----------------------------------------------------------------------
    # Run feature engineering
    # -----------------------------------------------------------------------
    df_feat = add_graph_features(df_raw)

    # -----------------------------------------------------------------------
    # Print statistics for each new feature
    # -----------------------------------------------------------------------
    GRAPH_COLS = [
        "card_degree_1h",
        "merchant_degree_1h",
        "card_tx_count_24h",
        "card_total_amt_24h",
        "amt_zscore_24h",
        "card_merchant_tx_count_24h",
    ]

    has_target = "isFraud" in df_feat.columns

    print("\n== New Feature Statistics ==================================================")
    header = f"  {'Feature':<30}  {'Mean':>10}  {'Max':>12}"
    if has_target:
        header += f"  {'Corr(isFraud)':>14}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    for col in GRAPH_COLS:
        mean_val = df_feat[col].mean()
        max_val  = df_feat[col].max()
        line     = f"  {col:<30}  {mean_val:>10.4f}  {max_val:>12.4f}"
        if has_target:
            corr = df_feat[[col, "isFraud"]].corr().iloc[0, 1]
            line += f"  {corr:>14.6f}"
        print(line)

    print("=" * 72 + "\n")

    # -----------------------------------------------------------------------
    # Build and print sample graph statistics
    # -----------------------------------------------------------------------
    print("Building sample bipartite graph on first 1 000 transactions ...")
    G = build_sample_graph(df_feat, n_sample=1000)
    print_graph_stats(G)

    # -----------------------------------------------------------------------
    # Sanity check: first few rows of new features
    # -----------------------------------------------------------------------
    print("-- Sample rows (first 10) --------------------------------------------------")
    display_cols = (
        ["TransactionDT", "card1", "ProductCD", "TransactionAmt"] + GRAPH_COLS
    )
    print(df_feat[display_cols].head(10).to_string(index=False))
    print("-" * 74)
