"""
feature_layer/graph_features.py
--------------------------------
Graph-based, time-windowed feature engineering for IEEE-CIS fraud detection.

Input
-----
    The full merged DataFrame (pre-split), sorted by ``TransactionDT``, that
    contains at minimum:

    ==================  =======================================================
    Column              Description
    ==================  =======================================================
    ``TransactionDT``   Unix-style timestamp in seconds.
    ``TransactionAmt``  Raw transaction amount (before any transformation).
    ``card1``           Card identifier (raw or label-encoded integer).
    ``ProductCD``       Merchant proxy (raw or label-encoded integer).
    ==================  =======================================================

Output
------
    The same DataFrame with **6 new columns** appended:

    ============================  ========  =====================================
    Column                        Window    Description
    ============================  ========  =====================================
    ``card_degree_1h``            1 h       Unique merchants this card hit in the
                                            past ``short_window_seconds``.
    ``merchant_degree_1h``        1 h       Unique cards that hit this merchant.
    ``card_tx_count_24h``         24 h      Transaction count for this card.
    ``card_total_amt_24h``        24 h      Total ``TransactionAmt`` for this card.
    ``amt_zscore_24h``            24 h      Z-score of current amount vs history.
                                            Returns 0 when std == 0 (single tx or
                                            identical past amounts).
    ``card_merchant_tx_count_24h``24 h      (card, merchant) pair count.
    ============================  ========  =====================================

Algorithm
---------
    For each row, **only past transactions** (strictly before the current
    ``TransactionDT``) contribute to its feature values (no lookahead).
    A deque-based sliding window is maintained per group key so the overall
    complexity is **O(n · k)**, where *k* is the average window occupancy —
    orders of magnitude faster than a naive O(n²) scan.

NetworkX graph
--------------
    :func:`build_sample_graph` builds a bipartite graph on a sample of rows
    for **visualisation only** — it is not used in feature computation.
    Card nodes are prefixed ``c_``; merchant nodes are prefixed ``m_`` to
    prevent ID collisions when the raw values overlap.
"""

import warnings
from collections import defaultdict, deque
from typing import Any, Optional

import networkx as nx
import numpy as np
import pandas as pd

from src.utils.logger import get_logger

warnings.filterwarnings("ignore")

_logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Core feature-engineering function
# ---------------------------------------------------------------------------

def add_graph_features(
    df: pd.DataFrame,
    cfg: Optional[dict[str, Any]] = None,
) -> pd.DataFrame:
    """
    Add 6 graph-inspired, time-windowed features using only past transactions.

    All window boundaries are drawn from *cfg* when provided; otherwise the
    canonical defaults (3 600 s / 86 400 s) are used so the function remains
    callable without a config file (e.g. in unit tests).

    Args:
        df:  Full merged DataFrame (pre train/test split).
             Must contain: ``TransactionDT``, ``TransactionAmt``,
             ``card1``, ``ProductCD``.
        cfg: Optional config dict from
             :func:`~src.utils.config_loader.load_config`.
             Reads keys ``graph_features.short_window_seconds`` and
             ``graph_features.long_window_seconds``.
             If ``None``, defaults to 3 600 and 86 400.

    Returns:
        Copy of *df* sorted by ``TransactionDT`` (index reset) with the
        6 new feature columns appended.  Any residual NaN values in the
        new columns are filled with 0.

    Raises:
        KeyError: If ``TransactionDT``, ``TransactionAmt``, ``card1``, or
                  ``ProductCD`` is absent from *df*.
    """
    short_window: int = (
        cfg["graph_features"]["short_window_seconds"] if cfg else 3_600
    )
    long_window: int = (
        cfg["graph_features"]["long_window_seconds"] if cfg else 86_400
    )

    # ------------------------------------------------------------------
    # 0. Sort and extract numpy arrays for speed in the hot loop
    # ------------------------------------------------------------------
    df = df.sort_values("TransactionDT").reset_index(drop=True)
    n: int = len(df)

    _logger.info(
        f"add_graph_features — {n:,} rows | "
        f"short_window={short_window}s | long_window={long_window}s"
    )

    dt_arr:   np.ndarray = df["TransactionDT"].to_numpy(dtype=np.int64)
    amt_arr:  np.ndarray = df["TransactionAmt"].to_numpy(dtype=np.float64)
    card_arr: np.ndarray = df["card1"].to_numpy()
    prod_arr: np.ndarray = df["ProductCD"].to_numpy()

    # ------------------------------------------------------------------
    # 1. Pre-allocate output arrays
    # ------------------------------------------------------------------
    card_degree_1h             = np.zeros(n, dtype=np.int32)
    merchant_degree_1h         = np.zeros(n, dtype=np.int32)
    card_tx_count_24h          = np.zeros(n, dtype=np.int32)
    card_total_amt_24h         = np.zeros(n, dtype=np.float64)
    amt_zscore_24h             = np.zeros(n, dtype=np.float64)
    card_merchant_tx_count_24h = np.zeros(n, dtype=np.int32)

    # ------------------------------------------------------------------
    # 2. Sliding-window deque structures (keyed by group identity)
    #
    #    card_1h_merchants[card]     → deque of (dt, merchant)
    #    merchant_1h_cards[merchant] → deque of (dt, card)
    #    card_24h_txs[card]          → deque of (dt, amount)
    #    pair_24h[(card, merchant)]  → deque of dt
    #
    #    Each deque is appended AFTER the current row's features are read,
    #    guaranteeing strict past-only semantics.
    # ------------------------------------------------------------------
    card_1h_merchants: dict = defaultdict(deque)
    merchant_1h_cards: dict = defaultdict(deque)
    card_24h_txs:      dict = defaultdict(deque)
    pair_24h:          dict = defaultdict(deque)

    # ------------------------------------------------------------------
    # 3. Main loop — O(n · k)
    # ------------------------------------------------------------------
    for i in range(n):
        dt:    int   = int(dt_arr[i])
        amt:   float = float(amt_arr[i])
        card:  Any   = card_arr[i]
        merch: Any   = prod_arr[i]
        pair:  tuple = (card, merch)

        cutoff_short: int = dt - short_window
        cutoff_long:  int = dt - long_window

        # ---- Feature 1: card_degree_1h --------------------------------
        dq = card_1h_merchants[card]
        while dq and dq[0][0] <= cutoff_short:
            dq.popleft()
        card_degree_1h[i] = len({entry[1] for entry in dq})

        # ---- Feature 2: merchant_degree_1h ----------------------------
        dq2 = merchant_1h_cards[merch]
        while dq2 and dq2[0][0] <= cutoff_short:
            dq2.popleft()
        merchant_degree_1h[i] = len({entry[1] for entry in dq2})

        # ---- Features 3, 4, 5: card 24 h history ---------------------
        dq3 = card_24h_txs[card]
        while dq3 and dq3[0][0] <= cutoff_long:
            dq3.popleft()

        if dq3:
            past_amounts: np.ndarray = np.array(
                [e[1] for e in dq3], dtype=np.float64
            )
            count_24h: int   = len(past_amounts)
            total_24h: float = float(past_amounts.sum())
            mean_24h:  float = float(past_amounts.mean())
            std_24h:   float = float(past_amounts.std())

            card_tx_count_24h[i]  = count_24h
            card_total_amt_24h[i] = total_24h

            # z-score: return 0 when std == 0 (div-by-zero guard)
            amt_zscore_24h[i] = (
                (amt - mean_24h) / std_24h if std_24h > 0.0 else 0.0
            )
        # else: first transaction for this card → all three stay at 0

        # ---- Feature 6: (card, merchant) pair count -------------------
        dq4 = pair_24h[pair]
        while dq4 and dq4[0] <= cutoff_long:
            dq4.popleft()
        card_merchant_tx_count_24h[i] = len(dq4)

        # ---- Update structures AFTER reading (strict past-only) -------
        card_1h_merchants[card].append((dt, merch))
        merchant_1h_cards[merch].append((dt, card))
        card_24h_txs[card].append((dt, amt))
        pair_24h[pair].append(dt)

    # ------------------------------------------------------------------
    # 4. Attach columns and fill any residual NaN
    # ------------------------------------------------------------------
    df = df.copy()
    df["card_degree_1h"]             = card_degree_1h
    df["merchant_degree_1h"]         = merchant_degree_1h
    df["card_tx_count_24h"]          = card_tx_count_24h
    df["card_total_amt_24h"]         = card_total_amt_24h
    df["amt_zscore_24h"]             = amt_zscore_24h
    df["card_merchant_tx_count_24h"] = card_merchant_tx_count_24h

    graph_cols: list[str] = [
        "card_degree_1h",
        "merchant_degree_1h",
        "card_tx_count_24h",
        "card_total_amt_24h",
        "amt_zscore_24h",
        "card_merchant_tx_count_24h",
    ]
    df[graph_cols] = df[graph_cols].fillna(0)

    _logger.info(f"Graph features added — output shape: {df.shape}")
    for col in graph_cols:
        _logger.info(
            f"  {col:<30}  "
            f"min={df[col].min():.4f}  "
            f"max={df[col].max():.4f}  "
            f"mean={df[col].mean():.4f}"
        )

    return df


# ---------------------------------------------------------------------------
# NetworkX sample graph (visualisation only)
# ---------------------------------------------------------------------------

def build_sample_graph(
    df: pd.DataFrame,
    n_sample: int = 1000,
) -> nx.Graph:
    """
    Build a weighted bipartite graph on the first *n_sample* rows of *df*.

    Node naming convention
    ----------------------
    ``'c_<card1>'``     — card node  (``bipartite=0``)
    ``'m_<ProductCD>'`` — merchant node (``bipartite=1``)

    The ``c_`` / ``m_`` prefixes prevent ID collisions when card and merchant
    values share the same numeric space.

    Edge weight counts how many times a given (card, merchant) pair appears
    in the sample.  This graph is **not** used for feature computation.

    Args:
        df:       DataFrame that contains ``card1`` and ``ProductCD`` columns.
        n_sample: Number of rows (from the head of *df*) to include in the
                  graph (default 1 000).

    Returns:
        A :class:`networkx.Graph` with ``node_type`` node attributes
        (``"card"`` or ``"merchant"``) and integer ``weight`` edge attributes.
    """
    sample = df.head(n_sample)
    G: nx.Graph = nx.Graph()

    for _, row in sample.iterrows():
        c_node: str = f"c_{row['card1']}"
        m_node: str = f"m_{row['ProductCD']}"

        if not G.has_node(c_node):
            G.add_node(c_node, bipartite=0, node_type="card")
        if not G.has_node(m_node):
            G.add_node(m_node, bipartite=1, node_type="merchant")

        if G.has_edge(c_node, m_node):
            G[c_node][m_node]["weight"] += 1
        else:
            G.add_edge(c_node, m_node, weight=1)

    return G


def log_graph_stats(G: nx.Graph) -> None:
    """
    Log basic structural statistics of a sample graph via the module logger.

    Metrics logged at INFO level: node counts (total, card, merchant),
    edge count, density, edge-weight range, connected-component count,
    largest component size, and degree distribution (min / max / mean).

    Args:
        G: :class:`networkx.Graph` produced by :func:`build_sample_graph`.
    """
    card_nodes     = [n for n, d in G.nodes(data=True) if d.get("node_type") == "card"]
    merchant_nodes = [n for n, d in G.nodes(data=True) if d.get("node_type") == "merchant"]
    weights        = [d["weight"] for _, _, d in G.edges(data=True)]

    _logger.info("Sample graph statistics:")
    _logger.info(f"  Total nodes      : {G.number_of_nodes():,}")
    _logger.info(f"    Card nodes     : {len(card_nodes):,}")
    _logger.info(f"    Merchant nodes : {len(merchant_nodes):,}")
    _logger.info(f"  Total edges      : {G.number_of_edges():,}")
    _logger.info(f"  Graph density    : {nx.density(G):.6f}")
    if weights:
        _logger.info(
            f"  Edge weight      : min={min(weights)}, "
            f"max={max(weights)}, mean={np.mean(weights):.2f}"
        )
    components = list(nx.connected_components(G))
    _logger.info(f"  Connected components   : {len(components)}")
    largest = max(components, key=len)
    _logger.info(f"  Largest component size : {len(largest)} nodes")
    deg_vals = [d for _, d in G.degree()]
    _logger.info(
        f"  Node degree      : min={min(deg_vals)}, "
        f"max={max(deg_vals)}, mean={np.mean(deg_vals):.2f}"
    )


# ---------------------------------------------------------------------------
# Quick-run block
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os
    import sys

    from src.utils.config_loader import load_config
    from src.data_layer.loader import load_raw

    _cfg = load_config()

    _t_path = _cfg["data"]["transaction_path"]
    _i_path = _cfg["data"]["identity_path"]

    if not os.path.exists(_t_path):
        _logger.error(f"File not found: {_t_path}")
        sys.exit(1)

    _df_raw = load_raw(_t_path, _i_path)

    # Minimal cleaning to preserve the four graph-feature columns
    _required = {"TransactionDT", "TransactionAmt", "card1", "ProductCD"}
    _mr       = _df_raw.isnull().mean()
    _drop     = [c for c in _mr[_mr > _cfg["preprocessing"]["missing_threshold"]].index
                 if c not in _required]
    _df_raw.drop(columns=_drop, inplace=True)
    _nulls    = [c for c in _df_raw.columns if _df_raw[c].isnull().any() and c not in _required]
    _df_raw[_nulls] = _df_raw[_nulls].fillna(_cfg["preprocessing"]["sentinel_fill_value"])
    _df_raw.sort_values("TransactionDT", inplace=True)
    _df_raw.reset_index(drop=True, inplace=True)

    _df_feat = add_graph_features(_df_raw, _cfg)

    _n_sample = _cfg["graph_features"]["graph_sample_size"]
    _logger.info(f"Building sample bipartite graph on {_n_sample} rows ...")
    _G = build_sample_graph(_df_feat, n_sample=_n_sample)
    log_graph_stats(_G)
