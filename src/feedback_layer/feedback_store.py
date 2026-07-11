"""
src/feedback_layer/feedback_store.py
-------------------------------------
Append-only SQLite event store for analyst feedback records.

Design principles
-----------------
* **Event Sourcing** — records are never updated or deleted; ``used_in_retraining``
  and ``retraining_run_id`` are the only mutable columns, updated only by
  :meth:`FeedbackStore.mark_as_used`.
* **Idempotent inserts** — a second ``add_feedback`` call for the same
  ``transaction_id`` is silently ignored (INSERT OR IGNORE), enforcing the
  business rule that an analyst can review a transaction only once.
* **Indexed reads** — the ``used_in_retraining`` column is indexed to make
  :meth:`get_unused_feedback` sub-millisecond even with large tables.

Usage::

    from src.feedback_layer.feedback_store import FeedbackStore, FeedbackRecord, AnalystDecision

    store = FeedbackStore()
    store.add_feedback(
        FeedbackRecord(
            transaction_id="txn-001",
            analyst_id="analyst-42",
            decision=AnalystDecision.CONFIRM_FRAUD,
            confidence=0.9,
            model_score=0.87,
            true_label=1,
        )
    )
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import List, Optional

from src.utils.config_loader import load_config
from src.utils.logger import get_logger

_logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class AnalystDecision(str, Enum):
    """Controlled vocabulary for analyst label decisions.

    Using ``str`` as a mixin so instances serialise naturally to/from SQLite
    TEXT columns without additional conversion.
    """

    CONFIRM_FRAUD = "CONFIRM_FRAUD"
    """Analyst confirmed the model's fraud flag as correct."""

    FALSE_POSITIVE = "FALSE_POSITIVE"
    """Analyst determined the flagged transaction was legitimate."""

    ESCALATE = "ESCALATE"
    """Analyst escalated for senior review; ground truth not yet determined."""


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------


@dataclass
class FeedbackRecord:
    """Immutable snapshot of a single analyst review event.

    All timestamps are stored as ISO-8601 UTC strings in SQLite.

    Attributes:
        transaction_id: Unique identifier of the reviewed transaction.
        analyst_id: Identifier of the analyst who submitted the label.
        decision: The analyst's label decision as :class:`AnalystDecision`.
        confidence: Analyst self-reported confidence in [0, 1]; used as a
            sample-weight multiplier during retraining.
        model_score: The ensemble fraud score at the time of flagging.
        true_label: Ground-truth binary label (1 = fraud, 0 = legitimate).
        notes: Optional free-text notes from the analyst.
        reviewed_at: UTC timestamp of label submission (auto-populated if
            not provided).
        model_version: The model version string active at the time the
            transaction was scored.
        drift_active: Whether the drift detector was active at score time.
        used_in_retraining: Flag set to ``True`` by
            :meth:`~FeedbackStore.mark_as_used` after the record is
            incorporated in a retraining run.
        retraining_run_id: MLflow run ID of the retraining run that consumed
            this record; ``None`` until :meth:`~FeedbackStore.mark_as_used`
            is called.
    """

    transaction_id: str
    analyst_id: str
    decision: AnalystDecision
    confidence: float
    model_score: float
    true_label: int
    notes: Optional[str] = None
    reviewed_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    model_version: str = "unknown"
    drift_active: bool = False
    used_in_retraining: bool = False
    retraining_run_id: Optional[str] = None


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------


class FeedbackStore:
    """Append-only SQLite store for :class:`FeedbackRecord` events.

    The store guarantees that:

    * Each ``transaction_id`` maps to at most one record (INSERT OR IGNORE).
    * Records are never physically deleted.
    * ``used_in_retraining`` and ``retraining_run_id`` are the only columns
      updated post-insert, and only via :meth:`mark_as_used`.

    Args:
        db_path: Filesystem path to the SQLite database file.  Defaults to
            the value of ``feedback.db_path`` in ``config/config.yaml``.
        config_path: Path to the YAML configuration file.
    """

    # SQL statements --------------------------------------------------------

    _CREATE_TABLE = """
        CREATE TABLE IF NOT EXISTS feedback_records (
            transaction_id      TEXT PRIMARY KEY,
            analyst_id          TEXT NOT NULL,
            decision            TEXT NOT NULL,
            confidence          REAL NOT NULL,
            model_score         REAL NOT NULL,
            true_label          INTEGER NOT NULL,
            notes               TEXT,
            reviewed_at         TEXT NOT NULL,
            model_version       TEXT NOT NULL DEFAULT 'unknown',
            drift_active        INTEGER NOT NULL DEFAULT 0,
            used_in_retraining  INTEGER NOT NULL DEFAULT 0,
            retraining_run_id   TEXT
        );
    """

    _CREATE_IDX_USED = """
        CREATE INDEX IF NOT EXISTS idx_used_in_retraining
        ON feedback_records (used_in_retraining);
    """

    _CREATE_IDX_REVIEWED = """
        CREATE INDEX IF NOT EXISTS idx_reviewed_at
        ON feedback_records (reviewed_at);
    """

    _INSERT = """
        INSERT OR IGNORE INTO feedback_records (
            transaction_id, analyst_id, decision, confidence,
            model_score, true_label, notes, reviewed_at,
            model_version, drift_active, used_in_retraining, retraining_run_id
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
    """

    _SELECT_UNUSED = """
        SELECT * FROM feedback_records
        WHERE used_in_retraining = 0
        ORDER BY reviewed_at ASC;
    """

    _UPDATE_USED = """
        UPDATE feedback_records
        SET used_in_retraining = 1,
            retraining_run_id  = ?
        WHERE transaction_id IN ({placeholders});
    """

    _SELECT_STATS = """
        SELECT
            COUNT(*)                                          AS total_records,
            SUM(CASE WHEN used_in_retraining = 0 THEN 1 ELSE 0 END) AS unused_count,
            SUM(CASE WHEN decision = 'CONFIRM_FRAUD'  THEN 1 ELSE 0 END) AS confirm_fraud_count,
            SUM(CASE WHEN decision = 'FALSE_POSITIVE' THEN 1 ELSE 0 END) AS false_positive_count,
            SUM(CASE WHEN decision = 'ESCALATE'       THEN 1 ELSE 0 END) AS escalate_count,
            AVG(confidence)                                   AS avg_confidence
        FROM feedback_records;
    """

    _SELECT_HISTORY = """
        SELECT * FROM feedback_records
        ORDER BY reviewed_at DESC
        LIMIT ?;
    """

    # Lifecycle -------------------------------------------------------------

    def __init__(
        self,
        db_path: Optional[str] = None,
        config_path: str = "config/config.yaml",
    ) -> None:
        """Initialise the store, creating the database and schema if needed.

        Args:
            db_path: Path to the SQLite file.  Auto-resolved from config if
                ``None``.
            config_path: Path to the YAML configuration file.
        """
        if db_path is None:
            cfg = load_config(config_path)
            db_path = cfg["feedback"]["db_path"]

        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)

        self._conn: sqlite3.Connection = sqlite3.connect(
            str(self._db_path),
            check_same_thread=False,
            isolation_level=None,   # autocommit; we manage transactions explicitly
        )
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.execute("PRAGMA foreign_keys=ON;")

        self._create_schema()
        _logger.info(
            "FeedbackStore initialised | db=%s", self._db_path.resolve()
        )

    def _create_schema(self) -> None:
        """Create the feedback_records table and indexes if they don't exist."""
        with self._conn:
            self._conn.execute(self._CREATE_TABLE)
            self._conn.execute(self._CREATE_IDX_USED)
            self._conn.execute(self._CREATE_IDX_REVIEWED)
        _logger.debug("FeedbackStore schema verified.")

    # Public API ------------------------------------------------------------

    def add_feedback(self, record: FeedbackRecord) -> None:
        """Append a feedback record to the store.

        If a record with the same ``transaction_id`` already exists, the
        insert is silently ignored (idempotent / event-sourcing guarantee).
        The analyst can review a transaction only once.

        Args:
            record: The :class:`FeedbackRecord` to persist.
        """
        decision_value = (
            record.decision.value
            if isinstance(record.decision, AnalystDecision)
            else str(record.decision)
        )
        params = (
            record.transaction_id,
            record.analyst_id,
            decision_value,
            record.confidence,
            record.model_score,
            record.true_label,
            record.notes,
            record.reviewed_at,
            record.model_version,
            int(record.drift_active),
            int(record.used_in_retraining),
            record.retraining_run_id,
        )
        with self._conn:
            cursor = self._conn.execute(self._INSERT, params)

        if cursor.rowcount == 0:
            _logger.warning(
                "FeedbackStore.add_feedback — duplicate transaction_id='%s' "
                "ignored (event-sourcing: analyst may review only once).",
                record.transaction_id,
            )
        else:
            _logger.info(
                "FeedbackStore.add_feedback — stored | tx=%s, analyst=%s, "
                "decision=%s, confidence=%.2f",
                record.transaction_id,
                record.analyst_id,
                decision_value,
                record.confidence,
            )

    def get_unused_feedback(self) -> List[FeedbackRecord]:
        """Return all records that have not yet been used in a retraining run.

        Returns:
            List of :class:`FeedbackRecord` ordered by ``reviewed_at`` ASC
            (oldest first, matching insertion order of the event log).
        """
        cursor = self._conn.execute(self._SELECT_UNUSED)
        rows = cursor.fetchall()
        records = [self._row_to_record(r) for r in rows]
        _logger.debug(
            "FeedbackStore.get_unused_feedback — returned %d records.", len(records)
        )
        return records

    def mark_as_used(self, transaction_ids: List[str], run_id: str) -> None:
        """Mark a batch of records as consumed by a specific retraining run.

        This is the *only* mutation permitted after initial insert, preserving
        the event-sourcing append-only contract for all other columns.

        Args:
            transaction_ids: List of ``transaction_id`` strings to mark.
            run_id: MLflow run ID of the retraining run that consumed them.
        """
        if not transaction_ids:
            _logger.debug("FeedbackStore.mark_as_used — empty list, no-op.")
            return

        placeholders = ", ".join(["?"] * len(transaction_ids))
        sql = self._UPDATE_USED.format(placeholders=placeholders)
        params = [run_id] + list(transaction_ids)

        with self._conn:
            cursor = self._conn.execute(sql, params)

        _logger.info(
            "FeedbackStore.mark_as_used — updated %d/%d records | run_id=%s",
            cursor.rowcount,
            len(transaction_ids),
            run_id,
        )

    def get_feedback_stats(self) -> dict:
        """Return aggregate counts and average confidence for all records.

        Returns:
            Dictionary with keys:

            * ``total_records``      — total rows in the store.
            * ``unused_count``       — rows not yet used in retraining.
            * ``confirm_fraud_count``— rows with decision CONFIRM_FRAUD.
            * ``false_positive_count``— rows with decision FALSE_POSITIVE.
            * ``escalate_count``     — rows with decision ESCALATE.
            * ``avg_confidence``     — mean analyst confidence (0-1).
        """
        cursor = self._conn.execute(self._SELECT_STATS)
        row = cursor.fetchone()
        stats = {
            "total_records": row["total_records"] or 0,
            "unused_count": row["unused_count"] or 0,
            "confirm_fraud_count": row["confirm_fraud_count"] or 0,
            "false_positive_count": row["false_positive_count"] or 0,
            "escalate_count": row["escalate_count"] or 0,
            "avg_confidence": round(row["avg_confidence"] or 0.0, 4),
        }
        _logger.debug("FeedbackStore.get_feedback_stats — %s", stats)
        return stats

    def get_feedback_history(self, limit: int = 100) -> List[FeedbackRecord]:
        """Return the most recent feedback records for display in Streamlit.

        Args:
            limit: Maximum number of records to return (default: 100).

        Returns:
            List of :class:`FeedbackRecord` ordered by ``reviewed_at`` DESC.
        """
        cursor = self._conn.execute(self._SELECT_HISTORY, (limit,))
        rows = cursor.fetchall()
        records = [self._row_to_record(r) for r in rows]
        _logger.debug(
            "FeedbackStore.get_feedback_history — returned %d records (limit=%d).",
            len(records),
            limit,
        )
        return records

    # Helpers ---------------------------------------------------------------

    @staticmethod
    def _row_to_record(row: sqlite3.Row) -> FeedbackRecord:
        """Convert a SQLite row into a :class:`FeedbackRecord` dataclass.

        Args:
            row: A :class:`sqlite3.Row` from ``feedback_records``.

        Returns:
            Populated :class:`FeedbackRecord`.
        """
        return FeedbackRecord(
            transaction_id=row["transaction_id"],
            analyst_id=row["analyst_id"],
            decision=AnalystDecision(row["decision"]),
            confidence=float(row["confidence"]),
            model_score=float(row["model_score"]),
            true_label=int(row["true_label"]),
            notes=row["notes"],
            reviewed_at=row["reviewed_at"],
            model_version=row["model_version"],
            drift_active=bool(row["drift_active"]),
            used_in_retraining=bool(row["used_in_retraining"]),
            retraining_run_id=row["retraining_run_id"],
        )

    def close(self) -> None:
        """Close the underlying SQLite connection."""
        self._conn.close()
        _logger.info("FeedbackStore connection closed.")

    def __repr__(self) -> str:  # pragma: no cover
        stats = self.get_feedback_stats()
        return (
            f"FeedbackStore(db='{self._db_path}', "
            f"total={stats['total_records']}, "
            f"unused={stats['unused_count']})"
        )
