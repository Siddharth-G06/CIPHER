"""
src/serving_layer/schemas.py
----------------------------
Pydantic v2 schemas defining the data contracts for Kafka messages in CIPHER.

All schemas allow extra fields (``extra="allow"``) to ensure forward
compatibility as new features or metadata are added to the upstream producer
without breaking downstream consumers.
"""

from __future__ import annotations

from typing import Any, Optional
from datetime import datetime
from pydantic import BaseModel, ConfigDict, Field

class TransactionMessage(BaseModel):
    """Raw transaction message arriving from the payment processor feed.
    Published to: `cipher.transactions.raw`
    """
    model_config = ConfigDict(extra="allow")

    transaction_id: str
    card1: int
    timestamp: datetime
    amount: float
    features: dict[str, Any] = Field(
        description="Dictionary of preprocessed raw + graph features."
    )


class FlaggedMessage(BaseModel):
    """Message emitted when a transaction is flagged as potential fraud.
    Published to: `cipher.transactions.flagged`
    """
    model_config = ConfigDict(extra="allow")

    transaction_id: str
    timestamp: datetime
    ensemble_score: float
    lgbm_score: float
    iso_score: float
    raw_features: dict[str, Any]
    graph_features: dict[str, Any]
    # For explanation, we can store just a summary or the raw JSON form of ExplanationResult
    explanation_summary: Optional[str] = None
    drift_active: bool
    drift_info: Optional[dict[str, Any]] = None
    model_version: str


class DriftEventMessage(BaseModel):
    """Message emitted when concept drift is detected by the DriftDetector.
    Published to: `cipher.drift.events`
    """
    model_config = ConfigDict(extra="allow")

    feature: str
    psi: float
    adwin_window_size: Optional[int] = None
    alert_type: str = "psi_critical"
    detected_at: datetime


class FeedbackMessage(BaseModel):
    """Analyst feedback confirming or refuting a fraud flag.
    Published to: `cipher.feedback.labels`
    """
    model_config = ConfigDict(extra="allow")

    transaction_id: str
    analyst_decision: str  # e.g., "CONFIRMED_FRAUD", "FALSE_POSITIVE"
    analyst_id: str
    submitted_at: datetime
