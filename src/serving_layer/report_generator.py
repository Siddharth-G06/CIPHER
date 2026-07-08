"""
src/serving_layer/report_generator.py
--------------------------------------
Professional PDF investigation report generator for the CIPHER fraud-detection
pipeline.

Architecture — Template Method Pattern
---------------------------------------
:class:`ReportGenerator` is an **abstract base class** whose public method
:meth:`generate_report` acts as the *template method*.  It calls a fixed
sequence of protected ``_build_*`` methods in a defined order, collects their
:class:`~reportlab.platypus.Flowable` outputs, and hands the assembled list to
ReportLab Platypus for PDF rendering.

Individual ``_build_*`` methods can be overridden in concrete subclasses to
customise specific sections (e.g. a ``ComplianceReportGenerator`` could override
:meth:`_build_audit_trail` to include regulatory fields) without touching the
orchestration logic.

Why single-underscore (protected)?
    Double-underscore mangling (``__build_*``) would break subclass access.
    PEP-8 convention for *intended-to-be-overridden-but-internal* methods is
    single-underscore.

Key design decisions
--------------------
* All colors, thresholds, and paths come from ``config/config.yaml``.
  Nothing is hardcoded.
* The risk tier palette (LOW / MEDIUM / HIGH / CRITICAL) drives the header
  band color on every page, making risk immediately visible.
* SHAP waterfall plots are embedded as images.  If the plot file is missing
  or corrupt, the section degrades gracefully to a "SHAP computation pending"
  message instead of crashing.
* Async generation via :class:`~concurrent.futures.ThreadPoolExecutor` lets
  the serving layer fire-and-forget PDF jobs without blocking the main
  prediction thread.

Typical usage::

    from src.serving_layer.report_generator import (
        FlaggedTransaction,
        ReportGenerator,
    )

    generator = ReportGenerator()
    path = generator.generate_report(transaction)          # synchronous
    generator.generate_report_async(transaction, callback) # non-blocking

Data contract
-------------
The generator consumes :class:`FlaggedTransaction`, which bundles together the
ensemble prediction scores, raw & graph feature dictionaries, the
:class:`~src.ml_layer.explainer.ExplanationResult` produced by Module 5, and
the drift state at the moment of flagging.
"""

from __future__ import annotations

import abc
import os
import re
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import A4, letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    HRFlowable,
    Image,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.platypus.flowables import Flowable

from src.utils.config_loader import load_config
from src.utils.logger import get_logger

_logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# FlaggedTransaction — data contract
# ---------------------------------------------------------------------------


@dataclass
class FlaggedTransaction:
    """All artefacts produced for a single flagged transaction.

    This dataclass is the sole input to :class:`ReportGenerator`.  It bundles
    every piece of information a fraud analyst needs: raw predictions, feature
    values, the SHAP explanation, and the drift state at flagging time.

    Attributes:
        transaction_id:  Unique identifier (e.g. ``"TX_000123"``).
        timestamp:       UTC datetime when the transaction was processed.
        ensemble_score:  Weighted fraud probability from the ensemble model
                         (range 0–1).
        lgbm_score:      LightGBM fraud probability component (range 0–1).
        iso_score:       Isolation-Forest anomaly score after normalisation
                         (range 0–1, where 1 = most anomalous).
        raw_features:    Dict of original transaction feature values
                         (e.g. ``{"TransactionAmt": 1200.0, ...}``).
        graph_features:  Dict of graph-derived feature values
                         (e.g. ``{"card_degree_1h": 11, ...}``).
        explanation:     :class:`~src.ml_layer.explainer.ExplanationResult`
                         produced by Module 5.
        drift_active:    ``True`` if a drift alert was active when this
                         transaction was flagged.
        drift_info:      Last drift event dict from the drift detector if
                         ``drift_active`` is ``True``, else ``None``.
        model_version:   MLflow model version string (e.g. ``"v3.1"``).
    """

    transaction_id: str
    timestamp: datetime
    ensemble_score: float
    lgbm_score: float
    iso_score: float
    raw_features: dict
    graph_features: dict
    explanation: Any  # ExplanationResult — typed as Any to avoid circular import
    drift_active: bool
    drift_info: Optional[dict]
    model_version: str


# ---------------------------------------------------------------------------
# _HexColor helper
# ---------------------------------------------------------------------------


def _hex_to_color(hex_str: str) -> colors.HexColor:
    """Convert a CSS hex colour string to a ReportLab HexColor.

    Args:
        hex_str: Hex string with or without leading ``#``
                 (e.g. ``"#28a745"`` or ``"28a745"``).

    Returns:
        :class:`reportlab.lib.colors.HexColor` instance.
    """
    return colors.HexColor(hex_str if hex_str.startswith("#") else f"#{hex_str}")


# ---------------------------------------------------------------------------
# ReportGenerator — Template Method Pattern
# ---------------------------------------------------------------------------


class ReportGenerator(abc.ABC if False else object):
    """Template-Method PDF report generator for CIPHER fraud investigations.

    Concrete usage — instantiate directly; subclass only to override specific
    ``_build_*`` sections::

        generator = ReportGenerator()
        path      = generator.generate_report(transaction)

    The ``_build_*`` methods are **intentionally not abstract** — they have
    default implementations so the class is usable without subclassing, while
    still enabling selective override.

    Attributes:
        _cfg:        The ``report`` sub-dict from ``config.yaml``.
        _output_dir: :class:`~pathlib.Path` to the ``reports/`` directory.
        _executor:   :class:`~concurrent.futures.ThreadPoolExecutor` for async
                     PDF generation.
        _styles:     Dict of named :class:`~reportlab.lib.styles.ParagraphStyle`
                     objects (title, heading, body, warning, critical).
        _reports:    Dict mapping ``transaction_id`` → report metadata dict,
                     used by :meth:`get_report_path` and :meth:`list_reports`.
        _reports_lock: :class:`threading.Lock` protecting ``_reports``.
    """

    def __init__(self, config_path: str = "config/config.yaml") -> None:
        """Load config, initialise executor, styles, and output directory.

        Args:
            config_path: Path to ``config/config.yaml``, resolved relative to
                         the current working directory.
        """
        full_cfg = load_config(config_path)
        self._cfg: dict[str, Any] = full_cfg["report"]
        self._output_dir = Path(self._cfg["output_dir"])
        self._output_dir.mkdir(parents=True, exist_ok=True)

        # Thread pool for async generation
        self._executor = ThreadPoolExecutor(
            max_workers=int(self._cfg["threadpool_workers"])
        )

        # In-memory report registry (populated by generate_report)
        self._reports: dict[str, dict] = {}
        self._reports_lock = threading.Lock()

        # Preload existing reports from disk into registry
        self._scan_existing_reports()

        # Build paragraph styles
        self._styles = self._build_styles()

        _logger.info(
            "ReportGenerator initialised | output_dir='%s', workers=%d",
            self._output_dir,
            int(self._cfg["threadpool_workers"]),
        )

    # ------------------------------------------------------------------
    # Style factory
    # ------------------------------------------------------------------

    def _build_styles(self) -> dict[str, ParagraphStyle]:
        """Create and return named ParagraphStyle objects.

        Returns:
            Dict with keys ``title``, ``heading``, ``subheading``, ``body``,
            ``body_bold``, ``warning``, ``critical``, ``mono``.
        """
        base = getSampleStyleSheet()
        clr = self._cfg["colors"]
        header_text = _hex_to_color(clr["header_text"])

        styles: dict[str, ParagraphStyle] = {
            "title": ParagraphStyle(
                "CipherTitle",
                parent=base["Title"],
                fontSize=22,
                textColor=header_text,
                spaceAfter=4,
                alignment=TA_CENTER,
                fontName="Helvetica-Bold",
            ),
            "subtitle": ParagraphStyle(
                "CipherSubtitle",
                parent=base["Normal"],
                fontSize=11,
                textColor=header_text,
                spaceAfter=2,
                alignment=TA_CENTER,
                fontName="Helvetica",
            ),
            "heading": ParagraphStyle(
                "CipherHeading",
                parent=base["Heading2"],
                fontSize=13,
                textColor=_hex_to_color(clr["table_header"]),
                spaceBefore=14,
                spaceAfter=6,
                fontName="Helvetica-Bold",
                borderPad=4,
            ),
            "subheading": ParagraphStyle(
                "CipherSubheading",
                parent=base["Heading3"],
                fontSize=11,
                textColor=_hex_to_color(clr["table_header"]),
                spaceBefore=8,
                spaceAfter=4,
                fontName="Helvetica-Bold",
            ),
            "body": ParagraphStyle(
                "CipherBody",
                parent=base["Normal"],
                fontSize=9,
                leading=14,
                textColor=colors.black,
                fontName="Helvetica",
            ),
            "body_bold": ParagraphStyle(
                "CipherBodyBold",
                parent=base["Normal"],
                fontSize=9,
                leading=14,
                textColor=colors.black,
                fontName="Helvetica-Bold",
            ),
            "warning": ParagraphStyle(
                "CipherWarning",
                parent=base["Normal"],
                fontSize=9,
                leading=14,
                textColor=_hex_to_color("#7d4e00"),
                backColor=_hex_to_color("#fff3cd"),
                borderPad=6,
                fontName="Helvetica-Bold",
            ),
            "critical": ParagraphStyle(
                "CipherCritical",
                parent=base["Normal"],
                fontSize=9,
                leading=14,
                textColor=_hex_to_color("#842029"),
                backColor=_hex_to_color("#f8d7da"),
                borderPad=6,
                fontName="Helvetica-Bold",
            ),
            "mono": ParagraphStyle(
                "CipherMono",
                parent=base["Code"],
                fontSize=8,
                leading=12,
                textColor=colors.black,
                fontName="Courier",
            ),
            "analyst": ParagraphStyle(
                "CipherAnalyst",
                parent=base["Normal"],
                fontSize=9,
                leading=20,
                textColor=colors.HexColor("#555555"),
                fontName="Helvetica",
            ),
        }
        return styles

    # ------------------------------------------------------------------
    # Public interface — Template Method
    # ------------------------------------------------------------------

    def generate_report(self, transaction: FlaggedTransaction) -> str:
        """Generate a PDF investigation report for one flagged transaction.

        This is the **template method**.  It calls the protected ``_build_*``
        methods in a fixed order, concatenates all
        :class:`~reportlab.platypus.Flowable` objects, and passes them to
        ReportLab's :class:`~reportlab.platypus.SimpleDocTemplate`.

        The generated file is registered in :attr:`_reports` for fast lookup
        by :meth:`get_report_path` and :meth:`list_reports`.

        Args:
            transaction: Fully-populated :class:`FlaggedTransaction` instance.

        Returns:
            Absolute path to the generated PDF file.
        """
        ts_str = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
        filename = (
            f"cipher_report_{transaction.transaction_id}_{ts_str}.pdf"
        )
        file_path = str(self._output_dir / filename)

        # Page size from config (default A4)
        page_size_name = self._cfg.get("page", {}).get("size", "A4").upper()
        page_size = A4 if page_size_name == "A4" else letter
        margin = float(self._cfg.get("page", {}).get("margin_inches", 0.75)) * inch

        doc = SimpleDocTemplate(
            file_path,
            pagesize=page_size,
            leftMargin=margin,
            rightMargin=margin,
            topMargin=margin,
            bottomMargin=margin,
            title=f"CIPHER Fraud Report — {transaction.transaction_id}",
            author="CIPHER Fraud Detection System",
            subject="Fraud Investigation Report",
        )

        # Assemble all sections in order — Template Method
        story: list[Flowable] = []
        story.extend(self._build_header(transaction))
        story.extend(self._build_risk_summary(transaction))
        story.extend(self._build_transaction_details(transaction))
        story.extend(self._build_shap_section(transaction))
        story.extend(self._build_audit_trail(transaction))

        doc.build(story)

        _logger.info(
            "ReportGenerator — PDF generated | transaction_id='%s', path='%s'",
            transaction.transaction_id,
            file_path,
        )

        # Register in in-memory registry
        metadata = {
            "transaction_id": transaction.transaction_id,
            "path": file_path,
            "size_kb": round(Path(file_path).stat().st_size / 1024, 2),
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }
        with self._reports_lock:
            self._reports[transaction.transaction_id] = metadata

        return file_path

    def generate_report_async(
        self,
        transaction: FlaggedTransaction,
        callback: Callable[[str], None],
    ) -> None:
        """Submit PDF generation to the thread pool (non-blocking).

        Returns immediately.  When the PDF is ready, ``callback`` is called
        on the worker thread with the absolute file path as its sole argument.

        Args:
            transaction: Fully-populated :class:`FlaggedTransaction` instance.
            callback:    Callable invoked with ``file_path: str`` on completion.
        """

        def _task() -> None:
            try:
                path = self.generate_report(transaction)
                callback(path)
            except Exception as exc:  # noqa: BLE001
                _logger.error(
                    "generate_report_async — failed for '%s': %s",
                    transaction.transaction_id,
                    exc,
                    exc_info=True,
                )

        self._executor.submit(_task)
        _logger.debug(
            "generate_report_async — submitted | transaction_id='%s'",
            transaction.transaction_id,
        )

    # ------------------------------------------------------------------
    # Template Method — Protected build steps
    # ------------------------------------------------------------------

    def _build_header(self, transaction: FlaggedTransaction) -> list[Flowable]:
        """Build the page header band with CIPHER branding and report metadata.

        The header background color is driven by the risk tier so analysts
        can identify risk level at a glance before reading any details.

        Args:
            transaction: Current :class:`FlaggedTransaction`.

        Returns:
            List of :class:`~reportlab.platypus.Flowable` objects.
        """
        tier_name, tier_color = self._classify_risk_tier(transaction.ensemble_score)
        band_color = _hex_to_color(tier_color)
        text_color = self._cfg["colors"]["header_text"]

        # Header band table — full-width colored background
        ts = transaction.timestamp.strftime("%Y-%m-%d %H:%M:%S UTC")
        header_data = [
            [
                Paragraph(
                    "CIPHER FRAUD DETECTION SYSTEM",
                    self._styles["title"],
                ),
            ],
            [
                Paragraph(
                    "FRAUD INVESTIGATION REPORT", self._styles["subtitle"]
                ),
            ],
            [
                Paragraph(
                    (
                        f"Transaction ID: <b>{transaction.transaction_id}</b>"
                        f"&nbsp;&nbsp;|&nbsp;&nbsp;"
                        f"Generated: {ts}"
                        f"&nbsp;&nbsp;|&nbsp;&nbsp;"
                        f"Model: {transaction.model_version}"
                    ),
                    self._styles["subtitle"],
                ),
            ],
        ]

        page_w = A4[0]
        margin = float(self._cfg.get("page", {}).get("margin_inches", 0.75)) * inch
        avail_w = page_w - 2 * margin

        header_table = Table(header_data, colWidths=[avail_w])
        header_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, -1), band_color),
                    ("TOPPADDING", (0, 0), (-1, -1), 10),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                    ("LEFTPADDING", (0, 0), (-1, -1), 12),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 12),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ]
            )
        )

        return [header_table, Spacer(1, 10)]

    def _build_risk_summary(self, transaction: FlaggedTransaction) -> list[Flowable]:
        """Build the risk summary section: tier badge, scores, drift warning.

        Args:
            transaction: Current :class:`FlaggedTransaction`.

        Returns:
            List of :class:`~reportlab.platypus.Flowable` objects.
        """
        tier_name, tier_color = self._classify_risk_tier(transaction.ensemble_score)
        tier_bg = _hex_to_color(tier_color)
        white = colors.white

        flowables: list[Flowable] = []
        flowables.append(
            Paragraph("Risk Assessment", self._styles["heading"])
        )
        flowables.append(HRFlowable(width="100%", thickness=1, color=_hex_to_color(self._cfg["colors"]["table_header"])))
        flowables.append(Spacer(1, 6))

        # Tier badge + score table
        badge_style = ParagraphStyle(
            "badge",
            fontName="Helvetica-Bold",
            fontSize=14,
            textColor=white,
            alignment=TA_CENTER,
        )
        score_data = [
            ["RISK TIER", "ENSEMBLE SCORE", "LGBM SCORE", "ISO SCORE"],
            [
                Paragraph(tier_name, badge_style),
                Paragraph(f"<b>{transaction.ensemble_score:.4f}</b>", self._styles["body_bold"]),
                Paragraph(f"{transaction.lgbm_score:.4f}", self._styles["body"]),
                Paragraph(f"{transaction.iso_score:.4f}", self._styles["body"]),
            ],
        ]

        score_table = Table(score_data, colWidths=["*", "*", "*", "*"])
        score_table.setStyle(
            TableStyle(
                [
                    # Header row
                    ("BACKGROUND", (0, 0), (-1, 0), _hex_to_color(self._cfg["colors"]["table_header"])),
                    ("TEXTCOLOR", (0, 0), (-1, 0), white),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, 0), 9),
                    ("ALIGN", (0, 0), (-1, 0), "CENTER"),
                    # Tier badge cell
                    ("BACKGROUND", (0, 1), (0, 1), tier_bg),
                    # Data row
                    ("ALIGN", (0, 1), (-1, 1), "CENTER"),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("TOPPADDING", (0, 0), (-1, -1), 8),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.lightgrey),
                    ("ROUNDEDCORNERS", [4, 4, 4, 4]),
                ]
            )
        )
        flowables.append(score_table)
        flowables.append(Spacer(1, 10))

        # Drift warning box (only when drift is active)
        if transaction.drift_active:
            drift_msg = (
                "⚠ DRIFT ALERT ACTIVE — Data distribution has shifted since model training. "
                "Prediction confidence may be reduced. Review drift details in the Audit Trail."
            )
            if transaction.drift_info:
                feature = transaction.drift_info.get("feature", "unknown")
                psi = transaction.drift_info.get("psi", "N/A")
                drift_msg += f" | Feature: {feature} | PSI: {psi}"
            flowables.append(
                Paragraph(drift_msg, self._styles["warning"])
            )
            flowables.append(Spacer(1, 6))

        return flowables

    def _build_transaction_details(
        self, transaction: FlaggedTransaction
    ) -> list[Flowable]:
        """Build a two-column feature table (raw | graph) with baseline row.

        Args:
            transaction: Current :class:`FlaggedTransaction`.

        Returns:
            List of :class:`~reportlab.platypus.Flowable` objects.
        """
        flowables: list[Flowable] = []
        flowables.append(
            Paragraph("Transaction Details", self._styles["heading"])
        )
        flowables.append(HRFlowable(width="100%", thickness=1, color=_hex_to_color(self._cfg["colors"]["table_header"])))
        flowables.append(Spacer(1, 6))

        alt_bg = _hex_to_color(self._cfg["colors"]["row_alt"])
        hdr_bg = _hex_to_color(self._cfg["colors"]["table_header"])
        white = colors.white

        raw = transaction.raw_features
        graph = transaction.graph_features

        # Interleave raw and graph features into rows
        raw_items = list(raw.items())
        graph_items = list(graph.items())
        max_rows = max(len(raw_items), len(graph_items))

        # Pad the shorter list
        while len(raw_items) < max_rows:
            raw_items.append(("", ""))
        while len(graph_items) < max_rows:
            graph_items.append(("", ""))

        table_data: list[list] = [
            [
                Paragraph("RAW FEATURE", self._styles["body_bold"]),
                Paragraph("VALUE", self._styles["body_bold"]),
                Paragraph("GRAPH FEATURE", self._styles["body_bold"]),
                Paragraph("VALUE", self._styles["body_bold"]),
            ]
        ]
        style_cmds = [
            ("BACKGROUND", (0, 0), (-1, 0), hdr_bg),
            ("TEXTCOLOR", (0, 0), (-1, 0), white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, 0), 9),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.lightgrey),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ]

        for i, ((rk, rv), (gk, gv)) in enumerate(zip(raw_items, graph_items), start=1):
            # Format values
            rv_str = f"{rv:.4g}" if isinstance(rv, float) else str(rv)
            gv_str = f"{gv:.4g}" if isinstance(gv, float) else str(gv)
            table_data.append([
                Paragraph(str(rk), self._styles["body"]),
                Paragraph(rv_str, self._styles["mono"]),
                Paragraph(str(gk), self._styles["body"]),
                Paragraph(gv_str, self._styles["mono"]),
            ])
            if i % 2 == 0:
                style_cmds.append(("BACKGROUND", (0, i), (-1, i), alt_bg))

        # Baseline comparison row (append at bottom)
        tx_amt = raw.get("TransactionAmt", raw.get("amount", "N/A"))
        card_avg = graph.get("card_avg_amount_24h", graph.get("total_amount_24h", "N/A"))
        if isinstance(tx_amt, (int, float)):
            tx_amt = f"${tx_amt:,.2f}"
        if isinstance(card_avg, (int, float)):
            card_avg = f"${card_avg:,.2f}"

        baseline_row_idx = len(table_data)
        table_data.append([
            Paragraph("⚖ Card Normal Avg (24h)", self._styles["body_bold"]),
            Paragraph(str(card_avg), self._styles["mono"]),
            Paragraph("⚖ This Transaction Amt", self._styles["body_bold"]),
            Paragraph(str(tx_amt), self._styles["mono"]),
        ])
        style_cmds.append(
            ("BACKGROUND", (0, baseline_row_idx), (-1, baseline_row_idx),
             _hex_to_color("#e8f4f8"))
        )
        style_cmds.append(
            ("LINEABOVE", (0, baseline_row_idx), (-1, baseline_row_idx), 1.5,
             _hex_to_color(self._cfg["colors"]["table_header"]))
        )

        details_table = Table(table_data, colWidths=["35%", "15%", "35%", "15%"])
        details_table.setStyle(TableStyle(style_cmds))
        flowables.append(details_table)
        flowables.append(Spacer(1, 12))

        return flowables

    def _build_shap_section(self, transaction: FlaggedTransaction) -> list[Flowable]:
        """Build the SHAP explainability section.

        Includes: plain-English summary paragraph, waterfall plot image (with
        graceful fallback for missing/corrupt plots), and a top-10 features
        table with alternating row colors and directional risk indicators.

        Args:
            transaction: Current :class:`FlaggedTransaction`.

        Returns:
            List of :class:`~reportlab.platypus.Flowable` objects.
        """
        flowables: list[Flowable] = []
        flowables.append(
            Paragraph("SHAP Explainability", self._styles["heading"])
        )
        flowables.append(HRFlowable(width="100%", thickness=1, color=_hex_to_color(self._cfg["colors"]["table_header"])))
        flowables.append(Spacer(1, 6))

        explanation = transaction.explanation

        # -- Plain-English Summary paragraph --
        summary_text = getattr(explanation, "plain_english_summary", None)
        if not summary_text:
            summary_text = "No SHAP summary available for this transaction."
        flowables.append(
            Paragraph(f"<b>Model Explanation:</b> {summary_text}", self._styles["body"])
        )
        flowables.append(Spacer(1, 8))

        # -- Waterfall plot image (graceful degradation) --
        waterfall_path = getattr(explanation, "waterfall_plot_path", None)
        img_w = float(self._cfg.get("shap_image", {}).get("width_points", 450))
        img_h = float(self._cfg.get("shap_image", {}).get("height_points", 300))

        plot_embedded = False
        if waterfall_path and Path(waterfall_path).is_file():
            try:
                img = Image(waterfall_path, width=img_w, height=img_h)
                flowables.append(img)
                plot_embedded = True
                _logger.debug("SHAP waterfall plot embedded: '%s'", waterfall_path)
            except Exception as exc:  # noqa: BLE001
                _logger.warning(
                    "Could not embed waterfall plot '%s': %s", waterfall_path, exc
                )

        if not plot_embedded:
            flowables.append(
                Paragraph(
                    "⏳ SHAP computation pending — waterfall plot not yet available.",
                    self._styles["warning"],
                )
            )

        flowables.append(Spacer(1, 10))

        # -- Top-10 SHAP features table --
        top_features: list[dict] = getattr(explanation, "top_features", [])[:10]
        if top_features:
            flowables.append(
                Paragraph("Top Contributing Features", self._styles["subheading"])
            )

            alt_bg = _hex_to_color(self._cfg["colors"]["row_alt"])
            hdr_bg = _hex_to_color(self._cfg["colors"]["table_header"])
            white = colors.white
            inc_color = _hex_to_color("#dc3545")  # red = increases risk
            dec_color = _hex_to_color("#28a745")  # green = decreases risk

            feat_data: list[list] = [
                [
                    Paragraph("#", self._styles["body_bold"]),
                    Paragraph("Feature Name", self._styles["body_bold"]),
                    Paragraph("Feature Value", self._styles["body_bold"]),
                    Paragraph("SHAP Value", self._styles["body_bold"]),
                    Paragraph("Direction", self._styles["body_bold"]),
                ]
            ]
            feat_style_cmds = [
                ("BACKGROUND", (0, 0), (-1, 0), hdr_bg),
                ("TEXTCOLOR", (0, 0), (-1, 0), white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, 0), 9),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.lightgrey),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("ALIGN", (0, 0), (0, -1), "CENTER"),
                ("ALIGN", (3, 1), (4, -1), "CENTER"),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
            ]

            for i, feat in enumerate(top_features, start=1):
                direction = feat.get("direction", "")
                arrow = "↑" if direction == "increases_risk" else "↓"
                fv = feat.get("feature_value", 0.0)
                sv = feat.get("shap_value", 0.0)
                fv_str = f"{fv:.4g}" if isinstance(fv, float) else str(fv)
                sv_str = f"{sv:+.4f}"

                dir_color = inc_color if direction == "increases_risk" else dec_color
                dir_style = ParagraphStyle(
                    f"dir_{i}",
                    parent=self._styles["body_bold"],
                    textColor=dir_color,
                    alignment=TA_CENTER,
                )

                feat_data.append([
                    Paragraph(str(i), self._styles["body"]),
                    Paragraph(str(feat.get("feature_name", "")), self._styles["body"]),
                    Paragraph(fv_str, self._styles["mono"]),
                    Paragraph(sv_str, self._styles["mono"]),
                    Paragraph(f"{arrow} {direction.replace('_', ' ')}", dir_style),
                ])
                if i % 2 == 0:
                    feat_style_cmds.append(
                        ("BACKGROUND", (0, i), (-1, i), alt_bg)
                    )

            feat_table = Table(
                feat_data,
                colWidths=["5%", "35%", "18%", "18%", "24%"],
            )
            feat_table.setStyle(TableStyle(feat_style_cmds))
            flowables.append(feat_table)

        else:
            flowables.append(
                Paragraph(
                    "⏳ Feature attribution data not yet available.",
                    self._styles["warning"],
                )
            )

        flowables.append(Spacer(1, 12))
        return flowables

    def _build_audit_trail(self, transaction: FlaggedTransaction) -> list[Flowable]:
        """Build the audit trail section: model metadata and analyst sign-off.

        Includes model version, drift state, ADWIN window size (if applicable),
        report generation time, CIPHER version, and blank analyst decision lines.

        Args:
            transaction: Current :class:`FlaggedTransaction`.

        Returns:
            List of :class:`~reportlab.platypus.Flowable` objects.
        """
        flowables: list[Flowable] = []
        flowables.append(
            Paragraph("Audit Trail", self._styles["heading"])
        )
        flowables.append(HRFlowable(width="100%", thickness=1, color=_hex_to_color(self._cfg["colors"]["table_header"])))
        flowables.append(Spacer(1, 6))

        now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        cipher_version = self._cfg.get("cipher_version", "1.0.0")

        # Collect audit entries
        audit_entries: list[tuple[str, str]] = [
            ("Report Generated At", now_str),
            ("Transaction ID", transaction.transaction_id),
            ("Transaction Timestamp", transaction.timestamp.strftime("%Y-%m-%d %H:%M:%S UTC")),
            ("Model Version", transaction.model_version),
            ("CIPHER Version", f"v{cipher_version}"),
            ("Drift Alert Active", "YES ⚠" if transaction.drift_active else "No"),
        ]

        # ADWIN window size from drift_info
        if transaction.drift_active and transaction.drift_info:
            adwin_w = transaction.drift_info.get("adwin_window_size", "N/A")
            psi_val = transaction.drift_info.get("psi", "N/A")
            feature = transaction.drift_info.get("feature", "N/A")
            audit_entries.append(("Drift Feature", str(feature)))
            audit_entries.append(("ADWIN Window Size", str(adwin_w)))
            audit_entries.append(("PSI Score", str(psi_val)))

        # SHAP computation time
        comp_ms = getattr(transaction.explanation, "computation_time_ms", None)
        if comp_ms is not None:
            audit_entries.append(("SHAP Computation Time", f"{comp_ms:.1f} ms"))

        hdr_bg = _hex_to_color(self._cfg["colors"]["table_header"])
        alt_bg = _hex_to_color(self._cfg["colors"]["row_alt"])
        white = colors.white

        audit_data: list[list] = [
            [Paragraph("FIELD", self._styles["body_bold"]),
             Paragraph("VALUE", self._styles["body_bold"])],
        ]
        audit_style = [
            ("BACKGROUND", (0, 0), (-1, 0), hdr_bg),
            ("TEXTCOLOR", (0, 0), (-1, 0), white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, 0), 9),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.lightgrey),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("LEFTPADDING", (0, 0), (-1, -1), 8),
            ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ]
        for i, (field_name, value) in enumerate(audit_entries, start=1):
            audit_data.append([
                Paragraph(field_name, self._styles["body_bold"]),
                Paragraph(str(value), self._styles["body"]),
            ])
            if i % 2 == 0:
                audit_style.append(("BACKGROUND", (0, i), (-1, i), alt_bg))

        audit_table = Table(audit_data, colWidths=["35%", "65%"])
        audit_table.setStyle(TableStyle(audit_style))
        flowables.append(audit_table)
        flowables.append(Spacer(1, 14))

        # -- Analyst decision placeholder --
        flowables.append(
            Paragraph("Analyst Review", self._styles["subheading"])
        )
        for label in [
            "Decision (circle): &nbsp;&nbsp; CONFIRMED FRAUD &nbsp;&nbsp; | &nbsp;&nbsp; FALSE POSITIVE &nbsp;&nbsp; | &nbsp;&nbsp; UNDER INVESTIGATION",
            "Analyst Name: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Date: _______________",
            "Notes: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;",
            "_" * 100,
            "_" * 100,
        ]:
            flowables.append(Paragraph(label, self._styles["analyst"]))
            flowables.append(Spacer(1, 4))

        flowables.append(Spacer(1, 10))
        flowables.append(
            Paragraph(
                f"— End of CIPHER Report | v{cipher_version} | "
                f"CONFIDENTIAL — FOR FRAUD ANALYST USE ONLY —",
                ParagraphStyle(
                    "footer",
                    parent=self._styles["body"],
                    alignment=TA_CENTER,
                    textColor=colors.grey,
                    fontSize=7,
                ),
            )
        )

        return flowables

    # ------------------------------------------------------------------
    # Risk tier classification
    # ------------------------------------------------------------------

    def _classify_risk_tier(self, score: float) -> tuple[str, str]:
        """Map an ensemble score to a risk tier name and hex color.

        All thresholds and colors are loaded from ``config.yaml`` — nothing is
        hardcoded.  The mapping is:

        =====================  ========  =========
        Score range            Tier      Color
        =====================  ========  =========
        ``[0.0, low_thr)``     LOW       green
        ``[low_thr, med_thr)`` MEDIUM    yellow
        ``[med_thr, hi_thr)``  HIGH      orange
        ``[hi_thr, 1.0]``      CRITICAL  red
        =====================  ========  =========

        Args:
            score: Ensemble fraud probability in ``[0.0, 1.0]``.

        Returns:
            Tuple of ``(tier_name, hex_color_string)``.
        """
        tiers = self._cfg["risk_tiers"]
        clr = self._cfg["colors"]
        low_thr = float(tiers["low_threshold"])
        med_thr = float(tiers["medium_threshold"])
        hi_thr = float(tiers["high_threshold"])

        if score < low_thr:
            return ("LOW", clr["low"])
        elif score < med_thr:
            return ("MEDIUM", clr["medium"])
        elif score < hi_thr:
            return ("HIGH", clr["high"])
        else:
            return ("CRITICAL", clr["critical"])

    # ------------------------------------------------------------------
    # Report file management
    # ------------------------------------------------------------------

    def get_report_path(self, transaction_id: str) -> Optional[str]:
        """Return the path of an existing report for the given transaction.

        Checks the in-memory registry first; falls back to a directory scan
        if the registry was populated from a previous session.

        Args:
            transaction_id: Transaction identifier string.

        Returns:
            Absolute path string to the PDF, or ``None`` if no report exists.
        """
        # Check in-memory registry
        with self._reports_lock:
            if transaction_id in self._reports:
                path = self._reports[transaction_id]["path"]
                if Path(path).is_file():
                    return path

        # Fallback: scan disk (covers reports from previous process runs)
        for pdf in self._output_dir.glob(f"cipher_report_{transaction_id}_*.pdf"):
            if pdf.is_file():
                return str(pdf)

        return None

    def list_reports(self) -> list[dict]:
        """Return metadata for all PDF reports in the output directory.

        Scans the ``reports/`` directory each call to stay accurate even when
        files are added or removed outside this process.

        Returns:
            List of dicts, each with keys:

            * ``transaction_id`` — extracted from the filename.
            * ``path``           — absolute path to the PDF.
            * ``size_kb``        — file size in kilobytes (rounded to 2 dp).
            * ``generated_at``   — ISO-8601 mtime string (UTC).
        """
        results: list[dict] = []
        pattern = re.compile(r"cipher_report_(.+?)_(\d{14})\.pdf$")

        for pdf in sorted(self._output_dir.glob("cipher_report_*.pdf")):
            if not pdf.is_file():
                continue
            m = pattern.match(pdf.name)
            tx_id = m.group(1) if m else pdf.stem
            mtime = pdf.stat().st_mtime
            generated_at = datetime.fromtimestamp(mtime, tz=timezone.utc).isoformat()
            results.append(
                {
                    "transaction_id": tx_id,
                    "path": str(pdf),
                    "size_kb": round(pdf.stat().st_size / 1024, 2),
                    "generated_at": generated_at,
                }
            )

        return results

    def _scan_existing_reports(self) -> None:
        """Populate ``_reports`` registry from existing PDFs on disk.

        Called once during ``__init__`` so that :meth:`get_report_path` works
        correctly even for reports generated in previous process invocations.
        """
        pattern = re.compile(r"cipher_report_(.+?)_(\d{14})\.pdf$")
        count = 0
        for pdf in self._output_dir.glob("cipher_report_*.pdf"):
            if not pdf.is_file():
                continue
            m = pattern.match(pdf.name)
            if not m:
                continue
            tx_id = m.group(1)
            mtime = pdf.stat().st_mtime
            with self._reports_lock:
                self._reports[tx_id] = {
                    "transaction_id": tx_id,
                    "path": str(pdf),
                    "size_kb": round(pdf.stat().st_size / 1024, 2),
                    "generated_at": datetime.fromtimestamp(
                        mtime, tz=timezone.utc
                    ).isoformat(),
                }
            count += 1
        if count:
            _logger.info(
                "ReportGenerator — loaded %d existing report(s) from '%s'",
                count,
                self._output_dir,
            )


# ---------------------------------------------------------------------------
# __main__ — demo runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import types

    # ---- Mock ExplanationResult (avoid importing shap in demo) ----------
    explanation = types.SimpleNamespace(
        transaction_id="DEMO_TX_001",
        shap_values=None,
        base_value=-2.5,
        prediction=0.87,
        plain_english_summary=(
            "This transaction was flagged primarily because "
            "card_degree_1h=11 (+0.52 risk contribution) contributed to this flag. "
            "The transaction amount was also anomalous: TransactionAmt=3200.0 "
            "(+0.41 risk contribution) raised the fraud probability. "
            "Additionally, amt_zscore_24h=6.8 (+0.29 risk contribution) was a contributing factor."
        ),
        waterfall_plot_path="/tmp/nonexistent_plot.png",  # graceful fallback demo
        top_features=[
            {"feature_name": "card_degree_1h",       "shap_value": 0.52,  "feature_value": 11.0,   "direction": "increases_risk"},
            {"feature_name": "TransactionAmt",        "shap_value": 0.41,  "feature_value": 3200.0, "direction": "increases_risk"},
            {"feature_name": "amt_zscore_24h",        "shap_value": 0.29,  "feature_value": 6.8,    "direction": "increases_risk"},
            {"feature_name": "merchant_degree_1h",   "shap_value": 0.18,  "feature_value": 32.0,   "direction": "increases_risk"},
            {"feature_name": "card_tx_count_24h",     "shap_value": 0.14,  "feature_value": 45.0,   "direction": "increases_risk"},
            {"feature_name": "card_pair_count_1h",    "shap_value": 0.09,  "feature_value": 3.0,    "direction": "increases_risk"},
            {"feature_name": "addr1",                 "shap_value": -0.07, "feature_value": 204.0,  "direction": "decreases_risk"},
            {"feature_name": "P_emaildomain",         "shap_value": -0.05, "feature_value": 0.0,    "direction": "decreases_risk"},
            {"feature_name": "card4",                 "shap_value": 0.04,  "feature_value": 1.0,    "direction": "increases_risk"},
            {"feature_name": "DeviceType",            "shap_value": 0.03,  "feature_value": 0.0,    "direction": "increases_risk"},
        ],
        computation_time_ms=412.7,
    )

    # ---- Synthetic FlaggedTransaction -----------------------------------
    transaction = FlaggedTransaction(
        transaction_id="DEMO_TX_001",
        timestamp=datetime.now(timezone.utc),
        ensemble_score=0.87,
        lgbm_score=0.91,
        iso_score=0.72,
        raw_features={
            "TransactionAmt":   3200.00,
            "ProductCD":        "W",
            "card1":            9500,
            "card4":            1,
            "card6":            0,
            "addr1":            204,
            "addr2":            87,
            "P_emaildomain":    0,
            "R_emaildomain":    1,
            "DeviceType":       0,
        },
        graph_features={
            "card_degree_1h":       11,
            "merchant_degree_1h":   32,
            "card_tx_count_24h":    45,
            "card_total_amount_24h": 12800.0,
            "card_avg_amount_24h":  284.44,
            "amt_zscore_24h":       6.8,
            "card_pair_count_1h":   3,
            "card_pair_count_24h":  7,
        },
        explanation=explanation,
        drift_active=True,
        drift_info={
            "feature": "card_degree_1h",
            "psi": 0.24,
            "adwin_window_size": 312,
            "alert_type": "psi_critical",
        },
        model_version="v3.1",
    )

    # ---- Generate report ------------------------------------------------
    print("\n" + "=" * 65)
    print("CIPHER — Demo PDF Report Generator")
    print("=" * 65)

    generator = ReportGenerator()
    pdf_path = generator.generate_report(transaction)
    size_kb = Path(pdf_path).stat().st_size / 1024

    print(f"\n[OK] Report generated successfully!")
    print(f"  Path  : {pdf_path}")
    print(f"  Size  : {size_kb:.1f} KB")
    print(f"  Tier  : {generator._classify_risk_tier(transaction.ensemble_score)[0]}")
    print()

    # ---- Open the PDF ---------------------------------------------------
    try:
        if sys.platform == "win32":
            os.startfile(pdf_path)  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.run(["open", pdf_path], check=True)
        else:
            subprocess.run(["xdg-open", pdf_path], check=True)
    except Exception as exc:
        print(f"  (Could not auto-open PDF: {exc})")
        print(f"  Open manually: {pdf_path}")
