"""
tests/test_report_generator.py
--------------------------------
Unit tests for the CIPHER PDF report generator.

All tests create synthetic :class:`FlaggedTransaction` objects with mocked
:class:`ExplanationResult` values so no trained model or SHAP computation
is required.  Reports are generated inside pytest's ``tmp_path`` fixture so
no files are left in the repo.

Tests
-----
* test_report_file_created                   — PDF exists at the expected path
* test_risk_tier_classification              — 0.1/0.4/0.6/0.8 → LOW/MEDIUM/HIGH/CRITICAL
* test_report_path_deterministic_on_id       — get_report_path() finds an existing report
* test_missing_shap_plot_handled_gracefully  — non-existent plot_path, no exception
* test_list_reports_returns_correct_fields   — 2 reports, dicts have all required keys
* test_async_generation_calls_callback       — threading.Event fires within 15 s
"""

from __future__ import annotations

import sys
import threading
import types
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import pytest

# Ensure project root is on path (handled by conftest.py in CI, added here
# defensively so the file can be run directly too).
sys.path.insert(0, str(Path(__file__).parent.parent))

# ---------------------------------------------------------------------------
# Helpers — synthetic objects
# ---------------------------------------------------------------------------


def _make_explanation(
    waterfall_path: str = "/tmp/nonexistent_shap.png",
) -> types.SimpleNamespace:
    """Return a mock ExplanationResult namespace."""
    return types.SimpleNamespace(
        transaction_id="TX_TEST",
        shap_values=None,
        base_value=-2.5,
        prediction=0.87,
        plain_english_summary=(
            "card_degree_1h=11 (+0.52) raised the fraud probability. "
            "TransactionAmt=3200.0 (+0.41) contributed. "
            "amt_zscore_24h=6.8 (+0.29) was a factor."
        ),
        waterfall_plot_path=waterfall_path,
        top_features=[
            {
                "feature_name": "card_degree_1h",
                "shap_value": 0.52,
                "feature_value": 11.0,
                "direction": "increases_risk",
            },
            {
                "feature_name": "TransactionAmt",
                "shap_value": 0.41,
                "feature_value": 3200.0,
                "direction": "increases_risk",
            },
            {
                "feature_name": "amt_zscore_24h",
                "shap_value": 0.29,
                "feature_value": 6.8,
                "direction": "increases_risk",
            },
            {
                "feature_name": "card_tx_count_24h",
                "shap_value": -0.12,
                "feature_value": 45.0,
                "direction": "decreases_risk",
            },
        ],
        computation_time_ms=312.5,
    )


def _make_transaction(
    tx_id: str = "TX_UNIT_001",
    ensemble_score: float = 0.87,
    waterfall_path: str = "/tmp/nonexistent.png",
) -> "FlaggedTransaction":
    """Return a synthetic FlaggedTransaction."""
    # Import here so the fixture can also patch the output_dir.
    from src.serving_layer.report_generator import FlaggedTransaction

    return FlaggedTransaction(
        transaction_id=tx_id,
        timestamp=datetime(2024, 6, 15, 14, 30, 0, tzinfo=timezone.utc),
        ensemble_score=ensemble_score,
        lgbm_score=ensemble_score + 0.04,
        iso_score=max(0.0, ensemble_score - 0.15),
        raw_features={
            "TransactionAmt": 3200.0,
            "ProductCD": "W",
            "card1": 9500,
            "addr1": 204,
        },
        graph_features={
            "card_degree_1h": 11,
            "merchant_degree_1h": 32,
            "card_tx_count_24h": 45,
            "card_avg_amount_24h": 284.44,
        },
        explanation=_make_explanation(waterfall_path),
        drift_active=True,
        drift_info={
            "feature": "card_degree_1h",
            "psi": 0.24,
            "adwin_window_size": 312,
        },
        model_version="v3.1",
    )


# ---------------------------------------------------------------------------
# Shared fixture — ReportGenerator with tmp_path output
# ---------------------------------------------------------------------------

MINIMAL_CFG = {
    "report": {
        "output_dir": "",          # overridden per-test with tmp_path
        "cipher_version": "1.0.0",
        "risk_tiers": {
            "low_threshold": 0.3,
            "medium_threshold": 0.5,
            "high_threshold": 0.75,
        },
        "colors": {
            "low": "#28a745",
            "medium": "#ffc107",
            "high": "#fd7e14",
            "critical": "#dc3545",
            "header_text": "#ffffff",
            "table_header": "#1a1a2e",
            "row_alt": "#f8f9fa",
        },
        "page": {"size": "A4", "margin_inches": 0.75},
        "shap_image": {
            "width_points": 450,
            "height_points": 300,
            "timeout_seconds": 10,
        },
        "threadpool_workers": 2,
    }
}


@pytest.fixture
def generator(tmp_path):
    """Provide a ReportGenerator writing to pytest tmp_path."""
    cfg = {k: v for k, v in MINIMAL_CFG.items()}
    cfg["report"] = {**MINIMAL_CFG["report"], "output_dir": str(tmp_path)}

    import src.serving_layer.report_generator as mod
    with patch.object(mod, "load_config", return_value=cfg):
        from src.serving_layer.report_generator import ReportGenerator
        gen = ReportGenerator()
    return gen


# ---------------------------------------------------------------------------
# Test 1 — PDF file is created at the expected path
# ---------------------------------------------------------------------------


def test_report_file_created(generator, tmp_path):
    """generate_report() must produce a real PDF file in the output directory."""
    tx = _make_transaction()
    path = generator.generate_report(tx)

    assert Path(path).exists(), f"Expected PDF at '{path}' but file not found."
    assert path.endswith(".pdf"), f"Expected .pdf extension, got: {path}"
    assert Path(path).parent == tmp_path, (
        f"PDF written to wrong directory: {Path(path).parent}"
    )
    assert Path(path).stat().st_size > 1024, (
        "PDF is unreasonably small — likely empty or corrupt."
    )


# ---------------------------------------------------------------------------
# Test 2 — Risk tier classification
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "score, expected_tier",
    [
        (0.10, "LOW"),
        (0.40, "MEDIUM"),
        (0.60, "HIGH"),
        (0.80, "CRITICAL"),
    ],
)
def test_risk_tier_classification(generator, score, expected_tier):
    """_classify_risk_tier() must map scores to the correct tier names."""
    tier_name, hex_color = generator._classify_risk_tier(score)
    assert tier_name == expected_tier, (
        f"Score {score}: expected tier '{expected_tier}', got '{tier_name}'"
    )
    assert hex_color.startswith("#"), (
        f"Expected a hex color string starting with '#', got: {hex_color}"
    )


# ---------------------------------------------------------------------------
# Test 3 — get_report_path() is deterministic on transaction_id
# ---------------------------------------------------------------------------


def test_report_path_deterministic_on_transaction_id(generator, tmp_path):
    """get_report_path() must return the existing PDF path for a given transaction_id."""
    tx = _make_transaction(tx_id="TX_PATH_CHECK")

    # No report yet — should return None
    assert generator.get_report_path("TX_PATH_CHECK") is None

    # Generate the report
    path = generator.generate_report(tx)
    assert Path(path).exists()

    # Now get_report_path must find it
    found = generator.get_report_path("TX_PATH_CHECK")
    assert found is not None, "get_report_path() returned None after report was generated"
    assert found == path or Path(found).name == Path(path).name, (
        f"Path mismatch: expected '{path}', got '{found}'"
    )


# ---------------------------------------------------------------------------
# Test 4 — Missing SHAP waterfall plot handled gracefully
# ---------------------------------------------------------------------------


def test_missing_shap_plot_handled_gracefully(generator):
    """If waterfall_plot_path points to a non-existent file, no exception is raised."""
    tx = _make_transaction(
        tx_id="TX_NO_SHAP",
        waterfall_path="/absolutely/nonexistent/path/shap_TX.png",
    )
    # Should not raise
    try:
        path = generator.generate_report(tx)
    except Exception as exc:
        pytest.fail(
            f"generate_report() raised an exception when the SHAP plot was "
            f"missing: {type(exc).__name__}: {exc}"
        )

    assert Path(path).exists(), "PDF should still be created when plot is missing"


# ---------------------------------------------------------------------------
# Test 5 — list_reports() returns dicts with all required keys
# ---------------------------------------------------------------------------


def test_list_reports_returns_correct_fields(generator):
    """list_reports() must return dicts with transaction_id, path, size_kb, generated_at."""
    required_keys = {"transaction_id", "path", "size_kb", "generated_at"}

    tx1 = _make_transaction(tx_id="TX_LIST_001", ensemble_score=0.82)
    tx2 = _make_transaction(tx_id="TX_LIST_002", ensemble_score=0.45)

    generator.generate_report(tx1)
    generator.generate_report(tx2)

    reports = generator.list_reports()
    assert len(reports) >= 2, (
        f"Expected at least 2 reports in list, got {len(reports)}"
    )

    for report in reports:
        missing = required_keys - set(report.keys())
        assert not missing, (
            f"Report dict missing keys: {missing}. Dict: {report}"
        )
        assert isinstance(report["transaction_id"], str)
        assert isinstance(report["path"], str)
        assert isinstance(report["size_kb"], (int, float))
        assert report["size_kb"] > 0, "size_kb must be positive"
        assert isinstance(report["generated_at"], str)
        # generated_at should be an ISO-8601 string
        try:
            datetime.fromisoformat(report["generated_at"].replace("Z", "+00:00"))
        except ValueError:
            pytest.fail(
                f"generated_at is not a valid ISO-8601 string: {report['generated_at']}"
            )


# ---------------------------------------------------------------------------
# Test 6 — Async generation calls callback within 15 seconds
# ---------------------------------------------------------------------------


def test_async_generation_calls_callback(generator):
    """generate_report_async() must invoke callback(file_path) within 15 seconds."""
    done_event = threading.Event()
    received_paths: list[str] = []

    def on_complete(file_path: str) -> None:
        received_paths.append(file_path)
        done_event.set()

    tx = _make_transaction(tx_id="TX_ASYNC_001", ensemble_score=0.61)
    generator.generate_report_async(tx, callback=on_complete)

    fired = done_event.wait(timeout=15.0)

    assert fired, (
        "generate_report_async() callback was NOT called within 15 seconds. "
        "This suggests the ThreadPoolExecutor task did not complete."
    )
    assert len(received_paths) == 1, (
        f"Expected exactly 1 callback invocation, got {len(received_paths)}"
    )
    assert received_paths[0].endswith(".pdf"), (
        f"Callback received non-PDF path: {received_paths[0]}"
    )
    assert Path(received_paths[0]).exists(), (
        f"PDF path returned by callback does not exist: {received_paths[0]}"
    )
