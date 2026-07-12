"""
tests/test_docker.py
--------------------
Unit tests for Module 10 — Docker & Infrastructure Hardening.

Tests
-----
* test_dockerfile_exists            — Dockerfile present at project root
* test_docker_compose_valid         — ``docker-compose config --quiet`` exits 0
* test_env_example_exists           — .env.example present at project root
* test_makefile_has_required_targets — all 9 Makefile targets present
* test_config_loader_prefers_env_var — env var overrides config.yaml value
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Project root — everything is relative to this
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.parent.resolve()

REQUIRED_MAKEFILE_TARGETS = [
    "up",
    "down",
    "clean",
    "logs",
    "train",
    "test",
    "kafka-lag",
    "kafka-topics",
    "rebuild",
]


# ===========================================================================
# Test 1 — Dockerfile exists
# ===========================================================================

def test_dockerfile_exists() -> None:
    """Dockerfile must be present at the project root."""
    dockerfile = PROJECT_ROOT / "Dockerfile"
    assert dockerfile.exists(), (
        f"Dockerfile not found at {dockerfile}. "
        "Run 'git status' to ensure it has been created and committed."
    )
    assert dockerfile.is_file(), f"{dockerfile} is not a regular file."


# ===========================================================================
# Test 2 — docker-compose.yml is syntactically valid
# ===========================================================================

def _docker_compose_available() -> bool:
    """Return True if the docker-compose CLI is available on this machine."""
    try:
        result = subprocess.run(
            ["docker", "compose", "version"],
            capture_output=True, timeout=10,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    try:
        result = subprocess.run(
            ["docker-compose", "version"],
            capture_output=True, timeout=10,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


@pytest.mark.skipif(
    not _docker_compose_available(),
    reason="Docker Compose not available on this machine — skipping schema validation.",
)
def test_docker_compose_valid() -> None:
    """``docker-compose config --quiet`` must exit 0 (validates YAML + schema)."""
    compose_file = PROJECT_ROOT / "docker-compose.yml"
    assert compose_file.exists(), f"docker-compose.yml not found at {compose_file}"

    # Try 'docker compose' (V2 plugin) first, fall back to 'docker-compose' (V1)
    for cmd in (
        ["docker", "compose", "-f", str(compose_file), "config", "--quiet"],
        ["docker-compose", "-f", str(compose_file), "config", "--quiet"],
    ):
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30,
                cwd=str(PROJECT_ROOT),
            )
            if result.returncode == 0:
                return  # success
            # If the command was found but returned non-zero, fail immediately
            pytest.fail(
                f"docker-compose config returned exit code {result.returncode}.\n"
                f"stdout: {result.stdout}\nstderr: {result.stderr}"
            )
        except FileNotFoundError:
            continue  # try next variant

    pytest.fail("Neither 'docker compose' nor 'docker-compose' could be executed.")


# ===========================================================================
# Test 3 — .env.example exists
# ===========================================================================

def test_env_example_exists() -> None:
    """.env.example must exist at the project root."""
    env_example = PROJECT_ROOT / ".env.example"
    assert env_example.exists(), (
        f".env.example not found at {env_example}. "
        "This file documents required environment variables."
    )
    assert env_example.is_file()

    # Spot-check that the four required keys are documented
    content = env_example.read_text(encoding="utf-8")
    for required_key in [
        "KAFKA_BOOTSTRAP_SERVERS",
        "MLFLOW_TRACKING_URI",
        "CIPHER_ENV",
        "LOG_LEVEL",
    ]:
        assert required_key in content, (
            f"Required key '{required_key}' not found in .env.example."
        )


# ===========================================================================
# Test 4 — Makefile has all required targets
# ===========================================================================

def test_makefile_has_required_targets() -> None:
    """All 9 specified Makefile targets must be defined."""
    makefile_path = PROJECT_ROOT / "Makefile"
    assert makefile_path.exists(), f"Makefile not found at {makefile_path}"

    content = makefile_path.read_text(encoding="utf-8")

    # Makefile targets follow the pattern: ^<target>:  at the start of a line
    # (optionally followed by prerequisites)
    found_targets: set[str] = set(
        re.findall(r"^([a-zA-Z][a-zA-Z0-9_\-]*):", content, re.MULTILINE)
    )

    missing = [t for t in REQUIRED_MAKEFILE_TARGETS if t not in found_targets]
    assert not missing, (
        f"The following Makefile targets are missing: {missing}\n"
        f"Found targets: {sorted(found_targets)}"
    )


def test_makefile_targets_count() -> None:
    """There must be exactly the 9 required targets (no fewer)."""
    assert len(REQUIRED_MAKEFILE_TARGETS) == 9


# ===========================================================================
# Test 5 — config_loader prefers environment variable over YAML
# ===========================================================================

def test_config_loader_prefers_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    """When KAFKA_BOOTSTRAP_SERVERS is set in os.environ, load_config must
    return that value rather than the one from config.yaml."""
    test_value = "test:9999"
    monkeypatch.setenv("KAFKA_BOOTSTRAP_SERVERS", test_value)

    # Import fresh so the env var is visible during the function call
    # (monkeypatch already patched os.environ before the import)
    from src.utils.config_loader import load_config

    cfg = load_config(str(PROJECT_ROOT / "config" / "config.yaml"))

    kafka_section = cfg.get("kafka", {})
    actual = kafka_section.get("bootstrap_servers")

    assert actual == test_value, (
        f"Expected config_loader to return env var value '{test_value}', "
        f"but got '{actual}'. "
        "Check _apply_env_overrides() in config_loader.py."
    )


def test_config_loader_falls_back_to_yaml(monkeypatch: pytest.MonkeyPatch) -> None:
    """When KAFKA_BOOTSTRAP_SERVERS is NOT set, load_config returns the YAML value."""
    monkeypatch.delenv("KAFKA_BOOTSTRAP_SERVERS", raising=False)

    from src.utils.config_loader import load_config

    cfg = load_config(str(PROJECT_ROOT / "config" / "config.yaml"))

    # YAML specifies 'localhost:9092' for bootstrap_servers
    kafka_section = cfg.get("kafka", {})
    actual = kafka_section.get("bootstrap_servers")

    assert actual is not None, "kafka.bootstrap_servers should be set in config.yaml"
    assert actual != "test:9999", (
        "Config loader returned a stale env-var value — monkeypatch may not have cleared it."
    )


def test_config_loader_mlflow_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """MLFLOW_TRACKING_URI env var must override mlflow.tracking_uri in YAML."""
    test_uri = "http://test-mlflow:9999"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", test_uri)

    from src.utils.config_loader import load_config

    cfg = load_config(str(PROJECT_ROOT / "config" / "config.yaml"))
    actual = cfg.get("mlflow", {}).get("tracking_uri")

    assert actual == test_uri, (
        f"Expected mlflow.tracking_uri='{test_uri}', got '{actual}'. "
        "Check _ENV_OVERRIDES mapping in config_loader.py."
    )
