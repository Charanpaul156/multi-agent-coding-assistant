"""Tests for application settings and configuration parsing."""

import json
from config.settings import Settings


def test_settings_allowed_repository_roots_json_array(monkeypatch) -> None:
    monkeypatch.setenv("ALLOWED_REPOSITORY_ROOTS", json.dumps(["C:/repo1", "C:/repo2"]))
    s = Settings()
    assert s.allowed_repository_roots == ["C:/repo1", "C:/repo2"]


def test_settings_allowed_repository_roots_comma_separated(monkeypatch) -> None:
    monkeypatch.setenv("ALLOWED_REPOSITORY_ROOTS", "C:/repo1, C:/repo2")
    s = Settings()
    assert s.allowed_repository_roots == ["C:/repo1", "C:/repo2"]


def test_settings_allowed_repository_roots_single_path(monkeypatch) -> None:
    monkeypatch.setenv("ALLOWED_REPOSITORY_ROOTS", "C:/single/repo")
    s = Settings()
    assert s.allowed_repository_roots == ["C:/single/repo"]


def test_settings_allowed_repository_roots_empty(monkeypatch) -> None:
    monkeypatch.setenv("ALLOWED_REPOSITORY_ROOTS", "")
    s = Settings()
    assert s.allowed_repository_roots == []
