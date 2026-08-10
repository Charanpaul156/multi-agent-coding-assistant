"""Tests for unified diff generation."""

from __future__ import annotations

from backend.domain.change_models import ChangeOperation, FileChange
from backend.infrastructure.diff_generator import diff_for_change, generate_diff


def test_generate_diff_modify_shows_changes():
    diff = generate_diff(
        "a.py",
        "modify",
        "x = 1\n",
        "x = 2\n",
    )
    assert "--- a.py" in diff
    assert "+++ a.py" in diff
    assert "-x = 1" in diff
    assert "+x = 2" in diff


def test_generate_diff_multi_line():
    old = "a = 1\nb = 2\nc = 3\n"
    new = "a = 1\nb = 20\nc = 3\nd = 4\n"
    diff = generate_diff("f.py", "modify", old, new)
    assert "-b = 2" in diff
    assert "+b = 20" in diff
    assert "+d = 4" in diff
    assert " a = 1" in diff


def test_generate_diff_create():
    diff = generate_diff("new.py", "create", None, "print('hi')\n")
    assert "+print('hi')" in diff


def test_generate_diff_unmodified_returns_empty_header():
    diff = generate_diff("a.py", "modify", "same\n", "same\n")
    # No changed lines, but headers may still appear. Just check no +/- content.
    assert "-same" not in diff
    assert "+same" not in diff


def test_diff_for_change_modify():
    change = FileChange(
        file_path="service.py",
        operation=ChangeOperation.MODIFY,
        new_content="b = 2\n",
        original_content="b = 1\n",
    )
    diff = diff_for_change(change)
    assert "-b = 1" in diff
    assert "+b = 2" in diff


def test_diff_for_change_create_no_original():
    change = FileChange(
        file_path="new.py",
        operation=ChangeOperation.CREATE,
        new_content="x = 1\n",
    )
    diff = diff_for_change(change)
    assert "+x = 1" in diff
