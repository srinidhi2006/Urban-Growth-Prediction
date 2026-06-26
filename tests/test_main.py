"""
Unit tests for the central CLI entry point validation functions.
"""

import sys
from unittest.mock import patch
import pytest
from main import verify_setup, main

def test_verify_setup_passes():
    """Asserts that verify_setup returns True when all paths exist in the environment."""
    assert verify_setup() is True

def test_main_help_arguments():
    """Asserts that running main.py with --help runs successfully without raising argparse errors."""
    with patch.object(sys, "argv", ["main.py", "--help"]):
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 0

def test_main_verify_setup_cmd():
    """Asserts that running main.py with --verify-setup executes the verification routine."""
    with patch.object(sys, "argv", ["main.py", "--verify-setup"]):
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 0

def test_main_future_command_warning():
    """Asserts that running main.py with future commands issues warnings and exits cleanly."""
    with patch.object(sys, "argv", ["main.py", "--ingest-data"]):
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 0
