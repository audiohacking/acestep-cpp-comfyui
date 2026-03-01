"""Tests for install.py helpers.

Validates the _binary_exists function that checks both build/ and build/bin/
to handle ggml's CMAKE_RUNTIME_OUTPUT_DIRECTORY default behaviour.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

# Load install.py by absolute path (it is a standalone script, not a package).
_repo_root = Path(__file__).parent.parent
_install_spec = importlib.util.spec_from_file_location(
    "install",
    str(_repo_root / "install.py"),
)
_install_module = importlib.util.module_from_spec(_install_spec)
# Prevent install() from running on import by making __name__ != "__main__"
sys.modules["install"] = _install_module
_install_spec.loader.exec_module(_install_module)

_binary_exists = _install_module._binary_exists


# ---------------------------------------------------------------------------
# _binary_exists
# ---------------------------------------------------------------------------

class TestBinaryExists:
    """_binary_exists checks build/ and build/bin/ (ggml's default output dir)."""

    def test_found_directly_in_build(self, tmp_path):
        """Binary placed directly in build/ is detected (new cmake configure)."""
        (tmp_path / "ace-qwen3").write_text("mock")
        assert _binary_exists(str(tmp_path), "ace-qwen3") is True

    def test_found_in_build_bin(self, tmp_path):
        """Binary placed in build/bin/ (ggml default) is detected."""
        bin_dir = tmp_path / "bin"
        bin_dir.mkdir()
        (bin_dir / "ace-qwen3").write_text("mock")
        assert _binary_exists(str(tmp_path), "ace-qwen3") is True

    def test_not_found_returns_false(self, tmp_path):
        """Returns False when binary is absent from both locations."""
        assert _binary_exists(str(tmp_path), "ace-qwen3") is False

    def test_both_locations_present(self, tmp_path):
        """Returns True when binary exists in build/ (checked before build/bin/)."""
        (tmp_path / "ace-qwen3").write_text("in_build")
        bin_dir = tmp_path / "bin"
        bin_dir.mkdir()
        (bin_dir / "ace-qwen3").write_text("in_bin")
        assert _binary_exists(str(tmp_path), "ace-qwen3") is True

    def test_wrong_name_not_found(self, tmp_path):
        """Binary with a different name is not falsely matched."""
        (tmp_path / "dit-vae").write_text("mock")
        assert _binary_exists(str(tmp_path), "ace-qwen3") is False

    def test_dit_vae_in_bin(self, tmp_path):
        """Works for dit-vae as well as ace-qwen3."""
        bin_dir = tmp_path / "bin"
        bin_dir.mkdir()
        (bin_dir / "dit-vae").write_text("mock")
        assert _binary_exists(str(tmp_path), "dit-vae") is True
