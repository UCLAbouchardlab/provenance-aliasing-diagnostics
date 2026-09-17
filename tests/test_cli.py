from __future__ import annotations

import os
import subprocess
import sys
import sysconfig
from importlib.metadata import distribution, version
from pathlib import Path


def test_installed_distribution_exposes_console_entry_point() -> None:
    entries = [
        entry for entry in distribution("provenance-aliasing-diagnostics").entry_points
        if entry.group == "console_scripts" and entry.name == "provenance-aliasing"
    ]
    assert len(entries) == 1
    assert entries[0].value == "provenance_aliasing.api.cli:main"


def test_installed_console_script_works_outside_checkout(tmp_path: Path) -> None:
    suffix = ".exe" if os.name == "nt" else ""
    executable = Path(sysconfig.get_path("scripts")) / f"provenance-aliasing{suffix}"
    completed = subprocess.run(
        [str(executable), "--version"], cwd=tmp_path, text=True, capture_output=True,
        check=True,
    )
    assert completed.stdout.strip() == (
        "provenance-aliasing " + version("provenance-aliasing-diagnostics")
    )
    assert not completed.stderr


def test_module_help_and_unknown_command(tmp_path: Path) -> None:
    help_result = subprocess.run(
        [sys.executable, "-m", "provenance_aliasing", "--help"],
        cwd=tmp_path, text=True, capture_output=True, check=True,
    )
    assert "--version" in help_result.stdout
    assert "planned" in help_result.stdout
    invalid = subprocess.run(
        [sys.executable, "-m", "provenance_aliasing", "diagnose"],
        cwd=tmp_path, text=True, capture_output=True,
    )
    assert invalid.returncode == 2
    assert "unrecognized arguments" in invalid.stderr


def test_import_and_help_do_not_load_optional_or_numerical_modules(tmp_path: Path) -> None:
    script = (
        "import sys; import provenance_aliasing; "
        "assert not {'numpy','pandas','scipy','openpyxl'} & sys.modules.keys(); "
        "from provenance_aliasing.cli import main; "
        "assert not {'numpy','pandas','scipy','openpyxl'} & sys.modules.keys()"
    )
    subprocess.run([sys.executable, "-c", script], cwd=tmp_path, check=True)


def test_required_import_failure_is_not_hidden(monkeypatch) -> None:
    import provenance_aliasing as package
    original = package.importlib.import_module

    def broken_import(name, package_name=None):
        if name == ".incidence":
            raise ImportError("missing required numerical dependency")
        return original(name, package_name)

    monkeypatch.setattr(package.importlib, "import_module", broken_import)
    import pytest
    with pytest.raises(ImportError, match="missing required numerical dependency"):
        package.__getattr__("Corpus")
