from __future__ import annotations

import ast
import importlib
from pathlib import Path

import pytest

import provenance_aliasing
import provenance_aliasing.api


# Wheel installation nests api/ inside provenance_aliasing/. Editable installation
# maps the same import namespace to sibling core/ and api/ source directories.
PACKAGE_ROOTS = {
    "provenance_aliasing": Path(provenance_aliasing.__file__).resolve().parent,
    "provenance_aliasing.api": Path(provenance_aliasing.api.__file__).resolve().parent,
}
TEST_DIR = Path(__file__).resolve().parent

MODULE_TEST_MAP = {
    "provenance_aliasing.__init__": "test_package_incidence.py",
    "provenance_aliasing.__main__": "test_cli.py",
    "provenance_aliasing.cli": "test_cli.py",
    "provenance_aliasing.incidence": "test_package_incidence.py",
    "provenance_aliasing.entropy": "test_entropy_ceiling_structure.py",
    "provenance_aliasing.ceiling": "test_entropy_ceiling_structure.py",
    "provenance_aliasing.structure": "test_entropy_ceiling_structure.py",
    "provenance_aliasing.grain": "test_grain_guards_report.py",
    "provenance_aliasing.guards": "test_grain_guards_report.py",
    "provenance_aliasing.report": "test_grain_guards_report.py",
    "provenance_aliasing.metrics": "test_metrics_calibration.py",
    "provenance_aliasing.calibration": "test_metrics_calibration.py",
    "provenance_aliasing.design": "test_design.py",
    "provenance_aliasing.adapters.__init__": "test_adapters.py",
    "provenance_aliasing.adapters.tidy": "test_adapters.py",
    "provenance_aliasing.adapters.pride": "test_adapters.py",
    "provenance_aliasing.adapters.aggregated_supplement": "test_adapters.py",
    "provenance_aliasing.api.__init__": "test_validation.py",
    "provenance_aliasing.api.config": "test_config.py",
    "provenance_aliasing.api.validation": "test_validation.py",
    "provenance_aliasing.api.diagnostics": "test_diagnostics.py",
    "provenance_aliasing.api.results": "test_results.py",
    "provenance_aliasing.api.cli": "test_cli_workflow.py",
    "provenance_aliasing.api.__main__": "test_cli_workflow.py",
    "provenance_aliasing.api._tabular": "test_tabular.py",
    "provenance_aliasing.api.run_record": "test_run_record.py",
}


def discovered_module_files() -> dict[str, Path]:
    modules: dict[str, Path] = {}
    for prefix, package_dir in PACKAGE_ROOTS.items():
        for path in package_dir.rglob("*.py"):
            path = path.resolve()
            relative = path.relative_to(package_dir).with_suffix("")
            name = prefix + "." + ".".join(relative.parts)
            # The wheel's API files appear under both traversals. Deduplicate the
            # same module/file pair, while still detecting conflicting sources.
            if name in modules:
                assert modules[name] == path, f"multiple source files for {name}: {modules[name]}, {path}"
            modules[name] = path
    return modules


MODULE_FILES = discovered_module_files()


def import_name(module_name: str) -> str:
    return "provenance_aliasing" if module_name == "provenance_aliasing.__init__" else module_name.removesuffix(".__init__")


def test_every_source_module_has_an_explicit_test_owner() -> None:
    assert set(MODULE_FILES) == set(MODULE_TEST_MAP)
    for test_file in MODULE_TEST_MAP.values():
        assert (TEST_DIR / test_file).is_file()


@pytest.mark.parametrize("module_name", sorted(MODULE_TEST_MAP))
def test_every_module_imports_from_its_expected_source(module_name: str) -> None:
    module = importlib.import_module(import_name(module_name))
    assert Path(module.__file__).resolve() == MODULE_FILES[module_name]


@pytest.mark.parametrize("module_name,path", sorted(MODULE_FILES.items()), ids=sorted(MODULE_FILES))
def test_no_duplicate_top_level_public_definitions(module_name: str, path: Path) -> None:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    definitions: dict[str, list[int]] = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and not node.name.startswith("_"):
            definitions.setdefault(node.name, []).append(node.lineno)
    duplicates = {name: lines for name, lines in definitions.items() if len(lines) > 1}
    assert not duplicates, f"duplicate public definitions in {module_name}: {duplicates}"
