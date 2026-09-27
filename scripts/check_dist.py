"""Check the fresh sdist and wheel against the repository's distribution contract."""
from pathlib import Path
import sys
import tarfile
import zipfile


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    directory = Path(sys.argv[1]) if len(sys.argv) > 1 else root / "dist"
    wheels = list(directory.glob("*.whl"))
    archives = list(directory.glob("*.tar.gz"))
    assert len(wheels) == len(archives) == 1, "expected one wheel and one sdist"
    required = {"README.md", "REPRODUCIBILITY.md", "RELEASING.md", "pyproject.toml",
                "MANIFEST.in", "CITATION_METADATA_NEEDED.md"}
    required.update(str(path.relative_to(root)).replace("\\", "/")
                    for path in (root / "tests").rglob("*")
                    if path.suffix in {".py", ".csv", ".json"})
    required.update(str(path.relative_to(root)).replace("\\", "/")
                    for path in (root / "scripts").glob("*.py"))
    required.update(name for name in ("LICENSE", "CITATION.cff") if (root / name).exists())
    with tarfile.open(archives[0]) as archive:
        names = {item.name.partition("/")[2] for item in archive.getmembers() if item.isfile()}
        assert required <= names, f"sdist missing: {required - names}"
    expected = {"provenance_aliasing/" + str(path.relative_to(root / "core")).replace("\\", "/")
                for path in (root / "core").rglob("*.py")}
    expected.update("provenance_aliasing/api/" + path.name for path in (root / "api").glob("*.py"))
    with zipfile.ZipFile(wheels[0]) as wheel:
        runtime = {name for name in wheel.namelist() if ".dist-info/" not in name}
        assert runtime == expected, f"unexpected/missing runtime content: {runtime ^ expected}"
        for item in wheel.infolist():
            assert item.file_size < 1_000_000, f"unexpected oversized wheel member: {item.filename}"
            assert "__pycache__" not in item.filename and not item.filename.endswith(".pyc")
    print(f"PASS: {archives[0].name}, {wheels[0].name}")


if __name__ == "__main__":
    main()
