"""Run outside the checkout with a clean wheel environment; optional --excel phase."""
from importlib.metadata import version
from importlib.util import find_spec
import json
import os
from pathlib import Path
import subprocess
import sys
import sysconfig
import tempfile

import pandas as pd
import provenance_aliasing as package
import provenance_aliasing.api as api
from provenance_aliasing.metrics import jensen_shannon


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    assert not Path.cwd().resolve().is_relative_to(root), "run outside checkout"
    assert Path(package.__file__).resolve().is_relative_to(Path(sysconfig.get_path("purelib")).resolve())
    assert package.__version__ == version("provenance-aliasing-diagnostics")
    for name in api.__all__:
        assert getattr(api, name) is not None
    assert jensen_shannon([[0.5, 0.5], [1.0, 0.0]]).distance.shape == (2, 2)
    fixtures = root / "tests" / "fixtures" / "synthetic"
    metadata = fixtures / "bulk-rna-seq.csv"
    config_path = fixtures / "bulk-rna-seq.mapping.json"
    frame = pd.read_csv(metadata, dtype=str, keep_default_na=False)
    config = api.AnalysisConfig.from_json(config_path)
    assert api.validate(frame, config=config).valid
    assert api.diagnose(frame, config=config).validation.valid
    with tempfile.TemporaryDirectory() as temporary:
        work = Path(temporary)
        if "--excel" in sys.argv:
            from provenance_aliasing.adapters import from_excel, from_long
            excel = work / "metadata.xlsx"
            frame.to_excel(excel, index=False)
            columns = dict(source="study_id", unit=["study_id", "sample_id"], group="condition")
            actual = from_excel(excel, **columns)
            expected = from_long(frame, **columns)
            pd.testing.assert_frame_equal(actual.incidence, expected.incidence)
            print("PASS: optional Excel adapter")
            return
        assert find_spec("openpyxl") is None, "base wheel must be tested without Excel extra"
        executable = Path(sysconfig.get_path("scripts")) / ("provenance-aliasing.exe" if os.name == "nt" else "provenance-aliasing")
        prefixes = [[str(executable)], [sys.executable, "-m", "provenance_aliasing"],
                    [sys.executable, "-m", "provenance_aliasing.api"]]
        for index, prefix in enumerate(prefixes):
            def invoke(*args: str, code: int = 0):
                result = subprocess.run(prefix + list(args), cwd=work, capture_output=True, text=True)
                assert result.returncode == code, result.stderr
                return result
            assert "{validate,diagnose,verify}" in invoke("--help").stdout
            common = [str(metadata), "--config", str(config_path), "--quiet"]
            assert json.loads(invoke("validate", *common).stdout)["valid"]
            report, record = work / f"report-{index}.json", work / f"record-{index}.json"
            invoke("diagnose", *common, "--output", str(report), "--run-record", str(record))
            args = [str(record), "--metadata", str(metadata), "--config", str(config_path), "--report", str(report), "--quiet"]
            assert json.loads(invoke("verify", *args).stdout)["verified"]
            assert "error:" in invoke("validate", code=2).stderr
            assert "already exists" in invoke("diagnose", *common, "--output", str(report), code=3).stderr
            report.write_text("{}", encoding="utf-8")
            assert not json.loads(invoke("verify", *args, code=2).stdout)["verified"]
    print(f"PASS: base wheel, API, SciPy, all three CLI interfaces; import={package.__file__}")


if __name__ == "__main__":
    main()
