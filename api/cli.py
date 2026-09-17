"""File-based validation and diagnosis using the public metadata API."""
from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Sequence
from importlib.metadata import version
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any

import provenance_aliasing.api as public_api

_PACKAGE = "provenance-aliasing-diagnostics"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="provenance-aliasing",
        description="Validate observation metadata and diagnose provenance/group-label aliasing.",
        epilog=(
            "Also available as: python -m provenance_aliasing.api. "
            "Exit codes: 0 completed/matching inputs; 2 invalid input/configuration/usage or verification mismatch; "
            "3 computation or report-output failure; 4 partial diagnosis limited by resources. "
            "These commands assess metadata, not assay measurements."
        ),
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {version(_PACKAGE)}")
    commands = parser.add_subparsers(dest="command", metavar="{validate,diagnose,verify}")
    for name, description in (
        ("validate", "Check mapped metadata and export its validation report."),
        ("diagnose", "Validate metadata, run selected diagnostics, and export the full report."),
    ):
        command = commands.add_parser(
            name, help=description, description=description,
            epilog=(
                "Identifiers are read as text. Missing values, duplicate handling, weighting, "
                "and diagnostic settings follow the JSON mapping. Issue row_positions are "
                "zero-based data-record positions, excluding the header. Invalid metadata "
                "produces a validation report (exit 2). Resource limits preserve partial "
                "diagnostic results (exit 4). Console summaries go to stderr; JSON goes "
                "only to the chosen output."
            ),
        )
        command.add_argument("metadata", type=Path, help="UTF-8 CSV or TSV observation metadata")
        command.add_argument("--config", type=Path, required=True, help="JSON analysis mapping")
        command.add_argument(
            "--format", choices=("csv", "tsv"),
            help="override the format inferred from the .csv/.tsv extension",
        )
        command.add_argument("--output", default="-", metavar="PATH", help="JSON report path; default '-' writes to stdout")
        command.add_argument("--run-record", type=Path, metavar="PATH", help="write a companion JSON record of inputs, settings, versions, and report fingerprint")
        command.add_argument("--overwrite", action="store_true", help="allow replacement of an existing report file")
        command.add_argument("--quiet", action="store_true", help="suppress summaries and warnings; keep errors on stderr")
    verification = commands.add_parser(
        "verify", help="Verify supplied inputs and optional report against a saved run record.",
        description="Check input identity without rerunning diagnostics. Checksums do not establish scientific validity or authenticity.",
        epilog="A missing --report leaves report contents explicitly unchecked. Environment differences are reported separately from content matches.",
    )
    verification.add_argument("record", type=Path, help="saved companion run-record JSON")
    verification.add_argument("--metadata", type=Path, required=True, help="metadata file to compare")
    verification.add_argument("--config", type=Path, required=True, help="configuration file to compare")
    verification.add_argument("--report", type=Path, help="also verify the scientific report JSON")
    verification.add_argument("--output", default="-", metavar="PATH", help="verification JSON path; default '-' writes to stdout")
    verification.add_argument("--overwrite", action="store_true", help="allow replacement of an existing verification report")
    verification.add_argument("--quiet", action="store_true", help="suppress summaries; keep failures on stderr")
    return parser


def _message(message: str) -> None:
    """Keep diagnostics readable even on a terminal with a legacy encoding."""
    encoding = getattr(sys.stderr, "encoding", None) or "utf-8"
    safe = message.encode(encoding, errors="backslashreplace").decode(encoding)
    print(safe, file=sys.stderr)


def _check_destination(destination: Path, protected: tuple[Path, ...], overwrite: bool) -> None:
    # Check the leaf before resolve(), which would hide a symbolic link.
    if destination.is_symlink():
        raise OSError(f"refusing symbolic-link output: {destination}")
    resolved = destination.resolve()
    for source in protected:
        if resolved == source.resolve() or (
            destination.exists() and source.exists() and destination.samefile(source)
        ):
            raise OSError(f"output must not replace a protected input or another output: {destination}")
    if not destination.parent.is_dir():
        raise OSError(f"output directory does not exist: {destination.parent}")
    if destination.exists():
        if not destination.is_file():
            raise OSError(f"output is not a regular file: {destination}")
        if not overwrite:
            raise FileExistsError(f"output already exists: {destination}; use --overwrite to replace it")


def _serialize_report(payload: dict[str, Any]) -> str:
    # ASCII JSON escapes preserve every Unicode value and also work with Windows
    # redirected stdout. Serialize before touching any destination file.
    return json.dumps(payload, ensure_ascii=True, allow_nan=False, indent=2) + "\n"


def _cleanup_temporary(temporary: Path) -> None:
    try:
        temporary.unlink(missing_ok=True)
    except OSError:
        pass  # Cleanup must not hide the original publication failure.


def _stage_report(serialized: str, destination: Path) -> Path:
    descriptor, temporary_name = tempfile.mkstemp(prefix=".provenance-aliasing-", suffix=".tmp", dir=destination.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(serialized)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        _cleanup_temporary(temporary)
        raise
    return temporary


def _publish_report(temporary: Path, destination: Path, protected: tuple[Path, ...], overwrite: bool) -> None:
    _check_destination(destination, protected, overwrite)
    if overwrite:
        os.replace(temporary, destination)
    else:
        # Exclusive creation prevents a report created since preflight from
        # being silently replaced.
        os.link(temporary, destination)


def _write_report(
    payload: dict[str, Any], destination: Path | None,
    protected: tuple[Path, ...], overwrite: bool,
) -> None:
    serialized = _serialize_report(payload)
    if destination is None:
        sys.stdout.write(serialized)
        sys.stdout.flush()
        return
    _check_destination(destination, protected, overwrite)
    temporary = _stage_report(serialized, destination)
    try:
        _publish_report(temporary, destination, protected, overwrite)
    finally:
        _cleanup_temporary(temporary)


def _write_recorded_run(
    run: public_api.RecordedRun, destination: Path | None, companion: Path,
    protected: tuple[Path, ...], overwrite: bool,
) -> None:
    """Stage both JSON files before publishing; report partial publication honestly."""
    scientific_text = _serialize_report(run.report)
    record_text = _serialize_report(run.record.to_dict())
    scientific_protected = (*protected, companion)
    record_protected = protected if destination is None else (*protected, destination)
    if destination is not None:
        _check_destination(destination, scientific_protected, overwrite)
    _check_destination(companion, record_protected, overwrite)
    temporary_files = []
    scientific_written = False
    try:
        if destination is not None:
            scientific_temporary = _stage_report(scientific_text, destination)
            temporary_files.append(scientific_temporary)
        record_temporary = _stage_report(record_text, companion)
        temporary_files.append(record_temporary)
        if destination is None:
            sys.stdout.write(scientific_text)
            sys.stdout.flush()
        else:
            _publish_report(scientific_temporary, destination, scientific_protected, overwrite)
        scientific_written = True
        _publish_report(record_temporary, companion, record_protected, overwrite)
    except (OSError, ValueError, RuntimeError) as exc:
        if scientific_written:
            location = "stdout" if destination is None else str(destination)
            raise OSError(f"Scientific report was written to {location}, but companion run record was not published: {exc}") from exc
        raise
    finally:
        for temporary in temporary_files:
            _cleanup_temporary(temporary)


def _validation_messages(report: dict[str, Any], quiet: bool) -> None:
    issues = [issue for issue in report["issues"] if not quiet or issue["severity"] == "error"]
    # Errors come first so a long list of warnings cannot conceal the failure.
    issues.sort(key=lambda issue: {"error": 0, "warning": 1, "info": 2}[issue["severity"]])
    for issue in issues[:10]:
        positions = ""
        if issue["row_positions"]:
            sample = ", ".join(map(str, issue["row_positions"][:10]))
            suffix = ", ..." if len(issue["row_positions"]) > 10 else ""
            positions = f" (zero-based row positions: {sample}{suffix})"
        _message(f"{issue['severity']}: {issue['code']}: {issue['message']}{positions}")
    if len(issues) > 10:
        _message(f"{len(issues) - 10} further issue(s); see the JSON report for full details.")


def _silence_failed_stdout() -> None:
    # A failed buffered flush can otherwise be retried at interpreter shutdown,
    # changing the documented exit code and printing an ignored exception.
    try:
        descriptor = sys.stdout.fileno()
        with open(os.devnull, "w") as sink:
            os.dup2(sink.fileno(), descriptor)
    except (AttributeError, OSError, ValueError):
        pass  # Embedded callers may supply streams without a file descriptor.


def _verify_main(args) -> int:
    destination = None if args.output == "-" else Path(args.output)
    protected = (args.record, args.metadata, args.config) + (() if args.report is None else (args.report,))
    try:
        if destination is not None:
            _check_destination(destination, protected, args.overwrite)
    except (OSError, ValueError, RuntimeError) as exc:
        _message(f"error: cannot write verification report: {exc}")
        return 3
    try:
        verification = public_api.verify_run(args.record, metadata=args.metadata, config=args.config, report=args.report)
        payload = verification.to_dict()
    except (OSError, ValueError, TypeError) as exc:
        _message(f"error: cannot verify run record: {exc}")
        return 2
    try:
        _write_report(payload, destination, protected, args.overwrite)
    except (OSError, ValueError, TypeError, RuntimeError) as exc:
        if destination is None and isinstance(exc, OSError):
            _silence_failed_stdout()
        _message(f"error: cannot write verification report: {exc}")
        return 3
    if not verification.verified:
        for check in payload["checks"]:
            if check["required"] and check["status"] != "match":
                _message(f"error: {check['name']}: {check['message']}")
    elif not args.quiet:
        _message("Input identity and report content verified." if args.report is not None
                 else "Input identity verified; report contents not checked.")
    if not args.quiet and not payload["environment"]["matches"]:
        _message("warning: current software versions differ from those recorded; see verification JSON.")
    return 0 if verification.verified else 2


def main(argv: Sequence[str] | None = None) -> int:
    """Run a command; return its documented exit code (argparse handles usage)."""
    parser = _parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0
    if args.command == "verify":
        return _verify_main(args)

    destination = None if args.output == "-" else Path(args.output)
    protected = (args.metadata, args.config)
    try:
        if destination is not None:
            extra = () if args.run_record is None else (args.run_record,)
            _check_destination(destination, (*protected, *extra), args.overwrite)
        if args.run_record is not None:
            extra = () if destination is None else (destination,)
            _check_destination(args.run_record, (*protected, *extra), args.overwrite)
    except (OSError, ValueError, RuntimeError) as exc:
        _message(f"error: cannot write report: {exc}")
        return 3
    try:
        run = public_api.run_analysis(args.metadata, config=args.config, command=args.command, format=args.format)
    except public_api.RunRecordError as exc:
        _message(f"error: cannot record analysis: {exc}")
        return 3
    except (OSError, ValueError, TypeError) as exc:
        _message(f"error: cannot read metadata or configuration: {exc}")
        return 2
    payload, code = run.report, run.exit_code
    validation = payload if payload["scope"] == "metadata_validation" else payload.get("validation")
    if code == 3:
        summary = f"error: {payload['error']['type']}: {payload['error']['message']}"
    elif code == 2:
        summary = "Validation failed."
    elif args.command == "validate":
        summary = "Validation passed."
    else:
        statuses = Counter(metric["status"] for metric in payload["metrics"])
        summary = "Diagnosis completed: " + ", ".join(f"{count} {status}" for status, count in sorted(statuses.items())) + "."
        if code == 4:
            summary += " Partial results; some diagnostics exceeded the configured resource budget."
    if validation is not None:
        _validation_messages(validation, args.quiet)
    try:
        if args.run_record is None:
            _write_report(payload, destination, protected, args.overwrite)
        else:
            _write_recorded_run(run, destination, args.run_record, protected, args.overwrite)
    except (OSError, ValueError, TypeError, RuntimeError) as exc:
        if destination is None and isinstance(exc, OSError):
            _silence_failed_stdout()
        _message(f"error: cannot write report: {exc}")
        return 3

    if not args.quiet or code in (2, 3):
        location = "stdout" if destination is None else str(destination)
        _message(f"{summary} JSON report: {location}")
        if args.run_record is not None:
            _message(f"Run record: {args.run_record}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
