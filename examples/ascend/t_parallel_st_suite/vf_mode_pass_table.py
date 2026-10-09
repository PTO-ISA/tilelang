#!/usr/bin/env python3
"""PASS/FAIL table for ST_VF_MODE runs (case tag × mode).

Reads oneshot SUMMARY files and, when those are absent, opsim/compile logs.
Does not launch opsim. SimdVF and PTO FAIL cells are expected for some cases.
Columns follow the mode directories under the regress root (the four
``simt-asc`` / ``simt-pto`` / ``simd-asc`` / ``simd-pto`` names when all ran).
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

SUITE = Path(__file__).resolve().parent
if str(SUITE) not in sys.path:
    sys.path.insert(0, str(SUITE))

from vf_mode import CANONICAL_VF_MODES, resolve_vf_mode  # noqa: E402

_COMPILE_FAIL = re.compile(r"^COMPILE_FAIL\s+(\S+)")
_COMPILE_OK = re.compile(r"^COMPILE_OK\s+(\S+)")
_COMPILE_HDR = re.compile(r"^==== COMPILE\s+(\S+)\s+====")
_OUTCOME = re.compile(r"^(\S+)\s+(PASS|FAIL)\b")
_TAG_FROM_COMPILE_LOG = re.compile(r"^compile_(.+)\.log$")
_TAG_FROM_OPSIM_LOG = re.compile(r"^opsim_(.+)\.log$")
_TAG_FROM_RESULT = re.compile(r"^opsim_(.+)_result\.txt$")

_MODE_ORDER = {name: idx for idx, name in enumerate(CANONICAL_VF_MODES)}


def _tag_sort_key(tag: str) -> tuple:
    match = re.match(r"([a-z]+)(\d+)_(.*)", tag)
    if not match:
        return (9, 0, tag)
    family = {"sv": 0, "cf": 1, "sp": 2}.get(match.group(1), 9)
    rest: list[tuple[int, int] | tuple[int, str]] = []
    for part in re.split(r"(\d+)", match.group(3)):
        if part.isdigit():
            rest.append((0, int(part)))
        else:
            rest.append((1, part))
    return (family, int(match.group(2)), rest)


def _summary_status(out: Path) -> dict[str, str]:
    """Read oneshot ``SUMMARY_*_raw.txt`` files.

    An opsim ``PASS``/``FAIL`` line wins. Otherwise a ``COMPILE_FAIL`` line
    wins over a later ``COMPILE_OK`` snippet grepped out of the compile log.
    """
    saw_fail: set[str] = set()
    saw_ok: set[str] = set()
    seen: set[str] = set()
    verdict: dict[str, str] = {}
    paths = sorted(out.glob("SUMMARY_*_raw.txt"))
    for path in paths:
        for line in path.read_text(errors="replace").splitlines():
            fail = _COMPILE_FAIL.match(line)
            if fail:
                tag = fail.group(1)
                seen.add(tag)
                saw_fail.add(tag)
                continue
            ok = _COMPILE_OK.match(line)
            if ok:
                tag = ok.group(1)
                seen.add(tag)
                saw_ok.add(tag)
                continue
            hdr = _COMPILE_HDR.match(line)
            if hdr:
                seen.add(hdr.group(1))
                continue
            outcome = _OUTCOME.match(line)
            if outcome:
                tag, word = outcome.group(1), outcome.group(2)
                seen.add(tag)
                verdict[tag] = "PASS" if word == "PASS" else "FAIL opsim"
    status: dict[str, str] = {}
    for tag in seen:
        if tag in verdict:
            status[tag] = verdict[tag]
        elif tag in saw_fail:
            status[tag] = "FAIL compile"
        elif tag in saw_ok:
            status[tag] = "FAIL no-pass-line"
        else:
            status[tag] = "MISSING"
    return status


def _opsim_verdict(out: Path, tag: str) -> str | None:
    log = out / f"opsim_{tag}.log"
    if log.is_file():
        last = None
        for line in log.read_text(errors="replace").splitlines():
            if line.startswith("PASS"):
                last = "PASS"
            elif line.startswith("FAIL"):
                last = "FAIL opsim"
        if last:
            return last
    result = out / f"opsim_{tag}_result.txt"
    if result.is_file():
        text = result.read_text(errors="replace")
        if "ok=True" in text or text.startswith("PASS"):
            return "PASS"
        if "ok=False" in text or "FAIL" in text:
            return "FAIL opsim"
    return None


def _artifact_tags(out: Path) -> set[str]:
    tags: set[str] = set()
    so_dir = out / "so"
    if so_dir.is_dir():
        tags.update(path.stem for path in so_dir.glob("*.so"))
    log_dir = out / "logs"
    if log_dir.is_dir():
        for path in log_dir.glob("compile_*.log"):
            match = _TAG_FROM_COMPILE_LOG.match(path.name)
            if match:
                tags.add(match.group(1))
    for path in out.glob("opsim_*.log"):
        match = _TAG_FROM_OPSIM_LOG.match(path.name)
        if match:
            tags.add(match.group(1))
    for path in out.glob("opsim_*_result.txt"):
        match = _TAG_FROM_RESULT.match(path.name)
        if match:
            tags.add(match.group(1))
    return tags


def _fallback_status(out: Path, tag: str) -> str:
    verdict = _opsim_verdict(out, tag)
    if verdict:
        return verdict
    so = out / "so" / f"{tag}.so"
    compile_log = out / "logs" / f"compile_{tag}.log"
    if compile_log.is_file() and not so.is_file():
        return "FAIL compile"
    if so.is_file():
        return "FAIL no-pass-line"
    if compile_log.is_file():
        return "FAIL compile"
    return "MISSING"


def status_for_out(out: Path) -> dict[str, str]:
    """Map case tag → cell text for one mode's OUT directory."""
    from_summary = _summary_status(out)
    tags = set(from_summary) | _artifact_tags(out)
    status: dict[str, str] = {}
    for tag in tags:
        recorded = from_summary.get(tag)
        if recorded and recorded != "MISSING":
            status[tag] = recorded
        else:
            status[tag] = _fallback_status(out, tag)
    return status


def _is_mode_dirname(name: str) -> bool:
    try:
        resolve_vf_mode(name)
    except ValueError:
        return False
    return True


def _has_records(path: Path) -> bool:
    if any(path.glob("SUMMARY_*_raw.txt")):
        return True
    if (path / "logs").is_dir() and any((path / "logs").glob("compile_*.log")):
        return True
    if (path / "so").is_dir() and any((path / "so").glob("*.so")):
        return True
    return any(path.glob("opsim_*.log"))


def mode_dirs_under(root: Path) -> list[tuple[str, Path]]:
    """``(canonical mode, directory)`` pairs under a regress root."""
    found: list[tuple[str, Path]] = []
    if not root.is_dir():
        return found
    for child in sorted(root.iterdir()):
        if child.is_dir() and _is_mode_dirname(child.name):
            found.append((resolve_vf_mode(child.name), child))
    if found:
        found.sort(key=lambda item: (_MODE_ORDER.get(item[0], 9), item[0]))
        return found
    if _has_records(root):
        if _is_mode_dirname(root.name):
            label = resolve_vf_mode(root.name)
        else:
            try:
                label = resolve_vf_mode(None)
            except ValueError:
                label = "simt-asc"
        return [(label, root)]
    return []


def render_table(columns: list[tuple[str, dict[str, str]]]) -> str:
    modes = [name for name, _ in columns]
    tags = sorted({tag for _, status in columns for tag in status}, key=_tag_sort_key)
    rows: list[list[str]] = []
    for tag in tags:
        rows.append([tag, *[status.get(tag, "MISSING") for _, status in columns]])
    header = ["tag", *modes]
    widths = [len(cell) for cell in header]
    for row in rows:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], len(cell))
    lines = [
        "# ST_VF_MODE pass table",
        "# PASS = opsim PASS line. FAIL compile / FAIL opsim / FAIL no-pass-line name the miss.",
        "# SimdVF and PTO modes may FAIL some ST cases; this table records the current outcome.",
        "# A missing PTO backend is FAIL compile. It is not remapped to ascend.",
        "",
    ]
    lines.append("  ".join(cell.ljust(widths[idx]) for idx, cell in enumerate(header)))
    lines.append("  ".join("-" * widths[idx] for idx in range(len(header))))
    for row in rows:
        lines.append("  ".join(cell.ljust(widths[idx]) for idx, cell in enumerate(row)))
    lines.append("")
    for name, status in columns:
        counts = {"PASS": 0, "FAIL": 0, "MISSING": 0}
        for cell in status.values():
            if cell == "PASS":
                counts["PASS"] += 1
            elif cell.startswith("FAIL"):
                counts["FAIL"] += 1
            else:
                counts["MISSING"] += 1
        # Tags present only in another mode.
        for tag in tags:
            if tag not in status:
                counts["MISSING"] += 1
        lines.append(f"{name}: PASS {counts['PASS']}  FAIL {counts['FAIL']}  MISSING {counts['MISSING']}")
    if not rows:
        lines.append("(no cases recorded)")
    lines.append("")
    return "\n".join(lines)


def _parse_out_arg(text: str) -> tuple[str, Path]:
    if "=" in text:
        label, raw_path = text.split("=", 1)
        try:
            label = resolve_vf_mode(label)
        except ValueError:
            label = label.strip() or "run"
        return label, Path(raw_path)
    path = Path(text)
    try:
        label = resolve_vf_mode(path.name)
    except ValueError:
        label = resolve_vf_mode(None)
    return label, path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help=(
            "Regress root containing per-mode OUT dirs such as simt-asc/ and simd-pto/ "
            "(default: $ST_VF_REGRESS_OUT or /tmp/t_parallel_st_suite_vf_mode)."
        ),
    )
    parser.add_argument(
        "--out",
        action="append",
        default=[],
        help="One mode directory, as PATH or LABEL=PATH. Repeatable. Overrides --root.",
    )
    parser.add_argument(
        "--write",
        type=Path,
        default=None,
        help="Table path. Default: <root>/VF_MODE_PASS_TABLE.txt when --root is used.",
    )
    args = parser.parse_args(argv)

    columns: list[tuple[str, dict[str, str]]] = []
    write_path = args.write
    if args.out:
        for item in args.out:
            label, path = _parse_out_arg(item)
            columns.append((label, status_for_out(path)))
    else:
        root = args.root
        if root is None:
            env_root = os.environ.get("ST_VF_REGRESS_OUT", "").strip()
            root = Path(env_root) if env_root else Path("/tmp/t_parallel_st_suite_vf_mode")
        pairs = mode_dirs_under(root)
        if not pairs:
            print(f"no ST_VF_MODE OUT dirs under {root}", file=sys.stderr)
            return 1
        for label, path in pairs:
            columns.append((label, status_for_out(path)))
        if write_path is None:
            write_path = root / "VF_MODE_PASS_TABLE.txt"

    text = render_table(columns)
    sys.stdout.write(text)
    if write_path is not None:
        write_path.parent.mkdir(parents=True, exist_ok=True)
        write_path.write_text(text)
        print(f"WROTE {write_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
