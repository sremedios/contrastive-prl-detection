#!/usr/bin/env python3
"""
check_versions.py — Report every place a project version is declared, so the
session-wrapup routine can verify they all agree before committing.

This script ONLY reports. It never edits files. Reconciling any mismatch is
left to the model so the edits are visible and reviewable in the normal flow.

Usage:
    python check_versions.py [ROOT]      # ROOT defaults to current directory

Exit codes:
    0  -> 0 or 1 distinct version found (nothing to reconcile), OR only dynamic
    1  -> 2+ distinct hard-coded versions found (a real mismatch to reconcile)
    2  -> bad invocation

Output is a small table plus a verdict line, designed to be skimmed.
"""

import os
import re
import sys

# Directories that never hold the source-of-truth version and only add noise.
SKIP_DIRS = {
    ".git", ".hg", ".svn", "node_modules", ".venv", "venv", "env",
    "__pycache__", ".mypy_cache", ".pytest_cache", ".tox", "build", "dist",
    ".eggs", "site-packages", ".idea", ".vscode", ".ipynb_checkpoints",
}

# Filenames (exact) -> list of (label, compiled regex with one capture group).
# Order of files in the report follows discovery order on disk.
NAMED_PATTERNS = {
    "pyproject.toml": [
        ("project.version", re.compile(r'(?m)^\s*version\s*=\s*["\']([^"\']+)["\']')),
    ],
    "setup.py": [
        ("setup(version=)", re.compile(r'version\s*=\s*["\']([^"\']+)["\']')),
    ],
    "setup.cfg": [
        ("metadata.version", re.compile(r'(?m)^\s*version\s*=\s*([^\s#]+)')),
    ],
    "VERSION": [("VERSION file", re.compile(r'^\s*v?([0-9][^\s]*)'))],
    "VERSION.txt": [("VERSION.txt", re.compile(r'^\s*v?([0-9][^\s]*)'))],
    "meta.yaml": [
        ("conda set version", re.compile(r'{%\s*set\s+version\s*=\s*["\']([^"\']+)["\']')),
    ],
    "CITATION.cff": [
        ("CITATION.cff", re.compile(r'(?m)^version:\s*["\']?([^"\'\n]+?)["\']?\s*$')),
    ],
    "package.json": [
        ("package.json", re.compile(r'"version"\s*:\s*"([^"]+)"')),
    ],
    "conf.py": [
        ("docs version/release", re.compile(r'(?m)^\s*(?:version|release)\s*=\s*["\']([^"\']+)["\']')),
    ],
}

# Any *.py file is scanned for a dunder version assignment.
DUNDER = re.compile(r'(?m)^\s*__version__\s*=\s*["\']([^"\']+)["\']')

# Detect dynamic / VCS-derived versioning so we don't try to "fix" a non-issue.
DYNAMIC_HINTS = (
    re.compile(r'(?m)^\s*dynamic\s*=\s*\[[^\]]*["\']version["\']'),
    re.compile(r'setuptools[_-]scm'),
    re.compile(r'hatch-vcs|hatch_vcs'),
    re.compile(r'versioneer'),
    re.compile(r'setuptools_scm|use_scm_version'),
)


def read(path):
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            return fh.read()
    except OSError:
        return ""


def rel(root, path):
    return os.path.relpath(path, root)


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "."
    if not os.path.isdir(root):
        print(f"error: {root!r} is not a directory", file=sys.stderr)
        return 2
    root = os.path.abspath(root)

    findings = []           # (relpath, label, version)
    dynamic_notes = []      # relpath where dynamic versioning was detected

    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for fn in filenames:
            full = os.path.join(dirpath, fn)
            r = rel(root, full)

            if fn in NAMED_PATTERNS:
                text = read(full)
                for label, pat in NAMED_PATTERNS[fn]:
                    m = pat.search(text)
                    if m:
                        findings.append((r, label, m.group(1).strip()))
                if fn in ("pyproject.toml", "setup.py", "setup.cfg"):
                    if any(h.search(text) for h in DYNAMIC_HINTS):
                        dynamic_notes.append(r)

            if fn.endswith(".py"):
                m = DUNDER.search(read(full))
                if m:
                    findings.append((r, "__version__", m.group(1).strip()))

    if not findings:
        print("No version declarations found.")
        if dynamic_notes:
            print("Dynamic/VCS-derived versioning detected in: "
                  + ", ".join(sorted(set(dynamic_notes))))
            print("Version is computed from git tags — nothing to sync by hand.")
        return 0

    width = max(len(r) for r, _, _ in findings)
    print(f"{'FILE'.ljust(width)}  LABEL                 VERSION")
    print(f"{'-' * width}  {'-' * 21} {'-' * 12}")
    for r, label, ver in findings:
        print(f"{r.ljust(width)}  {label.ljust(21)} {ver}")

    distinct = sorted({v for _, _, v in findings})
    print()
    if dynamic_notes:
        print("Dynamic/VCS-derived versioning detected in: "
              + ", ".join(sorted(set(dynamic_notes))))
        print("(Hard-coded versions below may be fallbacks; don't fight the VCS source.)")
        print()

    if len(distinct) <= 1:
        print(f"OK — all declarations agree on {distinct[0]}.")
        return 0
    print(f"MISMATCH — {len(distinct)} distinct versions: {', '.join(distinct)}")
    print("Reconcile these to a single source of truth before committing.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
