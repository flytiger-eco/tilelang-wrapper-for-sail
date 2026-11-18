#!/usr/bin/env python3
"""Apply the wrapper CI focus allowlist to the TileLang test runner (fail-closed).

Reads .github/scripts/wrapper_ci_focus.json (see wrapper_ci_focus.json for schema),
validates it, and injects an allowlist filter into test_tilelang.py right after the
existing skip filter:

    filtered = [t for t in collected if (t.file_path, t.test_filter) not in skip_set]
    filtered = _wrapper_ci_focus_filter(filtered)   # injected

Allowlist matching semantics:
  * test_filter without "[" -> function level: keeps every item of that function
    (exact name for non-parameterized items, "<func>[" prefix for parameterized);
  * test_filter with "["    -> item level: keeps only the exactly matching item.

Boards semantics (per case): every case declares its own "boards" list; the
injected filter keeps only the cases whose boards include the current platform.
The top-level "boards" must equal the union of the per-case boards (validated).
This mirrors the boards filtering of wrapper_skip.json, so a capability measured
on one board never silently changes the scope of another board.

If the focus file's top-level "boards" does not include the current platform,
this script exits 0 WITHOUT patching (that board runs the full scope, unchanged).

If the focus file is absent, ci.yml does not call this script at all (full scope).
Delete the file (or regenerate it when TILELANG_REF changes) to disable/refresh.

Fail-closed guard: the injected filter accumulates kept/seen counts across all
directory groups, and an injected aggregate check (called right after the scan
loop, before the runner's "if not test_configs:" warning) fails the run when
the allowlist kept none of the tests (kept == 0) - either because it matched
none of the collected tests (a schema-valid stale allowlist after a
TILELANG_REF bump) or because no tests survived collection and the board skip
filters in the first place. Either way a configured board must not pass CI
with zero coverage. Individual directory groups may legitimately keep 0 tests
(the allowlist does not cover every group); only the aggregate is fatal.
Every filter call also logs its kept/seen counts unconditionally.

Usage:
    apply_ci_focus.py <focus.json> <runner.py> <platform>
"""
import ast
import json
import os
import posixpath
import sys
import tempfile

KNOWN_BOARDS = frozenset({"ZW-M890P", "OAM-810E"})
ALLOWED_TOP = frozenset({"description", "boards", "cases"})
REQUIRED_TOP = frozenset({"boards", "cases"})
REQUIRED_KEYS = frozenset({"file_path", "test_filter", "boards"})

MARKER_FILTER = ("filtered = [t for t in collected "
                "if (t.file_path, t.test_filter) not in skip_set]")
MARKER_LOAD_SKIP = "def load_skip():"
MARKER_AGGREGATE = "    if not test_configs:"

FOCUS_FUNC = '''

_WRAPPER_CI_FOCUS_PATH = %r
_WRAPPER_CI_FOCUS_BOARD = %r
_WRAPPER_CI_FOCUS_SEEN = 0
_WRAPPER_CI_FOCUS_KEPT = 0


def _wrapper_ci_focus_filter(tests):
    """Keep only allowlisted tests (injected by wrapper CI, see apply_ci_focus.py)."""
    import json as _json

    global _WRAPPER_CI_FOCUS_SEEN, _WRAPPER_CI_FOCUS_KEPT
    with open(_WRAPPER_CI_FOCUS_PATH, encoding="utf-8") as _f:
        _doc = _json.load(_f)
    _cases = [c for c in _doc["cases"] if _WRAPPER_CI_FOCUS_BOARD in c["boards"]]
    _full = {(c["file_path"], c["test_filter"]) for c in _cases}
    _funcs = {}
    for _fp, _tf in _full:
        if "[" not in _tf:
            _funcs.setdefault(_fp, set()).add(_tf)
    _out = []
    for _t in tests:
        _fp, _node = _t.file_path, _t.test_filter
        if (_fp, _node) in _full:
            _out.append(_t)
        elif "[" in _node and _node.split("[", 1)[0] in _funcs.get(_fp, set()):
            _out.append(_t)
    print("ci-focus allowlist: kept %%d of %%d tests in this directory"
          %% (len(_out), len(tests)))
    _WRAPPER_CI_FOCUS_SEEN += len(tests)
    _WRAPPER_CI_FOCUS_KEPT += len(_out)
    return _out


def _wrapper_ci_focus_check_aggregate():
    """Fail closed when the allowlist kept none of the tests.

    A single directory group keeping 0 tests is legitimate (the allowlist does
    not cover every group); keeping 0 across ALL groups means either a stale
    allowlist for this board/TILELANG_REF (tests were collected but none
    matched) or no tests surviving collection/skip filtering at all - either
    way the run would otherwise pass CI with zero coverage.
    """
    import sys as _sys

    if not _WRAPPER_CI_FOCUS_KEPT:
        if _WRAPPER_CI_FOCUS_SEEN:
            _why = ("the allowlist is stale for this TILELANG_REF - regenerate "
                    "it with gen_ci_focus_unified.py")
        else:
            _why = ("no tests survived collection and the board skip filters "
                    "before the allowlist ran")
        print(
            "FATAL: ci-focus allowlist kept 0 of %%d collected tests on board "
            "%%r - %%s. Delete wrapper_ci_focus.json to run the full scope."
            %% (_WRAPPER_CI_FOCUS_SEEN, _WRAPPER_CI_FOCUS_BOARD, _why),
            file=_sys.stderr,
        )
        _sys.exit(1)
    print("ci-focus allowlist aggregate: kept %%d of %%d collected tests across "
          "all directory groups"
          %% (_WRAPPER_CI_FOCUS_KEPT, _WRAPPER_CI_FOCUS_SEEN))

'''


def fatal(msg):
    print(f"FATAL: {msg}", file=sys.stderr)
    sys.exit(1)


def validate(focus_path):
    with open(focus_path, encoding="utf-8") as f:
        try:
            doc = json.load(f)
        except (json.JSONDecodeError, ValueError) as exc:
            fatal(f"invalid JSON in {focus_path}: {exc}")

    if not isinstance(doc, dict):
        fatal(f"schema error – expected top-level object in {focus_path}")
    keys = set(doc.keys())
    missing = sorted(REQUIRED_TOP - keys)
    extra = sorted(keys - ALLOWED_TOP)
    if missing or extra:
        fatal(f"top-level key mismatch – missing: {missing}, unexpected: {extra}")

    boards = doc["boards"]
    if not isinstance(boards, list) or not boards:
        fatal(f'"boards" must be a non-empty array, got {boards!r}')
    for b in boards:
        if not isinstance(b, str) or b not in KNOWN_BOARDS:
            fatal(f'"boards" entry {b!r} is not in allowlist {sorted(KNOWN_BOARDS)}')
    board_union = set(boards)

    cases = doc["cases"]
    if not isinstance(cases, list) or not cases:
        fatal('"cases" must be a non-empty array')

    seen = set()
    case_boards_union = set()
    for i, c in enumerate(cases):
        if not isinstance(c, dict):
            fatal(f"cases[{i}] must be an object, got {type(c).__name__}")
        ckeys = set(c.keys())
        if ckeys != REQUIRED_KEYS:
            fatal(f"cases[{i}] key mismatch – required exactly {sorted(REQUIRED_KEYS)}, "
                  f"got {sorted(ckeys)}")
        fp, tf, cboards = c["file_path"], c["test_filter"], c["boards"]
        for field, val in (("file_path", fp), ("test_filter", tf)):
            if not isinstance(val, str) or not val.strip():
                fatal(f"cases[{i}].{field} must be a non-empty string, got {val!r}")
            bad = [ch for ch in val if not ch.isprintable()]
            if bad:
                fatal(f"cases[{i}].{field} contains non-printable characters: {bad!r}")
        if not isinstance(cboards, list) or not cboards:
            fatal(f"cases[{i}].boards must be a non-empty array, got {cboards!r}")
        for b in cboards:
            if not isinstance(b, str) or b not in KNOWN_BOARDS:
                fatal(f"cases[{i}].boards entry {b!r} is not in allowlist "
                      f"{sorted(KNOWN_BOARDS)}")
        case_boards_union.update(cboards)
        fpn = posixpath.normpath(fp)
        if posixpath.isabs(fpn):
            fatal(f"cases[{i}].file_path must be relative, got {fp!r}")
        if ".." in fp.split("/"):
            fatal(f"cases[{i}].file_path contains parent-directory traversal: {fp!r}")
        key = (fpn, tf)
        if key in seen:
            fatal(f"cases[{i}] duplicate after normalization: {fp!r} :: {tf!r}")
        seen.add(key)

    if board_union != case_boards_union:
        fatal(f'top-level "boards" {sorted(board_union)} != union of per-case boards '
              f'{sorted(case_boards_union)} – the file is inconsistent')

    return doc, len(seen)


def patch_runner(runner_path, focus_path, platform):
    with open(runner_path, encoding="utf-8") as f:
        src = f.read()

    if src.count(MARKER_FILTER) != 1:
        fatal(f"expected exactly one filter marker in {runner_path}, "
              f"found {src.count(MARKER_FILTER)}")
    if src.count(MARKER_LOAD_SKIP) != 1:
        fatal(f"expected exactly one {MARKER_LOAD_SKIP!r} in {runner_path}, "
              f"found {src.count(MARKER_LOAD_SKIP)}")
    if src.count(MARKER_AGGREGATE) != 1:
        fatal(f"expected exactly one aggregate-check anchor "
              f"{MARKER_AGGREGATE!r} in {runner_path}, "
              f"found {src.count(MARKER_AGGREGATE)}")
    if "_wrapper_ci_focus_filter" in src:
        fatal(f"{runner_path} already patched – refusing to double-patch")

    inject = FOCUS_FUNC % (os.path.abspath(focus_path), platform)
    new = src.replace(MARKER_LOAD_SKIP, inject + MARKER_LOAD_SKIP, 1)
    new = new.replace(
        MARKER_FILTER,
        MARKER_FILTER + "\n    filtered = _wrapper_ci_focus_filter(filtered)",
        1,
    )
    new = new.replace(
        MARKER_AGGREGATE,
        "    _wrapper_ci_focus_check_aggregate()\n" + MARKER_AGGREGATE,
        1,
    )
    try:
        ast.parse(new, filename=runner_path)
    except SyntaxError as exc:
        fatal(f"patched runner is not valid Python: {exc}")

    tmp = None
    try:
        with tempfile.NamedTemporaryFile(
                mode="w", delete=False,
                dir=os.path.dirname(runner_path) or ".", suffix=".tmp") as t:
            tmp = t.name
            t.write(new)
            t.flush()
            os.fsync(t.fileno())
        os.replace(tmp, runner_path)
    except BaseException:
        if tmp:
            try:
                os.unlink(tmp)
            except OSError:
                pass
        raise
    print(f"  Injected ci-focus allowlist filter into {runner_path}")


def main():
    if len(sys.argv) != 4:
        print(__doc__, file=sys.stderr)
        return 2
    focus_path, runner_path, platform = sys.argv[1:4]
    if platform not in KNOWN_BOARDS:
        fatal(f"platform {platform!r} is not in allowlist {sorted(KNOWN_BOARDS)}")

    if not os.path.isfile(focus_path):
        fatal(f"focus file not found: {focus_path}")
    if not os.path.isfile(runner_path):
        fatal(f"runner not found: {runner_path}")

    doc, n = validate(focus_path)

    if platform not in doc["boards"]:
        print(f"ci-focus not configured for board {platform!r} "
              f"(focus boards = {doc['boards']}); running full scope, runner untouched.")
        return 0

    patch_runner(runner_path, focus_path, platform)
    print(f"  ci-focus allowlist applied: {n} cases, boards={doc['boards']}, "
          f"platform={platform!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
