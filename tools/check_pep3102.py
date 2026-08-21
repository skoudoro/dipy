#!/usr/bin/env python
"""Enforce keyword-only arguments for parameters with defaults (PEP 3102).

Any parameter carrying a default must sit after the bare ``*`` marker, so that
callers always name it. This keeps DIPY's public API free to reorder or extend
optional arguments without silently changing the meaning of positional calls.

Some signatures are dictated by a protocol or a third-party framework that
calls back positionally (the descriptor protocol, VTK observers, SciPy
optimizer callbacks). Those are exempt: dunder methods are skipped
automatically, and anything else can opt out with a trailing
``# noqa: pep3102`` comment on the ``def`` line.
"""

import argparse
import ast
from pathlib import Path
import sys

NOQA = "# noqa: pep3102"


def find_violations(tree, lines):
    """Collect functions whose defaulted parameters are not keyword-only.

    Parameters
    ----------
    tree : ast.Module
        Parsed syntax tree of the file.
    lines : list of str
        Source lines of the file, used to detect ``noqa`` opt-outs.

    Returns
    -------
    list of tuple
        ``(lineno, function_name, offending_parameters)`` triples.
    """
    violations = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if node.name.startswith("__") and node.name.endswith("__"):
            continue
        if NOQA in lines[node.lineno - 1]:
            continue

        args = node.args
        positional = args.posonlyargs + args.args
        defaulted = positional[len(positional) - len(args.defaults) :]
        if defaulted:
            violations.append((node.lineno, node.name, [a.arg for a in defaulted]))
    return violations


def check_file(path):
    """Report PEP 3102 violations in a single file.

    Parameters
    ----------
    path : str
        Path to the Python file to check.

    Returns
    -------
    list of str
        Human-readable violation messages.
    """
    source = Path(path).read_text(encoding="utf-8")
    try:
        tree = ast.parse(source)
    except SyntaxError as err:
        return [f"{path}:{err.lineno}: could not parse: {err.msg}"]

    lines = source.splitlines()
    return [
        f"{path}:{lineno}: {name}() takes defaulted argument(s) "
        f"{', '.join(params)} positionally; move them after '*'"
        for lineno, name, params in find_violations(tree, lines)
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("filenames", nargs="*", help="files to check")
    args = parser.parse_args()

    messages = [msg for path in args.filenames for msg in check_file(path)]
    for msg in messages:
        print(msg)
    return 1 if messages else 0


if __name__ == "__main__":
    sys.exit(main())
