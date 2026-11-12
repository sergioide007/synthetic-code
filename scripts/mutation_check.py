#!/usr/bin/env python3
"""Tiny, dependency-free mutation tester: does the test suite notice when the code is deliberately broken?

    python scripts/mutation_check.py src/sintetico/guards.py tests/test_delivery_review.py [--max 60] [--seed 1] [--workers 4]

For each mutant (comparison flipped, and/or swapped, number nudged, True/False flipped, `not` dropped, +/- swapped)
the given tests run against a private copy of the package. A mutant is KILLED if any test fails or times out, and
SURVIVES if all tests still pass (= a behaviour your tests do not pin down). Exit code 1 if the kill rate < --min-score.
String constants are never mutated (mostly messages: they create equivalent mutants and noise).
"""
import argparse
import ast
import concurrent.futures as cf
import os
import random
import shutil
import subprocess  # noqa: S404
import sys
import tempfile
from pathlib import Path

CMP = {ast.Lt: ast.LtE, ast.LtE: ast.Lt, ast.Gt: ast.GtE, ast.GtE: ast.Gt, ast.Eq: ast.NotEq, ast.NotEq: ast.Eq}


def sites(tree):
    out = []
    for n in ast.walk(tree):
        if isinstance(n, ast.Compare) and type(n.ops[0]) in CMP: out.append(("cmp", n))
        elif isinstance(n, ast.BoolOp): out.append(("bool", n))
        elif isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.Not): out.append(("not", n))
        elif isinstance(n, ast.BinOp) and isinstance(n.op, (ast.Add, ast.Sub, ast.Mult)): out.append(("arith", n))
        elif isinstance(n, ast.Constant) and isinstance(n.value, bool): out.append(("bool_const", n))
        elif isinstance(n, ast.Constant) and isinstance(n.value, (int, float)) and not isinstance(n.value, bool): out.append(("num", n))
    return out


def in_fstring_or_doc(tree):
    skip = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.JoinedStr):
            for c in ast.walk(n): skip.add(id(c))
    return skip


def build(src, idx):
    tree = ast.parse(src); skip = in_fstring_or_doc(tree)
    cand = [(k, n) for k, n in sites(tree) if id(n) not in skip]
    kind, node = cand[idx]
    if kind == "cmp": node.ops[0] = CMP[type(node.ops[0])]()
    elif kind == "bool": node.op = ast.Or() if isinstance(node.op, ast.And) else ast.And()
    elif kind == "arith": node.op = ast.Sub() if isinstance(node.op, (ast.Add, ast.Mult)) else ast.Add()
    elif kind == "bool_const": node.value = not node.value
    elif kind == "num": node.value = node.value + 1 if isinstance(node.value, int) else round(node.value * 1.1 + 0.01, 6)
    elif kind == "not":
        parent = next(p for p in ast.walk(tree) for c in ast.iter_child_nodes(p) if c is node)
        for f, v in ast.iter_fields(parent):
            if v is node: setattr(parent, f, node.operand)
            elif isinstance(v, list): parent.__setattr__(f, [node.operand if x is node else x for x in v])
    ast.fix_missing_locations(tree)
    return kind, node.lineno, ast.unparse(tree)


def run_one(args):
    mod_rel, tests, idx, repo, timeout = args
    d = Path(tempfile.mkdtemp(prefix="mut_"))
    try:
        shutil.copytree(repo / "src", d / "src", ignore=shutil.ignore_patterns("__pycache__", "*.egg-info"))
        target = d / "src" / mod_rel
        kind, line, mutated = build((repo / "src" / mod_rel).read_text(encoding="utf-8"), idx)
        target.write_text(mutated, encoding="utf-8")
        env = {**os.environ, "PYTHONPATH": str(d / "src"), "PYTHONDONTWRITEBYTECODE": "1", "HYPOTHESIS_MAX_EXAMPLES": "20"}
        try:
            p = subprocess.run([  # noqa: S603
                               sys.executable, "-m", "pytest", "-x", "-q", "-p", "no:cacheprovider", "--no-header", *tests],
                               cwd=repo, env=env, capture_output=True, text=True, timeout=timeout)
            killed = p.returncode != 0
        except subprocess.TimeoutExpired:
            killed = True
        return idx, kind, line, killed
    finally:
        shutil.rmtree(d, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("module"); ap.add_argument("tests", nargs="+")
    ap.add_argument("--max", type=int, default=60); ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--workers", type=int, default=4); ap.add_argument("--timeout", type=int, default=60)
    ap.add_argument("--min-score", type=float, default=0.0)
    a = ap.parse_args()
    repo = Path(__file__).resolve().parent.parent
    mod_rel = Path(a.module).relative_to("src") if a.module.startswith("src") else Path(a.module)
    tree = ast.parse((repo / "src" / mod_rel).read_text(encoding="utf-8")); skip = in_fstring_or_doc(tree)
    total = len([1 for k, n in sites(tree) if id(n) not in skip])
    picks = sorted(random.Random(a.seed).sample(range(total), min(a.max, total)))  # noqa: S311 - not security related
    # sanity: unmutated code must pass, otherwise every mutant would count as 'killed'
    base = subprocess.run([  # noqa: S603
                          sys.executable, "-m", "pytest", "-x", "-q", "-p", "no:cacheprovider", "--no-header", *a.tests], cwd=repo,
                          env={**os.environ, "PYTHONPATH": str(repo / "src")}, capture_output=True, text=True)
    if base.returncode != 0:
        print("baseline tests fail; fix them first"); print(base.stdout[-600:]); return 2
    with cf.ThreadPoolExecutor(a.workers) as ex:
        res = list(ex.map(run_one, [(str(mod_rel), a.tests, i, repo, a.timeout) for i in picks]))
    survivors = [(ln, k) for _, k, ln, killed in res if not killed]
    score = 1 - len(survivors) / len(res) if res else 1.0
    print(f"{a.module}: {len(res)} of {total} mutants tried, killed {len(res) - len(survivors)}, survived {len(survivors)}, score {score:.0%}")
    lines = (repo / "src" / mod_rel).read_text(encoding="utf-8").splitlines()
    for ln, k in sorted(survivors):
        print(f"  SURVIVED line {ln} [{k}]: {lines[ln - 1].strip()[:110]}")
    return 0 if score >= a.min_score else 1


if __name__ == "__main__":
    sys.exit(main())
