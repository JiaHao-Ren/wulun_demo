#!/usr/bin/env python3
# 把测出来的 jsonl 收成表。

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

TIER_ORDER = ["torch_eager", "ort_python", "ort_cpp", "ort_cpp_cuda",
              "torch_eager_cuda", "ort_python_cuda"]

TIER_LABEL = {
    "torch_eager": "1. PyTorch eager  (Python, CPU)",
    "ort_python": "2. ONNX Runtime   (Python, CPU)",
    "ort_cpp": "3. ONNX Runtime   (C++,    CPU)",
    "ort_cpp_cuda": "4. ONNX Runtime   (C++,    CUDA)",
    "ort_python_cuda": "2b. ONNX Runtime  (Python, CUDA)",
    "torch_eager_cuda": "1b. PyTorch eager (Python, CUDA)",
}

def load(path: Path):
    recs = []
    for n, line in enumerate(path.read_text().splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            recs.append(json.loads(line))
        except json.JSONDecodeError as e:
            print(f"{path}:{n}: skipping unparseable line ({e})", file=sys.stderr)
    return recs

def check_comparable(rows):
    problems = []

    threads = {r.get("threads") for r in rows}
    if len(threads) > 1:
        problems.append(
            f"thread counts differ ({sorted(threads)}): a C++ run on more threads "
            f"than the Python run measures core count, not language")

    ort_versions = {r.get("ort_version") for r in rows
                    if r["tier"].startswith("ort") and r.get("ort_version") not in (None, "n/a")}
    if len(ort_versions) > 1:
        problems.append(
            f"ONNX Runtime versions differ ({sorted(ort_versions)}): the "
            f"Python-vs-C++ gap would include a release difference")

    shapes = {tuple(r["shape"]) for r in rows if "shape" in r}
    if len(shapes) > 1:
        problems.append(f"input shapes differ ({sorted(shapes)})")

    return problems

def fmt_ms(v):
    return f"{v:9.3f}"

def main():
    p = argparse.ArgumentParser()
    p.add_argument("jsonl", nargs="?", default="results/bench.jsonl")
    p.add_argument("--markdown", action="store_true", help="emit a Markdown table")
    args = p.parse_args()

    path = Path(args.jsonl)
    if not path.exists():
        print(f"no such file: {path}", file=sys.stderr)
        return 1

    recs = load(path)
    by_label = defaultdict(list)
    for r in recs:
        by_label[r["label"]].append(r)

    for label, rows in by_label.items():
        rows.sort(key=lambda r: TIER_ORDER.index(r["tier"]) if r["tier"] in TIER_ORDER else 99)
        problems = check_comparable(rows)

        print()
        print("=" * 78)
        print(f"  {label}")
        shape = rows[0].get("shape")
        meta = [f"input {tuple(shape)}" if shape else "",
                f"threads {rows[0].get('threads')}",
                f"iters {rows[0].get('iters')}"]
        print("  " + " | ".join(m for m in meta if m))
        print("=" * 78)

        if problems:
            print("  !! NOT COMPARABLE -- speedups suppressed:")
            for pr in problems:
                print(f"     - {pr}")
            print()

        print(f"  {'tier':34s} {'mean':>9s} {'p50':>9s} {'p95':>9s} {'p99':>9s}")
        print("  " + "-" * 74)
        for r in rows:
            print(f"  {TIER_LABEL.get(r['tier'], r['tier']):34s}"
                  f"{fmt_ms(r['mean_ms'])}{fmt_ms(r['p50_ms'])}"
                  f"{fmt_ms(r['p95_ms'])}{fmt_ms(r['p99_ms'])}")

        if problems:
            continue

        idx = {r["tier"]: r for r in rows}
        print()
        print("  attribution (each step measured against the one above it):")
        steps = [
            ("torch_eager", "ort_python", "runtime / graph optimisation"),
            ("ort_python", "ort_cpp", "language: Python -> C++"),
            ("ort_cpp", "ort_cpp_cuda", "execution provider: CPU -> CUDA"),
        ]
        any_step = False
        for a, b, what in steps:
            if a not in idx or b not in idx:
                continue
            any_step = True
            ta, tb = idx[a]["mean_ms"], idx[b]["mean_ms"]
            factor = ta / tb if tb > 0 else float("inf")
            sd = max(idx[a].get("sd_ms", 0.0), idx[b].get("sd_ms", 0.0))
            note = ""
            if abs(ta - tb) < sd:
                note = "  <- smaller than the run-to-run sd; not significant"
            elif factor < 1.0:
                note = "  <- SLOWER"
            print(f"    {what:34s} {factor:6.2f}x{note}")
        if not any_step:
            print("    (not enough tiers present)")

        if "torch_eager" in idx and "ort_cpp_cuda" in idx:
            total = idx["torch_eager"]["mean_ms"] / idx["ort_cpp_cuda"]["mean_ms"]
            lang = (idx["ort_python"]["mean_ms"] / idx["ort_cpp"]["mean_ms"]
                    if "ort_python" in idx and "ort_cpp" in idx else None)
            print()
            print(f"    end-to-end tier1 -> tier4         {total:6.2f}x")
            if lang is not None:
                print(f"    of which attributable to C++     {lang:6.2f}x"
                      f"   ({100 * (lang - 1):.1f}% of the total)")
        missing = [t for t in ("torch_eager",) if t not in idx]
        if missing:
            print()
            print("    tier 1 absent for this model: no PyTorch implementation exists")
            print("    (emotion-ferplus is a CNTK export). Benchmarking a substitute")
            print("    network here would not be a baseline, it would be a different")
            print("    measurement wearing the same label.")

    print()
    return 0

if __name__ == "__main__":
    sys.exit(main())
