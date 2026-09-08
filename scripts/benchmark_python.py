#!/usr/bin/env python3
# 四档对比的 Python 侧(tier 1/2)。线程数必须和 benchmark_cpp 一致。

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from pathlib import Path

def deterministic_input(n_elem: int):
    import numpy as np
    idx = np.arange(n_elem, dtype=np.int64) % 251
    return (idx.astype(np.float32) / np.float32(251.0)).astype(np.float32)

def percentile(sorted_vals, q: float) -> float:
    import math
    if not sorted_vals:
        return 0.0
    if q <= 0:
        return sorted_vals[0]
    if q >= 1:
        return sorted_vals[-1]
    rank = max(1, min(len(sorted_vals), math.ceil(q * len(sorted_vals))))
    return sorted_vals[rank - 1]

def summarise(samples_ms):
    s = sorted(samples_ms)
    return {
        "mean_ms": statistics.fmean(samples_ms),
        "sd_ms": statistics.stdev(samples_ms) if len(samples_ms) > 1 else 0.0,
        "p50_ms": percentile(s, 0.50),
        "p95_ms": percentile(s, 0.95),
        "p99_ms": percentile(s, 0.99),
        "min_ms": s[0],
    }

def run_ort(args):
    import numpy as np
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.intra_op_num_threads = args.threads
    so.inter_op_num_threads = args.threads
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    providers = ["CUDAExecutionProvider"] if args.cuda else ["CPUExecutionProvider"]
    if args.cuda and "CUDAExecutionProvider" not in ort.get_available_providers():
        print("WARNING: CUDA requested but not available; falling back to CPU",
              file=sys.stderr)
        providers = ["CPUExecutionProvider"]

    sess = ort.InferenceSession(args.model, sess_options=so, providers=providers)
    actual = sess.get_providers()[0]

    inp = sess.get_inputs()[0]
    shape = [d if isinstance(d, int) and d > 0 else 1 for d in inp.shape]
    if args.shape:
        shape = [int(x) for x in args.shape.split(",")]
    n_elem = 1
    for d in shape:
        n_elem *= d

    x = deterministic_input(n_elem).reshape(shape)
    feed = {inp.name: x}
    out_names = [o.name for o in sess.get_outputs()]

    for _ in range(args.warmup):
        sess.run(out_names, feed)

    samples = []
    for _ in range(args.iters):
        t0 = time.perf_counter()
        sess.run(out_names, feed)
        samples.append((time.perf_counter() - t0) * 1000.0)

    tier = "ort_python_cuda" if actual == "CUDAExecutionProvider" else "ort_python"
    return tier, actual, shape, samples

def run_torch(args):
    import numpy as np
    import torch

    if not args.torch_model:
        print("ERROR: --torch-model is required for the pytorch tier.\n"
              "There is no honest way to produce a PyTorch-eager number for a\n"
              "model that has no PyTorch implementation (emotion-ferplus is a\n"
              "CNTK export). Omit this tier for such models instead of\n"
              "substituting a different network.", file=sys.stderr)
        sys.exit(2)

    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(args.threads)

    try:
        model = torch.jit.load(args.torch_model, map_location=device).eval()
    except (RuntimeError, ValueError):
        obj = torch.load(args.torch_model, map_location=device, weights_only=False)
        model = obj.eval().to(device) if hasattr(obj, "eval") else obj

    if not args.shape:
        print("ERROR: --shape is required for the pytorch tier; guessing it would\n"
              "       silently benchmark a different input size from the ONNX tiers.",
              file=sys.stderr)
        sys.exit(2)
    shape = [int(x) for x in args.shape.split(",")]
    n_elem = 1
    for d in shape:
        n_elem *= d
    x = torch.from_numpy(deterministic_input(n_elem).reshape(shape)).to(device)

    def once():
        with torch.inference_mode():
            model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()

    for _ in range(args.warmup):
        once()

    samples = []
    for _ in range(args.iters):
        t0 = time.perf_counter()
        once()
        samples.append((time.perf_counter() - t0) * 1000.0)

    tier = "torch_eager_cuda" if device.type == "cuda" else "torch_eager"
    return tier, f"torch-{device.type}", shape, samples

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", help=".onnx file (for the onnxruntime tier)")
    p.add_argument("--torch-model", help="TorchScript/pickled nn.Module (pytorch tier)")
    p.add_argument("--tier", choices=["ort", "torch"], default="ort")
    p.add_argument("--shape", help="comma-separated input shape")
    p.add_argument("--iters", type=int, default=200)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--threads", type=int, default=1,
                   help="intra-op AND inter-op threads; must match the C++ run")
    p.add_argument("--cuda", action="store_true")
    p.add_argument("--label", default="model")
    p.add_argument("--out", help="append one JSON record to this file")
    args = p.parse_args()

    if args.tier == "ort":
        if not args.model:
            p.error("--model is required for the ort tier")
        tier, backend, shape, samples = run_ort(args)
        model_path = args.model
    else:
        tier, backend, shape, samples = run_torch(args)
        model_path = args.torch_model

    try:
        import onnxruntime as _ort
        ort_version = _ort.__version__
    except Exception:
        ort_version = "n/a"

    rec = {
        "tier": tier,
        "label": args.label,
        "model": model_path,
        "backend": backend,
        "threads": args.threads,
        "iters": args.iters,
        "warmup": args.warmup,
        "shape": shape,
        "python": platform.python_version(),
        "ort_version": ort_version,
        **summarise(samples),
    }

    print(f"label      : {rec['label']}")
    print(f"tier       : {rec['tier']}")
    print(f"backend    : {rec['backend']}")
    print(f"threads    : {rec['threads']}")
    print(f"iters      : {rec['iters']} (warmup {rec['warmup']})")
    print(f"mean       : {rec['mean_ms']:.3f} ms  (sd {rec['sd_ms']:.3f})")
    print(f"p50        : {rec['p50_ms']:.3f} ms")
    print(f"p95        : {rec['p95_ms']:.3f} ms")
    print(f"p99        : {rec['p99_ms']:.3f} ms")
    print(f"min        : {rec['min_ms']:.3f} ms")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"appended JSON record to {args.out}")

if __name__ == "__main__":
    main()
