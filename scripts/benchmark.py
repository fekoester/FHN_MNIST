#!/usr/bin/env python3
"""
Benchmark: what do the FHN dynamics and the learning rules buy over simple baselines?

For each dataset, trains (same validation split, best epoch by validation, test reported once):
  logreg             multinomial logistic regression on pixels
  mlp                784-512-10 ReLU MLP (backprop)
  fhn_reservoir      random FHN network, only the linear readout learns (backprop on the readout)
  fhn_bp_frozen_rec  backprop through time into W_in + readout, W_rec frozen
  fhn_bp             backprop through time into W_in, W_rec and readout
  bio_teacher_out    teacher signal, only the readout learns
  bio_teacher        teacher signal, three-factor plasticity in W_in and W_rec too
  bio_rl_out         reward-modulated, only the readout learns
  bio_rl             reward-modulated, three-factor plasticity in W_in and W_rec too

    python scripts/benchmark.py --datasets mnist fashion --out results/benchmark
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fhn_mnist_bioplasticity as bio  # noqa: E402
import fhn_mnist_train as bp  # noqa: E402

from fhn_mnist.backprop import MLP  # noqa: E402

BP_METHODS = {
    "logreg": dict(model=lambda: MLP(hidden=0)),
    "mlp": dict(model=lambda: MLP(hidden=512)),
    "fhn_reservoir": dict(train_in=0, train_rec=0),
    "fhn_bp_frozen_rec": dict(train_rec=0),
    "fhn_bp": dict(train_rec=1),
}
BIO_METHODS = {
    "bio_teacher_out": dict(mode="teacher", eta_in=0.0, eta_rec=0.0),
    "bio_teacher": dict(mode="teacher"),
    "bio_rl_out": dict(mode="reinforce", eta_in=0.0, eta_rec=0.0),
    "bio_rl": dict(mode="reinforce"),
}
LABELS = {
    "logreg": "Logistic regression (pixels)", "mlp": "MLP 784-512-10", "fhn_reservoir": "FHN, readout only (BP)",
    "fhn_bp_frozen_rec": "FHN, BPTT, W_rec frozen", "fhn_bp": "FHN, BPTT, all weights",
    "bio_teacher_out": "FHN, teacher, readout only", "bio_teacher": "FHN, teacher, 3-factor",
    "bio_rl_out": "FHN, reward, readout only", "bio_rl": "FHN, reward, 3-factor",
}


def run(dataset, method, a):
    if method in BP_METHODS:
        spec = dict(BP_METHODS[method])
        model_fn = spec.pop("model", None)
        args = bp.build_parser().parse_args([])
        args.epochs = a.bp_epochs
    else:
        spec = dict(BIO_METHODS[method])
        model_fn = None
        args = bio.build_parser().parse_args([])
        args.epochs = a.bio_epochs
    for k, v in spec.items():
        setattr(args, k, v)
    args.dataset, args.data_dir, args.seed, args.workers = dataset, a.data_dir, a.seed, a.workers
    args.neurons, args.T = a.neurons, a.T
    print(f"\n=== {dataset} / {method} ===", flush=True)
    if method in BP_METHODS:
        out, model = bp.train(args, model=model_fn() if model_fn else None)
    else:
        out, model = bio.train(args)
        out["trainable_params"] = sum(p.numel() for n, p in model.named_parameters()
                                      if not n.startswith("W_rec") or args.train_rec)
        if args.eta_in == 0:
            out["trainable_params"] = sum(p.numel() for p in model.W_out.parameters())
    out.pop("config", None)
    return {"dataset": dataset, "method": method, "label": LABELS[method], "test_acc": out["test_acc"],
            "best_epoch": out["best_epoch"], "train_time_s": out["train_time_s"],
            "trainable_params": out.get("trainable_params"), "history": out["history"]}


def table(results):
    lines = ["| method | " + " | ".join(sorted({r["dataset"] for r in results})) + " | train time (s) |",
             "|---|" + "---|" * (len({r["dataset"] for r in results}) + 1)]
    datasets = sorted({r["dataset"] for r in results})
    for m in LABELS:
        rows = {r["dataset"]: r for r in results if r["method"] == m}
        if not rows:
            continue
        accs = " | ".join(f"{rows[d]['test_acc']*100:.2f}%" if d in rows else "–" for d in datasets)
        times = "/".join(f"{rows[d]['train_time_s']:.0f}" for d in datasets if d in rows)
        lines.append(f"| {LABELS[m]} | {accs} | {times} |")
    return "\n".join(lines)


def plot(results, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    datasets = sorted({r["dataset"] for r in results})
    methods = [m for m in LABELS if any(r["method"] == m for r in results)]
    fig, ax = plt.subplots(figsize=(10, 4.2))
    width = 0.8 / len(datasets)
    for i, d in enumerate(datasets):
        acc = [next((100 * r["test_acc"] for r in results if r["method"] == m and r["dataset"] == d), np.nan)
               for m in methods]
        bars = ax.bar(np.arange(len(methods)) + (i - (len(datasets) - 1) / 2) * width, acc, width, label=d)
        for b, v in zip(bars, acc):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.3, f"{v:.1f}", ha="center", fontsize=7)
    ax.set_xticks(np.arange(len(methods)))
    ax.set_xticklabels([LABELS[m] for m in methods], rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("test accuracy (%)")
    lo = min(100 * r["test_acc"] for r in results)
    ax.set_ylim(max(0, lo - 10), 100)
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["mnist", "fashion"])
    ap.add_argument("--methods", nargs="+", default=list(LABELS))
    ap.add_argument("--bp_epochs", type=int, default=5)
    ap.add_argument("--bio_epochs", type=int, default=3)
    ap.add_argument("--neurons", type=int, default=512)
    ap.add_argument("--T", type=int, default=16)
    ap.add_argument("--data_dir", default="./data")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/benchmark")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    res_path = os.path.join(a.out, "results.json")
    results = json.load(open(res_path)) if os.path.exists(res_path) else []
    for d in a.datasets:
        for m in a.methods:
            if any(r["dataset"] == d and r["method"] == m for r in results):
                continue  # resume
            torch.manual_seed(a.seed)
            results.append(run(d, m, a))
            with open(res_path, "w") as f:
                json.dump(results, f, indent=2)
    md = table(results)
    with open(os.path.join(a.out, "results.md"), "w") as f:
        f.write(md + "\n")
    plot(results, os.path.join(a.out, "benchmark.png"))
    print("\n" + md)


if __name__ == "__main__":
    main()
