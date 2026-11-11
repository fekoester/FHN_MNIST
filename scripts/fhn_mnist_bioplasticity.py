#!/usr/bin/env python3
"""
Train a dense FitzHugh–Nagumo network on MNIST / Fashion-MNIST with three-factor plasticity (no backprop).

  teacher    broadcast output error through fixed random feedback (e-prop / feedback-alignment-like)
  reinforce  reward-modulated learning (dopamine-like reward-prediction error)

See fhn_mnist/plasticity.py for the learning rules. Learning-rate defaults depend on the mode
(tuned on the validation split; see the README).

Examples:
  python scripts/fhn_mnist_bioplasticity.py --mode teacher --epochs 3
  python scripts/fhn_mnist_bioplasticity.py --mode reinforce --epochs 3
  python scripts/fhn_mnist_bioplasticity.py --mode teacher --conn_video runs/conn.mp4 --conn_every 50
  # the original per-sample, sum-reduced update (learning rates as in v0.1)
  python scripts/fhn_mnist_bioplasticity.py --batch 1 --eta_in 2e-4 --eta_rec 2e-4 --eta_out 2e-3
"""

from __future__ import annotations

import argparse
import json
import os
import time

import torch

from fhn_mnist.data import build_loaders
from fhn_mnist.dynamics import FHNParams
from fhn_mnist.plasticity import FHNBioplasticNet, PlasticityCfg, evaluate_bio
from fhn_mnist.video import ConnectivityVideoLogger

MODE_DEFAULTS = {
    "teacher": dict(eta_in=1e-3, eta_rec=1e-3, eta_out=5e-3, rl_signal="global"),
    "reinforce": dict(eta_in=3e-6, eta_rec=3e-5, eta_out=1e-2, rl_signal="feedback"),
}


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", default="teacher", choices=["teacher", "reinforce"])
    ap.add_argument("--dataset", default="mnist", choices=["mnist", "fashion", "fake"])
    ap.add_argument("--data_dir", default="./data")
    ap.add_argument("--val_size", type=int, default=5000)

    # model / simulation
    ap.add_argument("--neurons", type=int, default=512)
    ap.add_argument("--T", type=int, default=16)
    ap.add_argument("--dt", type=float, default=0.10)
    ap.add_argument("--a", type=float, default=0.7)
    ap.add_argument("--b", type=float, default=0.8)
    ap.add_argument("--tau", type=float, default=12.5)
    ap.add_argument("--g", type=float, default=1.0, help="recurrent init scale (orthogonal * g/sqrt(N))")
    ap.add_argument("--pool", default="mean", choices=["mean", "last"])
    ap.add_argument("--bound", type=float, default=5.0)
    ap.add_argument("--bound_mode", default="tanh", choices=["tanh", "clamp"])
    ap.add_argument("--readout_on", default="v", choices=["v", "rate"])

    # spike proxy and traces
    ap.add_argument("--spk_theta", type=float, default=0.0)
    ap.add_argument("--spk_kappa", type=float, default=0.5)
    ap.add_argument("--tau_pre", type=float, default=8.0)
    ap.add_argument("--tau_post", type=float, default=8.0)
    ap.add_argument("--tau_e", type=float, default=30.0)
    ap.add_argument("--A_plus", type=float, default=1.0)
    ap.add_argument("--A_minus", type=float, default=0.8)

    # learning rates (None = mode default)
    ap.add_argument("--eta_in", type=float, default=None)
    ap.add_argument("--eta_rec", type=float, default=None)
    ap.add_argument("--eta_out", type=float, default=None)
    ap.add_argument("--train_rec", type=int, default=1, help="0 freezes W_rec")

    # safety
    ap.add_argument("--w_decay", type=float, default=0.0)
    ap.add_argument("--w_clip", type=float, default=0.0)
    ap.add_argument("--max_update", type=float, default=0.0, help="cap on the norm of each weight update")

    # reinforce
    ap.add_argument("--explore", default="sample", choices=["sample", "argmax"])
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--baseline_ema", type=float, default=0.99)
    ap.add_argument("--rl_signal", default=None, choices=["global", "feedback"])

    # teacher
    ap.add_argument("--teacher_inject", type=float, default=0.0,
                    help="label current during the learning pass; predictions/errors come from a free pass")

    # training
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", "--batch_io", dest="batch", type=int, default=16,
                    help="trials per weight update (1 = original online learning)")
    ap.add_argument("--batch_reduce", default=None, choices=["sum", "mean"],
                    help="combine per-trial updates (default: mean for batch>1, sum for batch=1)")
    ap.add_argument("--eval_batch", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out_dir", default="", help="write metrics.json and model.pt here")

    # connectivity video
    ap.add_argument("--conn_video", default="", help="output .mp4/.gif of W_rec over training (empty: off)")
    ap.add_argument("--conn_every", type=int, default=200, help="frame every N updates (0: off)")
    ap.add_argument("--conn_epoch", type=int, default=1)
    ap.add_argument("--conn_k", type=int, default=256)
    ap.add_argument("--conn_fps", type=int, default=15)
    ap.add_argument("--conn_max_frames", type=int, default=2000)
    ap.add_argument("--conn_vclip_pct", type=float, default=99.5)
    ap.add_argument("--conn_dpi", type=int, default=130)
    return ap


def resolve(args):
    for k, v in MODE_DEFAULTS[args.mode].items():
        if getattr(args, k) is None:
            setattr(args, k, v)
    if args.batch_reduce is None:
        args.batch_reduce = "mean" if args.batch > 1 else "sum"
    return args


def make_model(args):
    cfg = PlasticityCfg(
        spk_theta=args.spk_theta, spk_kappa=args.spk_kappa, tau_pre=args.tau_pre, tau_post=args.tau_post,
        tau_e=args.tau_e, A_plus=args.A_plus, A_minus=args.A_minus, eta_in=args.eta_in,
        eta_rec=args.eta_rec, eta_out=args.eta_out, w_decay=args.w_decay, w_clip=args.w_clip,
        max_update=args.max_update, readout_on=args.readout_on, teacher_inject=args.teacher_inject,
        explore=args.explore, temperature=args.temperature, baseline_ema=args.baseline_ema,
        rl_signal=args.rl_signal, batch_reduce=args.batch_reduce,
    )
    params = FHNParams(dt=args.dt, a=args.a, b=args.b, tau=args.tau, bound=args.bound, bound_mode=args.bound_mode)
    return FHNBioplasticNet(args.neurons, args.T, args.g, cfg, params, bool(args.train_rec), args.pool,
                            device=args.device)


def train(args, verbose=True):
    """Train with plasticity; returns a metrics dict (and writes it to out_dir if given)."""
    args = resolve(args)
    torch.manual_seed(args.seed)
    dev = args.device
    tr, va, te = build_loaders(args.dataset, args.data_dir, args.batch, args.eval_batch, args.workers,
                               args.seed, args.val_size)
    model = make_model(args)

    logger = None
    if args.conn_video:
        logger = ConnectivityVideoLogger(args.conn_video, args.conn_fps, args.conn_every, bool(args.conn_epoch),
                                         args.conn_k, args.seed, args.conn_max_frames, args.conn_vclip_pct,
                                         args.conn_dpi)

    history, best = [], {"val_acc": -1.0}
    t_start = time.time()
    step = 0
    try:
        for ep in range(1, args.epochs + 1):
            t0 = time.time()
            agg = {"n": 0, "ce_sum": 0.0, "correct": 0, "reward": 0.0}
            for x, y in tr:
                r = model.learn_batch(x.to(dev), y.to(dev), args.mode)
                for k in agg:
                    agg[k] += r.get(k, 0)
                step += 1
                if logger is not None:
                    logger.maybe_log_step(model.W_rec.weight, ep, step, step, r["ce_sum"] / r["n"], tag=args.mode)
            ev = va if va is not None else te
            v_ce, v_acc = evaluate_bio(model, ev, dev)
            row = {"epoch": ep, "train_ce": agg["ce_sum"] / agg["n"], "train_acc": agg["correct"] / agg["n"],
                   "val_ce": v_ce, "val_acc": v_acc, "time_s": time.time() - t0}
            if args.mode == "reinforce":
                row["train_reward"] = agg["reward"] / agg["n"]
            history.append(row)
            if v_acc > best["val_acc"]:
                best = {"val_acc": v_acc, "epoch": ep,
                        "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}
            if logger is not None:
                logger.log_epoch_end(model.W_rec.weight, ep, step, v_ce)
            if verbose:
                extra = f" reward={row['train_reward']*100:.1f}%" if "train_reward" in row else ""
                print(f"[epoch {ep:02d}] {args.mode} train_acc={row['train_acc']*100:.2f}%{extra} "
                      f"val_ce={v_ce:.4f} val_acc={v_acc*100:.2f}% ({row['time_s']:.1f}s)", flush=True)
    finally:
        if logger is not None:
            logger.close()

    model.load_state_dict(best["state"])
    te_ce, te_acc = evaluate_bio(model, te, dev)
    out = {"mode": args.mode, "dataset": args.dataset, "best_epoch": best["epoch"], "test_ce": te_ce,
           "test_acc": te_acc, "train_time_s": time.time() - t_start, "history": history,
           "config": {k: v for k, v in vars(args).items()}}
    if verbose:
        print(f"[test] best epoch {best['epoch']}: test_acc={te_acc*100:.2f}% test_ce={te_ce:.4f}")
    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)
        with open(os.path.join(args.out_dir, "metrics.json"), "w") as f:
            json.dump(out, f, indent=2)
        torch.save({"model": model.state_dict(), "config": out["config"]}, os.path.join(args.out_dir, "model.pt"))
    return out, model


def main():
    train(build_parser().parse_args())


if __name__ == "__main__":
    main()
