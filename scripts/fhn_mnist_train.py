#!/usr/bin/env python3
"""
Train a dense FitzHugh–Nagumo (FHN) network on MNIST / Fashion-MNIST with backprop through time.

  image -> constant drive u = W_in x -> T Euler steps with recurrent current W_rec v -> readout of mean v

  --train_rec 0   freeze W_rec (reservoir-like), learn W_in + readout
  --train_rec 1   learn W_rec too
  --train_in 0    also freeze W_in: only the linear readout learns (random FHN reservoir)

Examples:
  python scripts/fhn_mnist_train.py --epochs 10
  python scripts/fhn_mnist_train.py --dataset fashion --neurons 1024 --T 24
  python scripts/fhn_mnist_train.py --train_rec 1 --conn_video runs/conn.mp4 --conn_every 50 --conn_k 256
"""

from __future__ import annotations

import argparse
import json
import os
import time

import torch
import torch.nn.functional as F

from fhn_mnist.backprop import FHNClassifier, evaluate
from fhn_mnist.data import build_loaders
from fhn_mnist.video import ConnectivityVideoLogger


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="mnist", choices=["mnist", "fashion", "fake"])
    ap.add_argument("--data_dir", default="./data")
    ap.add_argument("--val_size", type=int, default=5000, help="held out from train for model selection")
    ap.add_argument("--neurons", type=int, default=512)
    ap.add_argument("--T", type=int, default=16, help="Euler steps per image")
    ap.add_argument("--dt", type=float, default=0.1)
    ap.add_argument("--a", type=float, default=0.7)
    ap.add_argument("--b", type=float, default=0.8)
    ap.add_argument("--tau", type=float, default=12.5)
    ap.add_argument("--g", type=float, default=1.0, help="recurrent init scale")
    ap.add_argument("--bound", type=float, default=5.0)
    ap.add_argument("--bound_mode", default="tanh", choices=["tanh", "clamp"])
    ap.add_argument("--pool", default="mean", choices=["mean", "last"])
    ap.add_argument("--train_rec", type=int, default=1, help="0 freeze W_rec, 1 train W_rec")
    ap.add_argument("--train_in", type=int, default=1, help="0 freeze W_in (readout-only reservoir)")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--clip", type=float, default=1.0, help="grad norm clip (0 disables)")
    ap.add_argument("--amp", type=int, default=1, help="bf16 autocast on CUDA")
    ap.add_argument("--compile", type=int, default=-1, help="torch.compile: 1 on, 0 off, -1 auto (CUDA only)")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out_dir", default="", help="write metrics.json and best.pt here")

    ap.add_argument("--conn_video", default="", help="output .mp4/.gif of W_rec over training (empty: off)")
    ap.add_argument("--conn_every", type=int, default=50, help="frame every N optimizer steps (0: off)")
    ap.add_argument("--conn_epoch", type=int, default=1, help="also a frame at each epoch end (0/1)")
    ap.add_argument("--conn_k", type=int, default=256, help="k x k submatrix to visualise")
    ap.add_argument("--conn_fps", type=int, default=15)
    ap.add_argument("--conn_max_frames", type=int, default=2000)
    ap.add_argument("--conn_vclip_pct", type=float, default=99.5)
    ap.add_argument("--conn_dpi", type=int, default=130)
    return ap


def make_model(args):
    return FHNClassifier(
        neurons=args.neurons, T=args.T, dt=args.dt, a=args.a, b=args.b, tau=args.tau, g=args.g,
        bound=args.bound, bound_mode=args.bound_mode, readout_pool=args.pool,
        train_rec=bool(args.train_rec), train_in=bool(args.train_in),
    )


def train(args, verbose=True, model=None):
    """Train with AdamW; `model` overrides the FHN network (any module mapping images to logits)."""
    torch.manual_seed(args.seed)
    dev = torch.device(args.device)
    tr, va, te = build_loaders(args.dataset, args.data_dir, args.batch, 1000, args.workers, args.seed, args.val_size)
    model = (model if model is not None else make_model(args)).to(dev)
    fwd = model
    if args.compile == 1 or (args.compile == -1 and dev.type == "cuda"):
        try:
            fwd = torch.compile(model)
        except Exception as e:  # pragma: no cover
            print(f"[warn] torch.compile failed: {e}")

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)
    use_amp = bool(args.amp) and dev.type == "cuda"

    logger = None
    if args.conn_video:
        logger = ConnectivityVideoLogger(args.conn_video, args.conn_fps, args.conn_every, bool(args.conn_epoch),
                                         args.conn_k, args.seed, args.conn_max_frames, args.conn_vclip_pct,
                                         args.conn_dpi)
        if not args.train_rec:
            print("[warn] --train_rec 0: W_rec is frozen, the connectivity video will be constant.")

    history, best = [], {"val_acc": -1.0}
    step, t_start = 0, time.time()
    try:
        for ep in range(1, args.epochs + 1):
            model.train()
            t0, run_loss, seen = time.time(), 0.0, 0
            for x, y in tr:
                x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
                opt.zero_grad(set_to_none=True)
                with torch.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=use_amp):
                    loss = F.cross_entropy(fwd(x).float(), y)
                loss.backward()
                if args.clip > 0:
                    torch.nn.utils.clip_grad_norm_(params, max_norm=args.clip)
                opt.step()
                run_loss += loss.item() * x.size(0)
                seen += x.size(0)
                step += 1
                if logger is not None and hasattr(model, "W_rec"):
                    logger.maybe_log_step(model.W_rec.weight, ep, step, step, loss.item())
            ev = va if va is not None else te
            v_loss, v_acc = evaluate(fwd, ev, dev)
            row = {"epoch": ep, "train_loss": run_loss / seen, "val_loss": v_loss, "val_acc": v_acc,
                   "time_s": time.time() - t0}
            history.append(row)
            if v_acc > best["val_acc"]:
                best = {"val_acc": v_acc, "epoch": ep,
                        "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}
            if logger is not None and hasattr(model, "W_rec"):
                logger.log_epoch_end(model.W_rec.weight, ep, step, v_loss)
            if verbose:
                print(f"[epoch {ep:02d}] train_loss={row['train_loss']:.4f} val_loss={v_loss:.4f} "
                      f"val_acc={v_acc*100:.2f}% ({row['time_s']:.1f}s)", flush=True)
    finally:
        if logger is not None:
            logger.close()

    model.load_state_dict(best["state"])
    te_loss, te_acc = evaluate(model, te, dev)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    out = {"dataset": args.dataset, "best_epoch": best["epoch"], "test_loss": te_loss, "test_acc": te_acc,
           "trainable_params": n_params, "train_time_s": time.time() - t_start, "history": history,
           "config": vars(args)}
    if verbose:
        print(f"[test] best epoch {best['epoch']}: test_acc={te_acc*100:.2f}% test_loss={te_loss:.4f}")
    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)
        with open(os.path.join(args.out_dir, "metrics.json"), "w") as f:
            json.dump(out, f, indent=2)
        torch.save({"model": model.state_dict(), "config": vars(args)}, os.path.join(args.out_dir, "best.pt"))
    return out, model


def main():
    train(build_parser().parse_args())


if __name__ == "__main__":
    main()
