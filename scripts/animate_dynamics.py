#!/usr/bin/env python3
"""
Animate what a trained FHN network does with a digit: membrane voltages v(t) of a few neurons,
their (v, w) phase-plane trajectories, and the class probabilities read out from the running mean
of v at every step.

    python scripts/animate_dynamics.py --checkpoint runs/bp/best.pt --out docs/media/dynamics.gif
    python scripts/animate_dynamics.py --train_epochs 1 --out dynamics.gif   # trains a model first
    python scripts/animate_dynamics.py --checkpoint runs/bp/best.pt --T 64   # run past the trained horizon
"""

from __future__ import annotations

import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fhn_mnist_train as bp  # noqa: E402

from fhn_mnist.data import CLASSES, STATS, build_datasets  # noqa: E402


def load_or_train(a):
    if a.checkpoint:
        ck = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
        args = bp.build_parser().parse_args([])
        for k, v in ck["config"].items():
            setattr(args, k, v)
        model = bp.make_model(args)
        model.load_state_dict(ck["model"])
        return model.eval(), args
    args = bp.build_parser().parse_args([])
    args.epochs, args.dataset, args.data_dir, args.device = a.train_epochs, a.dataset, a.data_dir, "cpu"
    _, model = bp.train(args)
    return model.eval(), args


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="")
    ap.add_argument("--train_epochs", type=int, default=1)
    ap.add_argument("--dataset", default="mnist")
    ap.add_argument("--data_dir", default="./data")
    ap.add_argument("--n_digits", type=int, default=3)
    ap.add_argument("--n_neurons", type=int, default=6, help="neurons to trace (most input-selective)")
    ap.add_argument("--T", type=int, default=0, help="steps to simulate (0: as trained)")
    ap.add_argument("--fps", type=int, default=8)
    ap.add_argument("--out", default="dynamics.gif")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    model, args = load_or_train(a)
    if a.T:
        model.T = a.T
    T = model.T
    _, _, test = build_datasets(args.dataset, a.data_dir, 0, a.seed)
    g = torch.Generator().manual_seed(a.seed)
    idx = torch.randperm(len(test), generator=g)[: a.n_digits].tolist()
    xs = torch.stack([test[i][0] for i in idx])
    ys = [test[i][1] for i in idx]
    names = CLASSES.get(args.dataset, [str(i) for i in range(10)])

    with torch.no_grad():
        _, traj = model.features(xs, record=True)
        V, W = traj["v"].numpy(), traj["w"].numpy()               # [T,B,N]
        run_mean = torch.cumsum(traj["v"], 0) / torch.arange(1, T + 1).view(-1, 1, 1)
        probs = torch.softmax(model.readout(run_mean), -1).numpy()  # [T,B,C]
    # neurons with the largest spread of mean activity across the chosen digits
    sel = np.argsort(-V.mean(0).std(0))[: a.n_neurons]
    mean, std = STATS.get(args.dataset, ((0.0,), (1.0,)))
    dt = model.p.dt

    frames = []
    import imageio.v2 as imageio

    colors = plt.cm.tab10(np.arange(a.n_neurons))
    for t in range(T):
        fig, axes = plt.subplots(a.n_digits, 4, figsize=(12, 2.6 * a.n_digits),
                                 gridspec_kw={"width_ratios": [0.7, 2, 1.3, 1.6]})
        axes = np.atleast_2d(axes)
        for r in range(a.n_digits):
            ax_img, ax_v, ax_ph, ax_p = axes[r]
            ax_img.imshow(xs[r, 0].numpy() * std[0] + mean[0], cmap="gray")
            ax_img.set_title(f"label: {names[ys[r]]}", fontsize=9)
            ax_img.axis("off")
            tt = (np.arange(t + 1) + 1) * dt
            for k, n in enumerate(sel):
                ax_v.plot(tt, V[: t + 1, r, n], color=colors[k], lw=1.2)
                ax_ph.plot(V[: t + 1, r, n], W[: t + 1, r, n], color=colors[k], lw=1.0)
                ax_ph.plot(V[t, r, n], W[t, r, n], "o", color=colors[k], ms=3)
            ax_v.set_xlim(0, T * dt)
            ax_v.set_ylim(V[:, :, sel].min() - 0.2, V[:, :, sel].max() + 0.2)
            ax_v.set_ylabel("v")
            ax_ph.set_xlim(V[:, :, sel].min() - 0.2, V[:, :, sel].max() + 0.2)
            ax_ph.set_ylim(W[:, :, sel].min() - 0.05, W[:, :, sel].max() + 0.05)
            ax_ph.set_xlabel("v")
            ax_ph.set_ylabel("w")
            p = probs[t, r]
            bars = ax_p.barh(range(len(p)), p, color="lightgray")
            bars[int(p.argmax())].set_color("tab:green" if int(p.argmax()) == ys[r] else "tab:red")
            ax_p.set_yticks(range(len(p)))
            ax_p.set_yticklabels(names, fontsize=7)
            ax_p.set_xlim(0, 1)
            ax_p.invert_yaxis()
            if r == 0:
                ax_v.set_title(f"membrane voltage, t = {(t + 1) * dt:.1f}", fontsize=9)
                ax_ph.set_title("phase plane (v, w)", fontsize=9)
                ax_p.set_title("readout of running mean v", fontsize=9)
            if r == a.n_digits - 1:
                ax_v.set_xlabel("time")
        fig.tight_layout()
        fig.canvas.draw()
        frames.append(np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy())
        if t == T - 1:
            fig.savefig(os.path.splitext(a.out)[0] + "_final.png", dpi=110)
        plt.close(fig)
    frames += [frames[-1]] * a.fps  # hold the last frame
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    imageio.mimsave(a.out, frames, fps=a.fps, loop=0)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
