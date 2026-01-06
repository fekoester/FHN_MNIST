#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fhn_mnist_train.py

Train a dense FitzHugh–Nagumo (FHN) network to classify MNIST on GPU.

Extended:
  - Logs evolution of recurrent connectivity (W_rec.weight) into a video.

Core idea:
  - Image x -> constant drive u = W_in(x)
  - Unroll FHN Euler for T steps with recurrent current W_rec(v)
  - Readout from v -> logits -> CE loss

Modes:
  --train_rec 0  : reservoir-ish (freeze W_rec), learn W_in + readout
  --train_rec 1  : learn W_rec too (connectivity changes)

Run examples:
  python fhn_mnist_train.py --neurons 2048 --T 32 --train_rec 0
  python fhn_mnist_train.py --neurons 4096 --T 24 --train_rec 1 --g 0.5 --dt 0.03

Connectivity video examples:
  # log every 50 Adam steps + each epoch end, record 256x256 submatrix to MP4
  python fhn_mnist_train.py --train_rec 1 --conn_video conn.mp4 --conn_every 50 --conn_epoch 1 --conn_k 256

  # log only at epoch end, record full matrix (WARNING: huge) to GIF
  python fhn_mnist_train.py --train_rec 1 --neurons 1024 --conn_video conn.gif --conn_epoch 1 --conn_k 1024
"""

import argparse, math, time, os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# --------------------------- Connectivity video logger ---------------------------

class ConnectivityVideoLogger:
    """
    Writes frames of a kxk submatrix of W_rec.weight over training.

    - Uses a fixed set of neuron indices for consistent visuals.
    - Supports MP4 (via imageio + ffmpeg) or GIF.
    - If MP4 writing fails, falls back to GIF automatically.
    """
    def __init__(
        self,
        out_path: str,
        fps: int = 15,
        every_steps: int = 0,
        log_epoch_end: bool = False,
        k: int = 256,
        seed: int = 0,
        max_frames: int = 2000,
        vclip_pct: float = 99.5,
        dpi: int = 130,
    ):
        self.out_path = out_path
        self.fps = int(fps)
        self.every_steps = int(every_steps)
        self.log_epoch_end_enabled = bool(log_epoch_end)
        self.k = int(k)
        self.seed = int(seed)
        self.max_frames = int(max_frames)
        self.vclip_pct = float(vclip_pct)
        self.dpi = int(dpi)

        self._writer = None
        self._writer_kind = None
        self._frames = 0
        self._idx = None  # chosen neuron indices, torch.LongTensor on same device as weight

        self._ensure_dir()

    def _ensure_dir(self):
        d = os.path.dirname(os.path.abspath(self.out_path))
        if d and not os.path.exists(d):
            os.makedirs(d, exist_ok=True)

    def _lazy_init_idx(self, W: torch.Tensor):
        # W: [N,N]
        N = W.shape[0]
        k = min(self.k, N)

        # choose fixed indices on CPU then move to device
        g = torch.Generator(device="cpu")
        g.manual_seed(self.seed)
        perm = torch.randperm(N, generator=g)[:k]
        self._idx = perm.to(device=W.device)

    def _lazy_open_writer(self):
        if self._writer is not None:
            return

        # Import here so script runs even if user doesn't enable logging
        import imageio.v2 as imageio

        ext = os.path.splitext(self.out_path)[1].lower()
        try:
            if ext == ".mp4":
                # needs ffmpeg installed / imageio-ffmpeg available
                self._writer = imageio.get_writer(
                    self.out_path,
                    fps=self.fps,
                    codec="libx264",
                    quality=8,
                    macro_block_size=None,  # avoid dimension constraints
                )
                self._writer_kind = "mp4"
            elif ext == ".gif":
                self._writer = imageio.get_writer(self.out_path, fps=self.fps)
                self._writer_kind = "gif"
            else:
                # default: gif
                gif_path = self.out_path + ".gif"
                self._writer = imageio.get_writer(gif_path, fps=self.fps)
                self._writer_kind = "gif"
                self.out_path = gif_path
        except Exception as e:
            # fallback to gif if mp4 fails
            if ext == ".mp4":
                gif_path = os.path.splitext(self.out_path)[0] + ".gif"
                self.out_path = gif_path
                self._writer = imageio.get_writer(self.out_path, fps=self.fps)
                self._writer_kind = "gif"
                print(f"[warn] MP4 writer failed ({e}). Falling back to GIF: {self.out_path}")
            else:
                raise

    @staticmethod
    def _to_frame_image(mat_cpu: torch.Tensor, title: str, vclip_pct: float, dpi: int):
        """
        mat_cpu: [k,k] float tensor on CPU
        Returns: HxWx3 uint8 numpy array
        """
        import numpy as np
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        A = mat_cpu.numpy()

        # robust symmetric clipping based on percentile of |A|
        absA = np.abs(A).reshape(-1)
        if absA.size == 0:
            vmax = 1.0
        else:
            vmax = np.percentile(absA, vclip_pct) if vclip_pct > 0 else absA.max()
            vmax = float(max(vmax, 1e-8))
        vmin = -vmax

        # draw
        fig = plt.figure(figsize=(5.2, 5.0), dpi=dpi)
        ax = fig.add_subplot(111)
        im = ax.imshow(A, vmin=vmin, vmax=vmax, interpolation="nearest")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("neuron j")
        ax.set_ylabel("neuron i")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # convert to RGB image (robust across matplotlib versions)
        fig.canvas.draw()

        # Preferred: renderer buffer -> RGBA
        try:
            rgba = np.asarray(fig.canvas.buffer_rgba())  # (H, W, 4), uint8
            img = rgba[:, :, :3].copy()                  # RGB
        except Exception:
            # Fallback: ARGB string -> RGB
            w, h = fig.canvas.get_width_height()
            argb = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8).reshape(h, w, 4)
            img = argb[:, :, [1, 2, 3]].copy()           # drop A, reorder to RGB

        plt.close(fig)
        return img


    def maybe_log_step(self, W: torch.Tensor, epoch: int, step_in_epoch: int, global_step: int, loss_val: float):
        if self.every_steps <= 0:
            return
        if (global_step % self.every_steps) != 0:
            return
        self._log(W, epoch=epoch, step_in_epoch=step_in_epoch, global_step=global_step, loss_val=loss_val, tag="step")

    def log_epoch_end(self, W: torch.Tensor, epoch: int, global_step: int, loss_val: float):
        if not self.log_epoch_end_enabled:
            return
        self._log(W, epoch=epoch, step_in_epoch=-1, global_step=global_step, loss_val=loss_val, tag="epoch_end")

    def _log(self, W: torch.Tensor, epoch: int, step_in_epoch: int, global_step: int, loss_val: float, tag: str):
        if self.out_path is None or self.out_path == "":
            return
        if self._frames >= self.max_frames:
            return

        if self._idx is None:
            self._lazy_init_idx(W)

        # gather kxk submatrix without moving full NxN to CPU
        idx = self._idx
        sub = W.index_select(0, idx).index_select(1, idx).detach().float().cpu()

        # basic stats
        with torch.no_grad():
            mean = float(sub.mean())
            std = float(sub.std(unbiased=False))
            mx = float(sub.abs().max())

        title = f"W_rec submatrix ({tag}) | ep={epoch} step={step_in_epoch if step_in_epoch>=0 else 'end'} gstep={global_step} | loss={loss_val:.4f} | mean={mean:.3e} std={std:.3e} max|w|={mx:.3e}"

        img = self._to_frame_image(sub, title=title, vclip_pct=self.vclip_pct, dpi=self.dpi)

        self._lazy_open_writer()
        self._writer.append_data(img)
        self._frames += 1

    def close(self):
        if self._writer is not None:
            try:
                self._writer.close()
            except Exception:
                pass
            self._writer = None


# --------------------------- Model ---------------------------

class FHNClassifier(nn.Module):
    """
    Dense FHN network classifier.
    x: [B, 1, 28, 28] -> flatten -> u = W_in(x) [B, N]
    Unroll T Euler steps:
      v' = v - v^3/3 - w + (u + W_rec(v))
      w' = (v + a - b w)/tau
    Readout: logits = W_out( pool(v over time) )
    """
    def __init__(
        self,
        neurons: int,
        T: int,
        dt: float,
        a: float,
        b: float,
        tau: float,
        g: float,
        bound: float,
        bound_mode: str,
        readout_pool: str,
        train_rec: bool,
    ):
        super().__init__()
        self.N = neurons
        self.T = T
        self.dt = dt

        self.bound = bound
        self.bound_mode = bound_mode
        self.readout_pool = readout_pool

        # Input drive from MNIST (784 dims)
        self.W_in = nn.Linear(28 * 28, neurons, bias=True)

        # Recurrent coupling on v
        self.W_rec = nn.Linear(neurons, neurons, bias=False)

        # Readout to 10 classes
        self.readout = nn.Linear(neurons, 10, bias=True)

        # FHN parameters (shared scalars). You can also make them learnable per-neuron later.
        self.register_buffer("a", torch.tensor(float(a)))
        self.register_buffer("b", torch.tensor(float(b)))
        self.register_buffer("tau", torch.tensor(float(tau)))

        # Init recurrent weights: orthogonal then scaled
        nn.init.orthogonal_(self.W_rec.weight)
        self.W_rec.weight.data *= (g / math.sqrt(neurons))

        # Optionally freeze W_rec
        if not train_rec:
            for p in self.W_rec.parameters():
                p.requires_grad_(False)

        # Mild init for W_in
        nn.init.normal_(self.W_in.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.W_in.bias)

    def _bound_states(self, v, w):
        if self.bound <= 0:
            return v, w
        if self.bound_mode == "tanh":
            b = self.bound
            v = b * torch.tanh(v / b)
            w = b * torch.tanh(w / b)
        elif self.bound_mode == "clamp":
            b = self.bound
            v = v.clamp(-b, b)
            w = w.clamp(-b, b)
        return v, w

    def forward(self, x):
        # x: [B,1,28,28]
        B = x.shape[0]
        x = x.view(B, -1)  # [B, 784]

        # constant drive over time
        u = self.W_in(x)  # [B, N]

        # initial states
        v = torch.zeros(B, self.N, device=x.device, dtype=x.dtype)
        w = torch.zeros(B, self.N, device=x.device, dtype=x.dtype)

        inv_tau = 1.0 / self.tau

        if self.readout_pool == "last":
            for _ in range(self.T):
                I = u + self.W_rec(v)
                dv = v - (v * v * v) / 3.0 - w + I
                dw = (v + self.a - self.b * w) * inv_tau
                v = v + self.dt * dv
                w = w + self.dt * dw
                v, w = self._bound_states(v, w)
            logits = self.readout(v)
            return logits

        elif self.readout_pool == "mean":
            v_sum = 0.0
            for _ in range(self.T):
                I = u + self.W_rec(v)
                dv = v - (v * v * v) / 3.0 - w + I
                dw = (v + self.a - self.b * w) * inv_tau
                v = v + self.dt * dv
                w = w + self.dt * dw
                v, w = self._bound_states(v, w)
                v_sum = v_sum + v
            v_mean = v_sum / float(self.T)
            logits = self.readout(v_mean)
            return logits

        else:
            raise ValueError("readout_pool must be 'last' or 'mean'")


# --------------------------- Train / Eval ---------------------------

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    loss_sum = 0.0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        loss_sum += float(loss) * x.size(0)

        pred = logits.argmax(dim=-1)
        correct += int((pred == y).sum())
        total += int(x.size(0))

    return loss_sum / total, correct / total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--neurons", type=int, default=512)
    ap.add_argument("--T", type=int, default=16, help="Euler steps per image")
    ap.add_argument("--dt", type=float, default=0.1)
    ap.add_argument("--a", type=float, default=0.7)
    ap.add_argument("--b", type=float, default=0.8)
    ap.add_argument("--tau", type=float, default=12.5)
    ap.add_argument("--g", type=float, default=1.0, help="recurrent init scale")
    ap.add_argument("--bound", type=float, default=5.0)
    ap.add_argument("--bound_mode", type=str, default="tanh", choices=["tanh", "clamp"])
    ap.add_argument("--pool", type=str, default="mean", choices=["mean", "last"])
    ap.add_argument("--train_rec", type=int, default=1, help="0 freeze W_rec, 1 train W_rec")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--clip", type=float, default=1.0, help="grad norm clip (0 disables)")
    ap.add_argument("--amp", type=int, default=1)
    ap.add_argument("--compile", type=int, default=1)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)

    # ---- connectivity video args ----
    ap.add_argument("--conn_video", type=str, default="FHN_connectivity.mp4", help="Path to output video (.mp4 or .gif). Empty disables.")
    ap.add_argument("--conn_every", type=int, default=10, help="Log a frame every N optimizer steps (0 disables).")
    ap.add_argument("--conn_epoch", type=int, default=1, help="Also log a frame at each epoch end (0/1).")
    ap.add_argument("--conn_k", type=int, default=256, help="Size of submatrix to visualize (k x k).")
    ap.add_argument("--conn_fps", type=int, default=15, help="Video fps.")
    ap.add_argument("--conn_max_frames", type=int, default=2000, help="Safety cap on frames.")
    ap.add_argument("--conn_vclip_pct", type=float, default=99.5, help="Percentile for symmetric color clipping of |W|.")
    ap.add_argument("--conn_dpi", type=int, default=130, help="Frame DPI (higher = bigger frames).")

    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[info] device: {device}")

    # MNIST loaders
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    train_ds = datasets.MNIST(root="./data", train=True, download=True, transform=tfm)
    test_ds  = datasets.MNIST(root="./data", train=False, download=True, transform=tfm)

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=args.workers, pin_memory=True,
                              persistent_workers=args.workers > 0)
    test_loader  = DataLoader(test_ds, batch_size=args.batch, shuffle=False,
                              num_workers=args.workers, pin_memory=True,
                              persistent_workers=args.workers > 0)

    model = FHNClassifier(
        neurons=args.neurons,
        T=args.T,
        dt=args.dt,
        a=args.a,
        b=args.b,
        tau=args.tau,
        g=args.g,
        bound=args.bound,
        bound_mode=args.bound_mode,
        readout_pool=args.pool,
        train_rec=bool(args.train_rec),
    ).to(device)

    if args.compile and hasattr(torch, "compile"):
        try:
            model = torch.compile(model)
            print("[info] torch.compile enabled")
        except Exception as e:
            print(f"[warn] torch.compile failed: {e}")

    # Optimizer: only params that require grad
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)

    scaler = torch.cuda.amp.GradScaler(enabled=bool(args.amp) and device.type == "cuda")

    # Connectivity logger (optional)
    conn_logger = None
    if args.conn_video:
        conn_logger = ConnectivityVideoLogger(
            out_path=args.conn_video,
            fps=args.conn_fps,
            every_steps=args.conn_every,
            log_epoch_end=bool(args.conn_epoch),
            k=args.conn_k,
            seed=args.seed,
            max_frames=args.conn_max_frames,
            vclip_pct=args.conn_vclip_pct,
            dpi=args.conn_dpi,
        )
        print(f"[info] connectivity logging enabled: {conn_logger.out_path}")
        if not bool(args.train_rec):
            print("[warn] --train_rec=0 so W_rec is frozen; connectivity video will be (nearly) constant.")

    # Baseline eval
    te_loss, te_acc = evaluate(model, test_loader, device)
    print(f"[eval] before train: loss={te_loss:.4f} acc={te_acc*100:.2f}%")

    global_step = 0

    try:
        for ep in range(1, args.epochs + 1):
            model.train()
            t0 = time.time()
            running = 0.0
            seen = 0

            for step_in_ep, (x, y) in enumerate(train_loader):
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

                opt.zero_grad(set_to_none=True)

                with torch.cuda.amp.autocast(
                    enabled=bool(args.amp) and device.type == "cuda",
                    dtype=torch.bfloat16
                ):
                    logits = model(x)
                    loss = F.cross_entropy(logits, y)

                scaler.scale(loss).backward()

                if args.clip and args.clip > 0:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(params, max_norm=args.clip)

                scaler.step(opt)
                scaler.update()

                running += float(loss) * x.size(0)
                seen += x.size(0)

                # log connectivity after the optimizer update (so you see the effect of this Adam step)
                if conn_logger is not None:
                    # handle torch.compile wrapping: compiled model still has .W_rec, but be safe
                    W = model.W_rec.weight if hasattr(model, "W_rec") else None
                    if W is not None:
                        conn_logger.maybe_log_step(
                            W=W,
                            epoch=ep,
                            step_in_epoch=step_in_ep,
                            global_step=global_step,
                            loss_val=float(loss),
                        )

                global_step += 1

            tr_loss = running / max(seen, 1)
            te_loss, te_acc = evaluate(model, test_loader, device)
            dt_ep = time.time() - t0
            print(f"[epoch {ep:02d}] train_loss={tr_loss:.4f} test_loss={te_loss:.4f} acc={te_acc*100:.2f}%  ({dt_ep:.1f}s)")

            if conn_logger is not None:
                W = model.W_rec.weight if hasattr(model, "W_rec") else None
                if W is not None:
                    conn_logger.log_epoch_end(
                        W=W,
                        epoch=ep,
                        global_step=global_step,
                        loss_val=te_loss,
                    )

    finally:
        if conn_logger is not None:
            conn_logger.close()
            print(f"[info] connectivity video written: {conn_logger.out_path}")

    print("[done]")


if __name__ == "__main__":
    main()
