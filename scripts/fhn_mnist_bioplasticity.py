#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fhn_mnist_bioplasticity.py

Biology-ish learning for a dense FitzHugh–Nagumo (FHN) network on MNIST, on GPU.

Implements TWO learning modes (no Adam/backprop):

1) Reinforcement (dopamine-like 3-factor learning)
   - run network on an image (a "trial")
   - choose an action from softmax(logits) (sample or argmax)
   - reward r = 1 if correct else 0
   - global modulatory signal M = r - baseline
   - synapses updated by: ΔW ∝ M * eligibility
   - output layer uses REINFORCE-style policy gradient (still 3-factor: pre * post * reward)

2) Teacher-signal (broadcast error / e-prop-like 3-factor learning)
   - compute err = onehot(label) - softmax(logits)
   - compute neuron-wise learning signal l = B_fb @ err  (fixed random feedback matrix)
   - synapses updated by: ΔW_ij ∝ l_i * eligibility_ij
   - output layer updated by local delta rule with err

Connectivity logging:
  - Saves frames of a kxk submatrix of W_rec.weight during training into MP4 or GIF.

IMPORTANT PRACTICAL NOTE (dense plasticity is expensive):
  - Dense W_rec plasticity is O(N^2) per simulation step.
  - To keep it feasible, defaults are small (neurons=512, T=16, batch_io=32 but updates happen per-sample).
  - The code processes each sample in the batch sequentially to keep sample-specific reward/teacher signals correct.

Requires:
  pip install torch torchvision matplotlib imageio

MP4 requires ffmpeg available.

Examples:
  # Reinforcement learning (slow, exploratory)
  python fhn_mnist_bioplasticity.py --mode reinforce --neurons 512 --T 16 --epochs 5 --train_rec 1

  # Teacher-signal learning (much faster learning)
  python fhn_mnist_bioplasticity.py --mode teacher --neurons 512 --T 16 --epochs 5 --train_rec 1

  # Log connectivity evolution
  python fhn_mnist_bioplasticity.py --mode teacher --train_rec 1 --conn_video conn.mp4 --conn_every 50 --conn_k 256
"""

import argparse, math, time, os
from dataclasses import dataclass
from typing import Optional, Tuple

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
        self._idx = None  # chosen neuron indices (torch.LongTensor on device)

        self._ensure_dir()

    def _ensure_dir(self):
        d = os.path.dirname(os.path.abspath(self.out_path))
        if d and not os.path.exists(d):
            os.makedirs(d, exist_ok=True)

    def _lazy_init_idx(self, W: torch.Tensor):
        N = W.shape[0]
        k = min(self.k, N)
        g = torch.Generator(device="cpu")
        g.manual_seed(self.seed)
        perm = torch.randperm(N, generator=g)[:k]
        self._idx = perm.to(device=W.device)

    def _lazy_open_writer(self):
        if self._writer is not None:
            return
        import imageio.v2 as imageio
        ext = os.path.splitext(self.out_path)[1].lower()
        try:
            if ext == ".mp4":
                self._writer = imageio.get_writer(
                    self.out_path,
                    fps=self.fps,
                    codec="libx264",
                    quality=8,
                    macro_block_size=None,
                )
                self._writer_kind = "mp4"
            elif ext == ".gif":
                self._writer = imageio.get_writer(self.out_path, fps=self.fps)
                self._writer_kind = "gif"
            else:
                gif_path = self.out_path + ".gif"
                self._writer = imageio.get_writer(gif_path, fps=self.fps)
                self._writer_kind = "gif"
                self.out_path = gif_path
        except Exception as e:
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
        import numpy as np
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        A = mat_cpu.numpy()
        absA = np.abs(A).reshape(-1)
        if absA.size == 0:
            vmax = 1.0
        else:
            vmax = np.percentile(absA, vclip_pct) if vclip_pct > 0 else absA.max()
            vmax = float(max(vmax, 1e-8))
        vmin = -vmax

        fig = plt.figure(figsize=(5.2, 5.0), dpi=dpi)
        ax = fig.add_subplot(111)
        im = ax.imshow(A, vmin=vmin, vmax=vmax, interpolation="nearest")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("neuron j")
        ax.set_ylabel("neuron i")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        fig.canvas.draw()
        try:
            rgba = np.asarray(fig.canvas.buffer_rgba())
            img = rgba[:, :, :3].copy()
        except Exception:
            w, h = fig.canvas.get_width_height()
            argb = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8).reshape(h, w, 4)
            img = argb[:, :, [1, 2, 3]].copy()

        plt.close(fig)
        return img

    def maybe_log_step(self, W: torch.Tensor, epoch: int, step_in_epoch: int, global_step: int, metric: float, tag: str = "step"):
        if self.every_steps <= 0:
            return
        if (global_step % self.every_steps) != 0:
            return
        self._log(W, epoch, step_in_epoch, global_step, metric, tag=tag)

    def log_epoch_end(self, W: torch.Tensor, epoch: int, global_step: int, metric: float):
        if not self.log_epoch_end_enabled:
            return
        self._log(W, epoch, -1, global_step, metric, tag="epoch_end")

    def _log(self, W: torch.Tensor, epoch: int, step_in_epoch: int, global_step: int, metric: float, tag: str):
        if not self.out_path:
            return
        if self._frames >= self.max_frames:
            return
        if self._idx is None:
            self._lazy_init_idx(W)

        idx = self._idx
        sub = W.index_select(0, idx).index_select(1, idx).detach().float().cpu()

        mean = float(sub.mean())
        std = float(sub.std(unbiased=False))
        mx = float(sub.abs().max())

        title = (
            f"W_rec submatrix ({tag}) | ep={epoch} step={step_in_epoch if step_in_epoch>=0 else 'end'} "
            f"gstep={global_step} | metric={metric:.4f} | mean={mean:.3e} std={std:.3e} max|w|={mx:.3e}"
        )
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


# --------------------------- FHN core + plasticity ---------------------------

@dataclass
class PlasticityCfg:
    # FHN dynamics
    dt: float
    a: float
    b: float
    tau: float
    bound: float
    bound_mode: str

    # spike proxy
    spk_theta: float
    spk_kappa: float

    # traces / eligibility
    tau_pre: float
    tau_post: float
    tau_e: float
    A_plus: float
    A_minus: float

    # learning rates
    eta_in: float
    eta_rec: float
    eta_out: float

    # weight regularization / safety
    w_decay: float       # multiplicative decay per update step (approx)
    w_clip: float        # clip weights to [-w_clip, w_clip], 0 disables

    # teacher mode
    teacher_inject: float  # additional current amplitude from label (0 disables)

    # reinforcement mode
    explore: str         # "sample" or "argmax"
    temperature: float   # sampling temperature
    baseline_ema: float  # reward baseline EMA


class FHNBioplasticNet(nn.Module):
    """
    Dense FHN network with:
      - W_in: 784 -> N
      - W_rec: N -> N  (dense)
      - W_out: N -> 10

    Simulation returns pooled activity for logits.
    Plasticity uses:
      - eligibility traces for W_in and W_rec (STDP-like, continuous proxy)
      - output updates:
          * reinforce: REINFORCE (policy gradient) with reward-baseline
          * teacher: delta rule with err = target - prob
    """
    def __init__(self, neurons: int, T: int, g: float, cfg: PlasticityCfg, train_rec: bool, device: torch.device):
        super().__init__()
        self.N = neurons
        self.T = T
        self.cfg = cfg

        self.W_in = nn.Linear(28 * 28, neurons, bias=True)
        self.W_rec = nn.Linear(neurons, neurons, bias=False)
        self.W_out = nn.Linear(neurons, 10, bias=True)

        # init
        nn.init.normal_(self.W_in.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.W_in.bias)

        nn.init.orthogonal_(self.W_rec.weight)
        self.W_rec.weight.data *= (g / math.sqrt(neurons))

        nn.init.normal_(self.W_out.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.W_out.bias)

        # no autograd; we update manually
        for p in self.parameters():
            p.requires_grad_(False)

        if not train_rec:
            self.W_rec.weight.requires_grad_(False)

        # teacher feedback matrix B_fb: N x 10 (fixed random)
        B = torch.randn(neurons, 10, device=device, dtype=torch.float32) / math.sqrt(10.0)
        self.register_buffer("B_fb", B)

        # reward baseline (reinforcement)
        self.register_buffer("reward_baseline", torch.tensor(0.0, device=device))

    def _bound_states(self, v, w):
        bnd = self.cfg.bound
        if bnd <= 0:
            return v, w
        if self.cfg.bound_mode == "tanh":
            v = bnd * torch.tanh(v / bnd)
            w = bnd * torch.tanh(w / bnd)
        else:
            v = v.clamp(-bnd, bnd)
            w = w.clamp(-bnd, bnd)
        return v, w

    def _spike_proxy(self, v):
        # smooth "spike-ish" activity in [0,1]
        return torch.sigmoid((v - self.cfg.spk_theta) / self.cfg.spk_kappa)

    @torch.no_grad()
    def forward_logits(self, x_img: torch.Tensor, pool: str = "mean") -> torch.Tensor:
        """
        Forward (no learning), possibly batched.
        x_img: [B,1,28,28]
        returns logits: [B,10]
        """
        B = x_img.shape[0]
        x = x_img.view(B, -1).float()
        u = self.W_in(x)

        v = torch.zeros(B, self.N, device=x.device)
        w = torch.zeros(B, self.N, device=x.device)

        inv_tau = 1.0 / float(self.cfg.tau)

        if pool == "last":
            for _ in range(self.T):
                I = u + self.W_rec(v)
                dv = v - (v * v * v) / 3.0 - w + I
                dw = (v + self.cfg.a - self.cfg.b * w) * inv_tau
                v = v + self.cfg.dt * dv
                w = w + self.cfg.dt * dw
                v, w = self._bound_states(v, w)
            return self.W_out(v)

        if pool == "mean":
            v_sum = torch.zeros_like(v)
            for _ in range(self.T):
                I = u + self.W_rec(v)
                dv = v - (v * v * v) / 3.0 - w + I
                dw = (v + self.cfg.a - self.cfg.b * w) * inv_tau
                v = v + self.cfg.dt * dv
                w = w + self.cfg.dt * dw
                v, w = self._bound_states(v, w)
                v_sum += v
            v_mean = v_sum / float(self.T)
            return self.W_out(v_mean)

        raise ValueError("pool must be 'mean' or 'last'")

    # --------- Learning: run ONE trial (one image) and update weights ---------

    @torch.no_grad()
    def learn_one_reinforce(self, x_img_1: torch.Tensor, y_1: torch.Tensor, pool: str, train_rec: bool) -> Tuple[float, int, int]:
        """
        One sample trial. Returns (ce_loss, correct, reward).
        """
        device = x_img_1.device
        x = x_img_1.view(1, -1).float()  # [1,784]
        y = int(y_1.item())

        # drive
        u = self.W_in(x)  # [1,N]

        # states
        v = torch.zeros(1, self.N, device=device)
        w = torch.zeros(1, self.N, device=device)

        # traces (per neuron)
        alpha_pre  = math.exp(-self.cfg.dt / self.cfg.tau_pre)
        alpha_post = math.exp(-self.cfg.dt / self.cfg.tau_post)
        alpha_e    = math.exp(-self.cfg.dt / self.cfg.tau_e)

        x_pre = torch.zeros(1, self.N, device=device)
        y_post = torch.zeros(1, self.N, device=device)

        # eligibility matrices
        E_rec = torch.zeros(self.N, self.N, device=device)
        E_in  = torch.zeros(self.N, 28 * 28, device=device)

        inv_tau = 1.0 / float(self.cfg.tau)

        # input trace (simple low-pass of image)
        x_in = x  # [1,784]
        x_in_trace = torch.zeros_like(x_in)
        alpha_in = alpha_pre

        # accumulate pooled activity for output update
        if pool == "mean":
            v_sum = torch.zeros_like(v)

        for _ in range(self.T):
            # spike proxy from current v
            s = self._spike_proxy(v)   # [1,N]

            # update traces
            x_pre  = alpha_pre  * x_pre  + s
            y_post = alpha_post * y_post + s

            x_in_trace = alpha_in * x_in_trace + x_in  # [1,784]

            # eligibility updates (STDP-ish)
            # E_rec <- alpha_e E_rec + A+ (s_post^T x_pre) - A- (y_post^T s_pre)
            # with batch=1, outer products are straightforward.
            if train_rec:
                E_rec.mul_(alpha_e)
                E_rec.add_(self.cfg.A_plus  * (s[0].unsqueeze(1) @ x_pre[0].unsqueeze(0)))
                E_rec.add_(-self.cfg.A_minus * (y_post[0].unsqueeze(1) @ s[0].unsqueeze(0)))

            # input eligibility: E_in <- alpha_e E_in + (s_post^T x_in_trace)
            E_in.mul_(alpha_e)
            E_in.add_(self.cfg.A_plus * (s[0].unsqueeze(1) @ x_in_trace[0].unsqueeze(0)))

            # FHN step
            I = u + self.W_rec(v)
            dv = v - (v * v * v) / 3.0 - w + I
            dw = (v + self.cfg.a - self.cfg.b * w) * inv_tau
            v = v + self.cfg.dt * dv
            w = w + self.cfg.dt * dw
            v, w = self._bound_states(v, w)

            if pool == "mean":
                v_sum += v

        v_feat = (v_sum / float(self.T)) if pool == "mean" else v  # [1,N]
        logits = self.W_out(v_feat)  # [1,10]
        probs = F.softmax(logits, dim=-1).squeeze(0)  # [10]

        # monitoring CE (not used for learning)
        ce = float(F.cross_entropy(logits, torch.tensor([y], device=device)))

        # choose action
        if self.cfg.explore == "argmax":
            a = int(torch.argmax(probs).item())
        else:
            # sample with temperature
            Ttemp = max(self.cfg.temperature, 1e-6)
            pT = F.softmax(torch.log(probs + 1e-12) / Ttemp, dim=-1)
            a = int(torch.multinomial(pT, num_samples=1).item())

        reward = 1 if (a == y) else 0
        correct = 1 if (torch.argmax(probs).item() == y) else 0  # greedy accuracy metric

        # dopamine-like modulatory signal
        baseline = float(self.reward_baseline.item())
        M = float(reward - baseline)

        # update baseline
        self.reward_baseline.mul_(self.cfg.baseline_ema).add_((1.0 - self.cfg.baseline_ema) * reward)

        # output update (REINFORCE): ΔW_out ∝ M * (onehot(a) - probs) ⊗ pre
        onehot_a = torch.zeros(10, device=device)
        onehot_a[a] = 1.0
        pg = (onehot_a - probs)  # [10]
        pre = self._spike_proxy(v_feat).squeeze(0)  # [N] as "pre" for output synapses

        self.W_out.weight.add_(self.cfg.eta_out * M * (pg.unsqueeze(1) @ pre.unsqueeze(0)))
        self.W_out.bias.add_(self.cfg.eta_out * M * pg)

        # input/recurrent updates via eligibility
        if self.cfg.eta_in != 0.0:
            self.W_in.weight.add_(self.cfg.eta_in * M * E_in)
            # (optional) bias on W_in: use post activity as proxy
            self.W_in.bias.add_(self.cfg.eta_in * M * pre)

        if train_rec and self.cfg.eta_rec != 0.0:
            self.W_rec.weight.add_(self.cfg.eta_rec * M * E_rec)

        # weight decay / clip (safety)
        self._post_update_safety(train_rec=train_rec)

        return ce, correct, reward

    @torch.no_grad()
    def learn_one_teacher(self, x_img_1: torch.Tensor, y_1: torch.Tensor, pool: str, train_rec: bool) -> Tuple[float, int]:
        """
        One sample trial with teacher/broadcast error.
        Returns (ce_loss, correct).
        """
        device = x_img_1.device
        x = x_img_1.view(1, -1).float()  # [1,784]
        y = int(y_1.item())

        # drive from image
        u = self.W_in(x)  # [1,N]

        # optional teacher injection current (label -> neurons)
        if self.cfg.teacher_inject != 0.0:
            y_oh = torch.zeros(10, device=device)
            y_oh[y] = 1.0
            u = u + self.cfg.teacher_inject * (y_oh @ self.B_fb.T).unsqueeze(0)  # [1,N]

        v = torch.zeros(1, self.N, device=device)
        w = torch.zeros(1, self.N, device=device)

        alpha_pre  = math.exp(-self.cfg.dt / self.cfg.tau_pre)
        alpha_post = math.exp(-self.cfg.dt / self.cfg.tau_post)
        alpha_e    = math.exp(-self.cfg.dt / self.cfg.tau_e)

        x_pre = torch.zeros(1, self.N, device=device)
        y_post = torch.zeros(1, self.N, device=device)

        E_rec = torch.zeros(self.N, self.N, device=device)
        E_in  = torch.zeros(self.N, 28 * 28, device=device)

        inv_tau = 1.0 / float(self.cfg.tau)

        x_in = x
        x_in_trace = torch.zeros_like(x_in)
        alpha_in = alpha_pre

        if pool == "mean":
            v_sum = torch.zeros_like(v)

        for _ in range(self.T):
            s = self._spike_proxy(v)  # [1,N]

            x_pre  = alpha_pre  * x_pre  + s
            y_post = alpha_post * y_post + s
            x_in_trace = alpha_in * x_in_trace + x_in

            if train_rec:
                E_rec.mul_(alpha_e)
                E_rec.add_(self.cfg.A_plus  * (s[0].unsqueeze(1) @ x_pre[0].unsqueeze(0)))
                E_rec.add_(-self.cfg.A_minus * (y_post[0].unsqueeze(1) @ s[0].unsqueeze(0)))

            E_in.mul_(alpha_e)
            E_in.add_(self.cfg.A_plus * (s[0].unsqueeze(1) @ x_in_trace[0].unsqueeze(0)))

            I = u + self.W_rec(v)
            dv = v - (v * v * v) / 3.0 - w + I
            dw = (v + self.cfg.a - self.cfg.b * w) * inv_tau
            v = v + self.cfg.dt * dv
            w = w + self.cfg.dt * dw
            v, w = self._bound_states(v, w)

            if pool == "mean":
                v_sum += v

        v_feat = (v_sum / float(self.T)) if pool == "mean" else v  # [1,N]
        logits = self.W_out(v_feat)  # [1,10]
        probs = F.softmax(logits, dim=-1).squeeze(0)  # [10]
        pred = int(torch.argmax(probs).item())
        correct = 1 if pred == y else 0
        ce = float(F.cross_entropy(logits, torch.tensor([y], device=device)))

        # teacher error
        y_oh = torch.zeros(10, device=device)
        y_oh[y] = 1.0
        err = (y_oh - probs)  # [10]

        # neuron-wise learning signal (broadcast error)
        l = (self.B_fb @ err)  # [N]

        # output update (delta rule for softmax readout):
        pre = self._spike_proxy(v_feat).squeeze(0)  # [N]
        self.W_out.weight.add_(self.cfg.eta_out * (err.unsqueeze(1) @ pre.unsqueeze(0)))
        self.W_out.bias.add_(self.cfg.eta_out * err)

        # hidden updates: ΔW_ij ∝ l_i * eligibility_ij  (3-factor)
        if self.cfg.eta_in != 0.0:
            self.W_in.weight.add_(self.cfg.eta_in * (l.unsqueeze(1) * E_in))
            self.W_in.bias.add_(self.cfg.eta_in * l)

        if train_rec and self.cfg.eta_rec != 0.0:
            self.W_rec.weight.add_(self.cfg.eta_rec * (l.unsqueeze(1) * E_rec))

        self._post_update_safety(train_rec=train_rec)
        return ce, correct

    @torch.no_grad()
    def _post_update_safety(self, train_rec: bool):
        # multiplicative decay ~ (1 - w_decay)
        if self.cfg.w_decay > 0:
            decay = max(0.0, 1.0 - self.cfg.w_decay)
            self.W_in.weight.mul_(decay)
            self.W_in.bias.mul_(decay)
            self.W_out.weight.mul_(decay)
            self.W_out.bias.mul_(decay)
            if train_rec:
                self.W_rec.weight.mul_(decay)

        # clip weights
        if self.cfg.w_clip > 0:
            c = self.cfg.w_clip
            self.W_in.weight.clamp_(-c, c)
            self.W_in.bias.clamp_(-c, c)
            self.W_out.weight.clamp_(-c, c)
            self.W_out.bias.clamp_(-c, c)
            if train_rec:
                self.W_rec.weight.clamp_(-c, c)


# --------------------------- Eval ---------------------------

@torch.no_grad()
def evaluate_ce_acc(model: FHNBioplasticNet, loader: DataLoader, device: torch.device, pool: str) -> Tuple[float, float]:
    model.eval()
    total = 0
    correct = 0
    loss_sum = 0.0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model.forward_logits(x, pool=pool)
        loss = F.cross_entropy(logits, y)
        loss_sum += float(loss) * x.size(0)
        pred = torch.argmax(logits, dim=-1)
        correct += int((pred == y).sum())
        total += int(x.size(0))

    return loss_sum / max(total, 1), correct / max(total, 1)


# --------------------------- Main ---------------------------

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--mode", type=str, default="teacher", choices=["teacher", "reinforce"],
                    help="teacher: broadcast-error 3-factor; reinforce: reward-modulated 3-factor")

    # model size / sim
    ap.add_argument("--neurons", type=int, default=512)
    ap.add_argument("--T", type=int, default=16)
    ap.add_argument("--dt", type=float, default=0.10)
    ap.add_argument("--a", type=float, default=0.7)
    ap.add_argument("--b", type=float, default=0.8)
    ap.add_argument("--tau", type=float, default=12.5)
    ap.add_argument("--g", type=float, default=1.0, help="recurrent init scale (orthogonal * g/sqrt(N))")
    ap.add_argument("--pool", type=str, default="mean", choices=["mean", "last"])

    # spike proxy
    ap.add_argument("--spk_theta", type=float, default=0.0)
    ap.add_argument("--spk_kappa", type=float, default=0.5)

    # traces / eligibility
    ap.add_argument("--tau_pre", type=float, default=8.0)
    ap.add_argument("--tau_post", type=float, default=8.0)
    ap.add_argument("--tau_e", type=float, default=30.0)
    ap.add_argument("--A_plus", type=float, default=1.0)
    ap.add_argument("--A_minus", type=float, default=0.8)

    # learning rates
    ap.add_argument("--eta_in", type=float, default=2e-4)
    ap.add_argument("--eta_rec", type=float, default=2e-4)
    ap.add_argument("--eta_out", type=float, default=2e-3)

    # what is plastic?
    ap.add_argument("--train_rec", type=int, default=1, help="0 freeze W_rec; 1 allow W_rec plasticity")

    # safety
    ap.add_argument("--bound", type=float, default=5.0)
    ap.add_argument("--bound_mode", type=str, default="tanh", choices=["tanh", "clamp"])
    ap.add_argument("--w_decay", type=float, default=0.0, help="per-update multiplicative weight decay (small, e.g. 1e-4)")
    ap.add_argument("--w_clip", type=float, default=0.0, help="clip weights to [-w_clip, w_clip], 0 disables")

    # reinforcement specifics
    ap.add_argument("--explore", type=str, default="sample", choices=["sample", "argmax"])
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--baseline_ema", type=float, default=0.99)

    # teacher specifics
    ap.add_argument("--teacher_inject", type=float, default=0.0, help="inject label->neurons current during sim (0 disables)")

    # training
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch_io", type=int, default=32, help="DataLoader batch size; learning still runs per-sample sequentially")
    ap.add_argument("--eval_batch", type=int, default=256)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)

    # connectivity video
    ap.add_argument("--conn_video", type=str, default="", help="Path to output video (.mp4 or .gif). Empty disables.")
    ap.add_argument("--conn_every", type=int, default=200, help="Log a frame every N update-steps (0 disables).")
    ap.add_argument("--conn_epoch", type=int, default=1, help="Also log a frame at each epoch end (0/1).")
    ap.add_argument("--conn_k", type=int, default=256, help="Size of submatrix to visualize (k x k).")
    ap.add_argument("--conn_fps", type=int, default=15)
    ap.add_argument("--conn_max_frames", type=int, default=2000)
    ap.add_argument("--conn_vclip_pct", type=float, default=99.5)
    ap.add_argument("--conn_dpi", type=int, default=130)

    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[info] device: {device}")

    # MNIST
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train_ds = datasets.MNIST(root="./data", train=True, download=True, transform=tfm)
    test_ds  = datasets.MNIST(root="./data", train=False, download=True, transform=tfm)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_io, shuffle=True,
        num_workers=args.workers, pin_memory=True, persistent_workers=args.workers > 0
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.eval_batch, shuffle=False,
        num_workers=args.workers, pin_memory=True, persistent_workers=args.workers > 0
    )

    cfg = PlasticityCfg(
        dt=args.dt, a=args.a, b=args.b, tau=args.tau,
        bound=args.bound, bound_mode=args.bound_mode,
        spk_theta=args.spk_theta, spk_kappa=args.spk_kappa,
        tau_pre=args.tau_pre, tau_post=args.tau_post, tau_e=args.tau_e,
        A_plus=args.A_plus, A_minus=args.A_minus,
        eta_in=args.eta_in, eta_rec=args.eta_rec, eta_out=args.eta_out,
        w_decay=args.w_decay, w_clip=args.w_clip,
        teacher_inject=args.teacher_inject,
        explore=args.explore, temperature=args.temperature, baseline_ema=args.baseline_ema,
    )

    model = FHNBioplasticNet(
        neurons=args.neurons, T=args.T, g=args.g,
        cfg=cfg, train_rec=bool(args.train_rec), device=device
    ).to(device)

    # Connectivity logger
    conn_logger: Optional[ConnectivityVideoLogger] = None
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
    te_loss, te_acc = evaluate_ce_acc(model, test_loader, device, pool=args.pool)
    print(f"[eval] before train: loss={te_loss:.4f} acc={te_acc*100:.2f}%")

    global_step = 0  # counts per-sample updates

    try:
        for ep in range(1, args.epochs + 1):
            model.train()
            t0 = time.time()

            # metrics
            ce_sum = 0.0
            seen = 0
            correct_sum = 0
            reward_sum = 0

            for step_in_ep, (xb, yb) in enumerate(train_loader):
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)

                # IMPORTANT: per-sample loop to preserve sample-specific reward/teacher signals
                B = xb.size(0)
                for i in range(B):
                    x1 = xb[i:i+1]
                    y1 = yb[i:i+1]

                    if args.mode == "reinforce":
                        ce, correct, reward = model.learn_one_reinforce(
                            x_img_1=x1, y_1=y1, pool=args.pool, train_rec=bool(args.train_rec)
                        )
                        reward_sum += reward
                    else:
                        ce, correct = model.learn_one_teacher(
                            x_img_1=x1, y_1=y1, pool=args.pool, train_rec=bool(args.train_rec)
                        )

                    ce_sum += ce
                    correct_sum += correct
                    seen += 1

                    # connectivity logging after update
                    if conn_logger is not None and hasattr(model, "W_rec"):
                        # use CE as "metric" in teacher mode; use running CE in reinforce as well
                        conn_logger.maybe_log_step(
                            W=model.W_rec.weight,
                            epoch=ep,
                            step_in_epoch=step_in_ep,
                            global_step=global_step,
                            metric=ce,
                            tag=args.mode,
                        )

                    global_step += 1

            tr_ce = ce_sum / max(seen, 1)
            tr_acc = correct_sum / max(seen, 1)
            tr_rew = reward_sum / max(seen, 1) if args.mode == "reinforce" else float("nan")

            te_loss, te_acc = evaluate_ce_acc(model, test_loader, device, pool=args.pool)
            dt_ep = time.time() - t0

            if args.mode == "reinforce":
                rb = float(model.reward_baseline.item())
                print(
                    f"[epoch {ep:02d}] mode=reinforce "
                    f"train_CE={tr_ce:.4f} train_acc(greedy)={tr_acc*100:.2f}% train_reward={tr_rew*100:.2f}% "
                    f"baseline={rb:.3f} | test_CE={te_loss:.4f} test_acc={te_acc*100:.2f}% ({dt_ep:.1f}s)"
                )
            else:
                print(
                    f"[epoch {ep:02d}] mode=teacher "
                    f"train_CE={tr_ce:.4f} train_acc={tr_acc*100:.2f}% | test_CE={te_loss:.4f} test_acc={te_acc*100:.2f}% ({dt_ep:.1f}s)"
                )

            if conn_logger is not None and hasattr(model, "W_rec"):
                conn_logger.log_epoch_end(
                    W=model.W_rec.weight,
                    epoch=ep,
                    global_step=global_step,
                    metric=te_loss,
                )

    finally:
        if conn_logger is not None:
            conn_logger.close()
            print(f"[info] connectivity video written: {conn_logger.out_path}")

    print("[done]")


if __name__ == "__main__":
    main()
