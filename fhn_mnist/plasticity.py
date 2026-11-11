"""
Three-factor (biologically motivated) learning for a dense FHN network, without backprop.

Each image is one trial. While the network runs, every synapse accumulates an STDP-like eligibility
trace from a smooth spike proxy s = sigmoid((v - theta) / kappa):

    x_pre  <- a_pre  x_pre  + s          (presynaptic trace)
    y_post <- a_post y_post + s          (postsynaptic trace)
    E_rec  <- a_e E_rec + A+ s_post x_pre^T - A- y_post s_pre^T
    E_in   <- a_e E_in  + A+ s_post x_in_trace^T

At the end of the trial a third factor turns eligibility into a weight change, dW_ij = eta l_i E_ij:

  teacher    l = B_fb (onehot(y) - p)                (fixed random feedback of the output error)
  reinforce  l = M = r - baseline                    (global reward-prediction error), or, with
             rl_signal="feedback", l = M * B_fb (onehot(a) - p_T)   (reward-modulated feedback)

The readout learns with the exact gradient of its own forward pass (delta rule for teacher, REINFORCE
for reinforce) on the same features it predicts from (readout_on="v": time-averaged voltage, or
"rate": its spike proxy).

Because the third factor arrives once per trial, the final eligibility is a weighted sum over steps,
so the per-trial update is computed exactly with a few batched matrix products instead of updating
N x N trace matrices every step. With batch size 1 the updates equal the original per-sample code.
Larger batches sum (batch_reduce="sum") or average the per-sample updates.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .dynamics import FHNParams, fhn_step


@dataclass
class PlasticityCfg:
    # spike proxy
    spk_theta: float = 0.0
    spk_kappa: float = 0.5
    # traces / eligibility
    tau_pre: float = 8.0
    tau_post: float = 8.0
    tau_e: float = 30.0
    A_plus: float = 1.0
    A_minus: float = 0.8
    # learning rates
    eta_in: float = 2e-4
    eta_rec: float = 2e-4
    eta_out: float = 2e-3
    # safety
    w_decay: float = 0.0
    w_clip: float = 0.0
    max_update: float = 0.0        # cap on the Frobenius norm of each weight update (0 = off)
    # readout
    readout_on: str = "v"          # "v" | "rate"
    # teacher mode
    teacher_inject: float = 0.0    # label current during the learning (clamped) pass; error from a free pass
    # reinforce mode
    explore: str = "sample"        # "sample" | "argmax"
    temperature: float = 1.0
    baseline_ema: float = 0.99
    rl_signal: str = "global"      # "global" | "feedback"
    # batching
    batch_reduce: str = "sum"      # "sum" | "mean"


class FHNBioplasticNet(nn.Module):
    def __init__(self, neurons=512, T=16, g=1.0, cfg: PlasticityCfg | None = None,
                 params: FHNParams | None = None, train_rec=True, pool="mean", n_in=28 * 28,
                 n_out=10, device="cpu"):
        super().__init__()
        self.N, self.T, self.pool = int(neurons), int(T), pool
        self.cfg = cfg or PlasticityCfg()
        self.p = params or FHNParams()
        self.train_rec = bool(train_rec)
        self.n_out = n_out

        self.W_in = nn.Linear(n_in, neurons, bias=True)
        self.W_rec = nn.Linear(neurons, neurons, bias=False)
        self.W_out = nn.Linear(neurons, n_out, bias=True)
        nn.init.normal_(self.W_in.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.W_in.bias)
        nn.init.orthogonal_(self.W_rec.weight)
        self.W_rec.weight.data *= g / math.sqrt(neurons)
        nn.init.normal_(self.W_out.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.W_out.bias)
        for prm in self.parameters():
            prm.requires_grad_(False)

        self.register_buffer("B_fb", torch.randn(neurons, n_out) / math.sqrt(float(n_out)))
        self.register_buffer("reward_baseline", torch.tensor(0.0))
        self.to(device)

    # ------------------------------------------------------------------ dynamics
    def _spike(self, v):
        return torch.sigmoid((v - self.cfg.spk_theta) / self.cfg.spk_kappa)

    def _readout_features(self, v_feat):
        return v_feat if self.cfg.readout_on == "v" else self._spike(v_feat)

    def _run(self, u, record: bool):
        """Simulate; when record, also return the per-step spike proxy and pre/post traces [T,B,N]."""
        c = self.cfg
        a_pre, a_post = math.exp(-self.p.dt / c.tau_pre), math.exp(-self.p.dt / c.tau_post)
        B = u.shape[0]
        v = torch.zeros(B, self.N, device=u.device)
        w = torch.zeros_like(v)
        x_pre, y_post = torch.zeros_like(v), torch.zeros_like(v)
        v_sum = torch.zeros_like(v)
        S, XP, YP = [], [], []
        for _ in range(self.T):
            if record:
                s = self._spike(v)
                x_pre = a_pre * x_pre + s
                y_post = a_post * y_post + s
                S.append(s)
                XP.append(x_pre)
                YP.append(y_post)
            v, w = fhn_step(v, w, u + self.W_rec(v), self.p)
            v_sum += v
        v_feat = v_sum / float(self.T) if self.pool == "mean" else v
        rec = (torch.stack(S), torch.stack(XP), torch.stack(YP)) if record else None
        return v_feat, rec

    @torch.no_grad()
    def forward_logits(self, x_img, pool: str | None = None):
        if pool is not None and pool != self.pool:
            raise ValueError("pool is fixed at construction")
        x = x_img.reshape(x_img.shape[0], -1).float()
        v_feat, _ = self._run(self.W_in(x), record=False)
        return self.W_out(self._readout_features(v_feat))

    forward = forward_logits

    # ------------------------------------------------------------------ learning
    def _step_weights(self, B):
        """w_t = a_e^(T-1-t) (decay of eligibility to the trial end); c_t = sum of the input trace."""
        c = self.cfg
        a_e = math.exp(-self.p.dt / c.tau_e)
        a_in = math.exp(-self.p.dt / c.tau_pre)
        t = torch.arange(self.T, dtype=torch.float32)
        w_t = a_e ** (self.T - 1 - t)
        c_t = (1 - a_in ** (t + 1)) / (1 - a_in)  # x_in_trace_t = c_t * x for a constant image
        return w_t, c_t

    def _cap(self, dW):
        m = self.cfg.max_update
        if m > 0:
            n = dW.norm()
            if n > m:
                dW = dW * (m / n)
        return dW

    @torch.no_grad()
    def learn_batch(self, x_img, y, mode: str = "teacher"):
        """
        One learning step on a batch of trials. Returns a dict of metrics for the batch
        (ce and correct from the free-running prediction, and reward for reinforce).
        """
        c = self.cfg
        dev = x_img.device
        B = x_img.shape[0]
        x = x_img.reshape(B, -1).float()
        y = y.to(dev)
        y_oh = F.one_hot(y, self.n_out).float()

        u = self.W_in(x)
        if mode == "teacher" and c.teacher_inject != 0.0:
            # free pass for the prediction/error, clamped (label-driven) pass for the eligibility
            v_free, _ = self._run(u, record=False)
            u_learn = u + c.teacher_inject * (y_oh @ self.B_fb.T)
            _, (S, XP, YP) = self._run(u_learn, record=True)
        else:
            v_free, (S, XP, YP) = self._run(u, record=True)

        f = self._readout_features(v_free)
        logits = self.W_out(f)
        probs = F.softmax(logits, dim=-1)
        ce = F.cross_entropy(logits, y, reduction="sum").item()
        correct = (probs.argmax(-1) == y).sum().item()
        out = {"n": B, "ce_sum": ce, "correct": correct}

        if mode == "teacher":
            err = y_oh - probs                       # [B,C] = -dCE/dlogits
            L = err @ self.B_fb.T                    # [B,N] neuron-wise learning signal
            dW_out, db_out = err.T @ f, err.sum(0)
            db_in = L.sum(0)
        elif mode == "reinforce":
            Ttemp = max(c.temperature, 1e-6)
            pT = F.softmax(logits / Ttemp, dim=-1)
            if c.explore == "argmax":
                act = probs.argmax(-1)
            else:
                act = torch.multinomial(pT, 1).squeeze(1)
            r = (act == y).float()
            M = r - self.reward_baseline             # baseline from before this batch
            for rb in r.tolist():                    # same as per-sample EMA updates
                self.reward_baseline.mul_(c.baseline_ema).add_((1.0 - c.baseline_ema) * rb)
            pg = (F.one_hot(act, self.n_out).float() - pT) / Ttemp   # d log pT(a) / d logits
            mod = M.unsqueeze(1)
            dW_out, db_out = (mod * pg).T @ f, (mod * pg).sum(0)
            if c.rl_signal == "feedback":
                L = mod * (pg @ self.B_fb.T)
                db_in = L.sum(0)
            elif c.rl_signal == "global":
                L = mod.expand(B, self.N)
                db_in = (mod * self._spike(v_free)).sum(0)
            else:
                raise ValueError("rl_signal must be 'global' or 'feedback'")
            out["reward"] = r.sum().item()
        else:
            raise ValueError("mode must be 'teacher' or 'reinforce'")

        scale = 1.0 / B if c.batch_reduce == "mean" else 1.0
        w_t, c_t = self._step_weights(B)
        w_t, c_t = w_t.to(dev), c_t.to(dev)

        # output layer
        self.W_out.weight.add_(self._cap(c.eta_out * scale * dW_out))
        self.W_out.bias.add_(c.eta_out * scale * db_out)

        # input layer: E_in = A+ (sum_t w_t c_t s_t) x^T, so dW_in = eta A+ (L * s_bar)^T x
        if c.eta_in != 0.0:
            s_bar = torch.einsum("t,tbn->bn", w_t * c_t, S)
            dW_in = c.A_plus * (L * s_bar).T @ x
            self.W_in.weight.add_(self._cap(c.eta_in * scale * dW_in))
            self.W_in.bias.add_(c.eta_in * scale * db_in)

        # recurrent layer: dW = eta sum_{b,t} w_t [A+ (L s_t) x_pre_t^T - A- (L y_post_t) s_t^T]
        if self.train_rec and c.eta_rec != 0.0:
            Tn, _, N = S.shape
            wl = w_t.view(Tn, 1, 1) * L.unsqueeze(0)            # [T,B,N]
            P = (wl * S).reshape(-1, N)
            Q = (wl * YP).reshape(-1, N)
            dW_rec = c.A_plus * P.T @ XP.reshape(-1, N) - c.A_minus * Q.T @ S.reshape(-1, N)
            self.W_rec.weight.add_(self._cap(c.eta_rec * scale * dW_rec))

        self._post_update_safety()
        return out

    @torch.no_grad()
    def _post_update_safety(self):
        c = self.cfg
        mats = [self.W_in.weight, self.W_in.bias, self.W_out.weight, self.W_out.bias]
        if self.train_rec:
            mats.append(self.W_rec.weight)
        if c.w_decay > 0:
            for m in mats:
                m.mul_(max(0.0, 1.0 - c.w_decay))
        if c.w_clip > 0:
            for m in mats:
                m.clamp_(-c.w_clip, c.w_clip)


@torch.no_grad()
def evaluate_bio(model: FHNBioplasticNet, loader, device):
    loss_sum, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model.forward_logits(x)
        loss_sum += F.cross_entropy(logits, y, reduction="sum").item()
        correct += (logits.argmax(-1) == y).sum().item()
        total += y.numel()
    return loss_sum / max(total, 1), correct / max(total, 1)
