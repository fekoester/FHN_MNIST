"""
FitzHugh–Nagumo network dynamics shared by the backprop and the plasticity models.

Per neuron i, with input current I_i = u_i + sum_j W_ij v_j (explicit Euler, step dt):

    dv/dt = v - v^3/3 - w + I
    dw/dt = (v + a - b w) / tau

followed by an optional soft (tanh) or hard (clamp) bound on v and w.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class FHNParams:
    dt: float = 0.1
    a: float = 0.7
    b: float = 0.8
    tau: float = 12.5
    bound: float = 5.0
    bound_mode: str = "tanh"  # "tanh" | "clamp"


def bound_states(v: torch.Tensor, w: torch.Tensor, bound: float, mode: str):
    if bound <= 0:
        return v, w
    if mode == "tanh":
        return bound * torch.tanh(v / bound), bound * torch.tanh(w / bound)
    if mode == "clamp":
        return v.clamp(-bound, bound), w.clamp(-bound, bound)
    raise ValueError(f"bound_mode must be 'tanh' or 'clamp', got {mode!r}")


def fhn_step(v, w, current, p: FHNParams):
    """One Euler step. Returns the new (v, w)."""
    dv = v - (v * v * v) / 3.0 - w + current
    dw = (v + p.a - p.b * w) * (1.0 / p.tau)
    v = v + p.dt * dv
    w = w + p.dt * dw
    return bound_states(v, w, p.bound, p.bound_mode)


def simulate(u: torch.Tensor, rec, p: FHNParams, T: int, pool: str = "mean", record: bool = False):
    """
    Run T steps from v = w = 0 under constant drive u [B, N].

    rec: callable v -> recurrent current (e.g. an nn.Linear or lambda v: v @ W.T).
    pool: "mean" (average of v over the T steps) or "last".
    Returns (features [B, N], trajectory) where trajectory, when record=True, is a dict of
    tensors [T, B, N]: "v_pre" (v before each step), "v", "w" (after each step).
    """
    if pool not in ("mean", "last"):
        raise ValueError("pool must be 'mean' or 'last'")
    B, N = u.shape
    v = torch.zeros(B, N, device=u.device, dtype=u.dtype)
    w = torch.zeros_like(v)
    v_sum = torch.zeros_like(v) if pool == "mean" else None
    traj = {"v_pre": [], "v": [], "w": []} if record else None
    for _ in range(int(T)):
        if record:
            traj["v_pre"].append(v)
        v, w = fhn_step(v, w, u + rec(v), p)
        if v_sum is not None:
            v_sum = v_sum + v
        if record:
            traj["v"].append(v)
            traj["w"].append(w)
    feat = v_sum / float(T) if pool == "mean" else v
    if record:
        traj = {k: torch.stack(x) for k, x in traj.items()}
    return feat, traj
