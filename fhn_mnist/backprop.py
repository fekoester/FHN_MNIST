"""FHN network trained end to end with backpropagation through time."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .dynamics import FHNParams, simulate


class FHNClassifier(nn.Module):
    """
    x [B,1,28,28] -> u = W_in x -> T Euler steps of the FHN network with recurrent current W_rec v
    -> readout of the time-averaged (or last) v.

    train_rec=False freezes W_rec (reservoir-like); train_in=False also freezes W_in, so only the
    linear readout learns (a random FHN "reservoir").
    """

    def __init__(self, neurons=512, T=16, dt=0.1, a=0.7, b=0.8, tau=12.5, g=1.0, bound=5.0,
                 bound_mode="tanh", readout_pool="mean", train_rec=True, train_in=True,
                 n_in=28 * 28, n_out=10):
        super().__init__()
        self.N, self.T = int(neurons), int(T)
        self.readout_pool = readout_pool
        self.p = FHNParams(dt=dt, a=a, b=b, tau=tau, bound=bound, bound_mode=bound_mode)

        self.W_in = nn.Linear(n_in, neurons, bias=True)
        self.W_rec = nn.Linear(neurons, neurons, bias=False)
        self.readout = nn.Linear(neurons, n_out, bias=True)

        nn.init.orthogonal_(self.W_rec.weight)
        self.W_rec.weight.data *= g / math.sqrt(neurons)
        self.W_rec.weight.requires_grad_(bool(train_rec))

        nn.init.normal_(self.W_in.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.W_in.bias)
        if not train_in:
            for prm in self.W_in.parameters():
                prm.requires_grad_(False)

    def features(self, x, record: bool = False):
        u = self.W_in(x.reshape(x.shape[0], -1))
        return simulate(u, self.W_rec, self.p, self.T, self.readout_pool, record=record)

    def forward(self, x):
        feat, _ = self.features(x)
        return self.readout(feat)


class MLP(nn.Module):
    """Pixel baseline: hidden=0 gives multinomial logistic regression."""

    def __init__(self, hidden: int = 512, n_in: int = 28 * 28, n_out: int = 10):
        super().__init__()
        self.net = (nn.Linear(n_in, n_out) if hidden <= 0 else
                    nn.Sequential(nn.Linear(n_in, hidden), nn.ReLU(), nn.Linear(hidden, n_out)))

    def forward(self, x):
        return self.net(x.reshape(x.shape[0], -1))


@torch.no_grad()
def evaluate(model, loader, device):
    """Returns (mean CE, accuracy)."""
    model.eval()
    loss_sum, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x).float()
        loss_sum += F.cross_entropy(logits, y, reduction="sum").item()
        correct += (logits.argmax(-1) == y).sum().item()
        total += y.numel()
    return loss_sum / max(total, 1), correct / max(total, 1)
