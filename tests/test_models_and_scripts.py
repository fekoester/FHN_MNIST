import json

import fhn_mnist_bioplasticity as bio
import fhn_mnist_train as bp
import pytest
import torch

from fhn_mnist.backprop import FHNClassifier
from fhn_mnist.dynamics import FHNParams, simulate


def test_simulate_matches_explicit_loop():
    torch.manual_seed(0)
    u = torch.randn(4, 16)
    Wr = torch.randn(16, 16) * 0.1
    p = FHNParams()
    feat, traj = simulate(u, lambda v: v @ Wr.T, p, T=5, pool="mean", record=True)
    v = w = torch.zeros(4, 16)
    acc = 0
    for _ in range(5):
        dv = v - v**3 / 3 - w + u + v @ Wr.T
        dw = (v + p.a - p.b * w) / p.tau
        v, w = v + p.dt * dv, w + p.dt * dw
        v, w = 5 * torch.tanh(v / 5), 5 * torch.tanh(w / 5)
        acc = acc + v
    torch.testing.assert_close(feat, acc / 5)
    assert traj["v"].shape == (5, 4, 16)


def test_frozen_flags():
    m = FHNClassifier(neurons=32, T=3, train_rec=False, train_in=False)
    trainable = {n for n, p in m.named_parameters() if p.requires_grad}
    assert trainable == {"readout.weight", "readout.bias"}
    assert m(torch.randn(2, 1, 28, 28)).shape == (2, 10)


@pytest.mark.parametrize("mode", ["teacher", "reinforce"])
def test_bio_cli_end_to_end(tmp_path, mode):
    args = bio.build_parser().parse_args(["--dataset", "fake", "--val_size", "64", "--neurons", "32", "--T", "4",
                                          "--epochs", "2", "--workers", "0", "--mode", mode,
                                          "--out_dir", str(tmp_path), "--conn_video", str(tmp_path / "c.gif"),
                                          "--conn_every", "10", "--conn_k", "16"])
    out, _ = bio.train(args, verbose=False)
    assert 0 <= out["test_acc"] <= 1 and len(out["history"]) == 2
    assert json.load(open(tmp_path / "metrics.json"))["mode"] == mode
    assert (tmp_path / "c.gif").exists()


def test_bp_cli_end_to_end(tmp_path):
    args = bp.build_parser().parse_args(["--dataset", "fake", "--val_size", "64", "--neurons", "32", "--T", "4",
                                         "--epochs", "2", "--workers", "0", "--out_dir", str(tmp_path)])
    out, _ = bp.train(args, verbose=False)
    assert (tmp_path / "best.pt").exists() and out["best_epoch"] in (1, 2)
    ck = torch.load(tmp_path / "best.pt", weights_only=False)
    m = bp.make_model(bp.build_parser().parse_args([]).__class__(**ck["config"]))
    m.load_state_dict(ck["model"])
