import importlib.util
import pathlib
import subprocess
import sys

import pytest
import torch
import torch.nn.functional as F

from fhn_mnist.dynamics import FHNParams
from fhn_mnist.plasticity import FHNBioplasticNet, PlasticityCfg

ROOT = pathlib.Path(__file__).resolve().parents[1]
KW = dict(spk_theta=0.1, spk_kappa=0.5, tau_pre=8, tau_post=6, tau_e=30, A_plus=1, A_minus=0.8, eta_in=1e-2,
          eta_rec=1e-2, eta_out=0.0, w_decay=1e-3, w_clip=0, teacher_inject=0, explore="argmax",
          temperature=1, baseline_ema=0.9)


def _original():
    """scripts/fhn_mnist_bioplasticity.py from the first commit (per-sample reference implementation)."""
    try:
        first = subprocess.run(["git", "rev-list", "--max-parents=0", "HEAD"], cwd=ROOT, capture_output=True,
                               text=True, check=True).stdout.split()[0]
        code = subprocess.run(["git", "show", f"{first}:scripts/fhn_mnist_bioplasticity.py"], cwd=ROOT,
                              capture_output=True, text=True, check=True).stdout
    except (subprocess.CalledProcessError, FileNotFoundError, IndexError):
        return None
    path = pathlib.Path(__file__).parent / "_orig_bio_tmp.py"
    path.write_text(code)
    try:
        spec = importlib.util.spec_from_file_location("_orig_bio_tmp", path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
    finally:
        path.unlink()
    return mod


@pytest.mark.parametrize("mode", ["teacher", "reinforce"])
def test_batch_of_one_matches_original_per_sample_rule(mode):
    orig = _original()
    if orig is None:
        pytest.skip("needs git history")
    ocfg = orig.PlasticityCfg(dt=0.1, a=0.7, b=0.8, tau=12.5, bound=5, bound_mode="tanh", **KW)
    torch.manual_seed(0)
    old = orig.FHNBioplasticNet(48, 8, 1.0, ocfg, True, torch.device("cpu"))
    torch.manual_seed(0)
    new = FHNBioplasticNet(48, 8, 1.0, PlasticityCfg(**KW), FHNParams(), True)
    xs = torch.randn(6, 1, 28, 28)
    ys = new.forward_logits(xs).argmax(-1)
    ys[3:] = (ys[3:] + 1) % 10  # mix of rewarded and unrewarded trials
    for i in range(6):
        if mode == "teacher":
            old.learn_one_teacher(xs[i:i + 1], ys[i:i + 1], "mean", True)
        else:
            old.learn_one_reinforce(xs[i:i + 1], ys[i:i + 1], "mean", True)
        new.learn_batch(xs[i:i + 1], ys[i:i + 1], mode)
    for k in ["W_in.weight", "W_in.bias", "W_rec.weight", "reward_baseline"]:
        torch.testing.assert_close(new.state_dict()[k], old.state_dict()[k], atol=1e-5, rtol=1e-4)


def test_batched_sum_equals_sum_of_single_updates_when_frozen_forward():
    """With batch_reduce='sum', a batch update is the sum of the per-trial updates (same starting weights)."""
    cfg = PlasticityCfg(eta_in=1e-3, eta_rec=1e-3, eta_out=1e-3, batch_reduce="sum")
    torch.manual_seed(1)
    base = FHNBioplasticNet(32, 6, 1.0, cfg)
    xs, ys = torch.randn(4, 1, 28, 28), torch.randint(0, 10, (4,))
    batched = FHNBioplasticNet(32, 6, 1.0, cfg)
    batched.load_state_dict(base.state_dict())
    batched.learn_batch(xs, ys, "teacher")
    total = {k: torch.zeros_like(v) for k, v in base.state_dict().items()}
    for i in range(4):
        m = FHNBioplasticNet(32, 6, 1.0, cfg)
        m.load_state_dict(base.state_dict())
        m.learn_batch(xs[i:i + 1], ys[i:i + 1], "teacher")
        for k, v in m.state_dict().items():
            total[k] += v - base.state_dict()[k]
    for k in ["W_in.weight", "W_rec.weight", "W_out.weight"]:
        torch.testing.assert_close(batched.state_dict()[k] - base.state_dict()[k], total[k], atol=1e-6, rtol=1e-4)


@pytest.mark.parametrize("readout_on", ["v", "rate"])
def test_teacher_output_update_is_the_ce_gradient_of_its_forward_pass(readout_on):
    cfg = PlasticityCfg(eta_in=0, eta_rec=0, eta_out=0.1, readout_on=readout_on, batch_reduce="sum")
    torch.manual_seed(0)
    m = FHNBioplasticNet(32, 6, 1.0, cfg)
    x, y = torch.randn(3, 1, 28, 28), torch.randint(0, 10, (3,))
    W = m.W_out.weight.clone().requires_grad_(True)
    b = m.W_out.bias.clone().requires_grad_(True)
    with torch.no_grad():
        v_feat, _ = m._run(m.W_in(x.reshape(3, -1)), record=False)
        f = m._readout_features(v_feat)
    loss = F.cross_entropy(f @ W.T + b, y, reduction="sum")
    gW, gb = torch.autograd.grad(loss, (W, b))
    m.learn_batch(x, y, "teacher")
    torch.testing.assert_close(m.W_out.weight, (W - 0.1 * gW).detach(), atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(m.W_out.bias, (b - 0.1 * gb).detach(), atol=1e-6, rtol=1e-5)


def test_teacher_inject_does_not_leak_label_into_prediction():
    cfg = PlasticityCfg(teacher_inject=50.0, eta_in=0, eta_rec=0, eta_out=0)
    torch.manual_seed(0)
    m = FHNBioplasticNet(32, 6, 1.0, cfg)
    x = torch.randn(64, 1, 28, 28)
    pred = m.forward_logits(x).argmax(-1)
    y = (pred + 1) % 10  # labels the free network gets wrong
    out = m.learn_batch(x, y, "teacher")
    assert out["correct"] == 0


def test_max_update_caps_the_step():
    cfg = PlasticityCfg(eta_in=1.0, eta_rec=1.0, eta_out=1.0, max_update=0.01)
    torch.manual_seed(0)
    m = FHNBioplasticNet(32, 6, 1.0, cfg)
    before = {k: v.clone() for k, v in m.state_dict().items()}
    m.learn_batch(torch.randn(8, 1, 28, 28), torch.randint(0, 10, (8,)), "teacher")
    for k in ["W_in.weight", "W_rec.weight", "W_out.weight"]:
        assert (m.state_dict()[k] - before[k]).norm() <= 0.01 + 1e-6
