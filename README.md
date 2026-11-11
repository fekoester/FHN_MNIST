# FHN-MNIST: FitzHugh–Nagumo neuron networks that learn to read digits

A dense recurrent network of **FitzHugh–Nagumo (FHN) neurons** classifies MNIST and Fashion-MNIST. It is trained in two ways:

1. **Backpropagation through time** (`scripts/fhn_mnist_train.py`)
2. **Biologically motivated three-factor plasticity**, with no backprop (`scripts/fhn_mnist_bioplasticity.py`):
   - a *teacher* mode, where a random feedback signal broadcasts the output error
   - a *reinforce* mode, where a dopamine-like reward-prediction error is the third factor

![Neuron dynamics for three test digits](docs/media/dynamics.gif)

*Membrane voltage of six neurons, their (v, w) phase-plane paths, and the class read out from the running mean of v, step by step.*

## Model

An image is flattened and projected once into a constant drive `u = W_in x`. For each neuron *i*, explicit Euler integration runs for *T* steps of size *dt*:

$$
\dot v_i = v_i - \tfrac{1}{3}v_i^3 - w_i + u_i + \sum_j W^{rec}_{ij} v_j,\qquad
\dot w_i = \frac{v_i + a - b\,w_i}{\tau}
$$

Both $v$ and $w$ are softly bounded with $5\tanh(\cdot/5)$. A linear readout of the time-averaged voltage $\bar v = \frac1T\sum_t v_t$ gives the class logits. Defaults: N = 512, T = 16, dt = 0.1, a = 0.7, b = 0.8, τ = 12.5, and orthogonal $W^{rec}$ scaled by $g/\sqrt N$.

**What the dynamics actually do.** With the defaults, the simulated time $T\,dt = 1.6$ is much shorter than the recovery time constant τ = 12.5, so $w$ barely moves and no neuron spikes. As the animation shows, each neuron relaxes to one of the two stable branches of the cubic nullcline, near $v \approx \pm 3$. The network is effectively a recurrent, *bistable* relaxation network that sorts its neurons into two groups depending on the input. Running a trained model 25× longer (`--T 400`) makes fewer than 4% of neurons oscillate, and accuracy drops from 94% to 79%. The regime where real spiking contributes to the computation (larger *T·dt*, smaller τ, or *a* chosen near the Hopf point) is open to explore. `scripts/animate_dynamics.py --T ...` makes that easy to look at.

## Learning without backprop: three-factor plasticity

During each image (a *trial*), every synapse builds an STDP-like **eligibility trace** from a smooth spike proxy $s=\sigma((v-\theta)/\kappa)$. It combines a presynaptic trace $x^{pre}$ and a postsynaptic trace $y^{post}$, which are low-pass filters of $s$:

$$
E^{rec}_{ij} \leftarrow \alpha_e E^{rec}_{ij} + A_+\, s_i\, x^{pre}_j - A_-\, y^{post}_i\, s_j,
\qquad
E^{in}_{ij} \leftarrow \alpha_e E^{in}_{ij} + A_+\, s_i\, \bar x_j
$$

At the end of the trial, a **third factor** $l_i$ turns eligibility into a weight change, $\Delta W_{ij} = \eta\, l_i\, E_{ij}$:

| mode | third factor $l_i$ | readout update |
|---|---|---|
| `teacher` | $(B_{fb}\,(\mathrm{onehot}(y) - p))_i$: the output error sent back through a fixed random matrix (feedback alignment / e-prop style) | delta rule = exact CE gradient |
| `reinforce` | $M = r - \bar r$: the reward-prediction error. With `--rl_signal feedback` (the default) it is $M\,(B_{fb}(\mathrm{onehot}(a) - p_T))_i$ | REINFORCE = exact policy gradient |

**Batched and exact.** The third factor arrives only once per trial, so the final eligibility is a weighted sum over steps: $E_T=\sum_t \alpha_e^{T-1-t}(\dots)_t$. The update is then computed with a few batched matrix products instead of updating an N×N trace matrix at every step. Because the image drive is constant, the input eligibility is even rank one. With `--batch 1`, the updates match the original per-sample implementation to float precision, and a test checks this. Larger batches sum or average the per-trial updates.

## Results

All methods share the same setup:
- 5,000 training images are held out for validation, the best epoch is chosen on them, and the test set is evaluated once
- N = 512, T = 16 for the FHN models
- 5 epochs for backprop and 3 epochs for plasticity, on 2 CPU cores, with one seed

Reproduce with `python scripts/benchmark.py`.

| method | trainable params | MNIST | Fashion-MNIST |
|---|---|---|---|
| Logistic regression on pixels | 7.9k | 92.18% | 83.68% |
| MLP 784-512-10 (backprop) | 407k | **97.75%** | **87.45%** |
| FHN, random network, readout only (backprop) | 5.1k | 92.32% | 84.26% |
| FHN, BPTT, W_rec frozen | 407k | 97.06% | 85.65% |
| FHN, BPTT, all weights | 669k | 96.24% | 86.02% |
| FHN, teacher plasticity, readout only | 5.1k | 90.96% | 81.83% |
| **FHN, teacher plasticity, three-factor** | 669k | **95.03%** | 83.59% |
| FHN, reward plasticity, readout only | 5.1k | 87.36% | 77.24% |
| FHN, reward plasticity, three-factor | 669k | 89.72% | 77.40% |

![Benchmark](docs/media/benchmark.png)

What this says:

- **Three-factor teacher plasticity works.** Without any backprop, learning in the hidden and recurrent layers adds +4.1 points on MNIST over training the readout alone (95.0% vs 91.0%). That gets within about 1–2 points of BPTT on the same network.
- **Reward-modulated plasticity is much harder.** One scalar reward per image is a weak signal. Hidden plasticity still adds +2.4 points on MNIST, but nothing on Fashion-MNIST.
- **On these budgets, a plain MLP is the best model.** The FHN dynamics add little over a feed-forward network trained by backprop, and training the recurrent weights does not help with this short a horizon (see *What the dynamics actually do*). A random FHN network with only the readout trained is no better than logistic regression on the pixels.

The numbers come from short CPU runs with one seed, so differences under about half a point should not be over-read.

## Installation and usage

```bash
pip install -e ".[video]"        # or: pip install -r requirements.txt -r requirements-optional.txt

# backprop through time
python scripts/fhn_mnist_train.py --epochs 10 --out_dir runs/bp
python scripts/fhn_mnist_train.py --dataset fashion --train_rec 0 --out_dir runs/bp_fashion

# three-factor plasticity
python scripts/fhn_mnist_bioplasticity.py --mode teacher --epochs 3 --out_dir runs/teacher
python scripts/fhn_mnist_bioplasticity.py --mode reinforce --epochs 3

# video of the recurrent weights during training (MP4 with ffmpeg, else GIF)
python scripts/fhn_mnist_train.py --conn_video runs/conn.mp4 --conn_every 50 --conn_k 256

# animate the neuron dynamics of a trained model (optionally past the trained horizon)
python scripts/animate_dynamics.py --checkpoint runs/bp/best.pt --out dynamics.gif --T 64

# full benchmark (resumable, writes results/benchmark/{results.json,results.md,benchmark.png})
python scripts/benchmark.py --datasets mnist fashion
```

Both training scripts:
- write `metrics.json` and a checkpoint to `--out_dir`
- keep the best epoch by validation and report the test set once at the end
- accept `--dataset {mnist,fashion}`
- run on CUDA automatically when it is available. The backprop script uses bf16 autocast there, and `torch.compile` is on by default only on CUDA.

Useful plasticity flags:
- `--batch` (1 = the original online learning) and `--batch_reduce`
- `--readout_on {v,rate}`
- `--rl_signal {global,feedback}`
- `--max_update` (caps the size of each update), `--w_decay`, `--w_clip`
- `--teacher_inject`: injects a label current during the learning pass only. The prediction and the error come from a separate free pass, so the label never leaks into the reported accuracy.

## Changes in 0.2

- **Fixed:** the plasticity readout predicted from $v$ but learned from $\sigma(v)$, so its update was neither the CE gradient (teacher) nor the policy gradient (reinforce). It now learns from the features it predicts with.
- **Fixed:** in reinforce, the learning rates were orders of magnitude too large for the eligibility scale (the input eligibility grows like $\sum_t c_t \approx 70$ times the image), so hidden updates wiped out the features. Test accuracy went from 24% (original code, 20k samples) to 89.7% (3 epochs). Defaults are now set per mode, and `--max_update` exists as a safety cap.
- **Fixed:** with `--teacher_inject`, the label was part of the input used for the reported prediction.
- **Plasticity is about 15× faster:** exact batched updates, 47 s per epoch on 2 CPU cores, against about 13 minutes before.
- **Backprop script:**
  - validation split and checkpoints
  - the weight video is off by default (it used to render a frame every 10 steps)
  - `torch.amp` replaces the deprecated `torch.cuda.amp`
  - `--train_in`, `--dataset fashion`, `--data_dir`
- **Structure:**
  - a shared package (`fhn_mnist/`) for the dynamics, models, plasticity, data and video logger
  - benchmark and animation scripts
  - tests (including bit-level equivalence with the original rule) and CI

## Tests

```bash
pip install -e ".[dev,video]" && pytest
```

## License

MIT, see [LICENSE](LICENSE).
