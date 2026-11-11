"""Video of a k x k submatrix of the recurrent weights over training (MP4 via ffmpeg, or GIF)."""

from __future__ import annotations

import os

import torch


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
        import matplotlib
        import numpy as np
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


    def maybe_log_step(self, W: torch.Tensor, epoch: int, step_in_epoch: int, global_step: int, loss_val: float,
                       tag: str = "step"):
        if self.every_steps <= 0:
            return
        if (global_step % self.every_steps) != 0:
            return
        self._log(W, epoch=epoch, step_in_epoch=step_in_epoch, global_step=global_step, loss_val=loss_val, tag=tag)

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

        step = step_in_epoch if step_in_epoch >= 0 else "end"
        title = (
            f"W_rec submatrix ({tag}) | ep={epoch} step={step} gstep={global_step}\n"
            f"metric={loss_val:.4f} | mean={mean:.2e} std={std:.2e} max|w|={mx:.2e}"
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
