import logging

import matplotlib

matplotlib.use("QtAgg")
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
from attrs import define, field
from typing import ClassVar, Any


@define
class Dynamic3DPlotter:
    """Utility for dynamic 3D plotting of tensor data.

    This class provides methods for visualizing tensors in 3D space using matplotlib.
    It supports multiple visualization modes including scatter plots, surface plots,
    voxel plots, and bar charts.

    Attributes:
        VALID_MODES (set): Set of valid visualization modes.
        fig (Any): Matplotlib figure object.
        ax (Any): Matplotlib axes object.
    """

    VALID_MODES: ClassVar[set] = set()
    fig: Any = field(default=None, init=False)
    ax: Any = field(default=None, init=False)

    def plot(self, tensor, mode, title="Tensor Visualizer"):
        # 1. Validasyon
        if hasattr(tensor, "cpu"):
            tensor = tensor.detach().cpu().numpy().astype(float)
        else:
            tensor = np.array(tensor, dtype=float)

        ndim = tensor.ndim
        if ndim not in (1, 2, 3):
            raise ValueError(
                f"Only 1D, 2D, or 3D tensors are supported, given: {ndim}D"
            )

        if ndim == 1:
            tensor = tensor[np.newaxis, np.newaxis, :]
        elif ndim == 2:
            tensor = tensor[np.newaxis, :, :]

        D, H, W = tensor.shape

        # 2. Dispatch tanımla
        dispatch = {
            "scatter": lambda: self._plot_scatter(tensor, D, H, W),
            "surface": lambda: self._plot_surface(tensor, H, W),
            "voxel": lambda: self._plot_voxel(tensor),
            "bar": lambda: self._plot_bar(tensor),
        }

        if mode not in dispatch:
            raise ValueError(f"Unknown mode: '{mode}'. Valid: {list(dispatch.keys())}")

        # 3. Fig ve ax set et
        self.fig = plt.figure(figsize=(12, 8))
        self.ax = self.fig.add_subplot(111, projection="3d")

        # 4. Plot et
        dispatch[mode]()

        # 5. Etiketler
        self.ax.set_xlabel("W (Width)")
        self.ax.set_ylabel("H (Height)")
        self.ax.set_zlabel("D (Depth)")
        self.ax.set_title(f"{title}\nShape: {tensor.shape} | Mode: {mode}")
        plt.tight_layout()
        plt.show()

    def _plot_scatter(self, tensor, D, H, W):
        """Plot a 3D scatter plot of the tensor values."""
        d_idx, h_idx, w_idx = np.meshgrid(
            np.arange(D), np.arange(H), np.arange(W), indexing="ij"
        )

        x = w_idx.ravel()
        y = h_idx.ravel()
        z = d_idx.ravel()
        values = tensor.ravel()

        # norm = plt.Normalize(values.min(), values.max())
        # colors = cm.viridis(norm(values))

        sc = self.ax.scatter(x, y, z, c=values, cmap="viridis", s=40)
        plt.colorbar(sc, ax=self.ax, shrink=0.5, label="Values")

    def _plot_surface(self, tensor, H, W):
        """Plot a surface for the first slice of the tensor."""
        slice_2d = tensor[0]
        X, Y = np.meshgrid(np.arange(W), np.arange(H))
        surf = self.ax.plot_surface(
            X, Y, slice_2d, cmap="plasma", edgecolor="none", alpha=0.9
        )
        plt.colorbar(surf, ax=self.ax, shrink=0.5, label="Values")

    def _plot_voxel(self, tensor):
        """Plot a voxel representation of the tensor."""
        norm_t = (tensor - tensor.min()) / (tensor.max() - tensor.min() + 1e-9)
        colors_rgba = cm.coolwarm(norm_t)
        colors_rgba[..., 3] = 0.4 + 0.5 * norm_t
        self.ax.voxels(
            norm_t > 0.3, facecolors=colors_rgba, edgecolor="k", linewidth=0.3
        )

    def _plot_bar(self, tensor):
        """Plot a 3D bar chart for the first slice of the tensor."""
        slice_2d = tensor[0]
        _h, _w = slice_2d.shape
        xpos, ypos = np.meshgrid(np.arange(_w), np.arange(_h))
        xpos = xpos.ravel()
        ypos = ypos.ravel()
        zpos = np.zeros_like(xpos)
        dz = slice_2d.ravel()

        norm = plt.Normalize(dz.min(), dz.max())
        colors = cm.magma(norm(dz))

        self.ax.bar3d(
            xpos,
            ypos,
            zpos,
            dx=0.8,
            dy=0.8,
            dz=dz,
            color=colors,
            alpha=0.85,
            shade=True,
        )

    def save(self, filename):
        """Save the current plot to a file."""
        if self.fig is None:
            raise RuntimeError("No plot to save. Please call plot() first.")
        self.fig.savefig(filename)
        logging.info(f"Plot saved to {filename}")
