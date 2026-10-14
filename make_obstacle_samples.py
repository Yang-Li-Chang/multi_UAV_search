"""Create representative obstacle-map samples for the course progress update."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Rectangle

from signal_search_env import SignalSearchEnv


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "signal_experiments" / "progress_figures" / "obstacle-sample-maps.png"


def draw_sample(axis: plt.Axes, env: SignalSearchEnv, sample_index: int) -> None:
    axis.set_facecolor("#f4f6f3")
    axis.set_xlim(-env.boundary, env.boundary)
    axis.set_ylim(-env.boundary, env.boundary)
    axis.set_aspect("equal")
    axis.set_xticks(np.linspace(-env.boundary, env.boundary, 7))
    axis.set_yticks(np.linspace(-env.boundary, env.boundary, 7))
    axis.set_xticklabels([])
    axis.set_yticklabels([])
    axis.tick_params(length=0)
    axis.grid(color="#d0d7d0", linewidth=0.55)
    mask = env._obstacle_mask(env._obstacle_samples()[sample_index])
    for x, y in np.argwhere(mask):
        lower_left = -env.boundary + np.array([x, y]) * env.cell_width
        axis.add_patch(Rectangle(lower_left, env.cell_width, env.cell_width, facecolor="#505b65", edgecolor="#f4f6f3", linewidth=0.45, zorder=3))
    for target in env._layouts()[sample_index]:
        axis.add_patch(Circle(target, 0.24, facecolor="#e9a86c", alpha=0.12, edgecolor="none", zorder=1))
        axis.scatter(target[0], target[1], marker="*", s=105, color="#d65b36", edgecolors="white", linewidths=0.7, zorder=4)
    axis.add_patch(Circle((0, -1.90), 0.13, facecolor="#46637d", edgecolor="white", linewidth=0.8, zorder=4))
    axis.text(0, -1.90, "Base", color="white", ha="center", va="center", fontsize=7.5, fontweight="bold", zorder=5)
    specs = env._obstacle_samples()[sample_index]
    shape_names = {"h": "bar", "v": "bar", "block": "block", "L": "L"}
    kinds = ", ".join(sorted(set(shape_names[item[0]] for item in specs)))
    axis.set_title(f"Sample {sample_index + 1}  |  {int(mask.sum())} obstacle cells\n{kinds}", fontsize=10.5, fontweight="semibold", pad=7)


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    env = SignalSearchEnv()
    figure, axes = plt.subplots(2, 2, figsize=(10.8, 9.2), facecolor="white")
    figure.suptitle("Diverse Obstacle Environment Samples", fontsize=20, fontweight="bold", y=0.98)
    figure.text(0.5, 0.945, "30 × 30 grid  •  bars, blocks, and L-shaped obstacles  •  target positions vary by sample", ha="center", color="#566159", fontsize=10)
    for axis, sample_index in zip(axes.flat, (0, 2, 5, 7), strict=True):
        draw_sample(axis, env, sample_index)
    figure.tight_layout(rect=(0, 0, 1, 0.92))
    figure.savefig(OUT, dpi=240, bbox_inches="tight")
    print(OUT)


if __name__ == "__main__":
    main()
