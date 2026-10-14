"""Create a concise Chinese schematic of the obstacle-search environment."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.patches import Circle, Rectangle

from signal_search_env import SignalSearchEnv


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "signal_experiments" / "progress_figures" / "场地示意图.png"


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    env = SignalSearchEnv(seed=5)
    env.reset(seed=5)
    data = env.render_data()
    figure, axis = plt.subplots(figsize=(8.6, 8.0), facecolor="white")
    axis.set_facecolor("#f4f6f3")
    axis.set_xlim(-env.boundary - 0.04, env.boundary + 0.04)
    axis.set_ylim(-env.boundary - 0.04, env.boundary + 0.04)
    axis.set_aspect("equal")
    ticks = np.linspace(-env.boundary, env.boundary, env.grid_size + 1)
    axis.set_xticks(ticks)
    axis.set_yticks(ticks)
    axis.set_xticklabels([])
    axis.set_yticklabels([])
    axis.tick_params(length=0)
    axis.grid(color="#cbd2ca", linewidth=0.5)
    axis.set_title("Multi-UAV Cooperative Search Environment", fontsize=19, fontweight="semibold", pad=18)
    axis.text(0, env.boundary + 0.13, "30 × 30 Grid   •   2 UAVs   •   3 Static Targets   •   6 × (1 × 3) Obstacles", ha="center", color="#566159", fontsize=10.5)

    for x, y in np.argwhere(data["obstacles"]):
        lower_left = -env.boundary + np.array([x, y]) * env.cell_width
        axis.add_patch(Rectangle(lower_left, env.cell_width, env.cell_width, facecolor="#555d65", edgecolor="white", linewidth=0.8, zorder=3))

    starts = data["positions"]
    base = Circle((0, -1.90), 0.17, facecolor="#46637d", edgecolor="white", linewidth=1.5, zorder=4)
    axis.add_patch(base)
    axis.text(0, -1.90, "Launch\nBase", ha="center", va="center", color="white", fontsize=9.5, fontweight="semibold", zorder=5)
    colors = ["#1f70b8", "#8c3f91"]
    for index, position in enumerate(starts):
        display_position = position + np.array([0.0, 0.17])
        axis.scatter(display_position[0], display_position[1], marker="^", s=95, color=colors[index], edgecolors="white", linewidths=1.0, zorder=6)

    for index, position in enumerate(data["targets"]):
        axis.add_patch(Circle(position, 0.33, facecolor="#ecad71", edgecolor="none", alpha=0.11, zorder=1))
        axis.scatter(position[0], position[1], marker="*", s=260, color="#d65b36", edgecolors="white", linewidths=1.0, zorder=6)
        axis.text(position[0], position[1] + 0.20, f"Target {index + 1}", ha="center", color="#b14225", fontsize=10, fontweight="semibold")

    figure.savefig(OUT, dpi=260, bbox_inches="tight")
    print(OUT)


if __name__ == "__main__":
    main()
