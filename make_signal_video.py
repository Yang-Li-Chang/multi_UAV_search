"""Create a GIF animation of the best local-signal MADDPG search policy."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter
from matplotlib.patches import Rectangle

from maddpg import MADDPG
from signal_search_env import SignalSearchEnv


ROOT = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=10_000, help="fixed layout seed for the displayed episode")
    parser.add_argument(
        "--model",
        type=Path,
        default=ROOT / "signal_experiments" / "diverse_obstacle_signal" / "maddpg_actor.pt",
        help="trained actor checkpoint",
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "signal_experiments" / "best_signal_search.mp4"
    )
    args = parser.parse_args()

    env = SignalSearchEnv(reward_variant="signal", seed=args.seed)
    agent = MADDPG(env.num_agents, env.obs_dim, env.action_dim, gamma=0.98, tau=0.01)
    agent.load(str(args.model))
    obs = env.reset(seed=args.seed)
    frames: list[dict[str, object]] = []
    paths = [[pos.copy()] for pos in env.render_data()["positions"]]
    done = False
    cumulative_rewards = np.zeros(env.num_agents, dtype=np.float32)
    while not done:
        actions = agent.select_actions(obs, noise_scale=0.0)
        obs, rewards, done, info = env.step(actions)
        cumulative_rewards += rewards
        data = env.render_data()
        for i, pos in enumerate(data["positions"]):
            paths[i].append(pos.copy())
        frames.append({"data": data, "paths": [[p.copy() for p in path] for path in paths], "info": info.copy()})

    if not frames[-1]["info"]["success"]:
        raise RuntimeError("Selected policy did not finish this displayed episode; choose another seed.")

    fig, ax = plt.subplots(figsize=(8.2, 7.4), facecolor="#f8f7f2")
    ax.set_facecolor("#edf0eb")
    ax.set_xlim(-env.boundary - 0.03, env.boundary + 0.03)
    ax.set_ylim(-env.boundary - 0.03, env.boundary + 0.03)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Two-UAV cooperative local-signal search (MADDPG)", pad=14)
    ticks = np.linspace(-env.boundary, env.boundary, env.grid_size + 1)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.grid(color="#c7cdc7", linewidth=0.45)

    base = plt.Circle((0.0, -1.90), 0.18, facecolor="#4e6274", edgecolor="#f8f7f2", linewidth=1.5, alpha=0.9, zorder=2)
    ax.add_patch(base)
    ax.text(0.0, -1.90, "Launch\nbase", color="white", ha="center", va="center", fontsize=8.5, zorder=3)

    # The field and hidden target symbols are visible only to the audience; the policy receives
    # only a 3x3 patch around each UAV, never this global map or target coordinates.
    signal_image = ax.imshow(
        np.zeros((160, 160)),
        extent=(-env.boundary, env.boundary, -env.boundary, env.boundary),
        origin="lower",
        cmap="magma",
        vmin=0,
        vmax=1.0,
        alpha=0.25,
        interpolation="bilinear",
        zorder=0,
    )
    covered_image = ax.imshow(
        np.ma.masked_all((env.grid_size, env.grid_size)),
        extent=(-env.boundary, env.boundary, -env.boundary, env.boundary),
        origin="lower",
        cmap="GnBu",
        vmin=0,
        vmax=1,
        alpha=0.22,
        interpolation="nearest",
        zorder=1,
    )
    colors = ["#1463b8", "#8a2a85"]
    path_lines = [ax.plot([], [], color=color, linewidth=2.2, alpha=0.84, zorder=4)[0] for color in colors]
    drones = [ax.plot([], [], marker="^", markersize=10, color=color, markeredgecolor="white", markeredgewidth=1.0, zorder=7)[0] for color in colors]
    sensing = [plt.Circle((0, 0), 0.25, facecolor=color, edgecolor=color, linewidth=1, alpha=0.08, zorder=3) for color in colors]
    for ring in sensing:
        ax.add_patch(ring)
    targets = [ax.plot([], [], marker="*", markersize=15, color="#e8751a", markeredgecolor="white", markeredgewidth=0.8, zorder=6)[0] for _ in range(env.num_targets)]
    status = ax.text(-env.boundary + 0.04, env.boundary - 0.04, "", va="top", ha="left", fontsize=10, bbox={"facecolor": "#f8f7f2", "alpha": 0.91, "edgecolor": "#c7cdc7"}, zorder=8)
    note = ax.text(env.boundary - 0.04, -env.boundary + 0.04, "Signal field shown for the viewer\nPolicy receives only local signal + obstacle patches", va="bottom", ha="right", fontsize=8.5, bbox={"facecolor": "#f8f7f2", "alpha": 0.91, "edgecolor": "#c7cdc7"}, zorder=8)
    _ = note

    initial_data = frames[0]["data"]
    for x, y in np.argwhere(initial_data["obstacles"]):
        lower_left = -env.boundary + np.array([x, y]) * env.cell_width
        ax.add_patch(Rectangle(lower_left, env.cell_width, env.cell_width, facecolor="#303942", edgecolor="#f8f7f2", linewidth=0.7, zorder=5))

    visual_axis = np.linspace(-env.boundary, env.boundary, 200)
    field_x, field_y = np.meshgrid(visual_axis, visual_axis)

    def smooth_signal_field(data: dict[str, object]) -> np.ndarray:
        field = np.zeros_like(field_x)
        for position, found in zip(data["targets"], data["found"], strict=True):
            if not found:
                distance_squared = (field_x - position[0]) ** 2 + (field_y - position[1]) ** 2
                field += np.exp(-distance_squared / (2 * 0.24**2))
        return np.clip(field, 0.0, 1.0)

    def update(frame_index: int):
        frame = frames[frame_index]
        data = frame["data"]
        signal_image.set_data(smooth_signal_field(data))
        covered_image.set_data(np.ma.masked_where(~data["coverage"].T, data["coverage"].T.astype(float)))
        for i, line in enumerate(path_lines):
            path = np.asarray(frame["paths"][i])
            line.set_data(path[:, 0], path[:, 1])
            drones[i].set_data([data["positions"][i, 0]], [data["positions"][i, 1]])
            sensing[i].center = data["positions"][i]
        for k, target in enumerate(targets):
            position = data["targets"][k]
            target.set_data([position[0]], [position[1]])
            target.set_color("#1a9850" if data["found"][k] else "#de2d26")
        info = frame["info"]
        status.set_text(
            f"Step {int(info['steps'])}   Targets found: {int(info['targets_found'])}/3\n"
            f"Coverage: {info['coverage']:.0%}   UAV collisions: {int(info['collisions'])}"
        )
        return [signal_image, covered_image, *path_lines, *drones, *sensing, *targets, status]

    animation = FuncAnimation(fig, update, frames=len(frames), interval=135, blit=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.suffix.lower() == ".mp4":
        import imageio_ffmpeg

        matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
        writer = FFMpegWriter(fps=7, metadata={"title": "Two-UAV local-signal search"}, bitrate=1500)
    elif args.output.suffix.lower() == ".gif":
        writer = PillowWriter(fps=7)
    else:
        raise ValueError("Output file must use .mp4 or .gif")
    animation.save(args.output, writer=writer, dpi=105)
    plt.close(fig)
    print(f"Saved {args.output} with {len(frames)} frames; success={frames[-1]['info']['success']}")


if __name__ == "__main__":
    main()
