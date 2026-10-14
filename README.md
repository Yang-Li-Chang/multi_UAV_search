# Multi-UAV Cooperative Search

A compact MADDPG baseline for two UAVs that cooperatively search for three hidden static targets in an MPE2 two-dimensional environment.

## Current task

- **Map:** 30 × 30 grid over a continuous 2D area.
- **Agents:** two UAVs launched from the same base area.
- **Targets:** three static targets per episode. Target coordinates are not included in the policy observation.
- **Obstacle samples:** eight layouts with different counts and combinations of bars, blocks, and L-shaped no-fly regions.
- **Observation:** own position and velocity, relative teammate position, found-target status, and local 3 × 3 patches for coverage, signal strength, and obstacles.
- **Communication baseline:** UAVs share coverage and found-target status without communication interference. Interference can be added after this baseline is stable.

## Reward

Each UAV receives its own reward:

\[
r_i = 15D_i + 20F_i + N_i + 0.8G_i - 0.1R_i - 5C_i - 2O_i - 0.005\|a_i\|^2 - 0.01
\]

Where `D` is a newly discovered target, `F` is the final target that completes the task, `N` is a newly covered grid cell, `G` is a local signal-strength increase, `R` is repeated coverage, `C` is a UAV collision, and `O` is a blocked obstacle entry.

## Project structure

```text
signal_search_env.py      # MPE2 environment, signals, obstacle samples, reward
maddpg.py                 # actor, centralized critic, replay buffer, MADDPG update
train_signal_search.py    # training and evaluation entry point
make_signal_video.py      # MP4 / GIF trajectory animation after training
make_site_diagram.py      # one environment-map image
make_obstacle_samples.py  # four representative obstacle-map images
samples/                  # generated map images for reports
```

## Install

```bash
python -m pip install -r requirements.txt
```

## Train

```bash
python train_signal_search.py --variant signal --episodes 400 --seed 53 --experiment diverse_obstacle_signal
```

Training records, checkpoints, and videos are written to `signal_experiments/`, which is intentionally ignored by Git.

## Generate figures

```bash
python make_site_diagram.py
python make_obstacle_samples.py
```

## Generate a video after training

```bash
python make_signal_video.py --model signal_experiments/diverse_obstacle_signal/maddpg_actor.pt --output search_demo.mp4
```
