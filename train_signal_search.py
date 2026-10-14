"""Train and compare MADDPG reward designs for two-UAV local-signal search.

The policy observation deliberately excludes target coordinates.  It contains only
each UAV's local 3x3 signal patch, its shared coverage memory, and basic motion
state.  Run one reward variant at a time so the experiment remains easy to repeat.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch

from maddpg import MADDPG, ReplayBuffer
from signal_search_env import RewardVariant, SignalSearchEnv


ROOT = Path(__file__).resolve().parent


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def evaluate(agent: MADDPG, variant: RewardVariant, episodes: int = 20) -> dict[str, float]:
    env = SignalSearchEnv(reward_variant=variant, seed=10_000)
    records: list[dict[str, float]] = []
    for episode in range(episodes):
        obs = env.reset(seed=10_000 + episode)
        done = False
        returns = np.zeros(env.num_agents, dtype=np.float32)
        final_info: dict[str, float] = {}
        while not done:
            actions = agent.select_actions(obs, noise_scale=0.0)
            obs, rewards, done, final_info = env.step(actions)
            returns += rewards
        records.append({**final_info, "return": float(returns.mean())})
    return {key: float(np.mean([record[key] for record in records])) for key in records[0]}


def train(variant: RewardVariant, episodes: int, seed: int, experiment: str | None = None) -> dict[str, float]:
    set_seed(seed)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    env = SignalSearchEnv(reward_variant=variant, seed=seed)
    agent = MADDPG(env.num_agents, env.obs_dim, env.action_dim, gamma=0.98, tau=0.01)
    buffer = ReplayBuffer(capacity=80_000)
    output = ROOT / "signal_experiments" / (experiment or variant)
    output.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, float]] = []
    environment_steps = 0

    for episode in range(1, episodes + 1):
        obs = env.reset()
        done = False
        returns = np.zeros(env.num_agents, dtype=np.float32)
        final_info: dict[str, float] = {}
        # Gradually reduce action exploration, while retaining a little exploration.
        noise = max(0.08, 0.36 * (1.0 - episode / episodes))
        while not done:
            actions = agent.select_actions(obs, noise_scale=noise)
            next_obs, rewards, done, final_info = env.step(actions)
            buffer.add(obs, actions, rewards, next_obs, done)
            obs = next_obs
            returns += rewards
            environment_steps += 1
            # Updating every fourth interaction keeps this larger 15x15 task practical
            # on a CPU while still giving the replay buffer enough diverse samples.
            if len(buffer) >= 256 and environment_steps % 4 == 0:
                agent.update(buffer, batch_size=128)
        record = {
            "episode": float(episode),
            "return": float(returns.mean()),
            **final_info,
        }
        history.append(record)
        if episode % 50 == 0 or episode == 1:
            recent = history[-50:]
            print(
                f"{variant:13s} episode {episode:3d}/{episodes}: "
                f"return={np.mean([r['return'] for r in recent]):6.2f}, "
                f"found={np.mean([r['targets_found'] for r in recent]):.2f}, "
                f"success={np.mean([r['success'] for r in recent]):.2%}",
                flush=True,
            )

    model_path = output / "maddpg_actor.pt"
    agent.save(str(model_path))
    with (output / "training_history.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    metrics = evaluate(agent, variant)
    summary = {
        "reward_variant": variant,
        "episodes": episodes,
        "seed": seed,
        "model": str(model_path),
        "evaluation": metrics,
        "observation_note": "Target coordinates are not part of the policy observation.",
    }
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("EVALUATION", json.dumps(summary, ensure_ascii=False), flush=True)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=["coverage", "signal", "shared_signal"], required=True)
    parser.add_argument("--episodes", type=int, default=250)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--experiment", type=str, default=None, help="subfolder for this training run")
    args = parser.parse_args()
    train(args.variant, args.episodes, args.seed, args.experiment)


if __name__ == "__main__":
    main()
