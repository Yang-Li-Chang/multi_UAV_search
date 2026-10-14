"""Small MADDPG implementation for continuous-control cooperative experiments."""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn


class Actor(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh(),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


class Critic(nn.Module):
    def __init__(self, joint_obs_dim: int, joint_action_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(joint_obs_dim + joint_action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, joint_obs: torch.Tensor, joint_actions: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([joint_obs, joint_actions], dim=-1))


class ReplayBuffer:
    def __init__(self, capacity: int = 100_000) -> None:
        self.data: deque[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]] = deque(maxlen=capacity)

    def __len__(self) -> int:
        return len(self.data)

    def add(self, obs: np.ndarray, actions: np.ndarray, rewards: np.ndarray, next_obs: np.ndarray, done: bool) -> None:
        self.data.append((obs.astype(np.float32), actions.astype(np.float32), rewards.astype(np.float32), next_obs.astype(np.float32), float(done)))

    def sample(self, batch_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        batch = random.sample(self.data, batch_size)
        obs, actions, rewards, next_obs, done = map(np.asarray, zip(*batch))
        return obs, actions, rewards, next_obs, done.astype(np.float32)


@dataclass
class Learner:
    actor: Actor
    actor_target: Actor
    critic: Critic
    critic_target: Critic
    actor_optim: torch.optim.Optimizer
    critic_optim: torch.optim.Optimizer


class MADDPG:
    def __init__(
        self,
        num_agents: int,
        obs_dim: int,
        action_dim: int,
        device: str = "cpu",
        actor_lr: float = 1e-3,
        critic_lr: float = 1e-3,
        gamma: float = 0.95,
        tau: float = 0.01,
    ) -> None:
        self.num_agents = num_agents
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.device = torch.device(device)
        self.gamma = gamma
        self.tau = tau
        self.learners: list[Learner] = []
        joint_obs_dim = num_agents * obs_dim
        joint_action_dim = num_agents * action_dim

        for _ in range(num_agents):
            actor = Actor(obs_dim, action_dim).to(self.device)
            actor_target = Actor(obs_dim, action_dim).to(self.device)
            critic = Critic(joint_obs_dim, joint_action_dim).to(self.device)
            critic_target = Critic(joint_obs_dim, joint_action_dim).to(self.device)
            actor_target.load_state_dict(actor.state_dict())
            critic_target.load_state_dict(critic.state_dict())
            self.learners.append(
                Learner(
                    actor=actor,
                    actor_target=actor_target,
                    critic=critic,
                    critic_target=critic_target,
                    actor_optim=torch.optim.Adam(actor.parameters(), lr=actor_lr),
                    critic_optim=torch.optim.Adam(critic.parameters(), lr=critic_lr),
                )
            )

    def select_actions(self, observations: np.ndarray, noise_scale: float = 0.0) -> np.ndarray:
        actions = []
        with torch.no_grad():
            for index, learner in enumerate(self.learners):
                obs = torch.as_tensor(observations[index], dtype=torch.float32, device=self.device).unsqueeze(0)
                action = learner.actor(obs).squeeze(0).cpu().numpy()
                if noise_scale:
                    action += np.random.normal(0.0, noise_scale, size=self.action_dim)
                actions.append(np.clip(action, -1.0, 1.0))
        return np.asarray(actions, dtype=np.float32)

    def update(self, buffer: ReplayBuffer, batch_size: int = 128) -> dict[str, float] | None:
        if len(buffer) < batch_size:
            return None
        obs, actions, rewards, next_obs, dones = buffer.sample(batch_size)
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        actions_t = torch.as_tensor(actions, dtype=torch.float32, device=self.device)
        rewards_t = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
        next_obs_t = torch.as_tensor(next_obs, dtype=torch.float32, device=self.device)
        dones_t = torch.as_tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(-1)
        joint_obs = obs_t.reshape(batch_size, -1)
        joint_actions = actions_t.reshape(batch_size, -1)
        next_joint_obs = next_obs_t.reshape(batch_size, -1)

        critic_losses: list[float] = []
        actor_losses: list[float] = []
        with torch.no_grad():
            target_actions = torch.cat(
                [learner.actor_target(next_obs_t[:, i]) for i, learner in enumerate(self.learners)], dim=-1
            )

        for i, learner in enumerate(self.learners):
            with torch.no_grad():
                target_q = learner.critic_target(next_joint_obs, target_actions)
                expected_q = rewards_t[:, i : i + 1] + self.gamma * (1.0 - dones_t) * target_q
            current_q = learner.critic(joint_obs, joint_actions)
            critic_loss = nn.functional.mse_loss(current_q, expected_q)
            learner.critic_optim.zero_grad()
            critic_loss.backward()
            nn.utils.clip_grad_norm_(learner.critic.parameters(), 1.0)
            learner.critic_optim.step()
            critic_losses.append(float(critic_loss.item()))

            actor_actions = []
            for j, other in enumerate(self.learners):
                policy_action = other.actor(obs_t[:, j])
                actor_actions.append(policy_action if i == j else policy_action.detach())
            actor_joint_actions = torch.cat(actor_actions, dim=-1)
            actor_loss = -learner.critic(joint_obs, actor_joint_actions).mean()
            learner.actor_optim.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(learner.actor.parameters(), 1.0)
            learner.actor_optim.step()
            actor_losses.append(float(actor_loss.item()))

        self._soft_update()
        return {"critic_loss": float(np.mean(critic_losses)), "actor_loss": float(np.mean(actor_losses))}

    def _soft_update(self) -> None:
        for learner in self.learners:
            for target, source in zip(learner.actor_target.parameters(), learner.actor.parameters(), strict=True):
                target.data.mul_(1.0 - self.tau).add_(source.data, alpha=self.tau)
            for target, source in zip(learner.critic_target.parameters(), learner.critic.parameters(), strict=True):
                target.data.mul_(1.0 - self.tau).add_(source.data, alpha=self.tau)

    def save(self, path: str) -> None:
        torch.save({"learners": [learner.actor.state_dict() for learner in self.learners]}, path)

    def load(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        for state, learner in zip(checkpoint["learners"], self.learners, strict=True):
            learner.actor.load_state_dict(state)
            learner.actor_target.load_state_dict(state)
