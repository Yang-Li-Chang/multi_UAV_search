"""Two-UAV local-signal cooperative search task built on MPE2 particle physics."""

from __future__ import annotations

from typing import Literal

import numpy as np
from mpe2._mpe_utils.core import Agent, Landmark, World


RewardVariant = Literal["coverage", "signal", "shared_signal"]


class SignalSearchEnv:
    """Search hidden static targets using only local signal and coverage observations.

    Targets and 1x3 no-fly obstacles are sampled from several layouts at reset. Their coordinates are never
    placed in the policy observation. Every undiscovered target emits a local field:
    3 at its cell, 2 in nearby cells, 1 farther away, 0 elsewhere. The UAV sees only
    its 3x3 neighbourhood. UAVs share their coverage maps and found-target status
    perfectly in this baseline (communication interference is intentionally absent).
    """

    def __init__(
        self,
        num_agents: int = 2,
        num_targets: int = 3,
        grid_size: int = 30,
        max_steps: int = 420,
        reward_variant: RewardVariant = "signal",
        seed: int | None = None,
    ) -> None:
        if num_agents != 2 or num_targets != 3:
            raise ValueError("This baseline is intentionally fixed to two UAVs and three targets")
        self.num_agents = num_agents
        self.num_targets = num_targets
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.reward_variant = reward_variant
        self.rng = np.random.default_rng(seed)
        self.boundary = 2.25
        self.detect_radius = 0.18
        self.cell_width = 2.0 * self.boundary / grid_size
        self.world = self._make_world()
        self.discovered = np.zeros(num_targets, dtype=bool)
        self.covered = np.zeros((grid_size, grid_size), dtype=bool)
        self.obstacles = np.zeros((grid_size, grid_size), dtype=bool)
        self.known_coverage = np.zeros((num_agents, grid_size, grid_size), dtype=bool)
        self.previous_cells = np.full((num_agents, 2), -1, dtype=int)
        self.signal_peak = np.zeros(num_agents, dtype=np.float32)
        self.previous_actions = np.zeros((num_agents, 2), dtype=np.float32)
        self.steps = 0

    @property
    def action_dim(self) -> int:
        return 2

    @property
    def obs_dim(self) -> int:
        # self pos/vel, teammate rel-pos, found status, and local coverage/signal/obstacle
        # patches, followed by shared coverage fraction and UAV id. No global route hint is given.
        return 4 + 2 + self.num_targets + 9 + 9 + 9 + 1 + self.num_agents

    def _make_world(self) -> World:
        world = World()
        world.dim_c = 0
        world.dt = 0.1
        world.damping = 0.25
        world.agents = []
        for i in range(self.num_agents):
            agent = Agent()
            agent.name = f"uav_{i}"
            agent.size = 0.05
            agent.accel = 1.0
            agent.max_speed = 0.8
            agent.collide = False  # collisions are measured explicitly for stable physics
            agent.silent = True
            agent.color = np.array([0.15 + 0.45 * i, 0.4, 0.85 - 0.35 * i])
            world.agents.append(agent)
        world.landmarks = []
        for k in range(self.num_targets):
            target = Landmark()
            target.name = f"hidden_target_{k}"
            target.size = 0.035
            target.collide = False
            target.movable = False
            target.color = np.array([0.85, 0.16, 0.14])
            world.landmarks.append(target)
        return world

    def _layouts(self) -> list[np.ndarray]:
        return [
            np.array([[-1.75, 1.60], [1.55, 1.20], [0.20, -0.20]], dtype=np.float32),
            np.array([[-1.55, -0.15], [1.70, 1.75], [-0.25, 1.10]], dtype=np.float32),
            np.array([[-1.80, 0.35], [1.35, -1.00], [1.40, 1.65]], dtype=np.float32),
            np.array([[-1.20, 1.75], [0.48, -1.12], [1.85, 0.25]], dtype=np.float32),
            np.array([[-1.72, 1.10], [0.72, 1.62], [1.72, -0.62]], dtype=np.float32),
            np.array([[-0.78, -0.55], [1.42, 0.62], [-1.55, 1.78]], dtype=np.float32),
            np.array([[-1.68, 0.20], [0.40, 1.62], [1.65, -0.72]], dtype=np.float32),
            np.array([[-1.22, -0.78], [1.52, 1.25], [-1.65, 1.50]], dtype=np.float32),
        ]

    def _obstacle_samples(self) -> list[list[tuple[str, int, int, int, int]]]:
        """Eight obstacle maps: bars, blocks, L-shapes, and irregular combinations.

        Tuple formats: ("h", x, y, length, 1), ("v", x, y, 1, length),
        ("block", x, y, width, height), and ("L", x, y, width, height).
        """
        return [
            [("v", 7, 18, 1, 5), ("h", 16, 20, 5, 1), ("block", 23, 7, 2, 2), ("L", 11, 7, 4, 4), ("h", 20, 14, 3, 1)],
            [("v", 10, 7, 1, 4), ("h", 17, 16, 5, 1), ("block", 22, 20, 2, 3), ("L", 6, 19, 4, 4), ("v", 26, 7, 1, 4)],
            [("h", 9, 14, 4, 1), ("L", 18, 7, 4, 4), ("block", 23, 18, 3, 2), ("v", 5, 22, 1, 4), ("h", 14, 20, 4, 1)],
            [("L", 8, 16, 4, 5), ("v", 18, 12, 1, 4), ("block", 23, 22, 3, 2), ("h", 5, 9, 4, 1), ("L", 25, 15, 3, 4)],
            [("h", 9, 12, 4, 1), ("L", 17, 18, 4, 4), ("block", 24, 9, 3, 2), ("v", 5, 18, 1, 4), ("h", 14, 6, 4, 1), ("v", 21, 14, 1, 3)],
            [("L", 8, 16, 4, 4), ("h", 18, 11, 4, 1), ("block", 24, 20, 2, 3), ("v", 5, 12, 1, 4), ("h", 13, 7, 4, 1), ("L", 20, 15, 4, 3)],
            [("block", 8, 17, 3, 3), ("L", 17, 10, 5, 4), ("v", 24, 19, 1, 5), ("h", 4, 11, 5, 1), ("block", 13, 6, 2, 2), ("h", 21, 15, 4, 1)],
            [("L", 7, 14, 5, 5), ("h", 16, 21, 5, 1), ("block", 23, 8, 3, 3), ("v", 5, 22, 1, 3), ("L", 19, 10, 4, 5), ("block", 13, 6, 2, 3)],
        ]

    def _obstacle_mask(self, specifications: list[tuple[str, int, int, int, int]]) -> np.ndarray:
        mask = np.zeros((self.grid_size, self.grid_size), dtype=bool)
        for shape, x, y, width, height in specifications:
            if shape == "h":
                cells = [(x + offset, y) for offset in range(width)]
            elif shape == "v":
                cells = [(x, y + offset) for offset in range(height)]
            elif shape == "block":
                cells = [(x + dx, y + dy) for dx in range(width) for dy in range(height)]
            elif shape == "L":
                cells = [(x + dx, y) for dx in range(width)] + [(x, y + dy) for dy in range(height)]
            else:
                raise ValueError(f"Unknown obstacle shape: {shape}")
            for cell_x, cell_y in cells:
                mask[cell_x, cell_y] = True
        return mask

    def reset(self, seed: int | None = None) -> np.ndarray:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.steps = 0
        self.discovered[:] = False
        self.covered[:] = False
        self.obstacles[:] = False
        self.known_coverage[:] = False
        self.previous_cells[:] = -1
        self.signal_peak[:] = 0.0
        self.previous_actions[:] = 0.0
        # A shared take-off base. The 0.18 m separation prevents an artificial
        # collision at time zero while still making the coordination problem symmetric.
        starts = [np.array([-0.09, -1.90], dtype=np.float32), np.array([0.09, -1.90], dtype=np.float32)]
        for agent, pos in zip(self.world.agents, starts, strict=True):
            agent.state.p_pos = pos.copy()
            agent.state.p_vel = np.zeros(2, dtype=np.float32)
            agent.state.c = np.zeros(0, dtype=np.float32)
            agent.action.u = np.zeros(2, dtype=np.float32)
            agent.action.c = np.zeros(0, dtype=np.float32)
        layout_index = int(self.rng.integers(len(self._layouts())))
        layout = self._layouts()[layout_index]
        self.obstacles[:] = self._obstacle_mask(self._obstacle_samples()[layout_index])
        for target, pos in zip(self.world.landmarks, layout, strict=True):
            target.state.p_pos = pos.copy()
            target.state.p_vel = np.zeros(2, dtype=np.float32)
        for i, pos in enumerate(self._positions()):
            cell = self._cell_of(pos)
            self.previous_cells[i] = cell
            self.covered[cell[0], cell[1]] = True
            self.known_coverage[i, cell[0], cell[1]] = True
        return self._observations()

    def _positions(self) -> np.ndarray:
        return np.asarray([agent.state.p_pos for agent in self.world.agents], dtype=np.float32)

    def _cell_of(self, pos: np.ndarray) -> np.ndarray:
        scaled = (np.clip(pos, -self.boundary, self.boundary) + self.boundary) / (2.0 * self.boundary)
        return np.minimum((scaled * self.grid_size).astype(int), self.grid_size - 1)

    def _signal_value(self, cell: np.ndarray) -> float:
        center = -self.boundary + (cell.astype(np.float32) + 0.5) * self.cell_width
        value = 0.0
        for found, target in zip(self.discovered, self.world.landmarks, strict=True):
            if found:
                continue
            distance = float(np.linalg.norm(center - target.state.p_pos))
            if distance <= 0.18:
                value = max(value, 3.0)
            elif distance <= 0.42:
                value = max(value, 2.0)
            elif distance <= 0.72:
                value = max(value, 1.0)
        return value

    def _patch(self, matrix: np.ndarray, cell: np.ndarray) -> np.ndarray:
        values: list[float] = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                x, y = cell[0] + dx, cell[1] + dy
                values.append(float(matrix[x, y]) if 0 <= x < self.grid_size and 0 <= y < self.grid_size else 0.0)
        return np.asarray(values, dtype=np.float32)

    def _signal_patch(self, cell: np.ndarray) -> np.ndarray:
        signal = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                signal[x, y] = self._signal_value(np.array([x, y])) / 3.0
        return self._patch(signal, cell)

    def _observation(self, i: int) -> np.ndarray:
        pos = self.world.agents[i].state.p_pos.astype(np.float32)
        vel = self.world.agents[i].state.p_vel.astype(np.float32)
        other = 1 - i
        cell = self._cell_of(pos)
        identifier = np.zeros(self.num_agents, dtype=np.float32)
        identifier[i] = 1.0
        return np.concatenate(
            [
                pos,
                vel,
                self.world.agents[other].state.p_pos.astype(np.float32) - pos,
                self.discovered.astype(np.float32),
                self._patch(self.known_coverage[i].astype(np.float32), cell),
                self._signal_patch(cell),
                self._patch(self.obstacles.astype(np.float32), cell),
                np.array([np.logical_and(self.known_coverage[i], ~self.obstacles).sum() / (~self.obstacles).sum()], dtype=np.float32),
                identifier,
            ]
        ).astype(np.float32)

    def _observations(self) -> np.ndarray:
        return np.stack([self._observation(i) for i in range(self.num_agents)]).astype(np.float32)

    def _collision_counts(self) -> np.ndarray:
        counts = np.zeros(self.num_agents, dtype=np.float32)
        positions = self._positions()
        if np.linalg.norm(positions[0] - positions[1]) < 2 * self.world.agents[0].size:
            counts[:] = 1.0
        return counts

    def step(self, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray, bool, dict[str, float]]:
        actions = np.clip(np.asarray(actions, dtype=np.float32), -1.0, 1.0)
        applied_actions = 0.72 * actions + 0.28 * self.previous_actions
        self.previous_actions = applied_actions.copy()
        previous_positions = self._positions().copy()
        for agent, action in zip(self.world.agents, applied_actions, strict=True):
            agent.action.u = action * float(agent.accel)
            agent.action.c = np.zeros(0, dtype=np.float32)
        self.world.step()
        self.steps += 1
        obstacle_hits = np.zeros(self.num_agents, dtype=np.float32)
        for i, agent in enumerate(self.world.agents):
            agent.state.p_pos = np.clip(agent.state.p_pos, -self.boundary, self.boundary)
            cell = self._cell_of(agent.state.p_pos)
            if self.obstacles[cell[0], cell[1]]:
                agent.state.p_pos = previous_positions[i]
                agent.state.p_vel = np.zeros(2, dtype=np.float32)
                obstacle_hits[i] = 1.0
        positions = self._positions()

        new_cells = np.zeros(self.num_agents, dtype=np.float32)
        repeats = np.zeros(self.num_agents, dtype=np.float32)
        signal_gain = np.zeros(self.num_agents, dtype=np.float32)
        for i, pos in enumerate(positions):
            cell = self._cell_of(pos)
            current_signal = self._signal_value(cell)
            signal_gain[i] = max(0.0, current_signal - self.signal_peak[i])
            self.signal_peak[i] = max(self.signal_peak[i], current_signal)
            if not np.array_equal(cell, self.previous_cells[i]):
                if self.covered[cell[0], cell[1]]:
                    repeats[i] = 1.0
                else:
                    self.covered[cell[0], cell[1]] = True
                    new_cells[i] = 1.0
                self.previous_cells[i] = cell
            self.known_coverage[i, cell[0], cell[1]] = True

        target_reward = np.zeros(self.num_agents, dtype=np.float32)
        final_finder = -1
        for k, target in enumerate(self.world.landmarks):
            if self.discovered[k]:
                continue
            distances = np.linalg.norm(positions - target.state.p_pos, axis=1)
            nearby = np.flatnonzero(distances <= self.detect_radius)
            if len(nearby):
                finder = int(nearby[np.argmin(distances[nearby])])
                self.discovered[k] = True
                target_reward[finder] += 15.0
                final_finder = finder
                self.signal_peak[:] = 0.0

        # Normal communication baseline: each step the two maps and target status are shared.
        combined_map = self.known_coverage[0] | self.known_coverage[1]
        self.known_coverage[:] = combined_map
        collisions = self._collision_counts()
        movement = np.sum(applied_actions * applied_actions, axis=1)
        individual = target_reward + new_cells - 0.1 * repeats - 5.0 * collisions - 2.0 * obstacle_hits - 0.005 * movement - 0.01
        if self.reward_variant in {"signal", "shared_signal"}:
            individual += 0.8 * signal_gain
        success = bool(self.discovered.all())
        if success and final_finder >= 0:
            individual[final_finder] += 20.0
        if self.reward_variant == "shared_signal":
            rewards = np.full(self.num_agents, individual.sum(), dtype=np.float32)
        else:
            rewards = individual.astype(np.float32)
        done = success or self.steps >= self.max_steps
        info = {
            "success": float(success),
            "targets_found": float(self.discovered.sum()),
            "coverage": float(np.logical_and(self.covered, ~self.obstacles).sum() / (~self.obstacles).sum()),
            "repeat_events": float(repeats.sum()),
            "collisions": float(collisions.sum() / 2.0),
            "obstacle_hits": float(obstacle_hits.sum()),
            "steps": float(self.steps),
        }
        return self._observations(), rewards, done, info

    def render_data(self) -> dict[str, np.ndarray]:
        signal = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                signal[x, y] = self._signal_value(np.array([x, y]))
        return {
            "positions": self._positions().copy(),
            "targets": np.asarray([target.state.p_pos for target in self.world.landmarks], dtype=np.float32).copy(),
            "found": self.discovered.copy(),
            "coverage": self.covered.copy(),
            "obstacles": self.obstacles.copy(),
            "signal": signal,
        }
