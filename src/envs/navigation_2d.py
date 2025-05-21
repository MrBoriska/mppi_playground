"""
Kohei Honda, 2023.
"""

from __future__ import annotations

from typing import Tuple, Union
from matplotlib import pyplot as plt

import torch
import numpy as np
import os


from moviepy.video.io.ImageSequenceClip import ImageSequenceClip

from envs.obstacle_map_2d import ObstacleMap, generate_random_obstacles


@torch.jit.script
def angle_normalize(x):
    return ((x + torch.pi) % (2 * torch.pi)) - torch.pi


class Navigation2DEnv:
    def __init__(
        self, device=torch.device("cuda"), dtype=torch.float32, seed: int = 42,
        min_distance: float = 0.2
    ) -> None:
        # device and dtype
        if torch.cuda.is_available() and device == torch.device("cuda"):
            self._device = torch.device("cuda")
        else:
            self._device = torch.device("cpu")
        self._dtype = dtype
        self.min_distance = min_distance

        self._obstacle_map = ObstacleMap(
            map_size=(20, 20), cell_size=0.1, device=self._device, dtype=self._dtype, min_distance=self.min_distance
        )
        self._seed = seed

        generate_random_obstacles(
            obstacle_map=self._obstacle_map,
            random_x_range=(-7.5, 7.5),
            random_y_range=(-7.5, 7.5),
            num_circle_obs=7,
            radius_range=(1, 1),
            num_rectangle_obs=7,
            width_range=(2, 2),
            height_range=(2, 2),
            max_iteration=1000,
            seed=seed,
        )
        self._obstacle_map.convert_to_torch()

        self._start_pos = torch.tensor(
            [-9.0, -9.0], device=self._device, dtype=self._dtype
        )
        self._goal_pos = torch.tensor(
            [9.0, 9.0], device=self._device, dtype=self._dtype
        )

        self._robot_state = torch.zeros(3, device=self._device, dtype=self._dtype)
        self._robot_state[:2] = self._start_pos
        self._robot_state[2] = angle_normalize(
            torch.atan2(
                self._goal_pos[1] - self._start_pos[1],
                self._goal_pos[0] - self._start_pos[0],
            )
        )

        # u: [v, omega] (m/s, rad/s)
        self.u_min = torch.tensor([-0.8, -0.2, -1.0], device=self._device, dtype=self._dtype)
        self.u_max = torch.tensor([0.8, 0.2, 1.0], device=self._device, dtype=self._dtype)
        
        self.robot_shape = torch.tensor([
            [-(0.5-0.2), 0.0],
            [0.0, 0.0],
            [(0.5-0.2), 0.0]
        ], device=self._device, dtype=self._dtype)

    def reset(self) -> torch.Tensor:
        """
        Reset robot state.
        Returns:
            torch.Tensor: shape (3,) [x, y, theta]
        """
        self._robot_state[:2] = self._start_pos
        self._robot_state[2] = angle_normalize(
            torch.atan2(
                self._goal_pos[1] - self._start_pos[1],
                self._goal_pos[0] - self._start_pos[0],
            )
        )

        self._fig = plt.figure(layout="tight")
        self._ax = self._fig.add_subplot()
        self._ax.set_xlim(self._obstacle_map.x_lim)
        self._ax.set_ylim(self._obstacle_map.y_lim)
        self._ax.set_aspect("equal")

        self._rendered_frames = []

        return self._robot_state

    def step(self, u: torch.Tensor) -> Tuple[torch.Tensor, bool]:
        """
        Update robot state based on differential drive dynamics.
        Args:
            u (torch.Tensor): control batch tensor, shape (2) [v, omega]
        Returns:
            Tuple[torch.Tensor, bool]: Tuple of robot state and is goal reached.
        """
        u = torch.clamp(u, self.u_min, self.u_max)

        self._robot_state = self.dynamics(
            state=self._robot_state.unsqueeze(0), action=u.unsqueeze(0)
        ).squeeze(0)

        # goal check
        goal_threshold = 0.5
        is_goal_reached = (
            torch.norm(self._robot_state[:2] - self._goal_pos) < goal_threshold
        )

        return self._robot_state, is_goal_reached

    def render(
        self,
        predicted_trajectory: torch.Tensor = None,
        is_collisions: torch.Tensor = None,
        top_samples: Tuple[torch.Tensor, torch.Tensor] = None,
        mode: str = "human",
    ) -> None:
        self._ax.set_xlabel("x [m]")
        self._ax.set_ylabel("y [m]")

        # obstacle map
        self._obstacle_map.render(self._ax, zorder=10)
        #self._obstacle_map.render_sdf(self._ax)

        # start and goal
        self._ax.scatter(
            self._start_pos[0].item(),
            self._start_pos[1].item(),
            marker="o",
            color="red",
            zorder=10,
        )
        self._ax.scatter(
            self._goal_pos[0].item(),
            self._goal_pos[1].item(),
            marker="o",
            color="orange",
            zorder=10,
        )

        # robot
        # self._ax.scatter(
        #     self._robot_state[0].item(),
        #     self._robot_state[1].item(),
        #     marker="o",
        #     color="green",
        #     zorder=100,
        # )
        # Получаем текущее состояние робота [x, y, theta]
        x = self._robot_state[0].item()
        y = self._robot_state[1].item()
        theta = self._robot_state[2].item()

        # Преобразуем форму робота в numpy array (3, 2)
        robot_shape = self.robot_shape.detach().cpu().numpy()

        # Вычисляем преобразование координат для каждой точки
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)

        # Преобразование всех точек
        global_points = np.array([
            [
                x + (p[0] * cos_theta - p[1] * sin_theta),
                y + (p[0] * sin_theta + p[1] * cos_theta)
            ] for p in robot_shape
        ])

        # Разделяем координаты X и Y
        xs = global_points[:, 0]
        ys = global_points[:, 1]

        # Рисуем три точки формы и центр
        self._ax.scatter(
            xs,
            ys,
            marker="o",
            color="green",
            s=40,  # Размер точек
            zorder=100
        )

        # Центральная точка (опционально)
        self._ax.scatter(
            x,
            y,
            marker="o",
            color="red",
            s=30,
            zorder=101
        )

        # visualize top samples with different alpha based on weights
        if top_samples is not None:
            top_samples, top_weights = top_samples
            top_samples = top_samples.cpu().numpy()
            top_weights = top_weights.cpu().numpy()
            top_weights = 0.7 * top_weights / np.max(top_weights)
            top_weights = np.clip(top_weights, 0.1, 0.7)
            for i in range(top_samples.shape[0]):
                self._ax.plot(
                    top_samples[i, :, 0],
                    top_samples[i, :, 1],
                    color="lightblue",
                    alpha=top_weights[i],
                    zorder=1,
                )

        # predicted trajectory
        if predicted_trajectory is not None:
            # if is collision color is red
            colors = np.array(["darkblue"] * predicted_trajectory.shape[1])
            if is_collisions is not None:
                is_collisions = is_collisions.cpu().numpy()
                is_collisions = np.any(is_collisions, axis=0)
                colors[is_collisions] = "red"

            self._ax.scatter(
                predicted_trajectory[0, :, 0].cpu().numpy(),
                predicted_trajectory[0, :, 1].cpu().numpy(),
                color=colors,
                marker="o",
                s=3,
                zorder=2,
            )

        if mode == "human":
            # online rendering
            plt.pause(0.001)
            plt.cla()
        elif mode == "rgb_array":
            # offline rendering for video
            # TODO: high resolution rendering
            self._fig.canvas.draw()
            data = np.frombuffer(self._fig.canvas.tostring_rgb(), dtype=np.uint8)
            data = data.reshape(self._fig.canvas.get_width_height()[::-1] + (3,))
            plt.cla()
            self._rendered_frames.append(data)

    def close(self, path: str = None) -> None:
        if path is None:
            # mkdir video if not exists

            if not os.path.exists("video"):
                os.mkdir("video")
            path = "video/" + "navigation_2d_" + str(self._seed) + ".gif"

        if len(self._rendered_frames) > 0:
            # save animation
            clip = ImageSequenceClip(self._rendered_frames, fps=10)
            # clip.write_videofile(path, fps=10)
            clip.write_gif(path, fps=10)

    def dynamics(
        self, state: torch.Tensor, action: torch.Tensor, delta_t: float = 0.25
    ) -> torch.Tensor:
        """
        Update robot state based on omnidirectional dynamics.
        Args:
            state (torch.Tensor): state batch tensor, shape (batch_size, 3) [x, y, theta]
            action (torch.Tensor): control batch tensor, shape (batch_size, 2) [v, omega]
            delta_t (float): time step interval [s]
        Returns:
            torch.Tensor: shape (batch_size, 3) [x, y, theta]
        """

        # Perform calculations as before
        x = state[:, 0].view(-1, 1)
        y = state[:, 1].view(-1, 1)
        theta = state[:, 2].view(-1, 1)
        vx = torch.clamp(action[:, 0].view(-1, 1), self.u_min[0], self.u_max[0])
        vy = torch.clamp(action[:, 1].view(-1, 1), self.u_min[1], self.u_max[1])
        omega = torch.clamp(action[:, 2].view(-1, 1), self.u_min[2], self.u_max[2])
        theta = angle_normalize(theta)

        new_x = x + vx * torch.cos(theta) * delta_t - vy * torch.sin(theta) * delta_t
        new_y = y + vx * torch.sin(theta) * delta_t + vy * torch.cos(theta) * delta_t
        new_theta = angle_normalize(theta + omega * delta_t)

        # Clamp x and y to the map boundary
        x_lim = torch.tensor(
            self._obstacle_map.x_lim, device=self._device, dtype=self._dtype
        )
        y_lim = torch.tensor(
            self._obstacle_map.y_lim, device=self._device, dtype=self._dtype
        )
        clamped_x = torch.clamp(new_x, x_lim[0], x_lim[1])
        clamped_y = torch.clamp(new_y, y_lim[0], y_lim[1])

        result = torch.cat([clamped_x, clamped_y, new_theta], dim=1)

        return result

    def cost_function(self, state: torch.Tensor, action: torch.Tensor, info: dict) -> torch.Tensor:
        """
        Calculate cost function with robot shape consideration
        """
        # Calculate goal cost
        goal_cost = torch.norm(state[:, :2] - self._goal_pos, dim=1)
        
        # Get distances for all robot shape points
        distances = self.get_distances(state.unsqueeze(1))  # Add trajectory dimension
        
        # Calculate obstacle proximity cost
        #safety_margin = self.min_distance
        obstacle_penalty = torch.clamp(self.min_distance - distances, min=0)
        
        # Sum penalties across all points and time steps
        obstacle_cost = distances.pow(2).sum(dim=1)
        
        # Combine costs
        total_cost = goal_cost + 1000 * obstacle_penalty.pow(2).sum(dim=1) + 10.0/obstacle_cost
        
        return total_cost

    def get_distances(self, state: torch.Tensor) -> torch.Tensor:
        """
        Calculate distances for robot shape points (batched version)
        Args:
            state: (batch_size, traj_size, 3) tensor with [x, y, theta]
        Returns:
            distances: (batch_size, traj_size*3) tensor of distances
        """
        batch_size, traj_size = state.shape[:2]
        
        # Calculate transformed points
        robot_shape = self.robot_shape.to(state.device)
        dx = robot_shape[:, 0].view(1, 1, -1, 1)
        dy = robot_shape[:, 1].view(1, 1, -1, 1)
        
        theta = state[..., 2]
        cos_theta = torch.cos(theta).unsqueeze(-1).unsqueeze(-1)
        sin_theta = torch.sin(theta).unsqueeze(-1).unsqueeze(-1)
        
        # Global coordinates calculation
        global_x = state[..., 0].unsqueeze(-1).unsqueeze(-1) + dx * cos_theta - dy * sin_theta
        global_y = state[..., 1].unsqueeze(-1).unsqueeze(-1) + dx * sin_theta + dy * cos_theta
        
        # Combine and reshape for distance query
        pos_points = torch.cat([global_x, global_y], dim=-1)
        pos_points = pos_points.view(batch_size, traj_size * 3, 2)
        
        # Query distances from obstacle map
        return self._obstacle_map.compute_cost(pos_points)
    
    
    def collision_check(self, state: torch.Tensor) -> torch.Tensor:
        """
        Check collisions considering safety margin
        """
        distances = self.get_distances(state)
        return (distances < self.min_distance).any(dim=1)
