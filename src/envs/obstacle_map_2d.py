"""
Kohei Honda, 2023.
"""

from __future__ import annotations

from typing import Callable, Tuple, List, Union
from dataclasses import dataclass
from math import ceil
from matplotlib import pyplot as plt
import torch
import numpy as np
from scipy.ndimage import distance_transform_edt as edt_transform

@dataclass
class CircleObstacle:
    """
    Circle obstacle used in the obstacle map.
    """

    center: np.ndarray
    radius: float

    def __init__(self, center: np.ndarray, radius: float) -> None:
        self.center = center
        self.radius = radius


@dataclass
class RectangleObstacle:
    """
    Rectangle obstacle used in the obstacle map.
    Not consider angle for now.
    """

    center: np.ndarray
    width: float
    height: float

    def __init__(self, center: np.ndarray, width: float, height: float) -> None:
        self.center = center
        self.width = width
        self.height = height


class ObstacleMap:
    def __init__(
        self,
        map_size: Tuple[int, int] = (20, 20),
        cell_size: float = 0.1,
        device=torch.device("cuda"),
        dtype=torch.float32,
        min_distance: float = 0.2
    ) -> None:
        self._device = device
        self._dtype = dtype
        self._cell_size = cell_size
        self.min_distance = min_distance

        # Initialize grid map
        cell_map_dim = (
            ceil(map_size[0] / cell_size),
            ceil(map_size[1] / cell_size)
        )
        self._map = np.zeros(cell_map_dim)
        self._cell_map_origin = np.zeros(2)
        self._cell_map_origin = np.array(
            [cell_map_dim[0] / 2, cell_map_dim[1] / 2]
        ).astype(int)
        
        # SDF properties
        self.sdf = None
        self.sdf_torch = None
        
        # Map boundaries
        self.x_lim = [-map_size[0]/2, map_size[0]/2]
        self.y_lim = [-map_size[1]/2, map_size[1]/2]
        
        # Obstacle lists
        self.circle_obs_list = []
        self.rectangle_obs_list = []

    def add_circle_obstacle(self, center: np.ndarray, radius: float) -> None:
        """
        Add a circle obstacle to the map.
        :param center: Center of the circle obstacle.
        :param radius: Radius of the circle obstacle.
        """
        assert len(center) == 2
        assert radius > 0

        # convert to cell map
        center_occ = (center / self._cell_size) + self._cell_map_origin
        center_occ = np.round(center_occ).astype(int)
        radius_occ = ceil(radius / self._cell_size)

        # add to occ map
        for i in range(-radius_occ, radius_occ + 1):
            for j in range(-radius_occ, radius_occ + 1):
                if i**2 + j**2 <= radius_occ**2:
                    i_bounded = np.clip(center_occ[0] + i, 0, self._map.shape[0] - 1)
                    j_bounded = np.clip(center_occ[1] + j, 0, self._map.shape[1] - 1)
                    self._map[i_bounded, j_bounded] = 1

        # add to circle obstacle list to use visualize
        self.circle_obs_list.append(CircleObstacle(center, radius))

    def add_rectangle_obstacle(
        self, center: np.ndarray, width: float, height: float
    ) -> None:
        """
        Add a rectangle obstacle to the map.
        :param center: Center of the rectangle obstacle.
        :param width: Width of the rectangle obstacle.
        :param height: Height of the rectangle obstacle.
        """
        assert len(center) == 2
        assert width > 0
        assert height > 0

        # convert to cell map
        center_occ = (center / self._cell_size) + self._cell_map_origin
        center_occ = np.ceil(center_occ).astype(int)
        width_occ = ceil(width / self._cell_size)
        height_occ = ceil(height / self._cell_size)

        # add to occ map
        x_init = center_occ[0] - ceil(width_occ / 2)
        x_end = center_occ[0] + ceil(width_occ / 2)
        y_init = center_occ[1] - ceil(height_occ / 2)
        y_end = center_occ[1] + ceil(height_occ / 2)

        # # deal with out of bound
        x_init = np.clip(x_init, 0, self._map.shape[0] - 1)
        x_end = np.clip(x_end, 0, self._map.shape[0] - 1)
        y_init = np.clip(y_init, 0, self._map.shape[1] - 1)
        y_end = np.clip(y_end, 0, self._map.shape[1] - 1)

        self._map[x_init:x_end, y_init:y_end] = 1

        # add to rectangle obstacle list to use visualize
        self.rectangle_obs_list.append(RectangleObstacle(center, width, height))

    def convert_to_torch(self) -> None:
        """Calculate SDF and convert to torch tensor"""
        # Calculate signed distance field
        obstacle_grid = self._map.astype(bool)
        self.sdf = edt_transform(obstacle_grid)# - edt_transform(obstacle_grid)
        self.sdf = self.sdf * self._cell_size  # Convert to meters
        
        # Convert to torch tensor
        self.sdf_torch = torch.tensor(self.sdf.T,  # Transpose for correct orientation
                                    device=self._device,
                                    dtype=self._dtype)

    def compute_cost(self, x: torch.Tensor) -> torch.Tensor:
        """Compute distance to nearest obstacle"""
        # Convert world coordinates to grid indices
        x_norm = (x[..., 0] - self.x_lim[0]) / self._cell_size
        y_norm = (x[..., 1] - self.y_lim[0]) / self._cell_size
        
        # Clamp indices to grid boundaries
        x_idx = torch.clamp(x_norm.long(), 0, self._map.shape[0]-1)
        y_idx = torch.clamp(y_norm.long(), 0, self._map.shape[1]-1)
        
        # Get distances from SDF
        distances = self.sdf_torch[y_idx, x_idx]  # Transposed access
        
        # Add penalty for out-of-bound points
        out_of_bounds = (x_norm < 0) | (x_norm >= self._map.shape[0]) | \
                        (y_norm < 0) | (y_norm >= self._map.shape[1])
        distances[out_of_bounds] = -self.min_distance
        
        return distances

    def render_occupancy(self, ax, cmap="binary") -> None:
        ax.imshow(self._map, cmap=cmap)

    def render_sdf(self, ax):
        im = ax.imshow(self.sdf.T, 
                    extent=[*self.x_lim, *self.y_lim],
                    cmap='coolwarm',
                    vmin=-self.min_distance, 
                    vmax=self.min_distance)
        #plt.colorbar(im, ax=ax)
    
    def render(self, ax, zorder: int = 0) -> None:
        """
        Render in continuous space.
        """
        ax.set_xlim(self.x_lim)
        ax.set_ylim(self.y_lim)
        ax.set_aspect("equal")

        # render circle obstacles
        for circle_obs in self.circle_obs_list:
            ax.add_patch(
                plt.Circle(
                    circle_obs.center, circle_obs.radius, color="gray", zorder=zorder
                )
            )

        # render rectangle obstacles
        for rectangle_obs in self.rectangle_obs_list:
            ax.add_patch(
                plt.Rectangle(
                    rectangle_obs.center
                    - np.array([rectangle_obs.width / 2, rectangle_obs.height / 2]),
                    rectangle_obs.width,
                    rectangle_obs.height,
                    color="gray",
                    zorder=zorder,
                )
            )


def generate_random_obstacles(
    obstacle_map: ObstacleMap,
    random_x_range: Tuple[float, float],
    random_y_range: Tuple[float, float],
    num_circle_obs: int,
    radius_range: Tuple[float, float],
    num_rectangle_obs: int,
    width_range: Tuple[float, float],
    height_range: Tuple[float, float],
    max_iteration: int,
    seed: int,
) -> None:
    """
    Generate random obstacles.
    """
    rng = np.random.default_rng(seed)

    # if random range is larger than map size, use map size
    if random_x_range[0] < obstacle_map.x_lim[0]:
        random_x_range[0] = obstacle_map.x_lim[0]
    if random_x_range[1] > obstacle_map.x_lim[1]:
        random_x_range[1] = obstacle_map.x_lim[1]
    if random_y_range[0] < obstacle_map.y_lim[0]:
        random_y_range[0] = obstacle_map.y_lim[0]
    if random_y_range[1] > obstacle_map.y_lim[1]:
        random_y_range[1] = obstacle_map.y_lim[1]

    for i in range(num_circle_obs):
        num_trial = 0
        while num_trial < max_iteration:
            center_x = rng.uniform(random_x_range[0], random_x_range[1])
            center_y = rng.uniform(random_y_range[0], random_y_range[1])
            center = np.array([center_x, center_y])
            radius = rng.uniform(radius_range[0], radius_range[1])

            # overlap check
            is_overlap = False
            for circle_obs in obstacle_map.circle_obs_list:
                if (
                    np.linalg.norm(circle_obs.center - center)
                    <= circle_obs.radius + radius
                ):
                    is_overlap = True

            for rectangle_obs in obstacle_map.rectangle_obs_list:
                if (
                    np.linalg.norm(rectangle_obs.center - center)
                    <= rectangle_obs.width / 2 + radius
                ):
                    if (
                        np.linalg.norm(rectangle_obs.center - center)
                        <= rectangle_obs.height / 2 + radius
                    ):
                        is_overlap = True

            if not is_overlap:
                break

            num_trial += 1

            if num_trial == max_iteration:
                raise RuntimeError(
                    "Cannot generate random obstacles due to reach max iteration."
                )

        obstacle_map.add_circle_obstacle(center, radius)

    for i in range(num_rectangle_obs):
        num_trial = 0
        while num_trial < max_iteration:
            center_x = rng.uniform(random_x_range[0], random_x_range[1])
            center_y = rng.uniform(random_y_range[0], random_y_range[1])
            center = np.array([center_x, center_y])
            width = rng.uniform(width_range[0], width_range[1])
            height = rng.uniform(height_range[0], height_range[1])

            # overlap check
            is_overlap = False
            for circle_obs in obstacle_map.circle_obs_list:
                if (
                    np.linalg.norm(circle_obs.center - center)
                    <= circle_obs.radius + width / 2
                ):
                    if (
                        np.linalg.norm(circle_obs.center - center)
                        <= circle_obs.radius + height / 2
                    ):
                        is_overlap = True

            for rectangle_obs in obstacle_map.rectangle_obs_list:
                if (
                    np.linalg.norm(rectangle_obs.center - center)
                    <= rectangle_obs.width / 2 + width / 2
                ):
                    if (
                        np.linalg.norm(rectangle_obs.center - center)
                        <= rectangle_obs.height / 2 + height / 2
                    ):
                        is_overlap = True

            if not is_overlap:
                break

            num_trial += 1

            if num_trial == max_iteration:
                raise RuntimeError(
                    "Cannot generate random obstacles due to reach max iteration."
                )

        obstacle_map.add_rectangle_obstacle(center, width, height)
