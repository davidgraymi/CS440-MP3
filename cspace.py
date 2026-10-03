from __future__ import annotations

import json

import numpy as np
from shapely.geometry import Point, Polygon
from matplotlib.axes import Axes

# Return True if an n-dimensional point is inside an n-dimensional boundary
# Boundary has shape (n, 2), where each row is [min, max]
# Touching the boundary counts as being inside
def point_in_boundary(x: np.ndarray, boundary: np.ndarray) -> bool:
    # TODO(III.1): implement this function, don't use a for loop
    pass

# Abstract base class for a Configuration Space with a 2D workspace
class CSpace2D:
    def __init__(self, workspace_boundary: np.ndarray, cspace_boundary: np.ndarray,
                 is_angular: np.ndarray, obstacles: list[np.ndarray],
                 interpolation_delta: float = 0.1) -> None:
        # shape (2, 2), [[x_min, x_max], [y_min, y_max]]
        self.workspace_boundary = workspace_boundary
        # shape (n, 2), [[dim1_min, dim1_max], ..., [dimn_min, dimn_max]]
        self.cspace_boundary = cspace_boundary
        # boolean array of shape (n,), True if dimension i is angular, meaning it wraps around
        self.is_angular = is_angular
        # list of obstacles, each obstacle i is an array of shape (m_i, 2) 
        # representing its vertices in workspace in counter-clockwise order
        self.obstacles = obstacles
        # a resolution parameter for how finely we check collision along motion between two configurations
        self.interpolation_delta = interpolation_delta

    # draw the workspace with boundary and obstacles on given Axes
    def draw_workspace(self, ax: Axes) -> None:
        ax.set_xlim(self.workspace_boundary[0])
        ax.set_ylim(self.workspace_boundary[1])
        # draw obstacles boundaries in black
        for obstacle in self.obstacles:
            # close the polygon by repeating the first point at the end
            ax.plot(np.append(obstacle[:,0], obstacle[0,0]), np.append(obstacle[:,1], obstacle[0,1]), color="black")

    # Return direction vectors from start to end, taking the short way around angular dimensions
    # start and end must have the same final dimension, but they do not need the same shape:
    # for example, start can be a batch of points with shape (m, n) while end has shape (n,)
    def point_to_point_direction(self, start: np.ndarray, end: np.ndarray) -> np.ndarray:
        assert start.shape[-1] == end.shape[-1], "start and end must have the same final dimension"
        direction = np.asarray(end, dtype=float) - np.asarray(start, dtype=float)
        if len(direction.shape) == 1:
            direction[self.is_angular] = (direction[self.is_angular] + np.pi) % (2 * np.pi) - np.pi
            return direction
        else:
            # can replace this ":" with "..." to allow broadcasting for any number of leading dimensions, 
            # but we will just assume the first dimension is the batch dimension for simplicity
            direction[:, self.is_angular] = (direction[:, self.is_angular] + np.pi) % (2 * np.pi) - np.pi
            return direction
    
    # The Euclidean distance between two points, taking the short way around angular dimensions
    def point_to_point_distance(self, start: np.ndarray, end: np.ndarray) -> np.ndarray:
        direction = self.point_to_point_direction(start, end)
        return np.linalg.norm(direction, axis=-1)

    # Return n uniformly random configurations shaped (n, cspace_dim)
    def sample_n_configs_in_boundary(self, n: int) -> np.ndarray:
        # TODO(III.2): implement this function, don't use a for loop
        return np.zeros((n, self.cspace_boundary.shape[0]))
    
    # Return intermediate configurations defining an unvalidated straight-line motion 
    # from start_config to end_config, excluding the endpoints
    def straight_line_local_planner(self, start_config: np.ndarray,
                                    end_config: np.ndarray) -> np.ndarray | list[np.ndarray]:
        # TODO(III.5): implement this function
        pass

    # generic implementation that checks the intermediate configurations returned by the local planner
    # callers are responsible for validating both endpoints
    def is_valid_edge(self, start_config: np.ndarray, end_config: np.ndarray) -> bool:
        intermediates = self.straight_line_local_planner(start_config, end_config)
        for config in intermediates:
            if not self.is_valid(config):
                return False
        return True
    
    # draw the robot at given configuration on given Axes
    def draw_state(self, ax: Axes, config: np.ndarray) -> None:
        pass  # to be implemented in subclasses

    # get workspace points corresponding to the given configuration
    def forward_kinematics(self, config: np.ndarray) -> np.ndarray | list[np.ndarray]:
        pass  # to be implemented in subclasses

    # check if the robot at given configuration is valid (inside boundary and not colliding with obstacles)
    def is_valid(self, config: np.ndarray) -> bool:
        pass # to be implemented in subclasses

# Polygonal CSpace where the robot and obstacles are interpreted as (shapely) polygons 
# for the purpose of collision checking (not necessarily efficient but simple and generic)
class PolygonalCSpace(CSpace2D):
    def __init__(self, workspace_boundary: np.ndarray, cspace_boundary: np.ndarray,
                 is_angular: np.ndarray, obstacles: list[np.ndarray],
                 interpolation_delta: float = 0.1) -> None:
        super().__init__(workspace_boundary, cspace_boundary, is_angular, obstacles, interpolation_delta)
        # a list of shapely Polygons representing obstacles
        self.shapely_obstacles = [Polygon(o) for o in obstacles]

    # A configuration is valid if it is in the cspace boundary, the robot is
    # inside the workspace boundary, and the robot does not collide with any obstacle
    def is_valid(self, config: np.ndarray) -> bool:
        # TODO(III.4): implement this function
        return False

# A PolygonalCSpace where the robot is a rectangle that can translate and rotate in the plane (x,y,theta configuration space)
# Dynamics are handled by Robot classes, not by CSpace
class RectangularCSpace(PolygonalCSpace):
    def __init__(self, workspace_boundary: np.ndarray, cspace_boundary: np.ndarray,
                 is_angular: np.ndarray | None = None, obstacles: list[np.ndarray] = [],
                 rectangle_height: float = 1.0, rectangle_width: float = 2.0,
                 interpolation_delta: float = 0.1) -> None:
        assert cspace_boundary.shape[0] == 3, "Expected 3D configuration space"  # we expect (x,y,theta) configuration space

        if is_angular is None:
            print("Assuming third dimension is angular since is_angular not provided, overriding cspace_boundary to be [0, 2pi] for third dimension")
            is_angular = np.array([False, False, True])  # only the angle dimension is angular
            cspace_boundary[2] = np.array([0, 2 * np.pi])  # override to ensure angle wraps around correctly
        super().__init__(workspace_boundary, cspace_boundary, is_angular, obstacles, interpolation_delta)
        
        self.rectangle_height = rectangle_height
        self.rectangle_width = rectangle_width

    # draw the car at given configuration on given Axes
    def draw_state(self, ax: Axes, config: np.ndarray, **kwargs: object) -> None:
        # do forward kinematics to get the four corners of the rectangle in workspace
        corners = self.forward_kinematics(config)
        # close the rectangle by repeating the first corner at the end
        ax.plot(corners[[0,1,2,3,0],0], corners[[0,1,2,3,0],1], **kwargs)
        # draw an arrow to indicate orientation
        arrow_length = min(self.rectangle_width, self.rectangle_height) / 2
        ax.arrow(config[0], config[1], arrow_length * np.cos(config[2]), arrow_length * np.sin(config[2]), 
                 head_width=arrow_length/2, head_length=arrow_length/2, fc='k', ec='k')
        
    # Return the four corners of the rectangle at configuration (x, y, theta)
    def forward_kinematics(self, config: np.ndarray) -> np.ndarray:
        # TODO(III.3): implement this function
        W, H = self.rectangle_width, self.rectangle_height
        # 1. corners of rectangle centered at origin
        corners = np.array([[-W/2,-H/2],
                            [-W/2, H/2],
                            [ W/2, H/2],
                            [ W/2,-H/2]])
        return corners

# Load a problem from a JSON file and return the CSpace, start, goal, and data dictionary
def load_problem(params_file: str) -> tuple[RectangularCSpace, np.ndarray, np.ndarray, dict]:
    with open(params_file, "r") as f:
        data = json.load(f)

    workspace_boundary = np.array([
        [data["boundary"]["x_min"], data["boundary"]["x_max"]],
        [data["boundary"]["y_min"], data["boundary"]["y_max"]],
    ])
    cspace_boundary = np.concatenate(
        [workspace_boundary, np.array([[0, 2 * np.pi]])],
        axis=0,
    )
    obstacles = [np.array(o) for o in data["obstacles"]]
    cspace = RectangularCSpace(
        workspace_boundary=workspace_boundary,
        cspace_boundary=cspace_boundary,
        is_angular=np.array([False, False, True]),
        obstacles=obstacles,
        rectangle_height=data["robot_height"],
        rectangle_width=data["robot_width"],
        interpolation_delta=data["dt"],
    )
    start = np.array(data["start"], dtype=np.float64)
    goal = np.array(data["goal"], dtype=np.float64)
    return cspace, start, goal, data

# Some example basic tests for CSpace functionality - we encourage you to also write some of your own
# You can run this file directly to see the results
def cspace_functionality_tests(cspace: CSpace2D, ax: Axes) -> None:
    print("Testing CSpace helper functions...")

    boundary = cspace.cspace_boundary
    assert point_in_boundary(boundary[:, 0], boundary)
    assert point_in_boundary(boundary[:, 1], boundary)
    assert not point_in_boundary(boundary[:, 0] - 0.1, boundary)
    assert not point_in_boundary(boundary[:, 1] + 0.1, boundary)

    wrapped_start = np.array([1.0, 1.0, 0.1])
    wrapped_goal = np.array([1.0, 1.0, 2 * np.pi - 0.1])
    assert cspace.point_to_point_distance(wrapped_start, wrapped_goal) < 0.25

    random_configs = cspace.sample_n_configs_in_boundary(10)
    assert random_configs.shape == (10, cspace.cspace_boundary.shape[0])
    assert all(point_in_boundary(config, cspace.cspace_boundary) for config in random_configs)

    corners = [cspace.forward_kinematics(config) for config in random_configs]
    assert all(corner_set.shape == (4, 2) for corner_set in corners)

    edge_configs = cspace.straight_line_local_planner(wrapped_start, wrapped_goal)
    edge_configs = np.asarray(edge_configs)
    assert edge_configs.shape == (1, 3)
    assert np.allclose(edge_configs[0], [1.0, 1.0, 0.0])

    reverse_edge_configs = np.asarray(cspace.straight_line_local_planner(wrapped_goal, wrapped_start))
    assert reverse_edge_configs.shape == (1, 3)
    assert np.allclose(reverse_edge_configs[0], [1.0, 1.0, 0.0])

    configs = np.array([
        [4.5, 1, 0],
        [3, 3, np.pi / 4],
        [2, 8, 3 * np.pi / 4],
        [5, 7, np.pi / 4],
        [9.5, 9.75, 3 * np.pi / 2],
        [10.1, 1, 0],
        [2, 2, -0.1],
    ])
    expected = [False, False, True, False, False, False, False]
    for config, expected_valid in zip(configs, expected):
        actual_valid = cspace.is_valid(config)
        print(f"  {config}: expected {expected_valid}, got {actual_valid}")
        cspace.draw_state(ax, config, color="black" if actual_valid == expected_valid else "red")

    ax.scatter([], [], color="black", label="Correct CSpace checks")
    ax.scatter([], [], color="red", label="Incorrect CSpace checks")
    print("CSpace sanity checks completed. Add your own tests here as you debug.")


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    params_file = "data/clutter.json"
    cspace, start, goal, _ = load_problem(params_file)
    fig, ax = plt.subplots(figsize=(5, 5))
    cspace.draw_workspace(ax)
    cspace.draw_state(ax, start, color="blue", label="Start")
    cspace.draw_state(ax, goal, color="green", label="Goal")
    cspace_functionality_tests(cspace, ax)
    ax.set_aspect("equal", adjustable="box")
    ax.legend()
    plt.show()
