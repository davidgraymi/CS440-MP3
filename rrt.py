from __future__ import annotations

from robot import Robot
import numpy as np

TreePath = list[tuple[np.ndarray, np.ndarray | None]]
ExpansionResult = list[tuple[np.ndarray, np.ndarray]]

class GuidedTreeSearch:
    def __init__(self, robot: Robot, goal_tolerance: float) -> None:
        self.robot = robot
        # all the configurations of the tree (i.e., vertices), shape (num_nodes, num_dims)
        self.tree_configs = np.empty((0, robot.cspace.cspace_boundary.shape[0]))
        # mapping from tree node index to (state, action_from_parent, parent_index)
        # usually we would store children not parents to represent a tree but the main operation we care about is backtracking 
        # from a leaf not traversing down from the root, so storing parent indices is more efficient for our use case
        self.idx_to_state = {}

        self.goal_tolerance = goal_tolerance
        self.goal_config = None

    def is_goal(self, config: np.ndarray) -> bool:
        if self.goal_config is None:
            raise ValueError("Goal config not set yet, cannot check if config is goal. Must call initialize_search first.")
        return self.robot.cspace.point_to_point_distance(config, self.goal_config) <= self.goal_tolerance

    def _initialize_search(self, starting_config: np.ndarray, goal_config: np.ndarray) -> None:
        self.tree_configs = np.copy(starting_config).reshape(1, -1) # shape (num_nodes, num_dims)
        # config, control_from_parent, parent_idx
        self.idx_to_state = {0: (starting_config, None, None)}
        self.goal_config = goal_config

    # returns numpy array of shape (path_length, num_dims) from root of tree to given leaf index
    def backtrack_tree(self, leaf_idx: int) -> TreePath:
        path = []
        current = leaf_idx
        while current is not None:
            state, action, parent_idx = self.idx_to_state[current]
            path.append((state, action))
            current = parent_idx
        return path[::-1]  # reverse to get path from start to goal

    def add_to_tree(self, config: np.ndarray, action: np.ndarray | None, parent_idx: int | None) -> int:
        new_idx = len(self.tree_configs)
        self.tree_configs = np.vstack([self.tree_configs, config])
        self.idx_to_state[new_idx] = (config, action, parent_idx)
        return new_idx

    def select_node(self) -> int:
        raise NotImplementedError("Must implement select_node method in subclass to define how to select a node from the tree for expansion")
    
    def expand_node(self, node_idx: int) -> ExpansionResult:
        raise NotImplementedError("Must implement expand_node method in subclass to define how to get neighbors of a node")

    def search(self, starting_config: np.ndarray, goal_config: np.ndarray, max_iterations: int) -> TreePath:
        self._initialize_search(starting_config, goal_config)
        for iter_idx in range(max_iterations):
            print(f"Iteration {iter_idx}: Tree has {len(self.tree_configs)} nodes", end="\r")
            node_idx = self.select_node()
            new_configs = self.expand_node(node_idx)
            for config, action in new_configs:
                new_idx = self.add_to_tree(config, action, node_idx)
                if self.is_goal(config):
                    return self.backtrack_tree(new_idx)
        print(f"Reached max iterations ({max_iterations}) without finding a path")
        return []  # failed to find a path within max_iterations


class KinodynamicRRT(GuidedTreeSearch):
    def __init__(self, robot: Robot, goal_tolerance: float, control_duration: float = 1.0,
                 goal_sample_prob: float = 0.1, num_control_samples: int = 10) -> None:
        super().__init__(robot, goal_tolerance)
        self.control_duration = control_duration
        self.goal_sample_prob = goal_sample_prob
        self.num_control_samples = num_control_samples
        # we store the random sample used for selection to reuse it during expansion
        self.random_sample = None

    # (VI): implement select_node and expand_node for Kinodynamic RRT
    
    # Sample either the goal or one cspace configuration, then return the tree
    # node nearest to that sample using wrapped cspace distance
    def select_node(self) -> int:
        prob = np.random.random_sample()
        if prob <=  self.goal_sample_prob:
            self.random_sample = self.goal_config
        else:
            self.random_sample = self.robot.cspace.sample_n_configs_in_boundary(1)

        node = np.argmin(
            self.robot.cspace.point_to_point_distance(
                self.tree_configs,
                self.random_sample
            )
        )
        return node

    # Sample one batch of controls, simulate each for control_duration, reject
    # invalid trajectories, then return the valid endpoint closest to
    # self.random_sample without drawing replacements
    # Return [] if no sampled control is valid, otherwise return [(config, control)]
    def expand_node(self, node_idx: int) -> ExpansionResult:
        ctrls = self.robot.sample_random_controls(self.num_control_samples)
        node = self.tree_configs[node_idx, :]
        closest_node = None
        closest_dist = None
        closest_control = None
        for control in ctrls:
            trajectory = self.robot.apply_constant_control(
                node,
                control,
                self.control_duration
            )

            if any([not self.robot.cspace.is_valid(inter) for inter in trajectory]):
                continue

            dist = self.robot.cspace.point_to_point_distance(
                trajectory[-1],
                self.random_sample
            )

            if closest_node is None or dist < closest_dist:
                closest_node = trajectory[-1]
                closest_dist = dist
                closest_control = control

        if closest_node is None or closest_control is None:
            return []

        return [(closest_node, closest_control)]
