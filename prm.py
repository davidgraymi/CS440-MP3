from __future__ import annotations

from typing import Callable

from state import AbstractState
from cspace import CSpace2D, point_in_boundary
from search import best_first_search
import numpy as np

from shapely.geometry import Point, Polygon

# A version of PRM that only "pre-processes" the the configuration space by sampling a single roadmap graph
# for future queries the graph is static, so if no path exists in the graph from start to goal we fail instead of trying to add new nodes to the graph
class OneShotPRM:
    def __init__(self, cspace: CSpace2D, num_samples: int, num_neighbors: int) -> None:
        self.cspace = cspace
        self.num_samples = num_samples
        self.num_neighbors = num_neighbors
        # for bookkeeping of known graph distances so that future queries can be answered faster
        self.known_graph_distances = {}

        # (VII): draw one batch, keep its valid samples, and validate only
        # the nearest candidate edges for each retained vertex
        # self.vertices has shape (num_VALID_samples, num_dims)
        samples = cspace.sample_n_configs_in_boundary(num_samples)
        valid = np.array([cspace.is_valid(sample) for sample in samples], dtype=bool)
        self.vertices = samples[valid]

        num_vertices = len(self.vertices)
        k = min(self.num_neighbors, max(0, num_vertices - 1))

        # dict mapping vertex index to list of (neighbor_index, edge_length) of length AT MOST num_neighbors
        self.edges = {i: [] for i in range(len(self.vertices))}
        if k == 0:
            return

        for i in range(len(self.vertices)):
            cur = self.vertices[i]
            dists = cspace.point_to_point_distance(self.vertices, cur)
            dists[i] = np.inf
            candidate_indices = np.argpartition(dists, k)[:k]
            for j in candidate_indices:
                nbr = self.vertices[j]
                if cspace.is_valid_edge(cur, nbr):
                    self.edges[i].append((int(j), float(dists[j])))
        
    # Search the graph defined by the vertices and edges using GraphState and best_first_search
    def search(self, start_config: np.ndarray, goal_config: np.ndarray) -> tuple[list[np.ndarray], float]:
        # Find nearest vertex to start and goal and validate edge to neighbor, if invalid fail
        if len(self.vertices) == 0:
            print("Cannot search an empty PRM graph")
            return [], -1
        start_dists = self.cspace.point_to_point_distance(self.vertices, start_config)
        start_idx = int(np.argmin(start_dists))
        if not self.cspace.is_valid_edge(start_config, self.vertices[start_idx]):
            print("Invalid edge from start config to nearest PRM vertex")
            return [], -1
        goal_dists = self.cspace.point_to_point_distance(self.vertices, goal_config)
        goal_idx = int(np.argmin(goal_dists))
        if not self.cspace.is_valid_edge(self.vertices[goal_idx], goal_config):
            print("Invalid edge from nearest PRM vertex to goal config")
            return [], -1
        if (start_idx, goal_idx) in self.known_graph_distances:
            path, path_length = self.known_graph_distances[(start_idx, goal_idx)]
        else:
            # Use GraphState and best_first_search to find path from start to goal in PRM graph
            start_state = GraphState(start_idx, goal_idx, 
                                    dist_from_start=0.0, use_heuristic=True,
                                    vertices=self.vertices, edges=self.edges,
                                    vertex_distance=self.cspace.point_to_point_distance)
            path = best_first_search(start_state)
            if len(path) == 0:
                print("Failed to find path in PRM graph")
                return [], -1
            # Return full path including start and goal configs, along with path length
            path_length = path[-1].dist_from_start
            path = [self.vertices[state.state] for state in path]
            self.known_graph_distances[(start_idx, goal_idx)] = (path, path_length)
        
        if start_dists[start_idx] > 0:
            path = [start_config] + path
            path_length += start_dists[start_idx]
        if goal_dists[goal_idx] > 0:
            path = path + [goal_config]
            path_length += goal_dists[goal_idx]
        return path, path_length

class GraphState(AbstractState):
    def __init__(self, state: int, goal: int, dist_from_start: float, use_heuristic: bool,
                 vertices: np.ndarray, edges: dict[int, list[tuple[int, float]]],
                 vertex_distance: Callable[[np.ndarray, np.ndarray], float]) -> None:
        # vertices: features associated with each vertex (np.ndarray of shape (num_vertices, num_dims))
        self.vertices = vertices
        # edges: dict mapping vertex index to list of neighbor indices and costs
        self.edges = edges
        self.vertex_distance = vertex_distance
        super().__init__(state, goal, dist_from_start, use_heuristic)

    def get_neighbors(self) -> list[GraphState]:
        return [GraphState(
            self.edges[self.state][nbr_idx][0],  # neighbor vertex index
            self.goal,
            self.dist_from_start + self.edges[self.state][nbr_idx][1],  # cost to neighbor
            self.use_heuristic,
            self.vertices, self.edges, self.vertex_distance)
            for nbr_idx in range(len(self.edges[self.state]))
        ]

    def is_goal(self) -> bool:
        return self.state == self.goal
    
    # use the graph's supplied distance function
    def compute_heuristic(self) -> float:
        return self.vertex_distance(self.vertices[self.state], self.vertices[self.goal])
    
    def __hash__(self) -> int:
        return int(self.state)
    def __eq__(self, other: GraphState) -> bool:
        return self.state == other.state    


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
        robot_pt = Point(config[0], config[1])
        return (
            point_in_boundary(config[:2], self.cspace_boundary)
            and point_in_boundary(config[:2], self.workspace_boundary)
            and not any(poly.intersects(robot_pt) for poly in self.shapely_obstacles)
        )


class PointCSpace(PolygonalCSpace):
    def __init__(self,
                 workspace_boundary: np.ndarray,
                 obstacles: list[np.ndarray] = [],
                 interpolation_delta: float = 0.1) -> None:
        cspace_boundary = np.array(workspace_boundary, copy=True)
        is_angular = np.array([False, False], dtype=bool)
        super().__init__(workspace_boundary, cspace_boundary, is_angular, obstacles, interpolation_delta)

    # draw the car at given configuration on given Axes
    def draw_state(self, ax: Axes, config: np.ndarray, **kwargs: object) -> None:
        # do forward kinematics to get the four corners of the rectangle in workspace
        point = self.forward_kinematics(config)
        # close the rectangle by repeating the first corner at the end
        ax.plot(point[0], point[1], **kwargs)
        # draw an arrow to indicate orientation
        arrow_length = min(self.rectangle_width, self.rectangle_height) / 2
        ax.arrow(config[0], config[1], arrow_length * np.cos(config[2]), arrow_length * np.sin(config[2]), 
                    head_width=arrow_length/2, head_length=arrow_length/2, fc='k', ec='k')
        
    # Return the 2D center point (x, y) of the rectangle at configuration
    def forward_kinematics(self, config: np.ndarray) -> np.ndarray:
        pt = np.asarray(config[:2], dtype=float)
        return pt.reshape(1, 2)

# Use PRM distance over a 2D point-robot projection as a heuristic for 3D CSpace
# The returned heuristic function will be used by DubinsCarState to guide best-first search in the 3D CSpace
def create_prm_heuristic(cspace: CSpace2D, num_samples: int = 2000,
                         num_neighbors: int = 20) -> Callable[[np.ndarray, np.ndarray], float]:
    # ---- TODO(VIII) ----
    # Create a PolygonalCSpace whose configurations are just (x, y), using the
    # same workspace boundary and obstacles as the original cspace. Then create
    # a OneShotPRM on that 2D point-robot space. The heuristic below queries
    # the PRM using config[:2] and goal_config[:2]
    pcspace = PointCSpace(
        cspace.workspace_boundary,
        cspace.obstacles,
        cspace.interpolation_delta
    )

    prm = OneShotPRM(
        pcspace,
        num_samples,
        num_neighbors
    )

    # ----
    def prm_heuristic(config: np.ndarray, goal_config: np.ndarray) -> float:
        # return path length from PRM as heuristic
        path_length = prm.search(config[:2], goal_config[:2])[1]
        if path_length < 0: # if PRM fails to find a path, fall back to euclidean distance
            return cspace.point_to_point_distance(config, goal_config)
        return path_length
    return prm_heuristic
