from __future__ import annotations

from dataclasses import dataclass
from typing import Callable
import numpy as np
from cspace import CSpace2D
from state import AbstractState

# A Robot is defined by how it transitions between configurations in a CSpace
# namely it is a set of controls and ways to apply these controls to get new configurations
class Robot:
    # Store the CSpace, control bounds, and simulation resolution for this robot
    def __init__(self, cspace: CSpace2D, control_boundary: np.ndarray, interpolation_delta: float = 0.1) -> None:
        self.cspace = cspace
        self.control_boundary = control_boundary
        # a resolution parameter for how finely we simulate the trajectory when applying a control, smaller is more accurate but more expensive
        self.interpolation_delta = interpolation_delta

    # Return True if the control lies inside the robot's control bounds
    def is_valid_control(self, control: np.ndarray) -> bool:
        return np.all(control >= self.control_boundary[:, 0]) and np.all(control <= self.control_boundary[:, 1])

    # Return uniformly random controls from inside the control bounds
    def sample_random_controls(self, num_controls: int) -> np.ndarray:
        return np.random.uniform(low=self.control_boundary[:, 0], 
                                 high=self.control_boundary[:, 1], 
                                 size=(num_controls, self.control_boundary.shape[0]))

    # Return one next configuration after applying a control for dt time
    def dynamics(self, config: np.ndarray, control: np.ndarray, dt: float | None = None) -> np.ndarray:
        raise NotImplementedError("Must implement dynamics method in subclass")

    # Apply control (using self.dynamics) starting from config for t total time in increments of self.interpolation_delta 
    # and return a list of all intermediate configurations and final configuration
    def apply_constant_control(self, config: np.ndarray, control: np.ndarray, t: float) -> np.ndarray:
        poses = [np.copy(config)]
        # ---- (V.2) ----
        num_steps_cont = t / self.interpolation_delta
        num_steps = int(np.ceil(num_steps_cont))
        time_remaining = t

        for i in range(num_steps):
            time_remaining -=self.interpolation_delta
            poses.append(self.dynamics(poses[-1], control, self.interpolation_delta))

        if time_remaining > 0.0:
            poses.append(self.dynamics(poses[-1], control, time_remaining))
        # ----
        return np.array(poses[1:])  # exclude the initial config from the returned trajectory


class UnicycleRobot(Robot):
    # Create a unicycle robot whose controls are velocity and turn rate
    def __init__(self, cspace: CSpace2D, interpolation_delta: float = 0.1) -> None:
        # control consists of (velocity, turn rate)
        control_boundary = np.array([[0.0, 2.0], [-1.0, 1.0]]) # velocity between 0 and 2.0, turn rate between -1 and 1
        # make sure the given CSpace has 3 dimensions (x, y, angle) since that's what our dynamics assume
        # NOTE: alternatively we can define a specific CSpace class that guarantees these things
        assert cspace.cspace_boundary.shape[0] == 3, "UnicycleRobot requires a 3D CSpace"
        assert np.array_equal(cspace.is_angular, np.array([False, False, True])), "UnicycleRobot requires a CSpace with 2 translational and 1 angular dimension"
        super().__init__(cspace, control_boundary, interpolation_delta)

    # Return the next configuration after applying one unicycle control
    def dynamics(self, config: np.ndarray, control: np.ndarray, dt: float | None = None) -> np.ndarray:
        if dt is None:
            dt = self.interpolation_delta
        config = np.copy(config)  # avoid modifying in place
        # ---- (V.1) ----
        x, y, theta = config
        v, turn_rate = control

        x_hat = v * np.cos(theta)
        y_hat = v * np.sin(theta)

        x_next = x + x_hat * dt
        y_next = y + y_hat * dt
        theta_next = theta + turn_rate * dt
        theta_next = theta_next % (2 * np.pi)
        config = np.array([x_next, y_next, theta_next])
        # ---
        return config


# define a struct for car params
@dataclass
class DubinsCarParams:
    robot: UnicycleRobot # a Dubins car is a unicycle robot with fixed velocity and turn rate
    velocity: float = 1.0 # a FIXED velocity for all controls
    turn_rate: float = 1.0 # a FIXED turn rate for left and right turns
    control_duration: float = 1.0 # a FIXED duration for each control
    goal_tolerance: float = 1.0 # our goal condition will be to get within this distance of the goal configuration

class DubinsCarState(AbstractState):
    # Create a search state for the Dubins car planning problem
    def __init__(self, 
                 state: np.ndarray, # a configuration in CSpace (np.ndarray of shape (num_dims,))
                 goal: np.ndarray, # a configuration in CSpace (np.ndarray of shape (num_dims,))
                 dist_from_start: float, use_heuristic: bool, # standard AbstractState parameters
                 car_params: DubinsCarParams, # parameters defining the dynamics and constraints of the Dubins car 
                 heuristic_func: Callable[[np.ndarray, np.ndarray], float] | None = None # a generic function that takes in a configuration and a goal configuration and returns a value
                 ) -> None:
        # all params except self.state get passed to neighbors without modification
        self.car_params = car_params
        self.heuristic_func = heuristic_func
        # controls consist of (velocity, turn rate) and we have 3 options for a Dubins car
        self.controls = np.array([
            [self.car_params.velocity, self.car_params.turn_rate], 
            [self.car_params.velocity, 0.0], 
            [self.car_params.velocity, -self.car_params.turn_rate]])
        super().__init__(state, goal, dist_from_start, use_heuristic)
        # for comparing states we round to avoid precision issues, and we use a tuple so that it is hashable
        self._state_key = tuple(np.round(self.state, decimals=6))
    
    # Return valid left, straight, and right neighbors from this state
    def get_neighbors(self) -> list[DubinsCarState]:
        nbrs = []
        for control in self.controls:
            # ---- (V.3) ----
            inter_nbrs = self.car_params.robot.apply_constant_control(
                self.state,
                control,
                self.car_params.control_duration
            )

            if any([not self.car_params.robot.cspace.is_valid(inter) for inter in inter_nbrs]):
                continue

            neighbor = DubinsCarState(
                inter_nbrs[-1],
                self.goal,
                self.dist_from_start + self.car_params.control_duration * self.car_params.velocity,
                self.use_heuristic,
                self.car_params,
                self.heuristic_func,
            )

            nbrs.append(neighbor)
            # ----
        return nbrs
    
    # Return True if this state is within goal tolerance of the goal
    def is_goal(self) -> bool:
        # ---- (V.4) ----
        # maybe use self.robot.cspace.point_to_point_distance
        return np.allclose(self.state, self.goal, atol=self.car_params.goal_tolerance)
        # ---
    
    # Return the heuristic value for this state
    def compute_heuristic(self) -> float:
        if self.heuristic_func is None:
            return self.car_params.robot.cspace.point_to_point_distance(self.state, self.goal)
        else:
            return self.heuristic_func(self.state, self.goal)
    
    def __hash__(self) -> int:
        return hash(self._state_key)

    def __eq__(self, other: DubinsCarState) -> bool:
        return isinstance(other, DubinsCarState) and self._state_key == other._state_key
