from __future__ import annotations

from abc import ABC, abstractmethod
from itertools import count


class AbstractState(ABC):
    # This count increments every time a new AbstractState is created
    _tiebreak_count = count()

    def __init__(self, state: object, goal: object, dist_from_start: float = 0,
                 use_heuristic: bool = True) -> None:
        self.state = state
        self.goal = goal
        # we tiebreak based on the order that the state was created/found
        self.tiebreak_idx = next(AbstractState._tiebreak_count)
        # dist_from_start is "g" in A*, i.e., f = g + h
        self.dist_from_start = dist_from_start
        self.use_heuristic = use_heuristic
        if use_heuristic:
            # NOTE: we only want to compute the heuristic once per state -
            #       you should not call self.compute_heuristic anywhere else, just use self.h
            self.h = self.compute_heuristic()
        else:
            self.h = 0

    # Return a list of AbstractState objects that can be reached from the current state
    @abstractmethod
    def get_neighbors(self) -> list[AbstractState]:
        pass

    # Return True if the state is the goal
    @abstractmethod
    def is_goal(self) -> bool:
        pass

    # Return a float estimating the cost to reach the goal from this state
    @abstractmethod
    def compute_heuristic(self) -> float:
        return 0

    # The "less than" method ensures that states are comparable
    # self.dist_from_start is g, self.h is h, and self.tiebreak_idx is the tiebreaker
    def __lt__(self, other: AbstractState) -> bool:
        f_self = self.dist_from_start + self.h
        f_other = other.dist_from_start + other.h

        if f_self < f_other:
            return True
        elif f_self == f_other:
            # Prefer smaller h value
            if self.h < other.h:
                return True
            elif self.h > other.h:
                return False
            # Fallback to created first
            return self.tiebreak_idx < other.tiebreak_idx
        else:
            return False

    # The "hash" method allows us to keep track of visited states in a dictionary
    # You should hash states based on self.state (and sometimes self.goal, if it can change)
    @abstractmethod
    def __hash__(self) -> int:
        pass

    # __eq__ gets called during hashing collisions, without it Python checks object equality
    @abstractmethod
    def __eq__(self, other: AbstractState) -> bool:
        pass
