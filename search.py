import heapq
from state import AbstractState

# TODO(IV): copy the best_first_search and backtrack methods from MP1/MP2 here


def best_first_search(starting_state: AbstractState) -> list[AbstractState]:
    '''
    Implementation of best first search algorithm

    Input:
        starting_state: an AbstractState object

    Return:
        A path consisting of a list of AbstractState states
        The first state should be starting_state
        The last state should have state.is_goal() == True
    '''
    visited_states = {starting_state: (None, 0)}
    frontier = []
    heapq.heappush(frontier, starting_state)
    # Your code here ---------------

    # ------------------------------
    return []


def backtrack(visited_states: dict, goal_state: AbstractState) -> list[AbstractState]:
    '''
    Implementation of the backtrack method

    Input:
        visited_states: a dictionary mapping AbstractState objects to (parent_state, distance_from_start) tuples
        goal_state: an AbstractState object

    Return:
        A path consisting of a list of AbstractState states
        The first state should be starting_state
        The last state should have state.is_goal() == True
    '''
    path = []
    # Your code here ---------------

    # ------------------------------
    return path
