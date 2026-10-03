import argparse
import time

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection

from cspace import CSpace2D, load_problem
from prm import OneShotPRM, create_prm_heuristic
from robot import DubinsCarParams, DubinsCarState, UnicycleRobot
from rrt import KinodynamicRRT
from search import best_first_search


def draw_prm(ax: Axes, prm: OneShotPRM) -> None:
    edge_lines = []
    for i, nbrs in prm.edges.items():
        for nbr_idx, _ in nbrs:
            edge_lines.append([prm.vertices[i][:2], prm.vertices[nbr_idx][:2]])
    if edge_lines:
        ax.add_collection(LineCollection(edge_lines, colors="gray", linewidths=0.5, label="PRM Graph"))


def draw_path_states(ax: Axes, cspace: CSpace2D, path: list, label: str) -> None:
    for state in path[1:]:
        config = state.state if hasattr(state, "state") else state
        cspace.draw_state(ax, config, color="red", label=label if state is path[1] else None)


def main(args: argparse.Namespace) -> None:
    if args.seed is not None:
        np.random.seed(args.seed)

    cspace, start, goal, data = load_problem(args.params_file)

    print(f"Initializing configuration space from {args.params_file}")
    fig, ax = plt.subplots(figsize=(5, 5))
    cspace.draw_workspace(ax)
    cspace.draw_state(ax, start, color="blue", label="Start")
    cspace.draw_state(ax, goal, color="green", label="Goal")
    start_time = time.time()

    robot = UnicycleRobot(cspace, interpolation_delta=data["dt"])

    if args.search_type == "best_first_search":
        car_params = DubinsCarParams(
            robot=robot,
            velocity=data["velocity"],
            turn_rate=data["turn_rate"],
            control_duration=data["control_duration"],
            goal_tolerance=data["goal_tolerance"],
        )
        heuristic_func = None
        if args.heuristic_type == "PRM2D":
            prm_start_time = time.time()
            heuristic_func = create_prm_heuristic(
                cspace,
                num_samples=args.max_iterations,
                num_neighbors=args.num_neighbors,
            )
            print(f"PRM heuristic construction took {time.time() - prm_start_time:.2f} seconds")

        start_state = DubinsCarState(
            start,
            goal,
            dist_from_start=0.0,
            use_heuristic=True,
            car_params=car_params,
            heuristic_func=heuristic_func,
        )
        path = best_first_search(start_state)
        print(f"Best first search found a path with {len(path)} states in {time.time() - start_time:.2f} seconds")
        path_length = max(0, len(path) - 1) * car_params.control_duration * car_params.velocity
        print(f"Path length: {path_length:.2f}")
        draw_path_states(ax, cspace, path, "Best First Search Path")

    elif args.search_type == "rrt":
        rrt = KinodynamicRRT(
            robot,
            goal_tolerance=data["goal_tolerance"],
            control_duration=args.rrt_control_duration,
            goal_sample_prob=args.rrt_goal_sample_prob,
            num_control_samples=args.num_neighbors,
        )
        path = rrt.search(start, goal, max_iterations=args.max_iterations)
        print(f"RRT found a path with {len(path)} states in {time.time() - start_time:.2f} seconds")
        control_path = [action for _, action in path][1:]
        path = [state for state, _ in path]
        path_length = sum(rrt.control_duration * abs(control[0]) for control in control_path)
        print(f"Path length: {path_length:.2f}")

        for idx, (state, _, parent_idx) in rrt.idx_to_state.items():
            if parent_idx is not None:
                parent_state = rrt.idx_to_state[parent_idx][0]
                ax.plot([state[0], parent_state[0]], [state[1], parent_state[1]],
                        color="gray", linewidth=0.5, label="RRT Tree" if idx == 1 else None)
        draw_path_states(ax, cspace, path, "RRT Path")

    elif args.search_type == "prm":
        prm = OneShotPRM(cspace, num_samples=args.max_iterations, num_neighbors=args.num_neighbors)
        path, path_length = prm.search(start, goal)
        print(f"PRM completed in {time.time() - start_time:.2f} seconds")
        draw_prm(ax, prm)
        if len(path) == 0:
            print("PRM failed to find a path")
        else:
            print(f"PRM found a path with {len(path)} states and path length {path_length:.2f}")
            path_vertices = np.array(path)
            ax.plot(path_vertices[:, 0], path_vertices[:, 1], "r-", linewidth=2, label="PRM Path")
            for state in path[1:-1]:
                cspace.draw_state(ax, state, color="red")

    ax.set_aspect("equal", adjustable="box")
    ax.legend()
    if args.save_image:
        fig.savefig(args.save_image, dpi=150, bbox_inches="tight")
    if not args.no_show:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CS440 MP3 Robot Motion Planning", allow_abbrev=False)
    parser.add_argument("--h", action="help", help=argparse.SUPPRESS)
    parser.add_argument("--params_file", type=str, default="data/open.json",
                        help="path to JSON file containing environment and robot parameters")
    parser.add_argument("--search_type", type=str,
                        choices=["best_first_search", "rrt", "prm"],
                        default="best_first_search",
                        help="which planner to run")
    parser.add_argument("--heuristic_type", type=str, choices=["euclidean", "PRM2D"],
                        default="euclidean",
                        help="heuristic for best_first_search")
    parser.add_argument("--max_iterations", type=int, default=1000,
                        help="RRT iterations, PRM samples, or heuristic PRM samples")
    parser.add_argument("--num_neighbors", type=int, default=10,
                        help="RRT controls sampled per expansion or PRM neighbors per vertex")
    parser.add_argument("--rrt_control_duration", type=float, default=1.0,
                        help="duration for each sampled RRT control")
    parser.add_argument("--rrt_goal_sample_prob", type=float, default=0.1,
                        help="probability of sampling the goal during RRT")
    parser.add_argument("--seed", type=int, default=None,
                        help="random seed for reproducible local runs")
    parser.add_argument("--save_image", type=str, default=None,
                        help="save the visualization to this image path")
    parser.add_argument("--no_show", action="store_true",
                        help="do not open a matplotlib window")
    main(parser.parse_args())
