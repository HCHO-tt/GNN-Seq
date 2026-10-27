"""Profile a short N=210 episode without treating it as a performance result."""
import argparse
import cProfile
import io
import json
import pstats
import time

from gnn_seq.environment import Environment


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("graph", "stats"), required=True)
    parser.add_argument("--budget-ratio", type=int, default=4)
    parser.add_argument("--fast-igd", action="store_true")
    parser.add_argument("--fast-hv3", action="store_true")
    cli = parser.parse_args()
    config = {"key": "M_2_46_3", "population_size": 210,
              "budget_ratio": cli.budget_ratio, "n_ref_points": 1000,
              "max_objectives": 7, "weight_k": 10, "objective_k": 5,
              "dim_bias": 3, "operator_history": False,
              "fast_igd": cli.fast_igd, "fast_hv3": cli.fast_hv3}
    t0 = time.perf_counter()
    env = Environment(config, 2022, cli.mode)
    t1 = time.perf_counter()
    env.reset()
    t2 = time.perf_counter()
    profiler = cProfile.Profile()
    profiler.enable()
    steps = 0
    while True:
        _, _, done, _ = env.step([0, 0, 0, 0])
        steps += 1
        if done:
            break
    profiler.disable()
    t3 = time.perf_counter()
    stream = io.StringIO()
    pstats.Stats(profiler, stream=stream).sort_stats("cumtime").print_stats(20)
    print(json.dumps({"mode": cli.mode, "fast_igd": cli.fast_igd,
                      "fast_hv3": cli.fast_hv3,
                      "population_size": 210,
                      "budget_ratio": cli.budget_ratio, "steps": steps,
                      "construct_seconds": t1 - t0, "reset_seconds": t2 - t1,
                      "step_seconds": t3 - t2, "seconds_per_step": (t3 - t2) / steps},
                     indent=2))
    print(stream.getvalue())


if __name__ == "__main__":
    main()
