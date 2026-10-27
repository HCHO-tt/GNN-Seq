"""Measure actual one- and two-hop graph reach at a requested MOEA/D size."""
import sys
sys.dont_write_bytecode = True

import argparse
import numpy as np

from gnn_seq.environment import Environment
from diagnose_graph_state import adjacency


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--key", default="DTLZ2_5")
    parser.add_argument("--population-size", type=int, required=True)
    parser.add_argument("--seed", type=int, default=2022)
    parser.add_argument("--weight-k", type=int, default=10)
    parser.add_argument("--objective-k", type=int, default=5)
    cli = parser.parse_args()
    env = Environment({"key": cli.key, "population_size": cli.population_size,
                       "n_ref_points": 1000, "budget_ratio": 4,
                       "max_objectives": 7, "weight_k": cli.weight_k,
                       "objective_k": cli.objective_k, "dim_bias": 3}, cli.seed)
    state = env.reset()
    n = cli.population_size
    w, o = adjacency(state["weight_knn"], n), adjacency(state["objective_knn"], n)
    union = w | o
    two_hop = (union.astype(np.int16) @ union.astype(np.int16)) > 0
    print(f"key={cli.key} N={n} weight_k={cli.weight_k} objective_k={cli.objective_k} "
          f"average union neighbours={(union.sum(1)-1).mean():.3f} "
          f"two-hop reach={two_hop.mean():.5f}")


if __name__ == "__main__":
    main()
