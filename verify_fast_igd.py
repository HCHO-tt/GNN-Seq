"""Check the optional vectorized IGD against the original at N=210."""
import argparse
import json

import numpy as np

from gnn_seq.environment import Environment


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("graph", "stats"), default="graph")
    parser.add_argument("--fast-hv3", action="store_true")
    cli = parser.parse_args()
    if cli.fast_hv3 and cli.mode != "stats":
        parser.error("HV verification requires --mode stats")
    config = {"key": "M_2_46_3", "population_size": 210,
              "budget_ratio": 1, "n_ref_points": 1000,
              "max_objectives": 7, "weight_k": 10, "objective_k": 5,
              "dim_bias": 3, "operator_history": False,
              "fast_igd": cli.fast_hv3}
    original = Environment(config, 2022, cli.mode)
    accelerated = Environment({**config, "fast_igd": True,
                               "fast_hv3": cli.fast_hv3}, 2022, cli.mode)
    original_state = original.reset()
    accelerated_state = accelerated.reset()
    deviations = [abs(original.core.last_value - accelerated.core.last_value)]
    rewards = []
    observations = [float(np.max(np.abs(original_state["state"] -
                                        accelerated_state["state"])))]
    hypervolumes = []
    for action in ([0, 0, 0, 0], [1, 2, 3, 1], [2, 1, 0, 0]):
        old_state, old_reward, old_done, old_info = original.step(action)
        new_state, new_reward, new_done, new_info = accelerated.step(action)
        assert old_done == new_done
        np.testing.assert_array_equal(
            [s.objectives for s in original.core.population],
            [s.objectives for s in accelerated.core.population])
        np.testing.assert_array_equal(original.core.weights, accelerated.core.weights)
        deviations.append(abs(old_info["last_igd"] - new_info["last_igd"]))
        rewards.append(abs(old_reward - new_reward))
        observations.append(float(np.max(np.abs(old_state["state"] - new_state["state"]))))
        if cli.fast_hv3:
            hypervolumes.append(abs(original.core.hv_his[-1] -
                                    accelerated.core.hv_his[-1]))
    assert max(deviations) < 1e-12 and max(rewards) < 1e-12
    assert max(observations) < 1e-6
    if cli.fast_hv3:
        assert max(hypervolumes) < 1e-12
    print(json.dumps({"mode": cli.mode, "population_size": 210,
                      "seed": 2022, "steps": 3, "fast_hv3": cli.fast_hv3,
                      "problem": original.core.env_name,
                      "max_absolute_igd_difference": max(deviations),
                      "max_absolute_reward_difference": max(rewards),
                      "max_absolute_observation_difference": max(observations),
                      "max_absolute_hv_difference": max(hypervolumes) if hypervolumes else None,
                      "population_and_weights_identical": True}, indent=2))


if __name__ == "__main__":
    main()
