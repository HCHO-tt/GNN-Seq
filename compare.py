"""Pair graph and original-statistics evaluations by problem and seed."""
import sys
sys.dont_write_bytecode = True

import argparse
import json
from pathlib import Path

import numpy as np

from gnn_seq.protocol import SEEN, UNSEEN


def metadata(report):
    path = Path(report["checkpoint"]) / "metadata.json"
    return json.loads(path.read_text(encoding="utf-8"))


def compare(graph, stats):
    if graph["mode"] != "graph" or stats["mode"] != "stats":
        raise ValueError("--graph and --stats must identify their respective modes")
    for key in ("training_key", "training_t_env", "training_steps", "suite", "evaluation_seeds"):
        if graph[key] != stats[key]:
            raise ValueError(f"Mismatched comparison field: {key}")
    gm, sm = metadata(graph), metadata(stats)
    for key in ("key", "population_size", "budget_ratio", "n_ref_points", "dim_bias"):
        if gm["environment"][key] != sm["environment"][key]:
            raise ValueError(f"Mismatched MOEA/D configuration: {key}")
    if gm["environment"].get("fast_igd", False) != sm["environment"].get("fast_igd", False):
        raise ValueError("Mismatched IGD implementation")
    if gm["environment"].get("fast_hv3", False) != sm["environment"].get("fast_hv3", False):
        raise ValueError("Mismatched HV implementation setting")
    for key in ("seed", "batch_size", "batch_size_run", "buffer_size", "lr", "gamma",
                "q_nstep", "epochs", "epsilon_start", "epsilon_finish",
                "epsilon_anneal_time", "target_update_interval_or_tau"):
        if gm["args"][key] != sm["args"][key]:
            raise ValueError(f"Mismatched training configuration: {key}")
    def index(report):
        keyed = {(row["task"], row["seed"]): row for row in report["evaluations"]}
        if len(keyed) != len(report["evaluations"]):
            raise ValueError("Duplicate problem and evaluation seed")
        return keyed
    g, s = index(graph), index(stats)
    if g.keys() != s.keys():
        raise ValueError("Both evaluations must contain the same problems and seeds")
    pairs = [{"task": task, "seed": seed,
              "graph_final_igd": g[task, seed]["last_igd"],
              "stats_final_igd": s[task, seed]["last_igd"],
              "graph_minus_stats": g[task, seed]["last_igd"] - s[task, seed]["last_igd"]}
             for task, seed in sorted(g)]
    grouped = {}
    for name, functions in (("seen", SEEN), ("unseen", UNSEEN)):
        tasks = {}
        for task in sorted({row["task"] for row in pairs if row["task"].split("_")[0] in functions}):
            rows = [row for row in pairs if row["task"] == task]
            tasks[task] = {"runs": len(rows),
                           "mean_graph_final_igd": float(np.mean([r["graph_final_igd"] for r in rows])),
                           "mean_stats_final_igd": float(np.mean([r["stats_final_igd"] for r in rows])),
                           "mean_paired_difference": float(np.mean([r["graph_minus_stats"] for r in rows])),
                           "graph_better_count": sum(r["graph_minus_stats"] < 0 for r in rows)}
        grouped[name] = {"tasks": tasks,
                         "mean_of_task_differences": (float(np.mean([
                             item["mean_paired_difference"] for item in tasks.values()]))
                             if tasks else None)}
    return {"training_key": graph["training_key"], "suite": graph["suite"],
            "training_t_env": graph["training_t_env"],
            "training_steps": graph["training_steps"],
            "training_seed": gm["args"]["seed"],
            "evaluation_seeds": graph["evaluation_seeds"],
            "difference": "graph final IGD minus original-statistics final IGD; lower is better",
            "groups": grouped, "pairs": pairs,
            "interpretation": "One trained-model pair; evaluation repeats are not independent training seeds."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", required=True, type=Path)
    parser.add_argument("--stats", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    cli = parser.parse_args()
    root = Path(__file__).resolve().parent
    output = cli.output.resolve()
    if not output.is_relative_to(root) or output.exists():
        parser.error("output must be a new file inside GNN-Seq")
    try:
        result = compare(json.loads(cli.graph.read_text(encoding="utf-8")),
                         json.loads(cli.stats.read_text(encoding="utf-8")))
    except (KeyError, ValueError) as error:
        parser.error(str(error))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    for group, values in result["groups"].items():
        if values["mean_of_task_differences"] is not None:
            print(f"{group}: graph - stats final IGD = "
                  f"{values['mean_of_task_differences']:.6f}")
    print(output)


if __name__ == "__main__":
    main()
