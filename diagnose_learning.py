"""Compare two policy checkpoints on matched evaluation tasks and seeds."""
import sys
sys.dont_write_bytecode = True

import argparse
import json
from pathlib import Path

import numpy as np
import torch


def inspect(before, after):
    for key in ("mode", "training_key", "suite", "evaluation_seeds"):
        if before[key] != after[key]:
            raise ValueError(f"Mismatched evaluation field: {key}")
    def index(report):
        rows = {(row["task"], row["seed"]): row for row in report["evaluations"]}
        if len(rows) != len(report["evaluations"]):
            raise ValueError("Duplicate task and seed")
        return rows
    a, b = index(before), index(after)
    if a.keys() != b.keys():
        raise ValueError("Evaluations must cover the same tasks and seeds")
    meta = [json.loads((Path(report["checkpoint"]) / "metadata.json").read_text(encoding="utf-8"))
            for report in (before, after)]
    for key in ("seed", "graph_hidden_dim", "graph_output_dim", "graph_policy",
                "graph_layers", "operator_history", "effect_loss_weight", "effect_head_lr"):
        if meta[0]["args"].get(key) != meta[1]["args"].get(key):
            raise ValueError(f"Mismatched model setup: {key}")
    if meta[0]["environment"] != meta[1]["environment"]:
        raise ValueError("Mismatched environment")
    files = ([f"agent{i}.th" for i in range(4)] +
             (["graph_encoder.th"] +
              (["action_readouts.th"] if meta[0]["args"].get("graph_policy", "pooled") == "action_attention" else [])
              if before["mode"] == "graph" else []))
    parameter_change = {}
    for filename in files:
        states = [torch.load(Path(report["checkpoint"]) / filename, map_location="cpu",
                             weights_only=True) for report in (before, after)]
        if states[0].keys() != states[1].keys():
            raise ValueError(f"Mismatched parameter keys: {filename}")
        differences = [(states[1][key].float() - states[0][key].float()).flatten()
                       for key in states[0]]
        parameter_change[filename] = {
            "l2": float(torch.linalg.vector_norm(torch.cat(differences))),
            "max_absolute": float(max(x.abs().max() for x in differences))}
    rows = []
    for task, seed in sorted(a):
        first, last = a[task, seed], b[task, seed]
        if len(first["actions"]) != len(last["actions"]):
            raise ValueError("Mismatched episode length")
        decisions = sum(x != y for aa, bb in zip(first["actions"], last["actions"])
                        for x, y in zip(aa, bb))
        rows.append({"task": task, "seed": seed,
                     "before_final_igd": first["last_igd"],
                     "after_final_igd": last["last_igd"],
                     "after_minus_before": last["last_igd"] - first["last_igd"],
                     "changed_decisions": decisions,
                     "decisions": 4 * len(first["actions"])})
    return {"before_updates": before["training_steps"],
            "after_updates": after["training_steps"],
            "episodes": len(rows),
            "changed_decisions": sum(r["changed_decisions"] for r in rows),
            "decisions": sum(r["decisions"] for r in rows),
            "changed_episodes": sum(r["changed_decisions"] > 0 for r in rows),
            "mean_final_igd_before": float(np.mean([r["before_final_igd"] for r in rows])),
            "mean_final_igd_after": float(np.mean([r["after_final_igd"] for r in rows])),
            "parameter_change": parameter_change, "pairs": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    cli = parser.parse_args()
    output = cli.output.resolve()
    if not output.is_relative_to(Path(__file__).resolve().parent) or output.exists():
        parser.error("output must be a new file inside GNN-Seq")
    try:
        report = inspect(json.loads(cli.before.read_text(encoding="utf-8")),
                         json.loads(cli.after.read_text(encoding="utf-8")))
    except (KeyError, ValueError) as error:
        parser.error(str(error))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"changed decisions: {report['changed_decisions']}/{report['decisions']}")
    print(f"changed episodes: {report['changed_episodes']}/{report['episodes']}")
    print(f"mean final IGD: {report['mean_final_igd_before']:.6f} -> "
          f"{report['mean_final_igd_after']:.6f}")
    print(output)


if __name__ == "__main__":
    main()
