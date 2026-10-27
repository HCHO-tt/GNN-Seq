"""Reload a policy and test explicit seen/unseen Seq MOEA/D problem suites."""
import sys
sys.dont_write_bytecode = True

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from gnn_seq.environment import Environment
from gnn_seq.model import GraphSeqMAC, SeqMAC
from gnn_seq.experiment import seed_all, scheme_for, collect, GROUPS, PREPROCESS
from gnn_seq.protocol import SEEN, UNSEEN, evaluation_tasks, summarize_evaluations
from components.episode_buffer import EpisodeBatch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--suite", choices=("single", "seen", "unseen", "all"), default="single")
    parser.add_argument("--objectives", type=int, choices=(3, 5, 7),
                        help="Required for mixed-objective checkpoints with a task suite")
    parser.add_argument("--seeds", nargs="+", type=int,
                        help="Explicit shared evaluation seeds; otherwise use --runs and --seed-start")
    parser.add_argument("--runs", type=int, help="Runs per function (default: 2 single, 30 suite)")
    parser.add_argument("--seed-start", type=int, default=9001)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--allow-untrained", action="store_true",
                        help="Diagnostic only: permit a zero-update checkpoint in a task suite")
    cli = parser.parse_args()
    root = Path(__file__).resolve().parent
    output = cli.output.resolve()
    if not output.is_relative_to(root) or output.exists():
        parser.error("output must be a new file inside GNN-Seq")
    if cli.seeds and cli.runs is not None:
        parser.error("specify either --seeds or --runs")
    runs = cli.runs if cli.runs is not None else (2 if cli.suite == "single" else 30)
    if runs < 1:
        parser.error("runs must be positive")
    seeds = cli.seeds if cli.seeds is not None else list(range(cli.seed_start, cli.seed_start + runs))
    if len(set(seeds)) != len(seeds):
        parser.error("evaluation seeds must be distinct")
    metadata_path = cli.model / "metadata.json"
    if not metadata_path.is_file():
        parser.error(f"missing checkpoint metadata: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    args = SimpleNamespace(**metadata["args"])
    args.device, args.use_cuda = "cpu", False
    torch.set_num_threads(1)
    config, progress = metadata["environment"], metadata["progress"]
    train_key, mode = config["key"], progress["mode"]
    try:
        tasks = evaluation_tasks(cli.suite, train_key, cli.objectives)
    except ValueError as error:
        parser.error(str(error))
    if cli.objectives is not None and cli.suite == "single":
        parser.error("--objectives applies to a seen/unseen suite")
    extra_path = cli.model / "learner_extra.pt"
    if not extra_path.is_file():
        parser.error("missing learner update metadata")
    updates = int(torch.load(extra_path, map_location="cpu", weights_only=True)["training_steps"])
    if cli.suite != "single" and updates == 0 and not cli.allow_untrained:
        parser.error("checkpoint has zero SADN updates; choose a trained checkpoint")
    scheme = scheme_for(config, mode)
    template = EpisodeBatch(scheme, GROUPS, 1, 2, preprocess=PREPROCESS)
    mac = (GraphSeqMAC if mode == "graph" else SeqMAC)(template.scheme, GROUPS, args)
    mac.load_models(str(cli.model))
    result, by_group = [], {"seen": {}, "unseen": {}}
    for task in tasks:
        task_config = {**config, "key": task}
        task_rows = []
        for seed in seeds:
            seed_all(seed)
            _, summaries = collect([Environment(task_config, seed, mode)], mac, scheme,
                                   args, progress["t_env"], test_mode=True)
            task_rows.append({"task": task, **summaries[0]})
        result.extend(task_rows)
        function = task.split("_")[0]
        group = "seen" if function in SEEN else "unseen" if function in UNSEEN else None
        if group is not None:
            by_group[group][task] = summarize_evaluations(task_rows)
        print(f"{task}: {len(task_rows)} runs, final IGD "
              f"{np.mean([row['last_igd'] for row in task_rows]):.6f}", flush=True)
    grouped = {}
    for group, task_results in by_group.items():
        grouped[group] = {
            "tasks": task_results,
            "mean_of_task_mean_final_igd": (
                float(np.mean([x["mean_final_igd"] for x in task_results.values()]))
                if task_results else None),
        }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({
        "mode": mode, "training_key": train_key, "checkpoint": str(cli.model.resolve()),
        "training_steps": updates, "training_t_env": progress["t_env"],
        "suite": cli.suite, "greedy": True, "evaluation_seeds": seeds,
        "untrained_diagnostic": updates == 0,
        "metric": "final IGD of last population is primary; best IGD is secondary",
        "groups": grouped, "evaluations": result,
        "mean_best_igd": float(np.mean([row["best_igd"] for row in result])),
        "mean_final_igd": float(np.mean([row["last_igd"] for row in result])),
    }, indent=2), encoding="utf-8")
    print(output, flush=True)


if __name__ == "__main__":
    main()
