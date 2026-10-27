"""Run the fixed three-seed, held-out N=35 screening evaluation."""
import sys
sys.dont_write_bytecode = True

import json
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from compare import compare
from gnn_seq.protocol import SEEN, UNSEEN


ROOT = Path(__file__).resolve().parent
SEEDS = (9101, 9102, 9103, 9104, 9105)
RUNS = {
    2022: ("action_m5_n35_t20_s2022_v2", "prelim_stats_m5_n35_t20_s2022"),
    2023: ("action_m5_n35_t20_s2023", "small_verify_stats_s2023"),
    2024: ("small_verify_graph_s2024", "small_verify_stats_s2024"),
}


def evaluation(name, stage):
    run = ROOT / "runs" / name
    model = run / stage
    output = run / f"small_gate_{stage}_9101_9105.json"
    metadata = json.loads((model / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["environment"]["key"] == "M_2_46_5"
    assert metadata["environment"]["population_size"] == 35
    assert metadata["environment"]["budget_ratio"] == 4
    assert metadata["args"]["seed"] in RUNS
    assert metadata["args"]["batch_size"] == 4
    assert metadata["args"]["batch_size_run"] == 2
    assert metadata["args"]["buffer_size"] == 32
    assert metadata["progress"]["training_steps"] == (0 if stage == "initial" else 38)
    if stage == "model":
        assert metadata["progress"]["t_env"] == 800
    if output.exists():
        report = json.loads(output.read_text(encoding="utf-8"))
        assert report["evaluation_seeds"] == list(SEEDS)
        assert report["checkpoint"] == str(model.resolve())
        assert len(report["evaluations"]) == 8 * len(SEEDS)
        return output
    command = [sys.executable, "-B", "evaluate.py", "--model", str(model),
               "--suite", "all", "--seeds", *map(str, SEEDS),
               "--output", str(output)]
    if stage == "initial":
        command.append("--allow-untrained")
    log = run / f"small_gate_{stage}_9101_9105.log"
    with log.open("w", encoding="utf-8") as stream:
        result = subprocess.run(command, cwd=ROOT, stdout=stream,
                                stderr=subprocess.STDOUT, check=False)
    if result.returncode:
        raise RuntimeError(f"Evaluation failed ({result.returncode}): {log}")
    return output


def relative_group(report, functions):
    values = []
    for task in sorted({row["task"] for row in report["pairs"]
                        if row["task"].split("_")[0] in functions}):
        rows = [row for row in report["pairs"] if row["task"] == task]
        graph = float(np.mean([row["graph_final_igd"] for row in rows]))
        stats = float(np.mean([row["stats_final_igd"] for row in rows]))
        values.append({"task": task, "graph_mean": graph, "stats_mean": stats,
                       "relative_difference": (graph - stats) / stats,
                       "graph_better_tests": sum(row["graph_minus_stats"] < 0
                                                 for row in rows)})
    return values


def main():
    jobs = [(name, stage) for pair in RUNS.values() for name in pair
            for stage in ("initial", "model")]
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {executor.submit(evaluation, name, stage): (name, stage)
                   for name, stage in jobs}
        for future in as_completed(futures):
            name, stage = futures[future]
            print(f"{name}/{stage}: {future.result()}", flush=True)
    summaries = []
    for seed, (graph_name, stats_name) in RUNS.items():
        stages = {}
        for stage in ("initial", "model"):
            graph = json.loads(evaluation(graph_name, stage).read_text(encoding="utf-8"))
            stats = json.loads(evaluation(stats_name, stage).read_text(encoding="utf-8"))
            paired = compare(graph, stats)
            path = ROOT / "runs" / f"small_gate_s{seed}_{stage}_comparison.json"
            path.write_text(json.dumps(paired, indent=2), encoding="utf-8")
            seen = relative_group(paired, SEEN)
            unseen = relative_group(paired, UNSEEN)
            stages[stage] = {
                "seen": seen, "unseen": unseen,
                "all_relative_mean": float(np.mean([
                    row["relative_difference"] for row in seen + unseen])),
                "seen_relative_mean": float(np.mean([
                    row["relative_difference"] for row in seen])),
                "unseen_relative_mean": float(np.mean([
                    row["relative_difference"] for row in unseen])),
                "comparison_file": str(path.relative_to(ROOT)),
            }
        summaries.append({"training_seed": seed, "initial": stages["initial"],
                          "trained": stages["model"],
                          "relative_gap_improvement": (stages["initial"]["all_relative_mean"]
                                                       - stages["model"]["all_relative_mean"])})
    gate = (all(item["trained"]["all_relative_mean"] < 0 for item in summaries)
            and np.mean([item["trained"]["seen_relative_mean"] for item in summaries]) < 0
            and np.mean([item["trained"]["unseen_relative_mean"] for item in summaries]) < 0
            and sum(item["relative_gap_improvement"] > 0 for item in summaries) >= 2)
    result = {"purpose": "Exploratory small-scale gate, not original-scale proof",
              "training_seeds": list(RUNS), "evaluation_seeds": list(SEEDS),
              "primary_metric": "Mean of per-function relative final-population IGD differences",
              "runs": summaries, "gate_passed": bool(gate)}
    output = ROOT / "runs" / "small_gate_summary_9101_9105.json"
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    for item in summaries:
        print(item["training_seed"], "trained_rel=",
              round(item["trained"]["all_relative_mean"], 6),
              "gap_gain=", round(item["relative_gap_improvement"], 6), flush=True)
    print(f"gate_passed={gate}; {output}", flush=True)


if __name__ == "__main__":
    main()
