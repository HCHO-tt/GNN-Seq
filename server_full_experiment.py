"""Run the fixed three-scale, three-seed Seq versus GNN-Seq protocol on Linux."""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RUNS = ROOT / "runs"
TOTAL_STEPS = 405000
COLLECT_EPISODES = 12
BATCH_EPISODES = 32
REPLAY_EPISODES = 5000
GRAPH_HIDDEN = 64


def source_digest():
    digest = hashlib.sha256()
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file() or "runs" in path.relative_to(ROOT).parts:
            continue
        if path.suffix not in (".py", ".yaml"):
            continue
        digest.update(path.relative_to(ROOT).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def name(scale, mode, seed):
    return f"formal_server_m{scale}_{mode}_s{seed}"


def train_command(scale, mode, seed, device, threads):
    return [sys.executable, "-B", "train.py", "--name", name(scale, mode, seed),
            "--mode", mode, "--seed", str(seed), "--key", f"M_2_46_{scale}",
            "--population-size", "210", "--budget-ratio", "100",
            "--total-steps", str(TOTAL_STEPS),
            "--batch-episodes", str(BATCH_EPISODES),
            "--collect-episodes", str(COLLECT_EPISODES),
            "--replay-episodes", str(REPLAY_EPISODES),
            "--graph-hidden-dim", str(GRAPH_HIDDEN),
            "--graph-policy", "action_attention", "--graph-layers", "2",
            "--parallel-envs", "--fast-igd", "--fast-hv3",
            "--device", device, "--threads", str(threads)]


def run_command(command, log):
    print("RUN", " ".join(command), flush=True)
    log.parent.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment.update(PYTHONUNBUFFERED="1", PYTHONDONTWRITEBYTECODE="1",
                       PYTORCH_NVML_BASED_CUDA_CHECK="1")
    with log.open("a", encoding="utf-8") as output:
        output.write("\n" + datetime.now(timezone.utc).isoformat() + "\n")
        output.write("COMMAND " + " ".join(command) + "\n")
        output.flush()
        result = subprocess.run(command, cwd=ROOT, env=environment, stdout=output,
                                stderr=subprocess.STDOUT, check=False)
    if result.returncode:
        tail = "\n".join(log.read_text(encoding="utf-8", errors="replace").splitlines()[-20:])
        raise RuntimeError(f"Command failed ({result.returncode}); see {log}\n{tail}")


def ensure_cuda(device):
    if device == "cpu":
        return
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in this Python environment")
    print("CUDA:", torch.cuda.get_device_name(0),
          "VRAM_GB:", round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2),
          flush=True)


def completed_model(run):
    metadata = run / "model" / "metadata.json"
    if not metadata.is_file():
        return False
    progress = json.loads(metadata.read_text(encoding="utf-8"))["progress"]
    if progress["t_env"] < TOTAL_STEPS or progress["training_steps"] <= 0:
        raise ValueError(f"Incomplete final model: {metadata}")
    return True


def check_disk(min_free_gb):
    free = shutil.disk_usage(ROOT).free / 1e9
    if free < min_free_gb:
        raise RuntimeError(f"Only {free:.1f} GB free; need {min_free_gb:.1f} GB before a new model")


def train_all(config):
    for scale in config["scales"]:
        for seed in config["seeds"]:
            for mode in ("graph", "stats"):
                run = RUNS / name(scale, mode, seed)
                if completed_model(run):
                    print("SKIP completed", run.name, flush=True)
                    continue
                check_disk(config["min_free_gb"])
                command = train_command(scale, mode, seed, config["device"], config["threads"])
                if run.exists():
                    if not (run / "resume.pt").is_file():
                        raise RuntimeError(f"Run exists without a committed resume snapshot: {run}")
                    command.append("--resume")
                run_command(command, RUNS / "server_logs" / f"{run.name}.train.log")
                if not completed_model(run):
                    raise RuntimeError(f"Training exited without a complete model: {run}")


def report_path(scale, mode, seed):
    return RUNS / name(scale, mode, seed) / "formal_all_30.json"


def evaluate_all(config):
    for scale in config["scales"]:
        # The reserved final test seeds are used only after all six models of
        # the scale are trained, never to select a training checkpoint.
        for seed in config["seeds"]:
            for mode in ("graph", "stats"):
                if not completed_model(RUNS / name(scale, mode, seed)):
                    raise RuntimeError(f"Train all m={scale} models before final evaluation")
        for seed in config["seeds"]:
            for mode in ("graph", "stats"):
                run = RUNS / name(scale, mode, seed)
                report = report_path(scale, mode, seed)
                if report.exists():
                    saved = json.loads(report.read_text(encoding="utf-8"))
                    expected = list(range(config["eval_seed_start"],
                                          config["eval_seed_start"] + config["eval_count"]))
                    if (saved["evaluation_seeds"] != expected or saved["suite"] != "all"
                            or saved["training_key"] != f"M_2_46_{scale}"
                            or saved["mode"] != mode):
                        raise ValueError(f"Existing evaluation does not match plan: {report}")
                    print("SKIP evaluated", run.name, flush=True)
                    continue
                command = [sys.executable, "-B", "evaluate.py", "--model", str(run / "model"),
                           "--suite", "all", "--runs", str(config["eval_count"]),
                           "--seed-start", str(config["eval_seed_start"]),
                           "--output", str(report)]
                run_command(command, RUNS / "server_logs" / f"{run.name}.eval.log")
        for seed in config["seeds"]:
            comparison = RUNS / f"formal_server_m{scale}_s{seed}_comparison.json"
            if comparison.exists():
                print("SKIP compared", comparison.name, flush=True)
                continue
            command = [sys.executable, "-B", "compare.py",
                       "--graph", str(report_path(scale, "graph", seed)),
                       "--stats", str(report_path(scale, "stats", seed)),
                       "--output", str(comparison)]
            run_command(command, RUNS / "server_logs" / f"m{scale}_s{seed}.compare.log")


def summarize(config):
    summary = {"primary_metric": "equal-weight mean of per-task relative final IGD differences",
               "negative_is_better": True, "scales": {}}
    for scale in config["scales"]:
        results = []
        for seed in config["seeds"]:
            path = RUNS / f"formal_server_m{scale}_s{seed}_comparison.json"
            if not path.is_file():
                raise FileNotFoundError(path)
            data = json.loads(path.read_text(encoding="utf-8"))
            if data["training_seed"] != seed or len(data["pairs"]) != 8 * config["eval_count"]:
                raise ValueError(f"Incomplete paired evaluation: {path}")
            per_task = {}
            for group in data["groups"].values():
                for task, values in group["tasks"].items():
                    per_task[task] = (values["mean_graph_final_igd"] -
                                      values["mean_stats_final_igd"]) / values["mean_stats_final_igd"]
            if len(per_task) != 8:
                raise ValueError(f"Expected eight tasks in {path}")
            results.append({"seed": seed, "relative_difference": sum(per_task.values()) / 8,
                            "per_task": per_task,
                            "seen": sum(v for k, v in per_task.items()
                                        if k.startswith(("DTLZ2_", "WFG4_", "WFG6_"))) / 3,
                            "unseen": sum(v for k, v in per_task.items()
                                          if k.startswith(("DTLZ4_", "WFG5_", "WFG7_",
                                                           "WFG8_", "WFG9_"))) / 5})
        summary["scales"][str(scale)] = {
            "seeds": results,
            "mean_relative_difference": sum(r["relative_difference"] for r in results) / len(results),
            "graph_better_training_seeds": sum(r["relative_difference"] < 0 for r in results),
        }
    output = RUNS / "server_formal_summary.json"
    temporary = output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    os.replace(temporary, output)
    print("SUMMARY", output, flush=True)


def preflight(config):
    ensure_cuda(config["device"])
    marker = RUNS / "server_preflight.json"
    expected = {"source_sha256": config["source_sha256"], "device": config["device"]}
    if marker.is_file() and json.loads(marker.read_text(encoding="utf-8")) == expected:
        print("SKIP completed GPU/CPU smoke preflight", flush=True)
        return
    for mode in ("graph", "stats"):
        unique = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S%f")
        run_name = f"server_smoke_{mode}_{unique}"
        command = [sys.executable, "-B", "train.py", "--name", run_name,
                   "--mode", mode, "--smoke", "--parallel-envs",
                   "--fast-igd", "--fast-hv3", "--device", config["device"],
                   "--threads", str(config["threads"])]
        run_command(command, RUNS / "server_logs" / f"{run_name}.log")
        if not completed_smoke(RUNS / run_name):
            raise RuntimeError(f"Smoke training did not finish: {run_name}")
    marker.write_text(json.dumps(expected, indent=2), encoding="utf-8")


def completed_smoke(run):
    path = run / "model" / "metadata.json"
    return path.is_file() and json.loads(path.read_text(encoding="utf-8"))["progress"]["training_steps"] > 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("preflight", "train", "evaluate", "all", "plan"),
                        default="all")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--scales", type=int, nargs="+", choices=(3, 5, 7), default=[3, 5, 7])
    parser.add_argument("--seeds", type=int, nargs="+", default=[2022, 2023, 2024])
    parser.add_argument("--eval-seed-start", type=int, default=10001)
    parser.add_argument("--eval-count", type=int, default=30)
    parser.add_argument("--min-free-gb", type=float, default=30)
    args = parser.parse_args()
    if args.threads < 1 or args.eval_count < 1 or args.min_free_gb < 0:
        parser.error("Threads and evaluation count must be positive; free-space limit cannot be negative")
    if len(set(args.scales)) != len(args.scales) or len(set(args.seeds)) != len(args.seeds):
        parser.error("Scales and training seeds must be unique")
    if args.eval_seed_start < 0 or min(args.seeds) < 0:
        parser.error("Seeds must be nonnegative")
    config = {"scales": args.scales, "seeds": args.seeds, "device": args.device,
              "threads": args.threads, "eval_seed_start": args.eval_seed_start,
              "eval_count": args.eval_count, "min_free_gb": args.min_free_gb,
              "total_steps": TOTAL_STEPS, "source_sha256": source_digest()}
    if args.stage == "plan":
        print(json.dumps(config, indent=2))
        return
    import fcntl  # Linux server lock; plan mode remains inspectable on Windows.
    RUNS.mkdir(exist_ok=True)
    with (RUNS / "server_full_experiment.lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Another full experiment driver is running") from error
        plan = RUNS / "server_formal_plan.json"
        if plan.exists():
            if json.loads(plan.read_text(encoding="utf-8")) != config:
                raise ValueError("Formal plan or source code changed; use the original frozen setup")
        else:
            plan.write_text(json.dumps(config, indent=2), encoding="utf-8")
        if args.stage in ("preflight", "all"):
            preflight(config)
        if args.stage in ("train", "all"):
            ensure_cuda(config["device"])
            train_all(config)
        if args.stage in ("evaluate", "all"):
            evaluate_all(config)
            summarize(config)


if __name__ == "__main__":
    main()
