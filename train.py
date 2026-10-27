"""Train GNN-Seq or its original 22-feature control in this isolated folder."""
import sys
sys.dont_write_bytecode = True

import argparse
import atexit
import json
import pickle
import platform
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from gnn_seq.environment import Environment
from gnn_seq.parallel_env import ParallelEnvironment
from gnn_seq.model import GraphSeqMAC, SeqMAC, SADNLearner
from gnn_seq.effect_learner import EffectSADNLearner
from gnn_seq.experiment import (defaults, seed_all, scheme_for, Logger, DiskReplay,
                                collect, save_checkpoint, GROUPS, PREPROCESS)
from gnn_seq.protocol import task_counts
from gnn_seq.resume import read_resume, restore_resume, save_resume
from components.episode_buffer import EpisodeBatch

ROOT = Path(__file__).resolve().parent


def main():
    base = defaults()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--resume", action="store_true",
                        help="Continue this run from its last committed rollout")
    parser.add_argument("--pause-after-rollouts", type=int,
                        help="Cleanly pause after this many total rollout batches")
    parser.add_argument("--mode", choices=["graph", "stats"], default="graph")
    parser.add_argument("--smoke", action="store_true", help="DTLZ2_5, N35, five generations; validation only")
    parser.add_argument("--seed", type=int, default=2022)
    parser.add_argument("--key")
    parser.add_argument("--population-size", type=int)
    parser.add_argument("--budget-ratio", type=int)
    parser.add_argument("--total-steps", type=int)
    parser.add_argument("--batch-episodes", type=int)
    parser.add_argument("--collect-episodes", type=int)
    parser.add_argument("--replay-episodes", type=int)
    parser.add_argument("--max-objectives", type=int, default=7)
    parser.add_argument("--weight-k", type=int, default=10)
    parser.add_argument("--objective-k", type=int, default=5)
    parser.add_argument("--graph-hidden-dim", type=int)
    parser.add_argument("--graph-policy", choices=["effect_attention", "action_attention", "pooled"],
                        default="action_attention")
    parser.add_argument("--graph-layers", type=int, choices=[0, 1, 2], default=2)
    parser.add_argument("--effect-loss-weight", type=float, default=0.1)
    parser.add_argument("--effect-head-lr", type=float, default=0.001)
    parser.add_argument("--checkpoint-interval", type=int,
                        help="Save a reviewable checkpoint after each interval (default: Seq 50000)")
    parser.add_argument("--eval-seeds", type=int, nargs="+", default=[9001, 9002])
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--parallel-envs", action="store_true",
                        help="Run independent MOEA/D environments in persistent processes")
    parser.add_argument("--fast-igd", action="store_true",
                        help="Use mathematically equivalent vectorized IGD for both modes")
    parser.add_argument("--fast-hv3", action="store_true",
                        help="Use mathematically equivalent exact 3D HV in the stats mode")
    cli = parser.parse_args()
    if not cli.name.replace("-", "").replace("_", "").isalnum():
        parser.error("name must contain only letters, digits, hyphens or underscores")
    if cli.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable")
    if cli.effect_loss_weight < 0 or cli.effect_head_lr <= 0:
        parser.error("effect loss weight must be nonnegative and head learning rate positive")
    if cli.pause_after_rollouts is not None and cli.pause_after_rollouts < 1:
        parser.error("pause-after-rollouts must be positive")
    torch.set_num_threads(cli.threads)
    config = {
        "key": cli.key or ("DTLZ2_5" if cli.smoke else base["env_args"]["key"]),
        "population_size": cli.population_size if cli.population_size is not None else (35 if cli.smoke else 210),
        "budget_ratio": cli.budget_ratio if cli.budget_ratio is not None else (1 if cli.smoke else base["env_args"]["budget_ratio"]),
        "n_ref_points": 1000, "max_objectives": cli.max_objectives,
        "weight_k": cli.weight_k, "objective_k": cli.objective_k, "dim_bias": 3,
        "operator_history": cli.mode == "graph" and cli.graph_policy == "effect_attention",
        "fast_igd": cli.fast_igd, "fast_hv3": cli.fast_hv3}
    total_steps = cli.total_steps if cli.total_steps is not None else (40 if cli.smoke else base["t_max"])
    batch_size = cli.batch_episodes if cli.batch_episodes is not None else (2 if cli.smoke else base["batch_size"])
    collect_size = cli.collect_episodes if cli.collect_episodes is not None else (2 if cli.smoke else base["batch_size_run"])
    capacity = cli.replay_episodes if cli.replay_episodes is not None else (8 if cli.smoke else base["buffer_size"])
    graph_hidden = cli.graph_hidden_dim if cli.graph_hidden_dim is not None else (16 if cli.smoke else 64)
    checkpoint_interval = (cli.checkpoint_interval if cli.checkpoint_interval is not None
                           else base["save_model_interval"])
    if min(config["population_size"], config["budget_ratio"], config["max_objectives"],
           config["weight_k"], config["objective_k"], total_steps, batch_size,
           collect_size, capacity, graph_hidden, cli.threads, checkpoint_interval) < 1:
        parser.error("sizes and budgets must be positive")
    if config["population_size"] < 31 or batch_size > capacity:
        parser.error("population must be >=31; batch size must be <= replay capacity")
    run = ROOT / "runs" / cli.name
    if cli.resume:
        if not (run / "resume.pt").is_file():
            parser.error(f"Missing committed resume snapshot: {run / 'resume.pt'}")
        saved = json.loads((run / "config.json").read_text(encoding="utf-8"))
        for key, value in saved["cli"].items():
            if key not in ("resume", "pause_after_rollouts") and getattr(cli, key) != value:
                parser.error(f"Resume configuration changed: {key}")
        if saved["environment"] != config:
            parser.error("Resume environment configuration changed")
        resume_state = read_resume(run)
    else:
        if run.exists():
            parser.error(f"Run already exists: {run}")
        run.mkdir(parents=True)
        resume_state = None
    seed_all(cli.seed)
    base.update(seed=cli.seed, device=cli.device, use_cuda=cli.device == "cuda",
                graph_hidden_dim=graph_hidden, max_objectives=cli.max_objectives,
                graph_policy=cli.graph_policy, graph_layers=cli.graph_layers,
                operator_history=config["operator_history"],
                effect_loss_weight=cli.effect_loss_weight,
                effect_head_lr=cli.effect_head_lr,
                critic_type="graph_cv_critic_ns" if cli.mode == "graph" else "cv_critic_ns",
                learner_log_interval=1, batch_size=batch_size, batch_size_run=collect_size,
                buffer_size=capacity, t_max=total_steps)
    base["env_args"] = {**base["env_args"], "key": config["key"], "seed": cli.seed,
                        "dim_bias": config["dim_bias"], "budget_ratio": config["budget_ratio"],
                        "wo_obs": cli.mode == "graph"}
    args = SimpleNamespace(**base)
    if not cli.resume:
        (run / "config.json").write_text(json.dumps({
            "cli": vars(cli), "environment": config, "learner": base,
            "runtime": {"python": platform.python_version(), "torch": torch.__version__,
                        "numpy": np.__version__, "cuda": torch.cuda.is_available()},
            "collector": "parallel persistent envs" if cli.parallel_envs else
                         "synchronous local envs, separate RNG per env"}, indent=2), encoding="utf-8")
    env_type = ParallelEnvironment if cli.parallel_envs else Environment
    envs = []
    for i in range(collect_size):
        snapshot = resume_state["environments"][i] if resume_state else None
        env = (env_type(config, cli.seed + i, cli.mode, snapshot=snapshot)
               if cli.parallel_envs else pickle.loads(snapshot) if snapshot is not None
               else env_type(config, cli.seed + i, cli.mode))
        envs.append(env)
        if hasattr(env, "close"):
            atexit.register(env.close)
    scheme = scheme_for(config, cli.mode)
    template = EpisodeBatch(scheme, GROUPS, 1, 2, preprocess=PREPROCESS)
    mac = (GraphSeqMAC if cli.mode == "graph" else SeqMAC)(template.scheme, GROUPS, args)
    logger = Logger()
    learner_type = (EffectSADNLearner if config["operator_history"] else SADNLearner)
    learner = learner_type(mac, template.scheme, logger, args)
    if args.use_cuda:
        learner.cuda()
    replay = DiskReplay(run / "replay", capacity, scheme)
    if cli.resume:
        progress = restore_resume(run, resume_state, learner, replay)
        t_env, episode_count = progress["t_env"], progress["episodes"]
        next_checkpoint = progress["next_checkpoint"]
        counts, rollout_count = progress["task_counts"].copy(), progress["rollouts"]
    else:
        t_env, episode_count, rollout_count = 0, 0, 0
        next_checkpoint = checkpoint_interval
        counts = {}
        # Preserve this run's exact zero-update policy for learning diagnostics.
        save_checkpoint(learner, run / "initial", args, config,
                        {"t_env": 0, "episodes": 0, "mode": cli.mode,
                         "training_steps": 0, "task_counts": {}})
    while t_env < total_steps:
        rollout_start = time.perf_counter()
        episodes, summaries = collect(envs, mac, scheme, args, t_env)
        collect_seconds = time.perf_counter() - rollout_start
        for episode in episodes:
            replay.insert(episode)
        t_env += sum(x["steps"] for x in summaries)
        episode_count += len(episodes)
        rollout_count += 1
        for task, count in task_counts(summaries).items():
            counts[task] = counts.get(task, 0) + count
        update_start = time.perf_counter()
        if len(replay.files) >= batch_size:
            # Same update count as original run_sequential after a parallel rollout.
            for _ in range(collect_size):
                learner.train(replay.sample(batch_size, args.device), t_env, episode_count)
        update_seconds = time.perf_counter() - update_start
        record = {"t_env": t_env, "episodes": episode_count,
                  "training_steps": learner.training_steps, "rollouts": summaries,
                  "task_counts": counts.copy(), "learner": logger.latest.copy(),
                  "timing": {"collect_seconds": collect_seconds,
                             "update_seconds": update_seconds}}
        if t_env >= next_checkpoint:
            save_checkpoint(learner, run / "checkpoints" / str(t_env), args, config,
                            {"t_env": t_env, "episodes": episode_count, "mode": cli.mode,
                             "training_steps": learner.training_steps, "task_counts": counts.copy()})
            while next_checkpoint <= t_env:
                next_checkpoint += checkpoint_interval
        save_resume(run, learner, replay, envs,
                    {"t_env": t_env, "episodes": episode_count,
                     "next_checkpoint": next_checkpoint, "task_counts": counts.copy(),
                     "rollouts": rollout_count}, record)
        with (run / "metrics.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        print(f"{cli.mode}: steps={t_env} updates={learner.training_steps} "
              f"collect_s={collect_seconds:.1f} update_s={update_seconds:.1f} "
              f"best_igd={np.mean([x['best_igd'] for x in summaries]):.6f}", flush=True)
        if cli.pause_after_rollouts is not None and rollout_count >= cli.pause_after_rollouts:
            print(f"Paused at committed rollout {rollout_count}; resume with --resume", flush=True)
            return
    save_checkpoint(learner, run / "model", args, config,
                    {"t_env": t_env, "episodes": episode_count, "mode": cli.mode,
                     "training_steps": learner.training_steps, "task_counts": counts.copy()})
    for env in envs:
        if hasattr(env, "close"):
            env.close()
    if config["key"].startswith("M_2_46_"):
        (run / "evaluation_plan.json").write_text(json.dumps({
            "status": "pending_explicit_problem_suite",
            "training_key": config["key"], "training_steps": learner.training_steps,
            "task_counts": counts,
            "example": f"python -B evaluate.py --model runs/{cli.name}/model "
                       f"--suite all --runs 30 --output runs/{cli.name}/evaluation_suite.json"
        }, indent=2), encoding="utf-8")
        print(f"Saved {run}; use evaluate.py --suite all for the eight-problem test", flush=True)
        return
    evaluations = []
    for seed in cli.eval_seeds:
        seed_all(seed)
        _, summary = collect([Environment(config, seed, cli.mode)], mac, scheme,
                             args, t_env, test_mode=True)
        evaluations.append({"seed": seed, **summary[0]})
    (run / "evaluation.json").write_text(json.dumps({
        "mode": cli.mode, "greedy": True, "evaluations": evaluations,
        "mean_best_igd": float(np.mean([s["best_igd"] for s in evaluations])),
        "claim": "Smoke runs validate execution, not optimization superiority."
    }, indent=2), encoding="utf-8")
    print(f"Saved {run}", flush=True)


if __name__ == "__main__":
    main()
