"""Experiment wiring: raw-state disk replay and synchronized episode collection."""
import json
import random
from collections import deque
from pathlib import Path

import numpy as np
import torch
import yaml

from .source import SEQ
from components.episode_buffer import EpisodeBatch
from components.transforms import OneHot


def defaults():
    data = {}
    for file in ("default.yaml", "algs/sadn_ns.yaml", "envs/moea.yaml"):
        data.update(yaml.safe_load((SEQ / "algos/madac/config" / file).read_text(encoding="utf-8")))
    data.update(n_agents=4, n_actions=4, device="cpu", use_cuda=False,
                graph_hidden_dim=64, graph_output_dim=64, max_objectives=7,
                graph_checkpoint=True, critic_time_chunk=16,
                graph_policy="action_attention")
    return data


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def scheme_for(config, mode):
    n, m = config["population_size"], config["max_objectives"]
    scheme = {
        "state": {"vshape": 22 if mode == "stats" else 1},
        "obs": {"vshape": 22 if mode == "stats" else 1, "group": "agents"},
        "actions": {"vshape": (1,), "group": "agents", "dtype": torch.long},
        "avail_actions": {"vshape": (4,), "group": "agents", "dtype": torch.int},
        "reward": {"vshape": (1,)},
        "terminated": {"vshape": (1,), "dtype": torch.uint8}}
    if mode == "graph":
        effect = config.get("operator_history", False)
        scheme.update({
            "nodes": {"vshape": (n, 4 * m + 9 + (32 if effect else 0))},
            "weight_knn": {"vshape": (n, min(config["weight_k"], n - 1)), "dtype": torch.int32},
            "objective_knn": {"vshape": (n, min(config["objective_k"], n - 1)), "dtype": torch.int32},
            "context": {"vshape": 20}})
        if effect:
            scheme.update({"local_gain": {"vshape": (n,)},
                           "local_success": {"vshape": (n,)}})
    return scheme


GROUPS = {"agents": 4}
PREPROCESS = {"actions": ("actions_onehot", [OneHot(out_dim=4)])}


class Logger:
    def __init__(self):
        self.latest = {}

    def log_stat(self, key, value, step):
        if not np.isfinite(value):
            raise FloatingPointError(f"Nonfinite learner metric {key}")
        self.latest[key] = float(value)


class DiskReplay:
    """Bounded episode count, no huge upfront allocation or cached embeddings."""
    def __init__(self, directory, capacity, scheme):
        self.directory, self.capacity, self.scheme = Path(directory), capacity, scheme
        self.directory.mkdir(exist_ok=True)
        self.files, self.sequence = deque(), 0

    def insert(self, episode):
        if episode.batch_size != 1:
            raise ValueError("Insert one episode at a time")
        filename = self.directory / f"{self.sequence:08d}.pt"
        # Plain tensors only: safe weights_only loading.
        torch.save({k: v.detach().cpu().clone()
                    for k, v in episode.data.transition_data.items()}, filename)
        self.files.append(filename)
        self.sequence += 1
        if len(self.files) > self.capacity:
            self.files.popleft().unlink()

    def sample(self, count, device):
        files = random.sample(list(self.files), count)
        episodes = [torch.load(f, map_location="cpu", weights_only=True) for f in files]
        length = max(x["filled"].shape[1] for x in episodes)
        batch = EpisodeBatch(self.scheme, GROUPS, count, length, preprocess=PREPROCESS)
        for i, tensors in enumerate(episodes):
            for key, value in tensors.items():
                batch.data.transition_data[key][i:i + 1, :value.shape[1]] = value
        batch.to(device)
        return batch


def collect(envs, mac, scheme, args, t_env, test_mode=False):
    """Synchronous batch stepping, with separate worker-equivalent env RNG.

    Original parallel runner holds t_env constant for a rollout and performs
    batch_size_run updates afterwards. Retain that schedule, using local envs.
    """
    length = max(env.episode_limit for env in envs) + 1
    batch = EpisodeBatch(scheme, GROUPS, len(envs), length,
                         preprocess=PREPROCESS, device=args.device)
    parallel = all(hasattr(env, "submit_step") for env in envs)
    if parallel:
        for env in envs:
            env.submit_reset()
        states = [env.receive_reset() for env in envs]
    else:
        states = [env.reset() for env in envs]
    done = [False] * len(envs)
    summaries = [{"seed": env.seed,
                  "function": env.metadata["function"] if parallel else env.core.env_name,
                  "objectives": env.metadata["objectives"] if parallel else env.core.n_objs,
                  "variables": env.metadata["variables"] if parallel else env.core.problem.nvars,
                  "population_size": env.metadata["population_size"] if parallel else env.core.population_size,
                  "steps": 0, "return": 0., "actions": [], "rewards": [],
                  "weight_updates": 0, "replacement_events": 0} for env in envs]
    step = 0
    while True:
        active = [i for i, flag in enumerate(done) if not flag]
        for i in active:
            batch.update({k: np.asarray(v)[None] for k, v in states[i].items()}, bs=i, ts=step)
        if not active:
            break
        # use_rnn=False: no memory is discarded here; original actors retained.
        mac.init_hidden(len(active))
        with torch.no_grad():
            # Actors use only the current observation and within-step action prefix.
            # Avoid copying the entire growing episode at every generation.
            actions = mac.select_actions(batch[active, step:step + 1], 0, t_env,
                                         test_mode=test_mode)
        action_array = actions.cpu().numpy()
        if parallel:
            for row, i in enumerate(active):
                envs[i].submit_step(action_array[row])
        for row, i in enumerate(active):
            action = action_array[row]
            if parallel:
                states[i], reward, done[i], info = envs[i].receive_step()
            else:
                states[i], reward, done[i], info = envs[i].step(action)
            transition = {"actions": action[None], "reward": [[reward]],
                          "terminated": [[done[i]]]}
            if "local_gain" in scheme:
                transition.update(local_gain=envs[i].last_local_gain[None],
                                  local_success=envs[i].last_local_success[None])
            batch.update(transition, bs=i, ts=step)
            summary = summaries[i]
            summary["steps"] += 1
            summary["return"] += reward
            summary["actions"].append(action.tolist())
            summary["rewards"].append(reward)
            summary.update({k: float(v) for k, v in info.items()})
            summary["weight_updates"] = (envs[i].weight_revision if parallel
                                         else envs[i].core.weight_revision)
            summary["replacement_events"] += (envs[i].replacement_events if parallel
                                              else len(envs[i].core.history.events))
            if done[i]:
                batch.update({k: np.asarray(v)[None] for k, v in states[i].items()}, bs=i, ts=step + 1)
                mac.init_hidden(1)
                with torch.no_grad():
                    final_action = mac.select_actions(batch[i:i + 1, step + 1:step + 2], 0, t_env,
                                                      test_mode=test_mode)
                batch.update({"actions": final_action.cpu().numpy()}, bs=i, ts=step + 1)
        step += 1
        if step >= length:
            raise RuntimeError("Episode exceeded the original budget")
    episodes = [batch[i:i + 1, :s["steps"] + 1] for i, s in enumerate(summaries)]
    return episodes, summaries


def save_checkpoint(learner, directory, args, config, progress):
    directory = Path(directory)
    directory.mkdir(parents=True)
    learner.save_models(str(directory))
    # Original save_models omits target V and running reward stats; include them
    # separately for faithful inspection/reuse of the trained model.
    extra = {"target": learner.target_Vnet.state_dict(),
             "training_steps": learner.training_steps,
             "last_target_update_step": learner.last_target_update_step}
    for attr in ("rew_ms", "ret_ms"):
        if hasattr(learner, attr):
            obj = getattr(learner, attr)
            extra[attr] = {k: getattr(obj, k) for k in ("mean", "var", "count")}
    torch.save(extra, directory / "learner_extra.pt")
    (directory / "metadata.json").write_text(json.dumps(
        {"args": vars(args), "environment": config, "progress": progress}, indent=2),
        encoding="utf-8")
