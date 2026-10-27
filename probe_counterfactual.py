"""One-generation paired action probes from identical MOEA/D state and RNG."""
import sys
sys.dont_write_bytecode = True

import argparse
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from gnn_seq.environment import Environment
from gnn_seq.experiment import GROUPS, PREPROCESS, scheme_for
from gnn_seq.model import GraphSeqMAC
from components.episode_buffer import EpisodeBatch


def state_batch(state, scheme):
    batch = EpisodeBatch(scheme, GROUPS, 1, 2, preprocess=PREPROCESS)
    batch.update({key: np.asarray(value)[None] for key, value in state.items()}, ts=0)
    return batch


def probe(env, state, mac, scheme, anchor=None):
    batch = state_batch(state, scheme)
    if anchor is None:
        mac.init_hidden(1)
        with torch.no_grad():
            base = mac.select_actions(batch, 0, 0, test_mode=True)[0].cpu().tolist()
    else:
        base = list(anchor)
        if any(not state["avail_actions"][i, action] for i, action in enumerate(base)):
            raise ValueError(f"Anchor action is illegal in this state: {base}")
    candidates = [base]
    for agent in range(4):
        for action in np.flatnonzero(state["avail_actions"][agent]):
            if action != base[agent]:
                variant = base.copy()
                variant[agent] = int(action)
                candidates.append(variant)
    rows = []
    for candidate in candidates:
        branch = copy.deepcopy(env)
        _, reward, _, info = branch.step(candidate)
        with torch.no_grad():
            predicted = mac.predict_local_effects(
                batch, 0, torch.tensor(candidate, dtype=torch.long)[None])[0].cpu().numpy()
        rows.append({"actions": candidate, "reward": reward,
                     "final_igd": float(info["last_igd"]),
                     "mean_source_gain": float(branch.last_local_gain.mean()),
                     "successful_sources": int(branch.last_local_success.sum()),
                     "predicted_mean_gain": float(predicted[:, 0].mean()),
                     "predicted_success_rate": float(predicted[:, 1].mean())})
    baseline = rows[0]
    for row in rows:
        row["igd_minus_baseline"] = row["final_igd"] - baseline["final_igd"]
        row["gain_minus_baseline"] = row["mean_source_gain"] - baseline["mean_source_gain"]
        row["predicted_gain_minus_baseline"] = (
            row["predicted_mean_gain"] - baseline["predicted_mean_gain"])
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=9301)
    parser.add_argument("--states", type=int, default=3)
    parser.add_argument("--anchor-action", type=int, nargs=4,
                        help="Use the same four-action anchor and trajectory for every model")
    cli = parser.parse_args()
    root = Path(__file__).resolve().parent
    output = cli.output.resolve()
    if not output.is_relative_to(root) or output.exists():
        parser.error("output must be a new file inside GNN-Seq")
    if cli.states < 1:
        parser.error("states must be positive")
    if cli.anchor_action is not None and any(a < 0 or a >= 4 for a in cli.anchor_action):
        parser.error("anchor actions must be in [0, 3]")
    metadata = json.loads((cli.model / "metadata.json").read_text(encoding="utf-8"))
    if metadata["args"].get("graph_policy") != "effect_attention":
        parser.error("requires an effect_attention checkpoint")
    args = SimpleNamespace(**metadata["args"])
    args.device, args.use_cuda = "cpu", False
    torch.set_num_threads(1)
    config = metadata["environment"]
    scheme = scheme_for(config, "graph")
    template = EpisodeBatch(scheme, GROUPS, 1, 2, preprocess=PREPROCESS)
    mac = GraphSeqMAC(template.scheme, GROUPS, args)
    mac.load_models(str(cli.model))
    env = Environment(config, cli.seed)
    state = env.reset()
    probes = []
    for step in range(min(cli.states, env.episode_limit)):
        rows = probe(env, state, mac, scheme, cli.anchor_action)
        probes.append({"generation": step, "candidates": rows})
        # An explicit anchor makes all compared checkpoints visit the same states.
        state, _, done, _ = env.step(rows[0]["actions"])
        if done:
            break
    pairs = [row for item in probes for row in item["candidates"][1:]]
    actual = np.asarray([row["gain_minus_baseline"] for row in pairs])
    predicted = np.asarray([row["predicted_gain_minus_baseline"] for row in pairs])
    correlation = (float(np.corrcoef(actual, predicted)[0, 1])
                   if len(pairs) >= 3 and actual.std() > 0 and predicted.std() > 0 else None)
    report = {"checkpoint": str(cli.model.resolve()), "seed": cli.seed,
              "states": len(probes), "pairs": len(pairs),
              "anchor_action": cli.anchor_action,
              "predicted_vs_actual_gain_delta_correlation": correlation,
              "note": "One-generation common-RNG probes; not long-horizon causal effects.",
              "probes": probes}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"states={len(probes)} pairs={len(pairs)} correlation={correlation}")
    print(output)


if __name__ == "__main__":
    main()
