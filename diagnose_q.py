"""Inspect fixed-replay Q values before and after learning, without changing evolution."""
import sys
sys.dont_write_bytecode = True

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from gnn_seq import source  # configure the frozen Seq module search path
from components.episode_buffer import EpisodeBatch
from gnn_seq.experiment import GROUPS, PREPROCESS, scheme_for
from gnn_seq.model import GraphSeqMAC, SeqMAC


def load_mac(checkpoint, scheme, mode):
    metadata = json.loads((checkpoint / "metadata.json").read_text(encoding="utf-8"))
    args = SimpleNamespace(**metadata["args"])
    args.device, args.use_cuda = "cpu", False
    template = EpisodeBatch(scheme, GROUPS, 1, 2, preprocess=PREPROCESS)
    mac = (GraphSeqMAC if mode == "graph" else SeqMAC)(template.scheme, GROUPS, args)
    mac.load_models(str(checkpoint))
    for agent in mac.agents:
        agent.eval()
    if mode == "graph":
        mac.encoder.eval()
        if mac.node_readouts is not None:
            mac.node_readouts.eval()
    return mac


def fixed_q(mac, files, scheme):
    outputs, masks = [], []
    for file in files:
        saved = torch.load(file, map_location="cpu", weights_only=True)
        length = saved["filled"].shape[1]
        batch = EpisodeBatch(scheme, GROUPS, 1, length, preprocess=PREPROCESS)
        for key, value in saved.items():
            batch.data.transition_data[key][:] = value
        actions = batch["actions"]
        for t in range(length - 1):
            mac.init_hidden(1)
            with torch.no_grad():
                q = mac.forward(batch, t, 0, actions=actions, test_mode=True)
            outputs.append(q[0].cpu())
            masks.append(batch["avail_actions"][0, t].bool().cpu())
    return torch.stack(outputs), torch.stack(masks)


def summarize(q, mask):
    masked = q.masked_fill(~mask, -1e10)
    best = masked.argmax(-1)
    margins = masked.topk(2, dim=-1).values.diff(dim=-1).abs().squeeze(-1)
    result = []
    for i in range(4):
        valid = mask[:, i].sum(-1) >= 2
        actual = margins[valid, i]
        result.append({"action_counts": {str(a): int((best[:, i] == a).sum()) for a in range(4)},
                       "states_with_two_or_more_actions": int(valid.sum()),
                       "mean_q_std_over_actions": float(q[:, i].std(0).mean()),
                       "median_top2_margin": float(actual.median()) if len(actual) else None,
                       "mean_top2_margin": float(actual.mean()) if len(actual) else None})
    return result


def graph_embedding_summary(mac, files):
    embeddings, node_means = [], []
    for file in files:
        saved = torch.load(file, map_location="cpu", weights_only=True)
        nodes = saved["nodes"][0, :-1]
        with torch.no_grad():
            z = mac.encoder(nodes, saved["weight_knn"][0, :-1],
                            saved["objective_knn"][0, :-1],
                            saved["context"][0, :-1])
        embeddings.append(z)
        node_means.append(nodes.mean(1))
    z, x = torch.cat(embeddings), torch.cat(node_means)
    return {"mean_feature_std": float(z.std(0).mean()),
            "active_dimensions": int((z.abs().max(0).values > 1e-8).sum()),
            "zero_fraction": float((z == 0).float().mean()),
            "mean_pooled_node_feature_std": float(x.std(0).mean())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    cli = parser.parse_args()
    root = Path(__file__).resolve().parent
    output = cli.output.resolve()
    if not output.is_relative_to(root) or output.exists():
        parser.error("output must be a new file inside GNN-Seq")
    metadata = [json.loads((checkpoint / "metadata.json").read_text(encoding="utf-8"))
                for checkpoint in (cli.before, cli.after)]
    if metadata[0]["environment"] != metadata[1]["environment"]:
        parser.error("checkpoints must have the same environment")
    mode = metadata[0]["progress"]["mode"]
    if mode != metadata[1]["progress"]["mode"]:
        parser.error("checkpoints must have the same mode")
    files = sorted(cli.replay.glob("*.pt"))
    if not files:
        parser.error("replay directory has no episodes")
    torch.set_num_threads(1)
    scheme = scheme_for(metadata[0]["environment"], mode)
    mac0, mac1 = load_mac(cli.before, scheme, mode), load_mac(cli.after, scheme, mode)
    q0, mask0 = fixed_q(mac0, files, scheme)
    q1, mask1 = fixed_q(mac1, files, scheme)
    if not torch.equal(mask0, mask1):
        raise RuntimeError("Fixed replay action masks changed")
    a0 = q0.masked_fill(~mask0, -1e10).argmax(-1)
    a1 = q1.masked_fill(~mask1, -1e10).argmax(-1)
    report = {"episodes": len(files), "states": len(q0),
              "checkpoint_updates": [m["progress"]["training_steps"] for m in metadata],
              "mean_absolute_q_change": float((q1 - q0).abs().mean()),
              "changed_greedy_actions_by_agent": (a0 != a1).sum(0).tolist(),
              "before": summarize(q0, mask0), "after": summarize(q1, mask1)}
    if mode == "graph":
        report["graph_policy"] = getattr(mac0.args, "graph_policy", "pooled")
        report["graph_embedding"] = {"before": graph_embedding_summary(mac0, files),
                                     "after": graph_embedding_summary(mac1, files)}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"fixed states: {report['states']}; action changes by agent: "
          f"{report['changed_greedy_actions_by_agent']}")
    print(f"mean absolute Q change: {report['mean_absolute_q_change']:.6f}")
    print(output)


if __name__ == "__main__":
    main()
