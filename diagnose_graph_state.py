"""Measure graph feature variation, edge reach, and encoder node dispersion."""
import sys
sys.dont_write_bytecode = True

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from gnn_seq.model import GraphEncoder, symmetric_edges


def adjacency(neighbours, size):
    matrix = np.eye(size, dtype=bool)
    for source, targets in enumerate(neighbours):
        matrix[source, targets] = True
        matrix[targets, source] = True
    return matrix


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    cli = parser.parse_args()
    output = cli.output.resolve()
    if not output.is_relative_to(Path(__file__).resolve().parent) or output.exists():
        parser.error("output must be a new file inside GNN-Seq")
    files = sorted(cli.replay.glob("*.pt"))
    if not files:
        parser.error("replay directory is empty")
    metadata = json.loads((cli.model / "metadata.json").read_text(encoding="utf-8"))
    if metadata["progress"]["mode"] != "graph":
        parser.error("requires a graph checkpoint")
    torch.set_num_threads(1)
    encoder = GraphEncoder(SimpleNamespace(**metadata["args"]))
    encoder.load_state_dict(torch.load(cli.model / "graph_encoder.th", map_location="cpu",
                                       weights_only=True))
    encoder.eval()
    nodes_list, contexts = [], []
    node_dispersion = [[] for _ in range(len(encoder.layers) + 1)]
    degree, two_hop, relation_overlap, objective_stability = [], [], [], []
    for file in files:
        data = torch.load(file, map_location="cpu", weights_only=True)
        nodes = data["nodes"][0, :-1]
        wk = data["weight_knn"][0, :-1]
        ok = data["objective_knn"][0, :-1]
        context = data["context"][0, :-1]
        nodes_list.append(nodes.numpy())
        contexts.append(context.numpy())
        with torch.no_grad():
            we, oe = symmetric_edges(wk), symmetric_edges(ok)
            h = torch.relu(encoder.input(nodes))
            node_dispersion[0].append(float(h.std(1, unbiased=False).mean()))
            for layer_no, layer in enumerate(encoder.layers, 1):
                updated = layer(h, we, oe)
                h = (h + updated) * .5 if encoder.residual else updated
                if encoder.initial_skip and layer_no == len(encoder.layers):
                    h = (torch.relu(encoder.input(nodes)) + h) * .5
                node_dispersion[layer_no].append(float(h.std(1, unbiased=False).mean()))
        weights, objectives = wk.numpy(), ok.numpy()
        n = nodes.shape[1]
        for t in range(len(nodes)):
            w, o = adjacency(weights[t], n), adjacency(objectives[t], n)
            union = w | o
            degree.append(float((union.sum(1) - 1).mean()))
            two_hop.append(float(((union.astype(np.int16) @ union.astype(np.int16)) > 0).mean()))
            relation_overlap.append(float(np.mean([
                len(set(weights[t, i]) & set(objectives[t, i])) /
                len(set(weights[t, i]) | set(objectives[t, i])) for i in range(n)])))
            if t:
                objective_stability.append(float(np.mean([
                    len(set(objectives[t - 1, i]) & set(objectives[t, i])) /
                    len(set(objectives[t - 1, i]) | set(objectives[t, i]))
                    for i in range(n)])))
    all_nodes, all_contexts = np.concatenate(nodes_list), np.concatenate(contexts)
    m = metadata["environment"]["max_objectives"]
    groups = {"weights": slice(0, m), "objectives": slice(m, 2 * m),
              "ideal_gaps": slice(2 * m, 3 * m), "mask": slice(3 * m, 4 * m),
              "fitness_and_distances": slice(4 * m, 4 * m + 3),
              "history": slice(4 * m + 3, 4 * m + 9)}
    if metadata["environment"].get("operator_history", False):
        groups["operator_history"] = slice(4 * m + 9, 4 * m + 41)
    features = {}
    for name, positions in groups.items():
        values = all_nodes[:, :, positions]
        features[name] = {
            "within_graph_node_std": float(values.std(1).mean()),
            "between_graph_mean_std": float(values.mean(1).std(0).mean()),
            "zero_fraction": float((values == 0).mean())}
    report = {"episodes": len(files), "states": len(all_nodes),
              "population": all_nodes.shape[1],
              "mean_union_neighbours": float(np.mean(degree)),
              "mean_two_hop_reachable_fraction": float(np.mean(two_hop)),
              "mean_directed_relation_jaccard": float(np.mean(relation_overlap)),
              "mean_objective_neighbour_jaccard_next_generation": float(np.mean(objective_stability)),
              "graph_layers": len(encoder.layers),
              "mean_node_hidden_std_by_stage": [float(np.mean(x)) for x in node_dispersion],
              "feature_groups": features,
              "context_mean_std": float(all_contexts.std(0).mean())}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"two-hop reach {report['mean_two_hop_reachable_fraction']:.3f}; "
          f"node dispersion {report['mean_node_hidden_std_by_stage']}")
    print(output)


if __name__ == "__main__":
    main()
