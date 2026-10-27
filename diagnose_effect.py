"""Measure factual local-effect prediction on fixed replay, without rollout changes."""
import sys
sys.dont_write_bytecode = True

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from diagnose_q import load_mac
from gnn_seq.experiment import GROUPS, PREPROCESS, scheme_for
from components.episode_buffer import EpisodeBatch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    cli = parser.parse_args()
    root = Path(__file__).resolve().parent
    output = cli.output.resolve()
    if not output.is_relative_to(root) or output.exists():
        parser.error("output must be a new file inside GNN-Seq")
    meta = json.loads((cli.model / "metadata.json").read_text(encoding="utf-8"))
    if meta["args"].get("graph_policy") != "effect_attention":
        parser.error("requires effect_attention")
    scheme = scheme_for(meta["environment"], "graph")
    mac = load_mac(cli.model, scheme, "graph")
    files = sorted(cli.replay.glob("*.pt"))
    if not files:
        parser.error("replay directory is empty")
    torch.set_num_threads(1)
    actual, predicted = [], []
    for file in files:
        saved = torch.load(file, map_location="cpu", weights_only=True)
        length = saved["filled"].shape[1]
        batch = EpisodeBatch(scheme, GROUPS, 1, length, preprocess=PREPROCESS)
        for key, value in saved.items():
            batch.data.transition_data[key][:] = value
        with torch.no_grad():
            for t in range(length - 1):
                a = batch["actions"][:, t, :, 0]
                predicted.append(mac.predict_local_effects(batch, t, a)[0].numpy())
                actual.append(np.stack((batch["local_gain"][0, t].numpy(),
                                        batch["local_success"][0, t].numpy()), -1))
    y, p = np.concatenate(actual), np.concatenate(predicted)
    def correlation(u, v):
        return float(np.corrcoef(u, v)[0, 1]) if u.std() > 0 and v.std() > 0 else None
    prevalence = y[:, 1].mean()
    report = {"checkpoint": str(cli.model.resolve()), "episodes": len(files),
              "node_transitions": len(y),
              "gain": {"actual_mean": float(y[:, 0].mean()),
                       "predicted_mean": float(p[:, 0].mean()),
                       "rmse": float(np.sqrt(np.mean((p[:, 0] - y[:, 0]) ** 2))),
                       "constant_rmse": float(y[:, 0].std()),
                       "correlation": correlation(y[:, 0], p[:, 0])},
              "success": {"actual_rate": float(prevalence),
                          "predicted_rate": float(p[:, 1].mean()),
                          "brier": float(np.mean((p[:, 1] - y[:, 1]) ** 2)),
                          "constant_brier": float(prevalence * (1 - prevalence)),
                          "correlation": correlation(y[:, 1], p[:, 1])},
              "note": "In-sample replay; prediction quality here is not generalization."}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"gain": report["gain"], "success": report["success"]}))
    print(output)


if __name__ == "__main__":
    main()
