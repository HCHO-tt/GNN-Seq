"""Measure one N=210 graph SADN update with the original batch size 32.

Resource probe only: repeats existing pilot episodes to match the tensor shape;
neither its gradient nor its result is a training or performance observation.
"""
import sys
sys.dont_write_bytecode = True

import json
import time
from pathlib import Path
from types import SimpleNamespace

import psutil
import torch

from gnn_seq.experiment import GROUPS, PREPROCESS, Logger, scheme_for
from gnn_seq.model import GraphSeqMAC, SADNLearner
from components.episode_buffer import EpisodeBatch


def main():
    root = Path(__file__).resolve().parent
    pilot = root / "runs" / "formal_parallel_slice_graph_m3_s2022"
    meta = json.loads((pilot / "model" / "metadata.json").read_text(encoding="utf-8"))
    config, args = meta["environment"], SimpleNamespace(**meta["args"])
    assert config["population_size"] == 210 and args.graph_layers == 2
    assert args.device == "cpu" and args.batch_size == 4
    torch.set_num_threads(1)
    scheme = scheme_for(config, "graph")
    paths = sorted((pilot / "replay").glob("*.pt"))
    assert len(paths) == 12
    available = psutil.virtual_memory().available
    print(f"free_ram_before_GB={available / 1e9:.2f}", flush=True)
    if available < 2_000_000_000:
        raise RuntimeError("Insufficient free RAM for the batch-32 resource probe")
    files = [torch.load(paths[i % len(paths)], map_location="cpu", weights_only=True)
             for i in range(32)]
    length = files[0]["filled"].shape[1]
    batch = EpisodeBatch(scheme, GROUPS, 32, length, preprocess=PREPROCESS)
    for i, tensors in enumerate(files):
        for key, value in tensors.items():
            batch.data.transition_data[key][i:i + 1, :value.shape[1]] = value
    del files
    template = EpisodeBatch(scheme, GROUPS, 1, 2, preprocess=PREPROCESS)
    mac = GraphSeqMAC(template.scheme, GROUPS, args)
    learner = SADNLearner(mac, template.scheme, Logger(), args)
    process = psutil.Process()
    print(f"loaded_batch_rss_GB={process.memory_info().rss / 1e9:.2f}", flush=True)
    start = time.perf_counter()
    learner.train(batch, 10800, 36)
    print(json.dumps({"batch_size": 32, "episode_length": length,
                      "elapsed_seconds": time.perf_counter() - start,
                      "end_rss_GB": process.memory_info().rss / 1e9,
                      "free_ram_after_GB": psutil.virtual_memory().available / 1e9,
                      "training_steps": learner.training_steps,
                      "resource_probe_only": True}), flush=True)


if __name__ == "__main__":
    main()
