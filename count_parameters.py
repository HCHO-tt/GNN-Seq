"""Count saved actor, encoder, and critic parameters without changing models."""
import sys
sys.dont_write_bytecode = True

import argparse
from pathlib import Path

import torch


def count(file):
    state = torch.load(file, map_location="cpu", weights_only=True)
    def elements(value):
        if isinstance(value, torch.Tensor):
            return value.numel()
        if isinstance(value, dict):
            return sum(elements(item) for item in value.values())
        if isinstance(value, (list, tuple)):
            return sum(elements(item) for item in value)
        return 0
    return elements(state)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    cli = parser.parse_args()
    actor = sum(count(cli.model / f"agent{i}.th") for i in range(4))
    encoder_file = cli.model / "graph_encoder.th"
    encoder = count(encoder_file) if encoder_file.exists() else 0
    readout_file = cli.model / "action_readouts.th"
    readout = count(readout_file) if readout_file.exists() else 0
    critic = count(cli.model / "critic.th")
    print(f"actor={actor} encoder={encoder} action_readouts={readout} "
          f"policy_total={actor + encoder + readout} "
          f"critic={critic} policy_plus_critic={actor + encoder + readout + critic}")


if __name__ == "__main__":
    main()
