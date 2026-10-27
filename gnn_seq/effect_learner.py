"""Observed joint-action effect supervision alongside unchanged original SADN."""
from pathlib import Path

import torch
from torch.nn import functional as F

from .model import SADNLearner


class EffectSADNLearner(SADNLearner):
    def __init__(self, mac, scheme, logger, args):
        super().__init__(mac, scheme, logger, args)
        if mac.graph_policy != "effect_attention":
            raise ValueError("EffectSADNLearner requires effect_attention")
        self.effect_weight = float(args.effect_loss_weight)
        head = [p for readout in mac.node_readouts
                for p in readout.effect_score.parameters()]
        head_ids = {id(p) for p in head}
        shared = [p for p in list(mac.encoder.parameters()) +
                  list(mac.node_readouts.parameters()) if id(p) not in head_ids]
        self.effect_optimizer = torch.optim.Adam(
            [{"params": shared, "lr": args.lr},
             {"params": head, "lr": getattr(args, "effect_head_lr", .001)}])

    def train(self, batch, t_env, episode_num):
        # Keep the original SADN actor/critic update byte-for-byte.
        super().train(batch, t_env, episode_num)
        if self.effect_weight == 0:
            return
        self.effect_optimizer.zero_grad()
        valid = batch["filled"][:, :-1].float()
        valid[:, 1:] *= 1 - batch["terminated"][:, :-2].float()
        normalizer = valid.sum() * batch["local_gain"].shape[-1]
        if normalizer <= 0:
            return
        total_gain, total_success = 0., 0.
        for t in range(batch.max_seq_length - 1):
            if not valid[:, t].any():
                continue
            prediction = self.mac.predict_local_effects(
                batch, t, batch["actions"][:, t, :, 0])
            mask = valid[:, t, None]
            gain = F.smooth_l1_loss(prediction[..., 0], batch["local_gain"][:, t],
                                    reduction="none")
            success = F.binary_cross_entropy(prediction[..., 1],
                                             batch["local_success"][:, t], reduction="none")
            gain_loss = (gain * mask).sum() / normalizer
            success_loss = (success * mask).sum() / normalizer
            (self.effect_weight * (gain_loss + success_loss)).backward()
            total_gain += float(gain_loss.detach())
            total_success += float(success_loss.detach())
        torch.nn.utils.clip_grad_norm_(
            list(self.mac.encoder.parameters()) + list(self.mac.node_readouts.parameters()),
            self.args.grad_norm_clip)
        self.effect_optimizer.step()
        self.logger.log_stat("effect_gain_loss", total_gain, t_env)
        self.logger.log_stat("effect_success_loss", total_success, t_env)

    def save_models(self, path):
        super().save_models(path)
        torch.save(self.effect_optimizer.state_dict(), Path(path) / "effect_opt.th")

    def load_models(self, path):
        super().load_models(path)
        state = Path(path) / "effect_opt.th"
        if state.exists():
            self.effect_optimizer.load_state_dict(torch.load(
                state, map_location="cpu", weights_only=True))
