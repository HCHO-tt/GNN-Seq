"""Original Seq actors with optional action-conditioned node readout."""
from pathlib import Path

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from .source import SADNLearner  # exact original class, not a new loss
from controllers.seq_controller import SeqMAC
from modules.critics import REGISTRY
from modules.critics.mlp import MLP


def symmetric_edges(neighbours):
    """Sparse union of directed kNN and its reverse, with duplicates removed."""
    b, n, k = neighbours.shape
    device = neighbours.device
    offset = torch.arange(b, device=device)[:, None, None] * n
    source = (torch.arange(n, device=device)[None, :, None] + offset).expand(b, n, k).reshape(-1)
    target = (neighbours.long() + offset).reshape(-1)
    size = b * n
    encoded = torch.cat((source * size + target, target * size + source))
    encoded = encoded[source.repeat(2) != target.repeat(2)]
    unique = torch.unique(encoded)
    return unique // size, unique % size


def neighbour_mean(hidden, edges):
    source, target = edges
    flat = hidden.reshape(-1, hidden.shape[-1])
    total = torch.zeros_like(flat).index_add(0, target, flat[source])
    count = flat.new_zeros(len(flat)).index_add(0, target, flat.new_ones(len(target)))
    return (total / count.clamp_min(1)[:, None]).reshape_as(hidden)


class RelationLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.self_map = nn.Linear(dim, dim)
        self.weight_map = nn.Linear(dim, dim, bias=False)
        self.objective_map = nn.Linear(dim, dim, bias=False)

    def forward(self, hidden, weight_edges, objective_edges):
        return torch.relu(self.self_map(hidden)
                          + self.weight_map(neighbour_mean(hidden, weight_edges))
                          + self.objective_map(neighbour_mean(hidden, objective_edges)))


class GraphEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        d = args.graph_hidden_dim
        self.input = nn.Linear(4 * args.max_objectives + 9 +
                               (32 if getattr(args, "operator_history", False) else 0), d)
        self.layers = nn.ModuleList([RelationLayer(d) for _ in range(
            getattr(args, "graph_layers", 2))])
        self.readout = nn.Sequential(nn.Linear(2 * d + 20, d), nn.ReLU(),
                                     nn.Linear(d, args.graph_output_dim), nn.ReLU())
        self.residual = getattr(args, "graph_policy", "pooled") in (
            "action_attention", "effect_attention")
        self.initial_skip = getattr(args, "graph_policy", "pooled") == "effect_attention"

    def forward(self, nodes, weights, objectives, context, return_nodes=False):
        we, oe = symmetric_edges(weights), symmetric_edges(objectives)
        hidden = torch.relu(self.input(nodes))
        initial = hidden
        for layer in self.layers:
            updated = layer(hidden, we, oe)
            hidden = (hidden + updated) * .5 if self.residual else updated
        if self.initial_skip and self.layers:
            hidden = (initial + hidden) * .5
        pooled = torch.cat((hidden.mean(1), hidden.max(1).values, context), -1)
        embedding = self.readout(pooled)
        return (embedding, hidden) if return_nodes else embedding


def encode(encoder, batch, ts, args, return_nodes=False):
    tensors = [batch[key][:, ts] for key in
               ("nodes", "weight_knn", "objective_knn", "context")]
    # An integer time index already has shape B,N,F.
    if not isinstance(ts, int):
        tensors = [x.flatten(0, 1) for x in tensors]
    def run(*inputs):
        return encoder(*inputs, return_nodes=return_nodes)
    if torch.is_grad_enabled() and args.graph_checkpoint:
        return checkpoint(run, *tensors, use_reentrant=False)
    return run(*tensors)


class ActionNodeReadout(nn.Module):
    """Give each candidate action its own query over the current population."""
    def __init__(self, args, agent_id):
        super().__init__()
        d = args.graph_hidden_dim
        self.agent_id = agent_id
        self.action = nn.Embedding(args.n_actions, d)
        self.global_query = nn.Linear(args.graph_output_dim, d, bias=False)
        self.prefix_query = nn.Linear(agent_id, d, bias=False) if agent_id else None
        self.key = nn.Linear(d, d, bias=False)
        self.value = nn.Linear(d, d, bias=False)
        self.node_norm = nn.LayerNorm(d)
        self.query_norm = nn.LayerNorm(d)
        self.score = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, 1))
        self.effect_score = (nn.Linear(d, 2) if getattr(args, "graph_policy", "pooled")
                             == "effect_attention" else None)
        self.effect_gate = nn.Parameter(torch.tensor(-2.)) if self.effect_score else None
        if self.effect_score is not None:
            nn.init.normal_(self.effect_score.weight, std=.02)
            with torch.no_grad():
                self.effect_score.bias.copy_(torch.tensor([-2.2, -1.4]))

    def _query_values(self, embedding, nodes, prefix):
        batch = embedding.shape[0]
        query = self.action.weight.unsqueeze(0) + self.global_query(embedding)[:, None]
        if self.agent_id:
            query = query + self.prefix_query(prefix.float().reshape(batch, self.agent_id))[:, None]
        query = self.query_norm(query)
        nodes = self.node_norm(nodes)
        values = self.value(nodes)
        logits = torch.matmul(query, self.key(nodes).transpose(1, 2)) / nodes.shape[-1] ** .5
        return query, values, logits.softmax(-1)

    def node_effects(self, embedding, nodes, prefix):
        if self.effect_score is None:
            raise ValueError("Effect prediction requires effect_attention")
        query, values, _ = self._query_values(embedding, nodes, prefix)
        return self.effect_score(query[:, :, None] * values[:, None])

    def forward(self, embedding, nodes, prefix):
        query, values, attention = self._query_values(embedding, nodes, prefix)
        attended = torch.matmul(attention, values)
        # The additional Q branch must use graph evidence for each action.
        scores = self.score(query * attended).squeeze(-1)
        if self.effect_score is not None:
            local_gain = self.effect_score(query[:, :, None] * values[:, None])[..., 0].sigmoid()
            scores = scores + self.effect_gate.sigmoid() * (attention * local_gain).sum(-1)
        return scores - scores.mean(-1, keepdim=True)


class GraphSeqMAC(SeqMAC):
    def __init__(self, scheme, groups, args):
        super().__init__(scheme, groups, args)
        self.encoder = GraphEncoder(args)
        self.graph_policy = getattr(args, "graph_policy", "pooled")
        if self.graph_policy not in ("pooled", "action_attention", "effect_attention"):
            raise ValueError(f"Unknown graph policy: {self.graph_policy}")
        self.node_readouts = (nn.ModuleList([ActionNodeReadout(args, i)
                                            for i in range(self.n_agents)])
                              if self.graph_policy != "pooled" else None)

    def predict_local_effects(self, batch, t, actions):
        if self.graph_policy != "effect_attention":
            raise ValueError("Effect prediction requires effect_attention")
        embedding, nodes = encode(self.encoder, batch, t, self.args, return_nodes=True)
        selected = []
        for agent, readout in enumerate(self.node_readouts):
            prefix = actions[:, :agent] if agent else None
            logits = readout.node_effects(embedding, nodes, prefix)
            index = actions[:, agent, None, None, None].expand(-1, 1, nodes.shape[1], 2)
            selected.append(logits.gather(1, index).squeeze(1))
        return torch.stack(selected).mean(0).sigmoid()

    def _get_input_shape_agent(self, agent_id, scheme):
        return self.args.graph_output_dim + agent_id

    def _build_inputs_agent(self, agent_id, pre_action, batch, t):
        self._prefix = pre_action
        parts = [self._embedding]
        if agent_id:
            parts.append(pre_action.reshape(batch.batch_size, -1))
        return torch.cat(parts, -1)

    def _forward_agent(self, agent_input, hidden_state, agent_id):
        scores, hidden = super()._forward_agent(agent_input, hidden_state, agent_id)
        if self.node_readouts is not None:
            scores = scores + self.node_readouts[agent_id](
                self._embedding, self._nodes, self._prefix)
        return scores, hidden

    def forward(self, ep_batch, t, t_env, actions=None, bs=slice(None), test_mode=False):
        if bs != slice(None):
            raise ValueError("Collector must slice active episodes before calling SeqMAC")
        if self.node_readouts is None:
            self._embedding = encode(self.encoder, ep_batch, t, self.args)
        else:
            self._embedding, self._nodes = encode(
                self.encoder, ep_batch, t, self.args, return_nodes=True)
        try:
            return super().forward(ep_batch, t, t_env, actions, bs, test_mode)
        finally:
            self._embedding = None
            self._nodes = None
            self._prefix = None

    def parameters(self):
        groups = super().parameters() + [{"params": self.encoder.parameters()}]
        if self.node_readouts is not None:
            groups.append({"params": self.node_readouts.parameters()})
        return groups

    def cuda(self, device="cuda:0"):
        super().cuda(device)
        self.encoder.cuda(device)
        if self.node_readouts is not None:
            self.node_readouts.cuda(device)

    def load_state(self, other_mac):
        super().load_state(other_mac)
        self.encoder.load_state_dict(other_mac.encoder.state_dict())
        if self.node_readouts is not None:
            self.node_readouts.load_state_dict(other_mac.node_readouts.state_dict())

    def save_models(self, path):
        super().save_models(path)
        torch.save(self.encoder.state_dict(), Path(path) / "graph_encoder.th")
        if self.node_readouts is not None:
            torch.save(self.node_readouts.state_dict(), Path(path) / "action_readouts.th")

    def load_models(self, path):
        super().load_models(path)
        self.encoder.load_state_dict(torch.load(Path(path) / "graph_encoder.th", map_location="cpu", weights_only=True))
        if self.node_readouts is not None:
            self.node_readouts.load_state_dict(torch.load(
                Path(path) / "action_readouts.th", map_location="cpu", weights_only=True))


class GraphCentralV(nn.Module):
    def __init__(self, scheme, args):
        super().__init__()
        self.args = args
        # No parameter sharing between branches or with the policy encoder.
        self.encoders = nn.ModuleList([GraphEncoder(args) for _ in range(4)])
        self.critics = nn.ModuleList([MLP(args.graph_output_dim, args.hidden_dim, 1)
                                     for _ in range(4)])

    def forward(self, batch, t=None):
        start, end = (0, batch.max_seq_length) if t is None else (t, t + 1)
        outputs = []
        for encoder, head in zip(self.encoders, self.critics):
            chunks = []
            for s in range(start, end, self.args.critic_time_chunk):
                stop = min(end, s + self.args.critic_time_chunk)
                z = encode(encoder, batch, slice(s, stop), self.args)
                chunks.append(head(z).reshape(batch.batch_size, stop - s, 1, 1))
            outputs.append(torch.cat(chunks, 1))
        return torch.cat(outputs, 2)


REGISTRY["graph_cv_critic_ns"] = GraphCentralV
