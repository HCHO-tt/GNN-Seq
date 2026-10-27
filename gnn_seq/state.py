"""Fixed initial-population coordinates, two relations, causal history."""
import numpy as np


def knn(values, k, uids):
    n = len(values)
    squared = ((values[:, None] - values[None, :]) ** 2).sum(-1)
    np.fill_diagonal(squared, np.inf)
    # Persistent UID breaks exact distance ties; not exposed as node features.
    return np.asarray([np.lexsort((uids, row))[:min(k, n - 1)]
                       for row in squared], dtype=np.int32)


class StateBuilder:
    def __init__(self, core, config):
        self.config = config
        self.max_m = config["max_objectives"]
        if core.n_objs > self.max_m:
            raise ValueError("max_objectives is smaller than the problem objective count")
        self.kw, self.ko = config["weight_k"], config["objective_k"]
        f = np.asarray([s.objectives for s in core.population], dtype=np.float64)
        self.lower = f.min(0)
        floor = 1e-8 * np.maximum(np.abs(f).max(0), 1.0)
        self.scale = np.maximum(np.ptp(f, axis=0), floor)
        g = np.asarray([core.moead_calculate_fitness(s, w)
                        for s, w in zip(core.population, core.weights)])
        self.g_scale = max(float(np.median(g[g > 0])) if np.any(g > 0) else 1., 1e-12)
        self.revision = None

    def build(self, core, previous_actions):
        m, n, max_m = core.n_objs, core.population_size, self.max_m
        f = np.asarray([s.objectives for s in core.population], dtype=np.float64)
        weights = np.asarray(core.weights, dtype=np.float64)
        if weights.shape != f.shape or not np.allclose(weights.sum(1), 1, atol=1e-5):
            raise ValueError("Population and normalized weights must align")
        normalized = (f - self.lower) / self.scale
        gaps = (f - np.asarray(core.ideal_point)) / self.scale
        uids = np.asarray(core.history.uids)
        if self.revision != core.weight_revision:
            self.weight_knn = knn(weights, self.kw, uids)
            self.revision = core.weight_revision
        objective_knn = knn(normalized, self.ko, uids)
        rho = np.linalg.norm(normalized[:, None] - normalized[objective_knn], axis=-1).mean(1) / np.sqrt(m)
        x = np.asarray([s.variables for s in core.population], dtype=np.float64)
        x = (x - core.lb) / np.maximum(core.ub - core.lb, 1e-12)
        spread = np.linalg.norm(x[:, None] - x[self.weight_knn], axis=-1).mean(1) / np.sqrt(x.shape[1])
        fitness = np.asarray([core.moead_calculate_fitness(s, w)
                              for s, w in zip(core.population, core.weights)])
        use_operator_history = self.config.get("operator_history", False)
        nodes = np.zeros((n, 4 * max_m + 9 + (32 if use_operator_history else 0)), np.float32)
        nodes[:, :m] = weights
        nodes[:, max_m:max_m + m] = normalized
        nodes[:, 2 * max_m:2 * max_m + m] = gaps
        nodes[:, 3 * max_m:3 * max_m + m] = 1.
        nodes[:, 4 * max_m:4 * max_m + 3] = np.stack(
            [np.log1p(np.maximum(fitness, 0) / self.g_scale), rho, spread], -1)
        generations = core.budget // n
        nodes[:, 4 * max_m + 3:4 * max_m + 9] = core.history.features(generations)
        if use_operator_history:
            nodes[:, 4 * max_m + 9:] = core.history.operator_features()
        context = np.zeros(20, np.float32)
        context[:5] = [core.moead_generation / max(generations, 1),
                       1 / m, 1 / core.problem.nvars,
                       max(core.adaptive_cooling_time, 0) / max(core.wag, 1),
                       len(core.EP) / max(core.EP_MaxSize, 1)]
        if previous_actions is not None:
            for offset, action in zip((5, 9, 13, 17), previous_actions):
                context[offset + action] = 1
            context[19] = 1
        if not np.isfinite(nodes).all() or not np.isfinite(context).all():
            raise FloatingPointError("Nonfinite graph state")
        return {"nodes": nodes, "weight_knn": self.weight_knn.copy(),
                "objective_knn": objective_knn, "context": context}
