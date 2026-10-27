"""Instrumentation only: original variation, selection, reward and weight update."""
import copy
import functools
import math
import random

import numpy as np
from scipy.spatial.distance import cdist
from platypus.core import normalize
from platypus.indicators import Hypervolume

from . import source  # noqa: F401
from .history import SubproblemHistory
from madacbench.mamo import moead_env

# Missing imports in the original file; patch its namespace, not its source.
moead_env.math = math
moead_env.functools = functools


class InstrumentedMOEAD(moead_env.MamoBase):
    @staticmethod
    def _origin_area_2d(points):
        ordered = sorted(points, key=lambda point: point[0], reverse=True)
        area, largest_y = 0.0, 0.0
        for index, (x, y) in enumerate(ordered):
            largest_y = max(largest_y, y)
            next_x = ordered[index + 1][0] if index + 1 < len(ordered) else 0.0
            area += (x - next_x) * largest_y
        return area

    def get_hypervolume(self, n_samples=1e5):
        if not getattr(self, "fast_hv3", False) or self.problem.nobjs != 3:
            return super().get_hypervolume(n_samples)
        hyp = Hypervolume(minimum=[0 for _ in range(self.n_objs)],
                          maximum=self.archive_maximum)
        feasible = [s for s in self.population if s.constraint_violation == 0.0]
        normalize(feasible, hyp.minimum, hyp.maximum)
        feasible = [s for s in feasible if all(o <= 1.0 for o in s.normalized_objectives)]
        for solution in feasible:
            hyp.invert(solution)
        points = sorted((tuple(s.normalized_objectives) for s in feasible),
                        key=lambda point: point[2], reverse=True)
        volume, active = 0.0, []
        for index, point in enumerate(points):
            active.append(point[:2])
            next_z = points[index + 1][2] if index + 1 < len(points) else 0.0
            if point[2] > next_z:
                volume += self._origin_area_2d(active) * (point[2] - next_z)
        self.hv_his.append(volume)
        self.hv_last5.add(volume)
        self.hv_running.update(np.array([volume]))
        return volume

    def get_igd(self):
        if not getattr(self, "fast_igd", False):
            return super().get_igd()
        indicator = self.igd_calculator
        feasible = [solution for solution in self.population
                    if solution.constraint_violation == 0.0]
        normalize(feasible, indicator.minimum, indicator.maximum)
        if not feasible:
            return math.inf
        if not hasattr(self, "_igd_reference_matrix"):
            self._igd_reference_matrix = np.asarray(
                [solution.normalized_objectives for solution in indicator.reference_set],
                dtype=np.float64)
        current = np.asarray([solution.normalized_objectives for solution in feasible],
                             dtype=np.float64)
        nearest = cdist(self._igd_reference_matrix, current, metric="euclidean").min(axis=1)
        return float(np.power(np.power(nearest, indicator.d).sum(), 1.0 / indicator.d)
                     / len(indicator.reference_set))

    def moead_reset(self):
        if hasattr(self, "_igd_reference_matrix"):
            del self._igd_reference_matrix
        self._weights_ready = False
        super().moead_reset()
        self.moead_initial()
        self.history = SubproblemHistory(self.population_size)
        self.weight_revision = 0

    def moead_initial(self):
        if getattr(self, "_weights_ready", False):
            return
        weights = moead_env.get_weights(self.n_objs, self.population_size)
        if len(weights) != self.population_size:
            raise ValueError("Population size must match a Das-Dennis lattice")
        super().moead_initial()
        self._weights_ready = True

    def moead_get_mating_indices(self, index):
        self._source_uid = self.history.uids[index]
        return super().moead_get_mating_indices(index)

    def moead_update_solution(self, solution, mating_indices):
        # Same branches, shuffle, deep copy, and replacement limit as Seq.
        replacements = []
        random.shuffle(mating_indices)
        for i in mating_indices:
            candidate, weight = self.population[i], self.weights[i]
            replace = False
            if solution.constraint_violation > 0 and candidate.constraint_violation > 0:
                replace = solution.constraint_violation < candidate.constraint_violation
            elif candidate.constraint_violation > 0:
                replace = True
            elif solution.constraint_violation > 0:
                pass
            elif self.moead_calculate_fitness(solution, weight) < self.moead_calculate_fitness(candidate, weight):
                replace = True
            if replace:
                # Both are evaluated using THIS target weight and SAME ideal.
                old = self.moead_calculate_fitness(candidate, weight)
                new = self.moead_calculate_fitness(solution, weight)
                replacements.append((self.history.uids[i], old, new))
                self.population[i] = copy.deepcopy(solution)
            if len(replacements) >= self.moead_eta:
                break
        self.history.child(self._source_uid, replacements)

    def update_weight(self):
        # Keep removed objects alive until remapping: Python could otherwise
        # reuse their ids for newly created weights in the upstream method.
        previous_weights = list(self.weights)
        previous = {id(weight): i for i, weight in enumerate(previous_weights)}
        super().update_weight()
        self.history.remap([previous.get(id(weight)) for weight in self.weights])
        self.weight_revision += 1

    def moead_step(self, action=None):
        self.history.begin()
        result = super().moead_step(action)
        self.history.finish(action)
        return result


class Environment:
    """Each instance owns Python/NumPy RNG, like an original worker process."""
    def __init__(self, config, seed, mode="graph"):
        self.config, self.mode, self.seed = config, mode, seed
        saved = random.getstate(), np.random.get_state()
        random.seed(seed)
        np.random.seed(seed)
        try:
            self.core = InstrumentedMOEAD(
                key=config["key"], n_ref_points=config["n_ref_points"],
                population_size=config["population_size"], seed=seed,
                budget_ratio=config["budget_ratio"], test=False,
                wo_obs=(mode != "stats"), save_history=False, early_stop=False,
                baseline=False, adaptive_open=True, replay=False, ban_agent=4,
                reward_type=0, dim_bias=config.get("dim_bias", 3))
            self.core.fast_igd = config.get("fast_igd", False)
            self.core.fast_hv3 = config.get("fast_hv3", False)
            self.rng = random.getstate(), np.random.get_state()
        finally:
            random.setstate(saved[0])
            np.random.set_state(saved[1])
        self.episode_limit = self.core.episode_limit

    def _call(self, fn, *args):
        saved = random.getstate(), np.random.get_state()
        random.setstate(self.rng[0])
        np.random.set_state(self.rng[1])
        try:
            return fn(*args)
        finally:
            self.rng = random.getstate(), np.random.get_state()
            random.setstate(saved[0])
            np.random.set_state(saved[1])

    def reset(self):
        from .state import StateBuilder
        self._call(self.core.moead_reset)
        if self.core.problem.nconstrs:
            raise ValueError("First graph feature definition supports unconstrained problems only")
        self.previous_actions = None
        self.builder = StateBuilder(self.core, self.config)
        return self.observe()

    def observe(self):
        if self.mode == "stats":
            obs = np.asarray(self.core.obs, dtype=np.float32)
            return {"state": obs, "obs": np.tile(obs, (4, 1)),
                    "avail_actions": self.available_actions()}
        return {"state": np.zeros(1, np.float32), "obs": np.zeros((4, 1), np.float32),
                "avail_actions": self.available_actions(),
                **self.builder.build(self.core, self.previous_actions)}

    def step(self, actions):
        actions = np.asarray(actions, dtype=np.int64).reshape(4).copy()
        if any(a < 0 or a >= 4 or not self.available_actions()[i, a]
               for i, a in enumerate(actions)):
            raise ValueError(f"Illegal action {actions}")
        source_uids = tuple(self.core.history.uids)
        reward, done, info = self._call(self.core.moead_step, actions)
        self.last_local_gain = np.asarray([
            -np.expm1(-self.core.history.outgoing.get(uid, 0.))
            for uid in source_uids], dtype=np.float32)
        self.last_local_success = np.asarray([
            float(self.core.history.successful.get(uid, 0) > 0)
            for uid in source_uids], dtype=np.float32)
        self.previous_actions = actions.copy()
        return self.observe(), float(reward), bool(done), info

    def available_actions(self):
        mask = np.ones((4, 4), dtype=np.int32)
        mask[3, 2:] = 0
        if (self.core.adaptive_cooling_time > 0 or
                self.core.action_count[0].sum() >=
                self.core.adaptive_end // self.core.population_size):
            mask[3, 1] = 0
        return mask
