"""Persistent process workers for independent MOEA/D environments."""
import multiprocessing as mp
import pickle
import traceback


def _worker(conn, config, seed, mode, snapshot):
    try:
        import torch
        torch.set_num_threads(1)
        from .environment import Environment
        env = pickle.loads(snapshot) if snapshot is not None else Environment(config, seed, mode)
        if env.seed != seed or env.mode != mode or env.config != config:
            raise ValueError("Restored environment does not match requested configuration")
        conn.send(("ready", {"episode_limit": env.episode_limit,
                             "seed": env.seed, "function": env.core.env_name,
                             "objectives": env.core.n_objs,
                             "variables": env.core.problem.nvars,
                             "population_size": env.core.population_size}))
        while True:
            command, payload = conn.recv()
            if command == "close":
                break
            if command == "reset":
                conn.send(("ok", env.reset()))
            elif command == "snapshot":
                conn.send(("ok", pickle.dumps(env, protocol=pickle.HIGHEST_PROTOCOL)))
            elif command == "step":
                state, reward, done, info = env.step(payload)
                conn.send(("ok", (state, reward, done, info,
                                  env.last_local_gain, env.last_local_success,
                                  env.core.weight_revision,
                                  len(env.core.history.events))))
            else:
                raise ValueError(f"Unknown worker command: {command}")
    except BaseException:
        try:
            conn.send(("error", traceback.format_exc()))
        except (BrokenPipeError, OSError):
            pass
    finally:
        conn.close()


class ParallelEnvironment:
    """Environment-like client with asynchronous reset and step operations."""
    def __init__(self, config, seed, mode="graph", snapshot=None):
        context = mp.get_context("spawn")
        self.conn, child = context.Pipe()
        self.process = context.Process(target=_worker, args=(child, config, seed, mode, snapshot))
        self.process.start()
        child.close()
        tag, metadata = self._receive()
        if tag != "ready":
            self.close()
            raise RuntimeError(metadata)
        self.metadata = metadata
        self.seed = seed
        self.episode_limit = metadata["episode_limit"]

    def _receive(self):
        try:
            tag, payload = self.conn.recv()
        except EOFError as exc:
            raise RuntimeError(f"Environment worker {self.process.pid} exited") from exc
        if tag == "error":
            raise RuntimeError(payload)
        return tag, payload

    def submit_reset(self):
        self.conn.send(("reset", None))

    def receive_reset(self):
        return self._receive()[1]

    def submit_step(self, actions):
        self.conn.send(("step", actions))

    def submit_snapshot(self):
        self.conn.send(("snapshot", None))

    def receive_snapshot(self):
        return self._receive()[1]

    def receive_step(self):
        (state, reward, done, info, self.last_local_gain,
         self.last_local_success, self.weight_revision,
         self.replacement_events) = self._receive()[1]
        return state, reward, done, info

    def close(self):
        if self.process.is_alive():
            try:
                self.conn.send(("close", None))
                self.process.join(timeout=5)
            except (BrokenPipeError, OSError):
                pass
            if self.process.is_alive():
                self.process.terminate()
                self.process.join(timeout=5)
        self.conn.close()
