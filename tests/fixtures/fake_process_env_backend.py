"""CPU simulator fixture used across a real process boundary; no CUDA imports."""

import json
import os
from pathlib import Path
import random

import numpy as np


# The adapter must restore the captured RNG *after* importing its backend.
random.random()
np.random.random()


class OffScreenRenderEnv:
    def __init__(self, *, event_file, camera_heights, camera_widths,
                 fail_constructor=False):
        self.event_file = Path(event_file)
        self.dimensions = (camera_heights, camera_widths)
        self.constructor_draws = (random.random(), float(np.random.random()))
        self.state = np.zeros(2, dtype=np.float32)
        self.seed_value = None
        self._event("construct")
        if fail_constructor:
            raise ValueError("synthetic constructor failure")

    def _event(self, name):
        with self.event_file.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"event": name, "pid": os.getpid()}) + "\n")

    def _observation(self):
        return {
            "state": self.state.copy(),
            "dimensions": self.dimensions,
            "seed": self.seed_value,
            "constructor_draws": self.constructor_draws,
            "draws": (random.random(), float(np.random.random())),
        }

    def seed(self, seed):
        self.seed_value = seed
        self._event("seed")
        return [seed]

    def reset(self, *, offset=0.0):
        self.state.fill(offset)
        self._event("reset")
        return self._observation()

    def set_init_state(self, state):
        self.state = np.asarray(state, dtype=np.float32).copy()
        self._event("set_init_state")
        return self._observation()

    def step(self, action, *, scale=1.0):
        self._event("step")
        if isinstance(action, str):
            if action == "raise":
                raise ValueError("synthetic step failure")
            if action == "exit":
                os._exit(17)
        self.state += np.asarray(action, dtype=np.float32) * scale
        return self._observation(), float(self.state.sum()), False, {"ok": True}

    def close(self):
        # Include cleanup's RNG consumption in the state returned to the parent.
        random.random()
        np.random.random()
        self._event("close")
