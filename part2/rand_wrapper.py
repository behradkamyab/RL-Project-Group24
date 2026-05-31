
from collections import deque

import gymnasium as gym

class RandomizationWrapper(gym.Wrapper):
    """
    Wrapper that applies randomization to the environment.
    """
    def __init__(
        self,
        env,
        mass_range=(1.0, 1.0),
        mode="none",
        adr_window=100,
        adr_target=0.7,
        adr_delta=0.1,
        adr_min_width=0.1,
    ):
        super().__init__(env)

        self.mode = mode
        self.mass_range = mass_range
        self.adr_window = adr_window
        self.adr_target = adr_target
        self.adr_delta = adr_delta
        self.adr_min_width = adr_min_width
        self.success_window = deque(maxlen=adr_window)
        self.last_sample_type = "none"

        # global limits
        self.mass_min_limit, self.mass_max_limit = mass_range
        self.mass_min = self.mass_min_limit
        self.mass_max = self.mass_max_limit

    # -----------------------
    # Mass Sampling
    # -----------------------

    def _sample_mass(self):

        if self.mode == "none":
            self.last_sample_type = "none"
            return None
        if self.mode == "udr":
            self.last_sample_type = "udr"
            return float(self.np_random.uniform(self.mass_min_limit, self.mass_max_limit))
        if self.mode == "adr":
            self.last_sample_type = "adr"
            return float(self.np_random.uniform(self.mass_min, self.mass_max))
        else:
            raise NotImplementedError(f"Sampling strategy '{self.mode}' is not implemented yet.")

    def step(self, action):

        obs, reward, terminated, truncated, info = self.env.step(action)

        done = terminated or truncated

        if done:
            success = info.get("is_success")
            if success is None:
                success = info.get("success")
            if success is not None:
                self.success_window.append(1.0 if success else 0.0)

        return obs, reward, terminated, truncated, info

    def _update_adr_range(self):
        if self.mode != "adr":
            return
        if len(self.success_window) < self.adr_window:
            return

        success_rate = sum(self.success_window) / len(self.success_window)
        width = self.mass_max - self.mass_min
        half_delta = self.adr_delta / 2.0

        if success_rate >= self.adr_target:
            # Expand the range within global limits
            self.mass_min = max(self.mass_min_limit, self.mass_min - half_delta)
            self.mass_max = min(self.mass_max_limit, self.mass_max + half_delta)
        else:
            # Shrink the range towards the center
            new_width = max(self.adr_min_width, width - self.adr_delta)
            center = (self.mass_min + self.mass_max) / 2.0
            self.mass_min = max(self.mass_min_limit, center - new_width / 2.0)
            self.mass_max = min(self.mass_max_limit, center + new_width / 2.0)

    # -----------------------
    # Reset
    # -----------------------

    def reset(self, **kwargs):

        self._update_adr_range()

        new_mass = self._sample_mass()

        if new_mass is not None:

            sim = self.env.unwrapped.task.sim
            object_body_id = sim._bodies_idx["object"]

            sim.physics_client.changeDynamics(
                bodyUniqueId=object_body_id,
                linkIndex=-1,
                mass=float(new_mass),
            )


        return super().reset(**kwargs)
