
from collections import deque

import gymnasium as gym

class RandomizationWrapper(gym.Wrapper):
    """
    Wrapper that applies randomization to the environment.
    """
    def __init__(
        self,
        env,
        mass_range=(0.5, 4.5),
        mode="none",
        adr_init_range=(0.9, 1.1),
        adr_window=100,
        adr_target=0.7,
        adr_delta=0.1,
        adr_min_width=0.1,
        adr_update_freq=1,
        adr_warmup_episodes=0,
        log_every=0,
        verify_mass=False,
        success_key="is_success",
    ):
        super().__init__(env)

        self.mode = mode
        self.mass_range = mass_range
        self.adr_window = adr_window
        self.adr_target = adr_target
        self.adr_delta = adr_delta
        self.adr_min_width = adr_min_width
        self.adr_update_freq = adr_update_freq
        self.adr_warmup_episodes = adr_warmup_episodes
        self.log_every = log_every
        self.verify_mass = verify_mass
        self.success_key = success_key
        self.success_window = deque(maxlen=adr_window)
        self.last_sample_type = "none"
        self.last_sampled_mass = None
        self.last_applied_mass = None
        self.last_mass_verified = None
        self.reset_count = 0
        
        # global limits
        self.mass_min_limit, self.mass_max_limit = mass_range
        init_min, init_max = adr_init_range
        if mode == 'adr':
            self.mass_min, self.mass_max= adr_init_range
        else:
            self.mass_min= self.mass_min_limit
            self.mass_max= self.mass_max_limit
                
        # init_min = max(self.mass_min_limit, init_min)
        # init_max = min(self.mass_max_limit, init_max)
        # if init_min >= init_max:
        #     raise ValueError(
        #         "adr_init_range must be within mass_range and have init_min < init_max"
        #     )
        # if self.mode == "adr":
        #     self.mass_min = init_min
        #     self.mass_max = init_max
        # else:
        #     self.mass_min = self.mass_min_limit
        #     self.mass_max = self.mass_max_limit
        self.range_history=[]
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
            success = info.get(self.success_key)
            if success is None:
                success = info.get("success")
            if success is not None:
                self.success_window.append(float(success))

        return obs, reward, terminated, truncated, info

    def _update_adr_range(self):
        if self.mode != "adr":
            return
        if self.adr_warmup_episodes and self.reset_count <= self.adr_warmup_episodes:
            return
        if len(self.success_window) < self.adr_window:
            return
        if self.adr_update_freq > 1 and (self.reset_count % self.adr_update_freq) != 0:
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
        self.reset_count += 1

        self._update_adr_range()

        new_mass = self._sample_mass()
        self.last_sampled_mass = new_mass
        self.range_history.append((self.mass_min,self.mass_max)) 
        obs, info = super().reset(**kwargs)

        if new_mass is not None:

            sim = self.env.unwrapped.task.sim
            object_body_id = sim._bodies_idx["object"]

            sim.physics_client.changeDynamics(
                bodyUniqueId=object_body_id,
                linkIndex=-1,
                mass=float(new_mass),
            )
            self.last_applied_mass = float(new_mass)

            if self.verify_mass:
                try:
                    dynamics = sim.physics_client.getDynamicsInfo(object_body_id, -1)
                    actual_mass = float(dynamics[0])
                    self.last_mass_verified = abs(actual_mass - float(new_mass)) < 1e-6
                except Exception:
                    self.last_mass_verified = None

        if self.log_every and (self.reset_count % self.log_every) == 0:
            success_rate = None
            if self.success_window:
                success_rate = sum(self.success_window) / len(self.success_window)
            print(
                f"[{self.mode}] reset={self.reset_count} mass={new_mass} "
                f"range=[{self.mass_min:.3f},{self.mass_max:.3f}] "
                f"success_rate={success_rate}"
            )
            

        return obs, info
