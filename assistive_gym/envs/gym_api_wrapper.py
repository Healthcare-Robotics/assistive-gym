try:
    import gymnasium as gym
    GYM_RESET_OLD_API = False
    GYM_STEP_OLD_API = False
except ImportError:
    import gym
    from packaging import version
    GYM_RESET_OLD_API = version.parse(gym.__version__) < version.parse("0.22.0")
    GYM_STEP_OLD_API = version.parse(gym.__version__) < version.parse("0.25.0")


class GymAPIWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, force_reset_old_api=False, force_step_old_api=False):
        self.reset_old_api = force_reset_old_api or GYM_RESET_OLD_API
        self.step_old_api = force_step_old_api or GYM_STEP_OLD_API
        super().__init__(env)
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if self.step_old_api:
            return obs, reward, terminated or truncated, info
        else:
            return obs, reward, terminated, truncated, info
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        if self.reset_old_api:
            return obs
        else:
            return obs, info