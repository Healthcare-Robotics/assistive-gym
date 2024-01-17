

'''
class GymAPIWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, force_reset_old_api=False, force_step_old_api=False):
        reset_old, step_old = get_gym_api_spec()
        self.reset_old_api = force_reset_old_api or reset_old
        self.step_old_api = force_step_old_api or step_old
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
'''