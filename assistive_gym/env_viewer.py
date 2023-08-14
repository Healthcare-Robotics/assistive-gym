import gym, sys, argparse
import numpy as np
from .learn import make_env
# import assistive_gym

if sys.version_info < (3, 0):
    print('Please use Python 3')
    exit()

def sample_action(env, coop):
    if coop:
        return {'robot': env.action_space_robot.sample(), 'human': env.action_space_human.sample()}
    return env.action_space.sample()

def viewer(env_name):
    coop = 'Human' in env_name
    env = make_env(env_name, coop=True) if coop else gym.make(env_name)
    env.reset()
    while True:
        done = False
        env.render()
        observation = env.reset()
        action = sample_action(env, coop)
        # if coop:
        #     print('Robot observation size:', np.shape(observation['robot']), 'Human observation size:', np.shape(observation['human']), 'Robot action size:', np.shape(action['robot']), 'Human action size:', np.shape(action['human']))
        # else:
        #     print('Observation size:', np.shape(observation), 'Action size:', np.shape(action))

        while 1:
            observation, reward, done, info = env.step(sample_action(env, coop))
            # if coop:
            #     done = done['__all__']

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Assistive Gym Environment Viewer')
    parser.add_argument('--env', default='ScratchItchJaco-v1',
                        help='Environment to test (default: ScratchItchJaco-v1)')
    args = parser.parse_args()

    viewer(args.env)
