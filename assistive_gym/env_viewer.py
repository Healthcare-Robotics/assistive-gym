import gym, sys, argparse
import numpy as np
# import assistive_gym

if sys.version_info < (3, 0):
    print('Please use Python 3')
    exit()

def viewer(env_name):
    env = gym.make(env_name)

    while True:
        done = False
        env.render()
        observation = env.reset()
        action = env.action_space.sample()
        print('Observation size:', np.shape(observation), 'Action size:', np.shape(action))
        while not done:
            observation, reward, done, info = env.step(env.action_space.sample())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Assistive Gym Environment Viewer')
    parser.add_argument('--env', default='ScratchItchJaco-v1',
                        help='Environment to test (default: ScratchItchJaco-v1)')
    args = parser.parse_args()

    viewer(args.env)
