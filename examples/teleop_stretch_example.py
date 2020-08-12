import gym, assistive_gym, argparse
import pybullet as p
import numpy as np

parser = argparse.ArgumentParser(description='Assistive Gym Environment Viewer')
parser.add_argument('--env', default='ScratchItchStretch-v1',
                    help='Environment to test (default: ScratchItchStretch-v1)')
args = parser.parse_args()

env = gym.make(args.env)
env.render()
observation = env.reset()
env.robot.print_joint_info()

# Arrow keys for moving the base, s/x for the lift, z/c for the prismatic joint, a/d for the wrist joint
keys_actions = {p.B3G_LEFT_ARROW: np.array([0.01, -0.01, 0, 0, 0]), p.B3G_RIGHT_ARROW: np.array([-0.01, 0.01, 0, 0, 0]), p.B3G_UP_ARROW: np.array([0.01, 0.01, 0, 0, 0]), p.B3G_DOWN_ARROW: np.array([-0.01, -0.01, 0, 0, 0]), ord('s'): np.array([0, 0, 0.01, 0, 0]), ord('x'): np.array([0, 0, -0.01, 0, 0]), ord('z'): np.array([0, 0, 0, -0.01, 0]), ord('c'): np.array([0, 0, 0, 0.01, 0]), ord('a'): np.array([0, 0, 0, 0, 0.01]), ord('d'): np.array([0, 0, 0, 0, -0.01])}

while True:
    env.render()

    action = np.zeros(env.action_robot_len)
    keys = p.getKeyboardEvents()
    for key, a in keys_actions.items():
        if key in keys and keys[key] & p.KEY_IS_DOWN:
            action += a

    observation, reward, done, info = env.step(action*100)

