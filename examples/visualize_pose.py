from time import sleep

import gym, assistive_gym, argparse
import pybullet as p
import numpy as np

parser = argparse.ArgumentParser(description='Assistive Gym Environment Viewer')
parser.add_argument('--env', default='HumanTesting-v1',
                    help='Environment to test (default: HumanTesting-v1)')
args = parser.parse_args()

env = gym.make(args.env)

env.reset()
env.render()
observation = env.reset()
env.human.print_joint_info()

angles = []
for i in range (0, 9):
    angles.append([i*45, 0, 0])
for i in range (0, 9):
    angles.append([0, i*45, 0])
for i in range (0, 9):
    angles.append([0, 0, i*45])


# joints = [[3, 4, 5], [13, 14, 15], [21, 22, 23], [25, 26, 27], [28, 29, 30], [32, 33, 34], [35, 36, 37],
#           [39, 40, 41]]
# dicts = ['right_shoulder', 'left_shoulder', 'head', 'waist', 'right_hip', 'right_ankle', 'left_hip',
#          'left_ankle']

joints = [[3, 4, 5], [13, 14, 15], [21, 22, 23]]
dicts = ['right_shoulder', 'left_shoulder', 'head']


text_id = None
env.reset()
for j in range(len(joints)):
    # for x in angles:
    #     for y in angles:
    #         for z in angles:
    for angle in angles:
        x, y, z = angle
        if text_id is not None:
            p.removeUserDebugItem(text_id)
        joint_x = joints[j][0]
        joint_y = joints[j][1]
        joint_z = joints[j][2]
        name = dicts[j]

        env.human.set_joint_angles([joint_x], [np.deg2rad(x)], use_limits=False)
        env.human.set_joint_angles([joint_y], [np.deg2rad(y)], use_limits=False)
        env.human.set_joint_angles([joint_z], [np.deg2rad(z)], use_limits=False)
        debug_text = 'name: ' + name + ' x: ' + str(x) + ' y: ' + str(y) + ' z: ' + str(z)
        text_id = p.addUserDebugText(debug_text, [0.2, 0, 0.2], textColorRGB=[1, 1, 1], textSize=1.5)
        sleep(1)
        env.render()