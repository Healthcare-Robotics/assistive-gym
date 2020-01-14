import gym, assistive_gym
import pybullet as p
import numpy as np

env = gym.make('FeedingJaco-v0')
env.render()
observation = env.reset()
env.world_creation.print_joint_info(env.robot)
keys_actions = {p.B3G_LEFT_ARROW: np.array([-0.01, 0, 0]), p.B3G_RIGHT_ARROW: np.array([0.01, 0, 0]), p.B3G_UP_ARROW: np.array([0, 0, 0.01]), p.B3G_DOWN_ARROW: np.array([0, 0, -0.01])}

# Get the position and orientation of the end effector
position, orientation = p.getLinkState(env.robot, 8, computeForwardKinematics=True)[:2]

while True:
    env.render()

    keys = p.getKeyboardEvents()
    for key, action in keys_actions.items():
        if key in keys and keys[key] & p.KEY_IS_DOWN:
            position += action

    # IK to get new joint positions (angles) for the robot
    target_joint_positions = p.calculateInverseKinematics(env.robot, 8, position, orientation)
    target_joint_positions = target_joint_positions[:7]

    # Get the joint positions (angles) of the robot arm
    joint_positions, joint_velocities, joint_torques = env.get_motor_joint_states(env.robot)
    joint_positions = np.array(joint_positions)[:7]

    # Set joint action to be the error between current and target joint positions
    joint_action = (target_joint_positions - joint_positions) * 10
    observation, reward, done, info = env.step(joint_action)

