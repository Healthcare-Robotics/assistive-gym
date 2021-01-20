import gym, assistive_gym
import pybullet as p
import numpy as np

env = gym.make('FeedingSawyer-v1')
env.render()
observation = env.reset()

# Map keys to position and orientation end effector movements
pos_keys_actions = {ord('j'): np.array([-0.01, 0, 0]), ord('l'): np.array([0.01, 0, 0]),
                    ord('u'): np.array([0, -0.01, 0]), ord('o'): np.array([0, 0.01, 0]),
                    ord('k'): np.array([0, 0, -0.01]), ord('i'): np.array([0, 0, 0.01])}
rpy_keys_actions = {ord('k'): np.array([-0.05, 0, 0]), ord('i'): np.array([0.05, 0, 0]),
                    ord('u'): np.array([0, -0.05, 0]), ord('o'): np.array([0, 0.05, 0]),
                    ord('j'): np.array([0, 0, -0.05]), ord('l'): np.array([0, 0, 0.05])}

start_pos, orient = env.robot.get_pos_orient(env.robot.right_end_effector)
start_rpy = env.get_euler(orient)
target_pos_offset = np.zeros(3)
target_rpy_offset = np.zeros(3)

while True:
    keys = p.getKeyboardEvents()
    # Process position movement keys ('u', 'i', 'o', 'j', 'k', 'l')
    for key, action in pos_keys_actions.items():
        if p.B3G_SHIFT not in keys and key in keys and keys[key] & p.KEY_IS_DOWN:
            target_pos_offset += action
    # Process rpy movement keys (shift + movement keys)
    for key, action in rpy_keys_actions.items():
        if p.B3G_SHIFT in keys and keys[p.B3G_SHIFT] & p.KEY_IS_DOWN and (key in keys and keys[key] & p.KEY_IS_DOWN):
            target_rpy_offset += action

    # print('Target position offset:', target_pos_offset, 'Target rpy offset:', target_rpy_offset)
    target_pos = start_pos + target_pos_offset
    target_rpy = start_rpy + target_rpy_offset

    # Use inverse kinematics to compute the joint angles for the robot's arm
    # so that its end effector moves to the target position.
    target_joint_angles = env.robot.ik(env.robot.right_end_effector, target_pos, env.get_quaternion(target_rpy), env.robot.right_arm_ik_indices, max_iterations=200, use_current_as_rest=True)
    # Get current joint angles of the robot's arm
    current_joint_angles = env.robot.get_joint_angles(env.robot.right_arm_joint_indices)
    # Compute the action as the difference between target and current joint angles.
    action = (target_joint_angles - current_joint_angles) * 10
    # Step the simulation forward
    observation, reward, done, info = env.step(action)

