import os
import numpy as np
import pybullet as p
from .robot import Robot

class Sawyer(Robot):
    def __init__(self, controllable_joints='right'):
        right_arm_joint_indices = [3, 8, 9, 10, 11, 13, 16] # Controllable arm joints
        left_arm_joint_indices = right_arm_joint_indices # Controllable arm joints
        wheel_joint_indices = []
        right_end_effector = 19 # Used to get the pose of the end effector
        left_end_effector = right_end_effector # Used to get the pose of the end effector
        right_gripper_indices = [20, 22] # Gripper actuated joints
        left_gripper_indices = right_gripper_indices # Gripper actuated joints
        right_tool_joint = 18 # Joint that tools are attached to
        left_tool_joint = right_tool_joint # Joint that tools are attached to
        right_gripper_collision_indices = [18, 20, 21, 22, 23] # Used to disable collision between gripper and tools
        left_gripper_collision_indices = right_gripper_collision_indices # Used to disable collision between gripper and tools
        gripper_pos = {'scratch_itch': [0.015, -0.015], # Gripper open position for holding tools
                       'feeding': [0, 0],
                       'drinking': [0.025, -0.025],
                       'bed_bathing': [0.0125, -0.0125],
                       'dressing': [0, 0],
                       'arm_manipulation': [0.01, -0.01]}
        tool_pos_offset = {'scratch_itch': [0, 0.125, 0], # Position offset between tool and robot tool joint
                           'feeding': [-0.1, 0.12, -0.02],
                           'drinking': [0.05, 0.125, 0],
                           'bed_bathing': [0, 0.1175, 0],
                           'arm_manipulation': [0.075, 0.235, 0]}
        tool_orient_offset = {'scratch_itch': [0, 0, np.pi/2.0], # RPY orientation offset between tool and robot tool joint
                              'feeding': [np.pi/2.0-0.1, 0, np.pi/2.0],
                              'drinking': [0, 0, np.pi/2.0],
                              'bed_bathing': [np.pi/2.0, 0, np.pi/2.0],
                              'arm_manipulation': [0, 0, np.pi/2.0]}
        toc_base_pos_offset = {'scratch_itch': [-0.1, 0, 0.975], # Robot base offset before TOC base pose optimization
                               'feeding': [-0.1, 0.2, 0.975],
                               'drinking': [-0.1, 0.2, 0.975],
                               'bed_bathing': [-0.2, 0, 0.975],
                               'dressing': [1.8, 0.7, 0.975],
                               'arm_manipulation': [-0.3, 0.6, 0.975]}
        toc_ee_orient_rpy = {'scratch_itch': [0, np.pi/2.0, 0], # Initial end effector orientation
                             'feeding': [np.pi/2.0, 0, np.pi/2.0],
                             'drinking': [0, -np.pi/2.0, np.pi],
                             'bed_bathing': [0, np.pi/2.0, 0],
                             'dressing': [[0, -np.pi/2.0, 0], [np.pi/2.0, -np.pi/2.0, 0]],
                             'arm_manipulation': [0, -np.pi/2.0, np.pi]}
        wheelchair_mounted = False

        super(Sawyer, self).__init__(controllable_joints, right_arm_joint_indices, left_arm_joint_indices, wheel_joint_indices, right_end_effector, left_end_effector, right_gripper_indices, left_gripper_indices, gripper_pos, right_tool_joint, left_tool_joint, tool_pos_offset, tool_orient_offset, right_gripper_collision_indices, left_gripper_collision_indices, toc_base_pos_offset, toc_ee_orient_rpy, wheelchair_mounted, half_range=False)

    def init(self, directory, id, np_random, fixed_base=True):
        self.body = p.loadURDF(os.path.join(directory, 'sawyer', 'sawyer.urdf'), useFixedBase=fixed_base, basePosition=[-1, -1, 0.975], flags=p.URDF_USE_SELF_COLLISION, physicsClientId=id)
        super(Sawyer, self).init(self.body, id, np_random)

        # Remove collisions between the various arm links for stability
        for i in range(3, 24):
            for j in range(3, 24):
                p.setCollisionFilterPair(self.body, self.body, i, j, 0, physicsClientId=id)
        for i in range(0, 3):
            for j in range(0, 9):
                p.setCollisionFilterPair(self.body, self.body, i, j, 0, physicsClientId=id)

