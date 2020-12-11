import os
import numpy as np
import pybullet as p
from .robot import Robot

class Baxter(Robot):
    def __init__(self, controllable_joints='right'):
        right_arm_joint_indices = [12, 13, 14, 15, 16, 18, 19] # Controllable arm joints
        left_arm_joint_indices = [34, 35, 36, 37, 38, 40, 41] # Controllable arm joints
        wheel_joint_indices = []
        right_end_effector = 26 # Used to get the pose of the end effector
        left_end_effector = 48 # Used to get the pose of the end effector
        right_gripper_indices = [27, 29] # Gripper actuated joints
        left_gripper_indices = [49, 51] # Gripper actuated joints
        right_tool_joint = 25 # Joint that tools are attached to
        left_tool_joint = 47 # Joint that tools are attached to
        right_gripper_collision_indices = [25, 27, 28, 29, 30] # Used to disable collision between gripper and tools
        left_gripper_collision_indices = [47, 49, 50, 51, 52] # Used to disable collision between gripper and tools
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
        toc_base_pos_offset = {'scratch_itch': [0, 0, 0.925], # Robot base offset before TOC base pose optimization
                               'feeding': [0, 0.2, 0.925],
                               'drinking': [0, 0.2, 0.925],
                               'bed_bathing': [-0.2, 0, 0.925],
                               'dressing': [1.7, 0.7, 0.925],
                               'arm_manipulation': [-0.3, 0.6, 0.925]}
        toc_ee_orient_rpy = {'scratch_itch': [0, np.pi/2.0, 0], # Initial end effector orientation
                             'feeding': [np.pi/2.0, 0, np.pi/2.0],
                             'drinking': [0, -np.pi/2.0, np.pi],
                             'bed_bathing': [0, np.pi/2.0, 0],
                             'dressing': [[0, -np.pi/2.0, 0], [np.pi/2.0, -np.pi/2.0, 0]],
                             'arm_manipulation': [0, -np.pi/2.0, np.pi]}
        wheelchair_mounted = False

        super(Baxter, self).__init__(controllable_joints, right_arm_joint_indices, left_arm_joint_indices, wheel_joint_indices, right_end_effector, left_end_effector, right_gripper_indices, left_gripper_indices, gripper_pos, right_tool_joint, left_tool_joint, tool_pos_offset, tool_orient_offset, right_gripper_collision_indices, left_gripper_collision_indices, toc_base_pos_offset, toc_ee_orient_rpy, wheelchair_mounted, half_range=True)

    def init(self, directory, id, np_random, fixed_base=True):
        self.body = p.loadURDF(os.path.join(directory, 'baxter', 'baxter_custom.urdf'), useFixedBase=fixed_base, basePosition=[-1, -1, 0.925], physicsClientId=id)
        super(Baxter, self).init(self.body, id, np_random)

        # Recolor robot
        for i in [20, 21, 23, 31, 32, 42, 43, 45, 53, 54]:
            p.changeVisualShape(self.body, i, rgbaColor=[1.0, 1.0, 1.0, 0.0], physicsClientId=id)

    def reset_joints(self):
        super(Baxter, self).reset_joints()
        # Position end effectors whith dual arm robots
        self.set_joint_angles(self.right_arm_joint_indices, [-0.75, 1, -0.5, 0.5, -1, -0.5, 0])
        self.set_joint_angles(self.left_arm_joint_indices, [0.75, 1, 0.5, 0.5, 1, -0.5, 0])

