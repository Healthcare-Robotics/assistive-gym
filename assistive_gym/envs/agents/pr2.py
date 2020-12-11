import os
import numpy as np
import pybullet as p
from .robot import Robot

class PR2(Robot):
    def __init__(self, controllable_joints='right'):
        right_arm_joint_indices = [42, 43, 44, 46, 47, 49, 50] # Controllable arm joints
        left_arm_joint_indices = [64, 65, 66, 68, 69, 71, 72] # Controllable arm joints
        wheel_joint_indices = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14] # Controllable wheel joints
        right_end_effector = 54 # Used to get the pose of the end effector
        left_end_effector = 76 # Used to get the pose of the end effector
        right_gripper_indices = [57, 58, 59, 60] # Gripper actuated joints
        left_gripper_indices = [79, 80, 81, 82] # Gripper actuated joints
        right_tool_joint = 54 # Joint that tools are attached to
        left_tool_joint = 76 # Joint that tools are attached to
        right_gripper_collision_indices = list(range(49, 64)) # Used to disable collision between gripper and tools
        left_gripper_collision_indices = list(range(71, 86)) # Used to disable collision between gripper and tools
        gripper_pos = {'scratch_itch': [0.25]*4, # Gripper open position for holding tools
                       'feeding': [0.03]*4,
                       'drinking': [0.45]*4,
                       'bed_bathing': [0.2]*4,
                       'dressing': [0]*4,
                       'arm_manipulation': [0.15]*4}
        tool_pos_offset = {'scratch_itch': [0, 0, 0], # Position offset between tool and robot tool joint
                           'feeding': [0, -0.03, -0.11],
                           'drinking': [-0.01, 0, -0.05],
                           'bed_bathing': [0, 0, 0],
                           'arm_manipulation': [0.125, 0, -0.075]}
        tool_orient_offset = {'scratch_itch': [0, 0, 0], # RPY orientation offset between tool and robot tool joint
                              'feeding': [-0.2, 0, 0],
                              'drinking': [np.pi/2.0, 0, 0],
                              'bed_bathing': [0, 0, 0],
                              'arm_manipulation': [np.pi/2.0, 0, 0]}
        toc_base_pos_offset = {'scratch_itch': [0.1, 0, 0], # Robot base offset before TOC base pose optimization
                               'feeding': [0.1, 0.2, 0],
                               'drinking': [0.2, 0.2, 0],
                               'bed_bathing': [-0.1, 0, 0],
                               'dressing': [1.7, 0.7, 0],
                               'arm_manipulation': [-0.3, 0.7, 0]}
        toc_ee_orient_rpy = {'scratch_itch': [0, 0, 0], # Initial end effector orientation
                             'feeding': [np.pi/2.0, 0, 0],
                             'drinking': [0, 0, 0],
                             'bed_bathing': [0, 0, 0],
                             'dressing': [[0, 0, np.pi], [0, 0, np.pi*3/2.0]],
                             'arm_manipulation': [0, 0, 0]}
        wheelchair_mounted = False

        super(PR2, self).__init__(controllable_joints, right_arm_joint_indices, left_arm_joint_indices, wheel_joint_indices, right_end_effector, left_end_effector, right_gripper_indices, left_gripper_indices, gripper_pos, right_tool_joint, left_tool_joint, tool_pos_offset, tool_orient_offset, right_gripper_collision_indices, left_gripper_collision_indices, toc_base_pos_offset, toc_ee_orient_rpy, wheelchair_mounted, half_range=False)

    def init(self, directory, id, np_random, fixed_base=True):
        self.body = p.loadURDF(os.path.join(directory, 'PR2', 'pr2_no_torso_lift_tall.urdf'), useFixedBase=fixed_base, basePosition=[-1, -1, 0], flags=p.URDF_USE_INERTIA_FROM_FILE, physicsClientId=id)
        super(PR2, self).init(self.body, id, np_random)

        # Recolor robot
        for i in [19, 42, 64]:
            p.changeVisualShape(self.body, i, rgbaColor=[1.0, 1.0, 1.0, 1.0], physicsClientId=id)
        for i in [43, 46, 49, 58, 60, 65, 68, 71, 80, 82]:
            p.changeVisualShape(self.body, i, rgbaColor=[0.4, 0.4, 0.4, 1.0], physicsClientId=id)
        for i in [45, 51, 67, 73]:
            p.changeVisualShape(self.body, i, rgbaColor=[0.7, 0.7, 0.7, 1.0], physicsClientId=id)
        p.changeVisualShape(self.body, 20, rgbaColor=[0.8, 0.8, 0.8, 1.0], physicsClientId=id)
        p.changeVisualShape(self.body, 40, rgbaColor=[0.6, 0.6, 0.6, 1.0], physicsClientId=id)

    def reset_joints(self):
        super(PR2, self).reset_joints()
        # Position end effectors whith dual arm robots
        self.set_joint_angles(self.right_arm_joint_indices, [-1.75, 1.25, -1.5, -0.5, -1, 0, -1])
        self.set_joint_angles(self.left_arm_joint_indices, [1.75, 1.25, 1.5, -0.5, 1, 0, 1])

