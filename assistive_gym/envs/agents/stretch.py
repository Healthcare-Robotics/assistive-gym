import os
import numpy as np
import pybullet as p
from .robot import Robot

class Stretch(Robot):
    def __init__(self, controllable_joints='right'):
        # right_arm_joint_indices = [0, 1, 3, 5, 6, 7, 8, 9] # Controllable arm joints
        right_arm_joint_indices = [3, 5, 9] # Controllable arm joints
        left_arm_joint_indices = right_arm_joint_indices # Controllable arm joints
        wheel_joint_indices = [0, 1] # Controllable wheel joints
        right_end_effector = 15 # Used to get the pose of the end effector
        left_end_effector = right_end_effector # Used to get the pose of the end effector
        right_gripper_indices = [11, 13] # Gripper actuated joints
        left_gripper_indices = right_gripper_indices # Gripper actuated joints
        right_tool_joint = 15 # Joint that tools are attached to
        left_tool_joint = right_tool_joint # Joint that tools are attached to
        # right_gripper_collision_indices = [9, 10, 11, 12, 13, 14, 15] # Used to disable collision between gripper and tools
        right_gripper_collision_indices = list(range(36)) # Used to disable collision between gripper and tools
        left_gripper_collision_indices = right_gripper_collision_indices # Used to disable collision between gripper and tools
        gripper_pos = {'scratch_itch': [0.1, 0.1], # Gripper open position for holding tools
                       'feeding': [0, 0],
                       'drinking': [0.2, 0.2],
                       'bed_bathing': [0.1, 0.1],
                       'dressing': [0, 0],
                       'arm_manipulation': [0.1, 0.1]}
        tool_pos_offset = {'scratch_itch': [0, 0, 0], # Position offset between tool and robot tool joint
                           'feeding': [0.1, 0, -0.02],
                           'drinking': [0, 0, -0.05],
                           'bed_bathing': [0, 0, 0],
                           'arm_manipulation': [0.11, 0, -0.07]}
        tool_orient_offset = {'scratch_itch': [0, 0, 0], # RPY orientation offset between tool and robot tool joint
                              'feeding': [np.pi/2.0-0.1, 0, -np.pi/2.0],
                              'drinking': [np.pi/2.0, 0, 0],
                              'bed_bathing': [0, 0, 0],
                              'arm_manipulation': [np.pi/2.0, 0, 0]}
        toc_base_pos_offset = {'scratch_itch': [-1.0, -0.1, 0.09], # Robot base offset before TOC base pose optimization
                               'feeding': [-0.9, -0.3, 0.09],
                               'drinking': [-0.9, -0.3, 0.09],
                               'bed_bathing': [-1.1, -0.1, 0.09],
                               'dressing': [0.75, -0.4, 0.09],
                               'arm_manipulation': [-1.3, 0.1, 0.09]}
        toc_ee_orient_rpy = {'scratch_itch': [0, 0, np.pi/2.0], # Initial end effector orientation
                             'feeding': [0, 0, np.pi/2.0],
                             'drinking': [0, 0, np.pi/2.0],
                             'bed_bathing': [0, 0, np.pi/2.0],
                             'dressing': [[0, 0, -np.pi/2.0]],
                             'arm_manipulation': [0, 0, np.pi/2.0]}
        wheelchair_mounted = False

        self.gains = [0.1]*2 + [0.01] + [0.025]*5
        self.forces = [10]*2 + [20] + [10]*5
        self.action_duplication = [1, 1, 1, 4, 1] if 'wheel' in controllable_joints else [1, 4, 1] # The urdf models the prismatic arm as multiple joints, but we want only 1 action to control all of them.
        self.action_multiplier = [3, 3, 2, 1, 2] # Adjust the speed of each motor by a multiplier of the default speed
        self.all_controllable_joints = [0, 1, 3, 5, 6, 7, 8, 9] if 'wheel' in controllable_joints else [3, 5, 6, 7, 8, 9]

        super(Stretch, self).__init__(controllable_joints, right_arm_joint_indices, left_arm_joint_indices, wheel_joint_indices, right_end_effector, left_end_effector, right_gripper_indices, left_gripper_indices, gripper_pos, right_tool_joint, left_tool_joint, tool_pos_offset, tool_orient_offset, right_gripper_collision_indices, left_gripper_collision_indices, toc_base_pos_offset, toc_ee_orient_rpy, wheelchair_mounted, half_range=False, action_duplication=self.action_duplication, action_multiplier=self.action_multiplier, flags='stretch')

    def randomize_init_joint_angles(self, task, offset=0):
        if task in ['bed_bathing', 'dressing']:
            self.set_joint_angles([3], [0.95+self.np_random.uniform(-0.1, 0.1)])
        else:
            self.set_joint_angles([3], [0.75+self.np_random.uniform(-0.1, 0.1)])

    def init(self, directory, id, np_random, fixed_base=False):
        # TODO: Inertia from urdf file is not correct.
        # It does not adhere to the property: ixx <= iyy+izz and iyy <= ixx+izz and izz <= ixx+iyy
        # self.body = p.loadURDF(os.path.join(directory, 'stretch', 'stretch_uncalibrated.urdf'), useFixedBase=False, basePosition=[-2, -2, 0.975], flags=p.URDF_USE_INERTIA_FROM_FILE, physicsClientId=id)
        self.body = p.loadURDF(os.path.join(directory, 'stretch', 'stretch_uncalibrated.urdf'), useFixedBase=False, basePosition=[-1, -1, 0.09], physicsClientId=id)
        super(Stretch, self).init(self.body, id, np_random)

        # Fix mass
        # print(p.getDynamicsInfo(self.body, 0, physicsClientId=id))
        # print('Robot mass:', np.sum([self.robot.get_mass(link) for link in self.robot.all_joint_indices]))
        for link in self.all_joint_indices:
            if self.get_mass(link) > 0:
                self.set_mass(link, 0.1)
        self.set_mass(-1, 10)
        self.set_mass(0, 10)
        self.set_mass(1, 10)

        # Disable friction of the robot base since it touches the ground
        self.set_friction(self.base, friction=0)

        # Recolor robot
        white = [1, 1, 1, 1]
        gray = [0.792, 0.82, 0.933, 1]
        dark_gray = [0.4, 0.4, 0.4, 1]
        black = [0.251, 0.251, 0.251, 1]
        for i in [0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 12, 14, 19, 20, 23, 35]:
            p.changeVisualShape(self.body, i, rgbaColor=black, physicsClientId=id)
        for i in [-1, 11, 13, 21, 16, 17, 18, 33, 34]:
            p.changeVisualShape(self.body, i, rgbaColor=gray, physicsClientId=id)
        # for i in [16, 17, 18, 33, 34]:
        #     p.changeVisualShape(self.body, i, rgbaColor=white, physicsClientId=id)
        for i in [3, 32]:
            p.changeVisualShape(self.body, i, rgbaColor=dark_gray, physicsClientId=id)

