import os
from time import sleep

from gym import spaces
import numpy as np
import pybullet as p

from .env import AssistiveEnv
from .agents import human
from .agents.human import Human

human_controllable_joint_indices = human.right_arm_joints + human.left_arm_joints
class HumanTestingEnv(AssistiveEnv):
    def __init__(self):
        super(HumanTestingEnv, self).__init__(robot=None, human=Human(human_controllable_joint_indices, controllable=True), task='human_testing', obs_robot_len=0, obs_human_len=0)
    def step(self, action):
        # TODO: Fix the self.take_step() function
        # self.take_step(action, gains=0.05, forces=1.0)
        return [], 0, False, {}

    def _get_obs(self, agent=None):
        return []

    def reset(self):
        super(HumanTestingEnv, self).reset()
        self.build_assistive_env(furniture_type=None, human_impairment='none')

        # Set joint angles for human joints (in degrees)
        joints_positions = []
        self.human.setup_joints(joints_positions, use_static_joints=True, reactive_force=None)
        human_height, human_base_height = self.human.get_heights()
        print('Human height:', human_height, 'm')
        self.human.set_base_pos_orient([0, 0, human_base_height], [0, 0, 0, 1])

        self.point = self.create_sphere(radius=0.01, mass=0.0, pos=[0, 0, human_height], visual=True, collision=False, rgba=[0, 1, 1, 1])

        p.setGravity(0, 0, 0, physicsClientId=self.id)

        p.resetDebugVisualizerCamera(cameraDistance=1.6, cameraYaw=0, cameraPitch=-30, cameraTargetPosition=[0, 0, human_height/2.0], physicsClientId=self.id)


        
        # Add debug lines 
        p.addUserDebugText("tip", [0,0,0.05],textColorRGB=[1,0,0],textSize=1.5)
        p.addUserDebugLine([0,0,0],[5.0,0,0],[1,0,0])
        p.addUserDebugLine([0,0,0],[0,5.0,0],[0,1,0])
        p.addUserDebugLine([0,0,0],[0,0,5.0],[0,0,1])

        # self.human.set_joint_angles(self.human.right_arm_joints, [np.deg2rad(198), np.deg2rad(    61), np.deg2rad(90), 0, 0, 0, 0])
        # p.resetJointState(self.human.body, jointIndex=0, targetValue=np.deg2rad(90), targetVelocity=0, physicsClientId=self.id)
        # angles = [0, 90, 180, 270, 0]
        # joints = [[3, 4, 5], [13, 14, 15], [21, 22, 23], [25, 26, 27], [28, 29, 30], [32, 33, 34], [35, 36, 37],
        #                [39, 40, 41]]
        # dicts = ['right_shoulder', 'left_shoulder', 'head', 'waist', 'right_hip', 'right_ankle', 'left_hip',
        #               'left_ankle']
        # text_id = None
        # for j in range(len(joints)):
        #
        #     for x in angles:
        #         for y in angles:
        #             for z in angles:
        #                 if text_id is not None:
        #                     p.removeUserDebugItem(text_id)
        #                 joint_x = joints[j][0]
        #                 joint_y = joints[j][1]
        #                 joint_z = joints[j][2]
        #                 name = dicts[j]
        #
        #                 self.human.set_joint_angles([joint_x], [np.deg2rad(x)], use_limits=False)
        #                 self.human.set_joint_angles([joint_y], [np.deg2rad(y)], use_limits=False)
        #                 self.human.set_joint_angles([joint_z], [np.deg2rad(z)], use_limits=False)
        #                 debug_text = 'name: ' + name + 'x: ' + str(x) + ' y: ' + str(y) + ' z: ' + str(z)
        #                 text_id = p.addUserDebugText(debug_text, [0.2, 0, 0.2], textColorRGB=[1, 1, 1], textSize=1.5)
        #
        #
        #                 # Enable rendering
        #                 p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1, physicsClientId=self.id)
        #                 self.init_env_variables()

        # Enable rendering
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1, physicsClientId=self.id)
        self.init_env_variables()
        return self._get_obs()

    def loop(self):
        angles = [0, 90, 180, 270]
        self.joints = [[3, 4, 5], [13, 14, 15], [21, 22, 23], [25, 26, 27], [28, 29, 30], [32, 33, 34], [35, 36, 37], [39, 40, 41]]
        self.dicts = ['right_shoulder', 'left_shoulder', 'head', 'waist', 'right_hip', 'right_ankle', 'left_hip', 'left_ankle']

        for j in range(len(self.joints)):
            self.j = j
            for x in angles:
                for y in angles:
                    for z in angles:
                        self.a = [x, y, z]
                        self.reset()


