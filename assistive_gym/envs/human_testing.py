import os
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
        self.take_step(action, gains=0.05, forces=1.0)
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

        # Enable rendering
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1, physicsClientId=self.id)

        self.init_env_variables()
        return self._get_obs()

