import os
from gym import spaces
import numpy as np
import time
import pybullet as p

from .env import AssistiveEnv

class HumanTestingEnv(AssistiveEnv):
    def __init__(self):
        super(HumanTestingEnv, self).__init__(robot_type=None, task='testing', human_control=False, frame_skip=5, time_step=0.02)

    def step(self, action):
        yaw = 0

        while True:
            yaw += -0.75
            p.resetDebugVisualizerCamera(cameraDistance=1.2, cameraYaw=yaw, cameraPitch=-20, cameraTargetPosition=[0, 0, 1.0], physicsClientId=self.id)
            indices = [4, 5, 6]
            # indices = [14, 15, 16]
            deltas = [0.01, 0.01, -0.01]
            indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
            deltas = [0, 0, 0, 0, 0.01, 0.01, -0.01, 0, 0, 0]
            # indices = []
            # deltas = []
            # indices = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]) + 10
            # deltas = [0, 0, 0, 0, -0.01, 0.01, -0.01, 0, 0, 0]
            for i, d in zip(indices, deltas):
                joint_position = p.getJointState(self.human, jointIndex=i, physicsClientId=self.id)[0]
                if joint_position + d > self.human_lower_limits[i] and joint_position + d < self.human_upper_limits[i]:
                    p.resetJointState(self.human, jointIndex=i, targetValue=joint_position+d, targetVelocity=0, physicsClientId=self.id)
            p.stepSimulation(physicsClientId=self.id)
            print('cameraYaw=%.2f, cameraPitch=%.2f, distance=%.2f' % p.getDebugVisualizerCamera(physicsClientId=self.id)[-4:-1])
            self.enforce_realistic_human_joint_limits()
            time.sleep(0.05)

        return [], None, None, None

    def _get_obs(self, forces, forces_human):
        return []

    def reset(self, robot_base_offset=[0, 0, 0], task='scratch_itch_pr2'):
        self.human, self.wheelchair, self.robot, self.robot_lower_limits, self.robot_upper_limits, self.human_lower_limits, self.human_upper_limits, self.robot_right_arm_joint_indices, self.robot_left_arm_joint_indices, self.gender = self.world_creation.create_new_world(furniture_type=None, static_human_base=True, human_impairment='none', print_joints=False, gender='random')

        p.resetDebugVisualizerCamera(cameraDistance=1.2, cameraYaw=0, cameraPitch=-20, cameraTargetPosition=[0, 0, 1.0], physicsClientId=self.id)

        joints_positions = []
        # self.human_controllable_joint_indices = []
        self.human_controllable_joint_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        # self.human_controllable_joint_indices = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
        self.world_creation.setup_human_joints(self.human, joints_positions, self.human_controllable_joint_indices, use_static_joints=True, human_reactive_force=None)

        p.setGravity(0, 0, 0, physicsClientId=self.id)

        # Enable rendering
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1, physicsClientId=self.id)

        return []

