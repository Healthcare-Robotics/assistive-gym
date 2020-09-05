import os
from gym import spaces
import numpy as np
import pybullet as p

from .env import AssistiveEnv
from .agents.human_mesh import HumanMesh

class SMPLXTestingEnv(AssistiveEnv):
    def __init__(self):
        super(SMPLXTestingEnv, self).__init__(robot=None, human=None, task='smplx_testing', obs_robot_len=0, obs_human_len=0)

    def step(self, action):
        self.take_step(action, gains=0.05, forces=1.0)
        return [], 0, False, {}

    def _get_obs(self, agent=None):
        return []

    def reset(self):
        super(SMPLXTestingEnv, self).reset()
        self.build_assistive_env(furniture_type='wheelchair2')
        self.furniture.set_on_ground()

        self.human_mesh = HumanMesh()

        h = self.human_mesh
        body_shape = 'female_1.pkl'
        body_shape = self.np_random.randn(1, self.human_mesh.num_body_shape)
        # joint_angles = [(h.j_left_hip_x, -90), (h.j_right_hip_x, -90), (h.j_left_knee_x, 70), (h.j_right_knee_x, 70), (h.j_left_shoulder_z, -45), (h.j_right_shoulder_z, 45), (h.j_left_elbow_y, -90), (h.j_right_elbow_y, 90)]
        # joint_angles = []
        joint_angles = [(self.human_mesh.j_left_hip_x, -90), (self.human_mesh.j_right_hip_x, -90), (self.human_mesh.j_left_knee_x, 70), (self.human_mesh.j_right_knee_x, 70), (self.human_mesh.j_left_shoulder_z, -45), (self.human_mesh.j_left_elbow_y, -90)]
        u = self.np_random.uniform
        joint_angles += [(self.human_mesh.j_right_pecs_y, u(-20, 20)), (self.human_mesh.j_right_pecs_z, u(-20, 20)), (self.human_mesh.j_right_shoulder_x, u(-45, 45)), (self.human_mesh.j_right_shoulder_y, u(-45, 45)), (self.human_mesh.j_right_shoulder_z, u(-45, 45)), (self.human_mesh.j_right_elbow_y, u(0, 90)), (self.human_mesh.j_waist_x, u(-30, 45)), (self.human_mesh.j_waist_y, u(-45, 45)), (self.human_mesh.j_waist_z, u(-30, 30))]
        self.human_mesh.init(self.directory, self.id, self.np_random, gender='female', height=1.7, body_shape=body_shape, joint_angles=joint_angles, position=[0, 0, 0], orientation=[0, 0, 0])

        # human_height, human_base_height = self.human_mesh.get_heights(set_on_ground=True)
        # print('Human height:', human_height, 'm')

        # self.human_mesh.set_base_pos_orient([0, -0.05, 1.0], [0, 0, 0, 1])
        chair_seat_position = np.array([0, 0.05, 0.6])
        self.human_mesh.set_base_pos_orient(chair_seat_position - self.human_mesh.get_vertex_positions(self.human_mesh.bottom_index), [0, 0, 0, 1])
        pos, orient = self.human_mesh.get_base_pos_orient()

        # spheres = self.create_spheres(radius=0.02, mass=0, batch_positions=self.human_mesh.get_joint_positions(list(range(22))), visual=True, collision=False, rgba=[0, 1, 0, 1])
        # spheres = self.create_spheres(radius=0.01, mass=0, batch_positions=self.human_mesh.get_vertex_positions(self.human_mesh.right_arm_vertex_indices), visual=True, collision=False, rgba=[0, 1, 0, 1])
        # spheres = self.create_spheres(radius=0.01, mass=0, batch_positions=self.human_mesh.get_vertex_positions(self.human_mesh.vert_indices), visual=True, collision=False, rgba=[0, 1, 0, 1])

        vertex_index = self.np_random.choice(self.human_mesh.right_arm_vertex_indices)
        target_pos = self.human_mesh.get_vertex_positions(vertex_index)
        self.target = self.create_sphere(radius=0.01, mass=0.0, pos=target_pos, visual=True, collision=False, rgba=[0, 1, 1, 1])

        # chair_seat_position = np.array([0, -0.03, 0.5])
        # self.human_mesh.set_base_pos_orient(chair_seat_position - np.array(vertex_positions[787]), [0, 0, 0, 1])

        # self.point = self.create_sphere(radius=0.01, mass=0.0, pos=[0, 0, 1.0], visual=True, collision=False, rgba=[0, 1, 1, 1])

        p.setGravity(0, 0, 0, physicsClientId=self.id)

        # p.resetDebugVisualizerCamera(cameraDistance=1.6, cameraYaw=0, cameraPitch=-30, cameraTargetPosition=[0, 0, 1.0], physicsClientId=self.id)

        # Enable rendering
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1, physicsClientId=self.id)

        # Generate spheres
        # radius = 0.01
        # mass = 0.001
        # batch_positions = []
        # x = 10
        # y = 20
        # for i in range(x):
        #     for j in range(y):
        #         batch_positions.append(np.array([i*2*radius-x*radius, j*2*radius-y*radius, 0.5]))
        # spheres = self.create_spheres(radius=radius, mass=mass, batch_positions=batch_positions, visual=False, collision=True)
        # p.setGravity(0, 0, -0.81, physicsClientId=self.id)

        self.init_env_variables()
        return self._get_obs()

