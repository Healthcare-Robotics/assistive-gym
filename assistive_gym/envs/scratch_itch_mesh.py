import os
from gym import spaces
import numpy as np
import pybullet as p

from .scratch_itch import ScratchItchEnv

class ScratchItchMeshEnv(ScratchItchEnv):
    def __init__(self, robot, human):
        super(ScratchItchMeshEnv, self).__init__(robot=robot, human=human)
        self.general_model = True
        # Parameters for personalized human participants
        self.gender = 'female'
        self.body_shape_filename = '%s_1.pkl' % self.gender
        self.human_height = 1.6

    def reset(self):
        super(ScratchItchEnv, self).reset()
        self.build_assistive_env('wheelchair')
        self.furniture.set_on_ground()
        self.prev_target_contact_pos = np.zeros(3)
        if self.robot.wheelchair_mounted:
            wheelchair_pos, wheelchair_orient = self.furniture.get_base_pos_orient()
            self.robot.set_base_pos_orient(wheelchair_pos + np.array(self.robot.toc_base_pos_offset[self.task]), [0, 0, -np.pi/2.0])

        if self.general_model:
            # Randomize the human body shape
            gender = self.np_random.choice(['male', 'female'])
            # body_shape = self.np_random.randn(1, self.human.num_body_shape)
            body_shape = self.np_random.uniform(-2, 5, (1, self.human.num_body_shape))
            # human_height = self.np_random.uniform(1.59, 1.91) if gender == 'male' else self.np_random.uniform(1.47, 1.78)
            human_height = self.np_random.uniform(1.5, 1.9)
        else:
            gender = self.gender
            body_shape = self.body_shape_filename
            human_height = self.human_height

        # Randomize human pose
        joint_angles = [(self.human.j_left_hip_x, -90), (self.human.j_right_hip_x, -90), (self.human.j_left_knee_x, 70), (self.human.j_right_knee_x, 70), (self.human.j_left_shoulder_z, -45), (self.human.j_left_elbow_y, -90)]
        joint_angles += [(self.human.j_right_shoulder_z, 45+self.np_random.uniform(-10, 10)), (self.human.j_right_elbow_y, 90+self.np_random.uniform(-10, 10))]
        # u = self.np_random.uniform
        # joint_angles += [(self.human.j_right_pecs_y, u(-20, 20)), (self.human.j_right_pecs_z, u(-20, 20)), (self.human.j_right_shoulder_x, u(-45, 45)), (self.human.j_right_shoulder_y, u(-45, 45)), (self.human.j_right_shoulder_z, u(-45, 45)), (self.human.j_right_elbow_y, u(0, 90)), (self.human.j_waist_x, u(-30, 45)), (self.human.j_waist_y, u(-45, 45)), (self.human.j_waist_z, u(-30, 30))]
        joint_angles += [(j, self.np_random.uniform(-10, 10)) for j in (self.human.j_right_pecs_y, self.human.j_right_pecs_z, self.human.j_right_shoulder_x, self.human.j_right_shoulder_y, self.human.j_waist_x, self.human.j_waist_y, self.human.j_waist_z)]

        # Set joint angles for human joints (in degrees)
        # joint_angles = [(self.human.j_left_hip_x, -90), (self.human.j_right_hip_x, -90), (self.human.j_left_knee_x, 70), (self.human.j_right_knee_x, 70), (self.human.j_left_shoulder_z, -45), (self.human.j_right_shoulder_z, 45), (self.human.j_left_elbow_y, -90), (self.human.j_right_elbow_y, 90)]
        self.human.init(self.directory, self.id, self.np_random, gender=gender, height=human_height, body_shape=body_shape, joint_angles=joint_angles, position=[0, 0, 0], orientation=[0, 0, 0])

        # Place human in chair
        chair_seat_position = np.array([0, 0.05, 0.6])
        self.human.set_base_pos_orient(self.furniture.get_base_pos_orient()[0] + chair_seat_position - self.human.get_vertex_positions(self.human.bottom_index), [0, 0, 0, 1])

        shoulder_pos = self.human.get_pos_orient(self.human.right_shoulder)[0]
        elbow_pos = self.human.get_pos_orient(self.human.right_elbow)[0]
        wrist_pos = self.human.get_pos_orient(self.human.right_wrist)[0]

        self.generate_target()

        target_ee_orient = np.array(p.getQuaternionFromEuler(np.array(self.robot.toc_ee_orient_rpy[self.task]), physicsClientId=self.id))
        while True:
            # Continually resample initial end effector poses until we find one where the robot isn't colliding with the person
            target_ee_pos = np.array([-0.5, 0, 0.8]) + np.array([self.np_random.uniform(-0.2, 0.05), self.np_random.uniform(-0.2, 0.2), self.np_random.uniform(-0.05, 0.2)])
            if self.robot.mobile:
                # Randomize robot base pose
                pos = np.array(self.robot.toc_base_pos_offset[self.task])
                pos[:2] += self.np_random.uniform(-0.1, 0.1, size=2)
                orient = np.array(self.robot.toc_ee_orient_rpy[self.task])
                orient[2] += self.np_random.uniform(-np.deg2rad(30), np.deg2rad(30))
                self.robot.set_base_pos_orient(pos, orient)
                # Randomize starting joint angles
                self.robot.set_joint_angles([3], [0.75+self.np_random.uniform(-0.1, 0.1)])

                # Randomly set friction of the ground
                self.plane.set_frictions(self.plane.base, lateral_friction=self.np_random.uniform(0.025, 0.5), spinning_friction=0, rolling_friction=0)
            elif self.robot.wheelchair_mounted:
                # Use IK to find starting joint angles for mounted robots
                self.robot.ik_random_restarts(right=False, target_pos=target_ee_pos, target_orient=target_ee_orient, max_iterations=1000, max_ik_random_restarts=40, success_threshold=0.03, step_sim=False, check_env_collisions=False)
            else:
                # Use TOC with JLWKI to find an optimal base position for the robot near the person
                self.robot.position_robot_toc(self.task, 'left', [(target_ee_pos, target_ee_orient)], [(self.target_pos, None)], self.human, step_sim=False, check_env_collisions=False, attempts=50)
            # Check if the robot is colliding with the person
            _, _, _, _, dists = self.robot.get_closest_points(self.human, distance=0)
            # print(len(dists))
            if not dists:
                break

        # Open gripper to hold the tool
        self.robot.set_gripper_open_position(self.robot.left_gripper_indices, self.robot.gripper_pos[self.task], set_instantly=True)
        # Initialize the tool in the robot's gripper
        self.tool.init(self.robot, self.task, self.directory, self.id, self.np_random, right=False, mesh_scale=[0.001]*3)

        if not self.robot.mobile:
            self.robot.set_gravity(0, 0, 0)
        self.human.set_gravity(0, 0, 0)
        self.tool.set_gravity(0, 0, 0)

        # Enable rendering
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1, physicsClientId=self.id)

        self.init_env_variables()
        return self._get_obs()

    def generate_target(self):
        # TODO: Pick a random vertex on the right arm
        vertex_index = self.np_random.choice(self.human.right_arm_vertex_indices)
        self.target_pos = self.human.get_vertex_positions(vertex_index)

        self.target = self.create_sphere(radius=0.01, mass=0.0, pos=self.target_pos, visual=True, collision=False, rgba=[0, 1, 1, 1])

    def update_targets(self):
        pass

