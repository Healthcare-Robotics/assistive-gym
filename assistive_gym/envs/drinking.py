import os
from gym import spaces
import numpy as np
import pybullet as p

from .env import AssistiveEnv
from .agents import human
from .agents.human import Human

class DrinkingEnv(AssistiveEnv):
    def __init__(self, robot, human_control=False):
        human_controllable_joint_indices = human.head_joints
        super(DrinkingEnv, self).__init__(robot=robot, human=Human(human_controllable_joint_indices), task='drinking', human_control=human_control, frame_skip=5, time_step=0.02, obs_robot_len=25, obs_human_len=(23 if human_control else 0))

    def step(self, action):
        self.take_step(action, gains=self.config('robot_gains'), forces=self.config('robot_forces'), human_gains=0.0005)

        robot_force_on_human, cup_force_on_human = self.get_total_force()
        total_force_on_human = robot_force_on_human + cup_force_on_human
        reward_water, water_mouth_velocities, water_hit_human_reward = self.get_water_rewards()
        end_effector_velocity = np.linalg.norm(self.robot.get_velocity(self.robot.right_end_effector))
        obs = self._get_obs([cup_force_on_human], [robot_force_on_human, cup_force_on_human])

        # Get human preferences
        preferences_score = self.human_preferences(end_effector_velocity=end_effector_velocity, total_force_on_human=robot_force_on_human, tool_force_at_target=cup_force_on_human, food_hit_human_reward=water_hit_human_reward, food_mouth_velocities=water_mouth_velocities)

        cup_pos, cup_orient = self.tool.get_base_pos_orient()
        cup_pos, cup_orient = p.multiplyTransforms(cup_pos, cup_orient, [0, 0.06, 0], p.getQuaternionFromEuler([np.pi/2.0, 0, 0], physicsClientId=self.id), physicsClientId=self.id)
        cup_top_center_pos, _ = p.multiplyTransforms(cup_pos, cup_orient, self.cup_top_center_offset, [0, 0, 0, 1], physicsClientId=self.id)
        reward_distance_mouth_target = -np.linalg.norm(self.target_pos - np.array(cup_top_center_pos)) # Penalize distances between top of cup and mouth
        reward_action = -np.linalg.norm(action) # Penalize actions

        # Encourage robot to have a tilted end effector / cup
        cup_euler = p.getEulerFromQuaternion(cup_orient, physicsClientId=self.id)
        reward_tilt = -abs(cup_euler[0] - np.pi/2)

        reward = self.config('distance_weight')*reward_distance_mouth_target + self.config('action_weight')*reward_action + self.config('cup_tilt_weight')*reward_tilt + self.config('drinking_reward_weight')*reward_water + preferences_score

        if self.gui and reward_water != 0:
            print('Task success:', self.task_success, 'Water reward:', reward_water)

        info = {'total_force_on_human': total_force_on_human, 'task_success': int(self.task_success >= self.total_water_count*self.config('task_success_threshold')), 'action_robot_len': self.action_robot_len, 'action_human_len': self.action_human_len, 'obs_robot_len': self.obs_robot_len, 'obs_human_len': self.obs_human_len}
        done = False

        return obs, reward, done, info

    def get_total_force(self):
        robot_force_on_human = np.sum(self.robot.get_contact_points(self.human)[-1])
        cup_force_on_human = np.sum(self.tool.get_contact_points(self.human)[-1])
        return robot_force_on_human, cup_force_on_human

    def get_water_rewards(self):
        # Check all water particles to see if they have entered the person's mouth or have left the scene
        # Delete such particles and give the robot a reward or penalty depending on particle status
        cup_pos, cup_orient = self.tool.get_base_pos_orient()
        cup_pos, cup_orient = p.multiplyTransforms(cup_pos, cup_orient, [0, 0.06, 0], p.getQuaternionFromEuler([np.pi/2.0, 0, 0], physicsClientId=self.id), physicsClientId=self.id)
        top_center_pos = np.array(p.multiplyTransforms(cup_pos, cup_orient, self.cup_top_center_offset, [0, 0, 0, 1], physicsClientId=self.id)[0])
        bottom_center_pos = np.array(p.multiplyTransforms(cup_pos, cup_orient, self.cup_bottom_center_offset, [0, 0, 0, 1], physicsClientId=self.id)[0])
        water_reward = 0
        water_hit_human_reward = 0
        water_mouth_velocities = []
        waters_to_remove = []
        for w in self.waters:
            water_pos, water_orient = w.get_base_pos_orient()
            if not self.util.points_in_cylinder(top_center_pos, bottom_center_pos, 0.05, np.array(water_pos)):
                distance_to_mouth = np.linalg.norm(self.target_pos - water_pos)
                if distance_to_mouth < 0.03: # hard
                # if distance_to_mouth < 0.05: # easy
                    # Delete particle and give robot a reward
                    water_reward += 10
                    self.task_success += 1
                    water_velocity = np.linalg.norm(w.get_velocity(w.base))
                    water_mouth_velocities.append(water_velocity)
                    waters_to_remove.append(w)
                    w.set_base_pos_orient(self.np_random.uniform(1000, 2000, size=3), [0, 0, 0, 1])
                    continue
                elif water_pos[-1] < 0.5:
                    # Delete particle and give robot a penalty for spilling water
                    water_reward -= 1
                    waters_to_remove.append(w)
                    continue
                if len(w.get_contact_points(self.human)) > 0:
                    # Record that this water particle just hit the person, so that we can penalize the robot
                    waters_to_remove.append(w)
                    water_hit_human_reward -= 1
        self.waters = [w for w in self.waters if w not in waters_to_remove]
        return water_reward, water_mouth_velocities, water_hit_human_reward

    def _get_obs(self, forces, forces_human):
        cup_pos, cup_orient = self.tool.get_base_pos_orient()
        cup_pos_real, cup_orient_real = self.robot.convert_to_realworld(cup_pos, cup_orient)
        robot_joint_angles = self.robot.get_joint_angles(self.robot.controllable_joint_indices)
        head_pos, head_orient = self.human.get_pos_orient(self.human.head)
        head_pos_real, head_orient_real = self.robot.convert_to_realworld(head_pos, head_orient)
        target_pos_real, _ = self.robot.convert_to_realworld(self.target_pos)
        if self.human_control:
            human_joint_angles = self.human.get_joint_angles(self.human.controllable_joint_indices)
            cup_pos_human, cup_orient_human = self.human.convert_to_realworld(cup_pos, cup_orient)
            head_pos_human, head_orient_human = self.human.convert_to_realworld(head_pos, head_orient)
            target_pos_human, _ = self.human.convert_to_realworld(self.target_pos)

        robot_obs = np.concatenate([cup_pos_real, cup_orient_real, cup_pos_real - target_pos_real, robot_joint_angles, head_pos_real, head_orient_real, forces]).ravel()
        if self.human_control:
            human_obs = np.concatenate([cup_pos_human, cup_orient_human, cup_pos_human - target_pos_human, human_joint_angles, head_pos_human, head_orient_human, forces_human]).ravel()
        else:
            human_obs = []

        return np.concatenate([robot_obs, human_obs]).ravel()

    def reset(self):
        self.setup_timing()
        self.task_success = 0
        self.wheelchair = self.world_creation.create_new_world(furniture_type='wheelchair', static_human_base=True, human_impairment='random', gender='random')
        if self.robot.wheelchair_mounted:
            wheelchair_pos, wheelchair_orient = self.wheelchair.get_base_pos_orient()
            self.robot.set_base_pos_orient(wheelchair_pos + np.array(self.robot.toc_base_pos_offset[self.task]), [0, 0, -np.pi/2.0], euler=True)

        joints_positions = [(self.human.j_right_elbow, -90), (self.human.j_left_elbow, -90), (self.human.j_right_hip_x, -90), (self.human.j_right_knee, 80), (self.human.j_left_hip_x, -90), (self.human.j_left_knee, 80)]
        joints_positions += [(self.human.j_head_x, self.np_random.uniform(-30, 30)), (self.human.j_head_y, self.np_random.uniform(-30, 30)), (self.human.j_head_z, self.np_random.uniform(-30, 30))]
        self.human.setup_joints(joints_positions, use_static_joints=True, reactive_force=None)

        # Set target on mouth
        self.mouth_pos = [0, -0.11, 0.03] if self.human.gender == 'male' else [0, -0.1, 0.03]
        head_pos, head_orient = self.human.get_pos_orient(self.human.head)
        target_pos, target_orient = p.multiplyTransforms(head_pos, head_orient, self.mouth_pos, [0, 0, 0, 1], physicsClientId=self.id)
        self.target = self.world_creation.create_sphere(radius=0.01, mass=0.0, pos=target_pos, collision=False, rgba=[0, 1, 0, 1])
        self.update_targets()

        p.resetDebugVisualizerCamera(cameraDistance=1.10, cameraYaw=55, cameraPitch=-45, cameraTargetPosition=[-0.2, 0, 0.75], physicsClientId=self.id)

        target_ee_pos = np.array([-0.2, -0.5, 1]) + self.np_random.uniform(-0.05, 0.05, size=3)
        target_ee_orient = np.array(p.getQuaternionFromEuler(np.array(self.robot.toc_ee_orient_rpy[self.task]), physicsClientId=self.id))
        if self.robot.wheelchair_mounted:
            # Use IK to find starting joint angles for mounted robots
            self.robot.ik_random_restarts(right=True, target_pos=target_ee_pos, target_orient=target_ee_orient, max_iterations=1000, max_ik_random_restarts=40, success_threshold=0.01, step_sim=True, check_env_collisions=True)
        else:
            # Use TOC with JLWKI to find an optimal base position for the robot near the person
            self.robot.position_robot_toc(self.task, 'right', [(target_ee_pos, target_ee_orient), (self.target_pos, None)], [(self.target_pos, target_ee_orient)], self.human, step_sim=True, check_env_collisions=False)
        # Open gripper to hold the tool
        self.robot.set_gripper_open_position(self.robot.right_gripper_indices, self.robot.gripper_pos[self.task], set_instantly=True)
        # Initialize the tool in the robot's gripper
        self.tool.init(self.robot, self.task, self.world_creation.directory, self.id, self.np_random, right=True, mesh_scale=[0.045]*3, alpha=0.75)
        self.cup_top_center_offset = np.array([0, 0, -0.055])
        self.cup_bottom_center_offset = np.array([0, 0, 0.07])

        p.setGravity(0, 0, -9.81, physicsClientId=self.id)
        p.setGravity(0, 0, 0, body=self.robot.body, physicsClientId=self.id)
        p.setGravity(0, 0, 0, body=self.human.body, physicsClientId=self.id)
        p.setGravity(0, 0, 0, body=self.tool.body, physicsClientId=self.id)

        p.setPhysicsEngineParameter(numSubSteps=4, numSolverIterations=10, physicsClientId=self.id)

        # Generate water
        cup_pos, cup_orient = self.tool.get_base_pos_orient()
        water_radius = 0.005
        water_mass = 0.001
        batch_positions = []
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    batch_positions.append(np.array([i*2*water_radius-0.02, j*2*water_radius-0.02, k*2*water_radius+0.075]) + cup_pos)
        self.waters = self.world_creation.create_spheres(radius=water_radius, mass=water_mass, batch_positions=batch_positions, visual=False, collision=True)
        for w in self.waters:
            p.changeVisualShape(w.body, -1, rgbaColor=[0.25, 0.5, 1, 1], physicsClientId=self.id)
        self.total_water_count = len(self.waters)

        # Enable rendering
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1, physicsClientId=self.id)

        # Drop water in the cup
        for _ in range(50):
            p.stepSimulation(physicsClientId=self.id)

        return self._get_obs([0], [0, 0])

    def update_targets(self):
        head_pos, head_orient = self.human.get_pos_orient(self.human.head)
        target_pos, target_orient = p.multiplyTransforms(head_pos, head_orient, self.mouth_pos, [0, 0, 0, 1], physicsClientId=self.id)
        self.target_pos = np.array(target_pos)
        self.target.set_base_pos_orient(self.target_pos, [0, 0, 0, 1])

