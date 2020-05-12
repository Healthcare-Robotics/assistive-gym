import os
from gym import spaces
import numpy as np
import pybullet as p

from .env import AssistiveEnv
from .agents import human, furniture
from .agents.human import Human
from .agents.furniture import Furniture

class FeedingEnv(AssistiveEnv):
    def __init__(self, robot, human_control=False):
        human_controllable_joint_indices = [20, 21, 22, 23]
        super(FeedingEnv, self).__init__(robot=robot, human=Human(human_controllable_joint_indices), task='feeding', human_control=human_control, frame_skip=5, time_step=0.02, obs_robot_len=25, obs_human_len=(23 if human_control else 0))

    # TODO: Food is not going in mouth. The human head is not static. The spoon does not generate in the Sawyer correctly.

    def step(self, action):
        self.take_step(action, robot_arm='right', gains=self.config('robot_gains'), forces=self.config('robot_forces'), human_gains=0.0005)

        robot_force_on_human, spoon_force_on_human = self.get_total_force()
        total_force_on_human = robot_force_on_human + spoon_force_on_human
        reward_food, food_mouth_velocities, food_hit_human_reward = self.get_food_rewards()
        end_effector_velocity = np.linalg.norm(self.robot.get_velocity(self.robot.left_end_effector))
        obs = self._get_obs([spoon_force_on_human], [robot_force_on_human, spoon_force_on_human])

        # Get human preferences
        preferences_score = self.human_preferences(end_effector_velocity=end_effector_velocity, total_force_on_human=robot_force_on_human, tool_force_at_target=spoon_force_on_human, food_hit_human_reward=food_hit_human_reward, food_mouth_velocities=food_mouth_velocities)

        spoon_pos, spoon_orient = self.tool.get_base_pos_orient()

        reward_distance_mouth_target = -np.linalg.norm(self.target_pos - spoon_pos) # Penalize robot for distance between the spoon and human mouth.
        reward_action = -np.sum(np.square(action)) # Penalize actions

        reward = self.config('distance_weight')*reward_distance_mouth_target + self.config('action_weight')*reward_action + self.config('food_reward_weight')*reward_food + preferences_score

        if self.gui and reward_food != 0:
            print('Task success:', self.task_success, 'Food reward:', reward_food)

        info = {'total_force_on_human': total_force_on_human, 'task_success': int(self.task_success >= self.total_food_count*self.config('task_success_threshold')), 'action_robot_len': self.action_robot_len, 'action_human_len': self.action_human_len, 'obs_robot_len': self.obs_robot_len, 'obs_human_len': self.obs_human_len}
        done = False

        return obs, reward, done, info

    def get_total_force(self):
        robot_force_on_human = np.sum(self.robot.get_contact_points(self.human)[-1])
        spoon_force_on_human = np.sum(self.tool.get_contact_points(self.human)[-1])
        return robot_force_on_human, spoon_force_on_human

    def get_food_rewards(self):
        # Check all food particles to see if they have left the spoon or entered the person's mouth
        # Give the robot a reward or penalty depending on food particle status
        food_reward = 0
        food_hit_human_reward = 0
        food_mouth_velocities = []
        foods_to_remove = []
        for f in self.foods:
            food_pos, food_orient = f.get_base_pos_orient()
            distance_to_mouth = np.linalg.norm(self.target_pos - food_pos)
            if distance_to_mouth < 0.03:
                # Food is close to the person's mouth. Delete particle and give robot a reward
                food_reward += 20
                self.task_success += 1
                food_velocity = np.linalg.norm(f.get_velocity(f.base))
                food_mouth_velocities.append(food_velocity)
                foods_to_remove.append(f)
                f.set_base_pos_orient(self.np_random.uniform(1000, 2000, size=3), [0, 0, 0, 1])
                continue
            elif food_pos[-1] < 0.5 or len(f.get_contact_points(self.table)[-1]) > 0 or len(f.get_contact_points(self.bowl)[-1]) > 0:
                # Delete particle and give robot a penalty for spilling food
                food_reward -= 5
                foods_to_remove.append(f)
                continue
            if len(f.get_contact_points(self.human)) > 0 and f not in self.foods_hit_person:
                # Record that this food particle just hit the person, so that we can penalize the robot
                self.foods_hit_person.append(f)
                food_hit_human_reward -= 1
        self.foods = [f for f in self.foods if f not in foods_to_remove]
        return food_reward, food_mouth_velocities, food_hit_human_reward

    def _get_obs(self, forces, forces_human):
        torso_pos = self.robot.get_pos_orient(self.robot.torso)[0]
        spoon_pos, spoon_orient = self.tool.get_base_pos_orient()
        robot_joint_angles = self.robot.get_joint_angles(self.robot.controllable_joint_indices)
        if self.human_control:
            human_pos = self.human.get_base_pos_orient()[0]
            human_joint_angles = self.human.get_joint_angles(self.human.controllable_joint_indices)
        head_pos, head_orient = self.human.get_pos_orient(self.human.head)

        robot_obs = np.concatenate([spoon_pos-torso_pos, spoon_orient, spoon_pos-self.target_pos, robot_joint_angles, head_pos-torso_pos, head_orient, forces]).ravel()
        if self.human_control:
            human_obs = np.concatenate([spoon_pos-human_pos, spoon_orient, spoon_pos-self.target_pos, human_joint_angles, head_pos-human_pos, head_orient, forces_human]).ravel()
        else:
            human_obs = []

        return np.concatenate([robot_obs, human_obs]).ravel()

    def reset(self):
        self.setup_timing()
        self.task_success = 0
        self.wheelchair = self.world_creation.create_new_world(furniture_type='wheelchair', static_human_base=True, human_impairment='random', gender='random')
        if self.robot.wheelchair_mounted:
            wheelchair_pos, wheelchair_orient = self.wheelchair.get_base_pos_orient()
            # TODO: Check if this default orientation works for all wheelchair mounted arms
            # TODO: Can we implement a personal function for getQuaternionFromEuler, i.e. util.get_quaternion()
            self.robot.set_base_pos_orient(wheelchair_pos + np.array(self.robot.toc_base_pos_offset[self.task]), p.getQuaternionFromEuler([0, 0, -np.pi/2.0], physicsClientId=self.id))

        joints_positions = [(self.human.j_right_elbow, -90), (self.human.j_left_elbow, -90), (self.human.j_right_hip_x, -90), (self.human.j_right_knee, 80), (self.human.j_left_hip_x, -90), (self.human.j_left_knee, 80)]
        joints_positions += [(self.human.j_head_x, self.np_random.uniform(-30, 30)), (self.human.j_head_y, self.np_random.uniform(-30, 30)), (self.human.j_head_z, self.np_random.uniform(-30, 30))]
        self.human.setup_joints(joints_positions, use_static_joints=True, reactive_force=None)

        # Place a bowl on a table
        self.table = Furniture()
        self.table.init('table', self.world_creation.directory, self.id, self.np_random)
        self.bowl = Furniture()
        self.bowl.init('bowl', self.world_creation.directory, self.id, self.np_random)

        # Set target on mouth
        self.mouth_pos = [0, -0.11, 0.03] if self.human.gender == 'male' else [0, -0.1, 0.03]
        head_pos, head_orient = self.human.get_pos_orient(self.human.head)
        target_pos, target_orient = p.multiplyTransforms(head_pos, head_orient, self.mouth_pos, [0, 0, 0, 1], physicsClientId=self.id)
        self.target = self.world_creation.create_sphere(radius=0.01, mass=0.0, pos=target_pos, collision=False, rgba=[0, 1, 0, 1])
        self.update_targets()

        p.resetDebugVisualizerCamera(cameraDistance=1.10, cameraYaw=40, cameraPitch=-45, cameraTargetPosition=[-0.2, 0, 0.75], physicsClientId=self.id)

        target_ee_pos = np.array([-0.15, -0.65, 1.15]) + self.np_random.uniform(-0.05, 0.05, size=3)
        target_ee_orient = np.array(p.getQuaternionFromEuler(np.array(self.robot.toc_ee_orient_rpy[self.task]), physicsClientId=self.id))
        if self.robot.wheelchair_mounted:
            # Use IK to find starting joint angles for mounted robots
            self.robot.ik_random_restarts(right=True, target_pos=target_ee_pos, target_orient=target_ee_orient, max_iterations=1000, max_ik_random_restarts=40, success_threshold=0.01, step_sim=True, check_env_collisions=True)
        else:
            # Use TOC with JLWKI to find an optimal base position for the robot near the person
            self.robot.position_robot_toc(self.task, 'right', [(target_ee_pos, target_ee_orient), (self.target_pos, None)], [(self.target_pos, target_ee_orient)], step_sim=True, check_env_collisions=False)
        # Open gripper to hold the tool
        self.robot.set_gripper_open_position(self.robot.right_gripper_indices, self.robot.gripper_pos[self.task], set_instantly=True)
        # Initialize the tool in the robot's gripper
        self.tool.init(self.robot, self.task, self.world_creation.directory, self.id, self.np_random, right=True, mesh_scale=[0.08]*3)

        p.setGravity(0, 0, -9.81, physicsClientId=self.id)
        p.setGravity(0, 0, 0, body=self.robot.body, physicsClientId=self.id)
        p.setGravity(0, 0, 0, body=self.human.body, physicsClientId=self.id)
        p.setGravity(0, 0, 0, body=self.tool.body, physicsClientId=self.id)

        p.setPhysicsEngineParameter(numSubSteps=4, numSolverIterations=10, physicsClientId=self.id)

        # Generate food
        spoon_pos, spoon_orient = self.tool.get_base_pos_orient()
        food_radius = 0.005
        food_mass = 0.001
        batch_positions = []
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    batch_positions.append(np.array([i*2*food_radius-0.005, j*2*food_radius, k*2*food_radius+0.01]) + spoon_pos)
        self.foods = self.world_creation.create_spheres(radius=food_radius, mass=food_mass, batch_positions=batch_positions, visual=False, collision=True)
        self.foods_hit_person = []
        self.total_food_count = len(self.foods)

        # Enable rendering
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1, physicsClientId=self.id)

        # Drop food in the spoon
        for _ in range(25):
            p.stepSimulation(physicsClientId=self.id)

        return self._get_obs([0], [0, 0])

    def update_targets(self):
        head_pos, head_orient = self.human.get_pos_orient(self.human.head)
        target_pos, target_orient = p.multiplyTransforms(head_pos, head_orient, self.mouth_pos, [0, 0, 0, 1], physicsClientId=self.id)
        self.target_pos = np.array(target_pos)
        self.target.set_base_pos_orient(self.target_pos, [0, 0, 0, 1])

