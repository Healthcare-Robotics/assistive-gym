import os
from gym import spaces
import numpy as np
import pybullet as p

from .env import AssistiveEnv
from .agents import human, furniture
from .agents.human import Human
from .agents.furniture import Furniture

class BedBathingEnv(AssistiveEnv):
    def __init__(self, robot, human_control=False):
        human_controllable_joint_indices = human.right_arm_joints
        super(BedBathingEnv, self).__init__(robot=robot, human=Human(human_controllable_joint_indices), task='bed_bathing', human_control=human_control, frame_skip=5, time_step=0.02, obs_robot_len=24, obs_human_len=(28 if human_control else 0))

    def step(self, action):
        self.take_step(action, gains=self.config('robot_gains'), forces=self.config('robot_forces'), human_gains=0.05)

        tool_force, tool_force_on_human, total_force_on_human, new_contact_points = self.get_total_force()
        end_effector_velocity = np.linalg.norm(self.robot.get_velocity(self.robot.left_end_effector))
        obs = self._get_obs([tool_force], [total_force_on_human, tool_force_on_human])

        # Get human preferences
        preferences_score = self.human_preferences(end_effector_velocity=end_effector_velocity, total_force_on_human=total_force_on_human, tool_force_at_target=tool_force_on_human)

        reward_distance = -min(self.tool.get_closest_points(self.human, distance=5.0)[-1])
        reward_action = -np.linalg.norm(action) # Penalize actions
        reward_new_contact_points = new_contact_points # Reward new contact points on a person

        reward = self.config('distance_weight')*reward_distance + self.config('action_weight')*reward_action + self.config('wiping_reward_weight')*reward_new_contact_points + preferences_score

        if self.gui and tool_force_on_human > 0:
            print('Task success:', self.task_success, 'Force at tool on human:', tool_force_on_human, reward_new_contact_points)

        info = {'total_force_on_human': total_force_on_human, 'task_success': int(self.task_success >= (self.total_target_count*self.config('task_success_threshold'))), 'action_robot_len': self.action_robot_len, 'action_human_len': self.action_human_len, 'obs_robot_len': self.obs_robot_len, 'obs_human_len': self.obs_human_len}
        done = False

        return obs, reward, done, info

    def get_total_force(self):
        total_force_on_human = np.sum(self.robot.get_contact_points(self.human)[-1])
        tool_force = np.sum(self.tool.get_contact_points()[-1])
        tool_force_on_human = 0
        new_contact_points = 0
        for linkA, linkB, posA, posB, force in zip(*self.tool.get_contact_points(self.human)):
            total_force_on_human += force
            if linkA in [1]:
                tool_force_on_human += force
                # Only consider contact with human upperarm, forearm, hand
                if linkB < 0 or linkB > len(self.human.all_joint_indices):
                    continue

                indices_to_delete = []
                for i, (target_pos_world, target) in enumerate(zip(self.targets_pos_upperarm_world, self.targets_upperarm)):
                    if np.linalg.norm(posB - target_pos_world) < 0.025:
                        # The robot made contact with a point on the person's arm
                        new_contact_points += 1
                        self.task_success += 1
                        target.set_base_pos_orient(self.np_random.uniform(1000, 2000, size=3), [0, 0, 0, 1])
                        indices_to_delete.append(i)
                self.targets_pos_on_upperarm = [t for i, t in enumerate(self.targets_pos_on_upperarm) if i not in indices_to_delete]
                self.targets_upperarm = [t for i, t in enumerate(self.targets_upperarm) if i not in indices_to_delete]
                self.targets_pos_upperarm_world = [t for i, t in enumerate(self.targets_pos_upperarm_world) if i not in indices_to_delete]

                indices_to_delete = []
                for i, (target_pos_world, target) in enumerate(zip(self.targets_pos_forearm_world, self.targets_forearm)):
                    if np.linalg.norm(posB - target_pos_world) < 0.025:
                        # The robot made contact with a point on the person's arm
                        new_contact_points += 1
                        self.task_success += 1
                        target.set_base_pos_orient(self.np_random.uniform(1000, 2000, size=3), [0, 0, 0, 1])
                        indices_to_delete.append(i)
                self.targets_pos_on_forearm = [t for i, t in enumerate(self.targets_pos_on_forearm) if i not in indices_to_delete]
                self.targets_forearm = [t for i, t in enumerate(self.targets_forearm) if i not in indices_to_delete]
                self.targets_pos_forearm_world = [t for i, t in enumerate(self.targets_pos_forearm_world) if i not in indices_to_delete]

        return tool_force, tool_force_on_human, total_force_on_human, new_contact_points

    def _get_obs(self, forces, forces_human):
        tool_pos, tool_orient = self.tool.get_pos_orient(1)
        tool_pos_real, tool_orient_real = self.robot.convert_to_realworld(tool_pos, tool_orient)
        robot_joint_angles = self.robot.get_joint_angles(self.robot.controllable_joint_indices)
        shoulder_pos = self.human.get_pos_orient(self.human.right_shoulder)[0]
        elbow_pos = self.human.get_pos_orient(self.human.right_elbow)[0]
        wrist_pos = self.human.get_pos_orient(self.human.right_wrist)[0]
        shoulder_pos_real, _ = self.robot.convert_to_realworld(shoulder_pos)
        elbow_pos_real, _ = self.robot.convert_to_realworld(elbow_pos)
        wrist_pos_real, _ = self.robot.convert_to_realworld(wrist_pos)
        if self.human_control:
            human_joint_angles = self.human.get_joint_angles(self.human.controllable_joint_indices)
            tool_pos_human, tool_orient_human = self.human.convert_to_realworld(tool_pos, tool_orient)
            shoulder_pos_human, _ = self.human.convert_to_realworld(shoulder_pos)
            elbow_pos_human, _ = self.human.convert_to_realworld(elbow_pos)
            wrist_pos_human, _ = self.human.convert_to_realworld(wrist_pos)

        robot_obs = np.concatenate([tool_pos_real, tool_orient_real, robot_joint_angles, shoulder_pos_real, elbow_pos_real, wrist_pos_real, forces]).ravel()
        if self.human_control:
            human_obs = np.concatenate([tool_pos_human, tool_orient_human, human_joint_angles, shoulder_pos_human, elbow_pos_human, wrist_pos_human, forces_human]).ravel()
        else:
            human_obs = []

        return np.concatenate([robot_obs, human_obs]).ravel()

    def reset(self):
        self.setup_timing()
        self.task_success = 0
        self.contact_points_on_arm = {}
        self.bed = self.world_creation.create_new_world(furniture_type='bed', static_human_base=False, human_impairment='no_tremor', gender='random')

        self.bed.set_friction(self.bed.base, friction=5)

        # Setup human in the air and let them settle into a resting pose on the bed
        joints_positions = [(self.human.j_right_shoulder_x, 30)]
        self.human.setup_joints(joints_positions, use_static_joints=False, reactive_force=None)
        self.human.set_base_pos_orient([-0.15, 0.2, 0.95], [-np.pi/2.0, 0, 0], euler=True)

        p.setGravity(0, 0, -1, physicsClientId=self.id)

        # Add small variation in human joint positions
        motor_indices, motor_positions, motor_velocities, motor_torques = self.human.get_motor_joint_states()
        self.human.set_joint_angles(motor_indices, self.np_random.uniform(-0.1, 0.1, size=len(motor_indices)))

        # Let the person settle on the bed
        for _ in range(100):
            p.stepSimulation(physicsClientId=self.id)

        # Lock human joints and set velocities to 0
        joints_positions = []
        self.human.setup_joints(joints_positions, use_static_joints=True, reactive_force=None, reactive_gain=0.01)
        self.human.set_mass(self.human.base, mass=0)
        self.human.set_base_velocity(linear_velocity=[0, 0, 0], angular_velocity=[0, 0, 0])

        shoulder_pos = self.human.get_pos_orient(self.human.right_shoulder)[0]
        elbow_pos = self.human.get_pos_orient(self.human.right_elbow)[0]
        wrist_pos = self.human.get_pos_orient(self.human.right_wrist)[0]

        target_ee_pos = np.array([-0.6, 0.2, 1]) + self.np_random.uniform(-0.05, 0.05, size=3)
        target_ee_orient = np.array(p.getQuaternionFromEuler(np.array(self.robot.toc_ee_orient_rpy[self.task]), physicsClientId=self.id))
        # Use TOC with JLWKI to find an optimal base position for the robot near the person
        base_position, _, _ = self.robot.position_robot_toc(self.task, 'left', [(target_ee_pos, target_ee_orient)], [(shoulder_pos, None), (elbow_pos, None), (wrist_pos, None)], self.human, step_sim=True, check_env_collisions=False)
        if self.robot.wheelchair_mounted:
            # Load a nightstand in the environment for the jaco arm
            self.nightstand = Furniture()
            self.nightstand.init('nightstand', self.world_creation.directory, self.id, self.np_random)
            self.nightstand.set_base_pos_orient(np.array([-0.9, 0.7, 0]) + base_position, [np.pi/2.0, 0, 0], euler=True)
        # Open gripper to hold the tool
        self.robot.set_gripper_open_position(self.robot.left_gripper_indices, self.robot.gripper_pos[self.task], set_instantly=True)
        # Initialize the tool in the robot's gripper
        self.tool.init(self.robot, self.task, self.world_creation.directory, self.id, self.np_random, right=False, mesh_scale=[1]*3)

        self.generate_targets()

        p.setGravity(0, 0, -9.81, physicsClientId=self.id)
        p.setGravity(0, 0, 0, body=self.robot.body, physicsClientId=self.id)
        # p.setGravity(0, 0, -1, body=self.human.body, physicsClientId=self.id)
        p.setGravity(0, 0, 0, body=self.tool.body, physicsClientId=self.id)

        # Enable rendering
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1, physicsClientId=self.id)

        return self._get_obs([0], [0, 0])

    def generate_targets(self):
        self.target_indices_to_ignore = []
        if self.human.gender == 'male':
            self.upperarm, self.upperarm_length, self.upperarm_radius = self.human.right_shoulder, 0.279, 0.043
            self.forearm, self.forearm_length, self.forearm_radius = self.human.right_elbow, 0.257, 0.033
        else:
            self.upperarm, self.upperarm_length, self.upperarm_radius = self.human.right_shoulder, 0.264, 0.0355
            self.forearm, self.forearm_length, self.forearm_radius = self.human.right_elbow, 0.234, 0.027

        self.targets_pos_on_upperarm = self.util.capsule_points(p1=np.array([0, 0, 0]), p2=np.array([0, 0, -self.upperarm_length]), radius=self.upperarm_radius, distance_between_points=0.03)
        self.targets_pos_on_forearm = self.util.capsule_points(p1=np.array([0, 0, 0]), p2=np.array([0, 0, -self.forearm_length]), radius=self.forearm_radius, distance_between_points=0.03)

        self.targets_upperarm = self.world_creation.create_spheres(radius=0.01, mass=0.0, batch_positions=[[0, 0, 0]]*len(self.targets_pos_on_upperarm), visual=True, collision=False, rgba=[0, 1, 1, 1])
        self.targets_forearm = self.world_creation.create_spheres(radius=0.01, mass=0.0, batch_positions=[[0, 0, 0]]*len(self.targets_pos_on_forearm), visual=True, collision=False, rgba=[0, 1, 1, 1])
        self.total_target_count = len(self.targets_pos_on_upperarm) + len(self.targets_pos_on_forearm)
        self.update_targets()

    def update_targets(self):
        upperarm_pos, upperarm_orient = self.human.get_pos_orient(self.upperarm)
        self.targets_pos_upperarm_world = []
        for target_pos_on_arm, target in zip(self.targets_pos_on_upperarm, self.targets_upperarm):
            target_pos = np.array(p.multiplyTransforms(upperarm_pos, upperarm_orient, target_pos_on_arm, [0, 0, 0, 1], physicsClientId=self.id)[0])
            self.targets_pos_upperarm_world.append(target_pos)
            target.set_base_pos_orient(target_pos, [0, 0, 0, 1])

        forearm_pos, forearm_orient = self.human.get_pos_orient(self.forearm)
        self.targets_pos_forearm_world = []
        for target_pos_on_arm, target in zip(self.targets_pos_on_forearm, self.targets_forearm):
            target_pos = np.array(p.multiplyTransforms(forearm_pos, forearm_orient, target_pos_on_arm, [0, 0, 0, 1], physicsClientId=self.id)[0])
            self.targets_pos_forearm_world.append(target_pos)
            target.set_base_pos_orient(target_pos, [0, 0, 0, 1])

