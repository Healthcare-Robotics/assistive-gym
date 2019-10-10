import os, time, datetime, configparser
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import pybullet as p
# import cv2
from keras.models import load_model

from .util import Util
from .world_creation import WorldCreation

class AssistiveEnv(gym.Env):
    def __init__(self, robot_type='pr2', task='scratch_itch', human_control=False, frame_skip=5, time_step=0.02, action_robot_len=7, action_human_len=0, obs_robot_len=30, obs_human_len=0):
        # Start the bullet physics server
        self.id = p.connect(p.DIRECT)
        # print('Physics server ID:', self.id)
        self.gui = False

        self.robot_type = robot_type
        self.task = task
        self.human_control = human_control
        self.action_robot_len = action_robot_len
        self.action_human_len = action_human_len
        self.obs_robot_len = obs_robot_len
        self.obs_human_len = obs_human_len
        self.action_space = spaces.Box(low=np.array([-1.0]*(self.action_robot_len+self.action_human_len)), high=np.array([1.0]*(self.action_robot_len+self.action_human_len)), dtype=np.float32)
        self.observation_space = spaces.Box(low=np.array([-1.0]*(self.obs_robot_len+self.obs_human_len)), high=np.array([1.0]*(self.obs_robot_len+self.obs_human_len)), dtype=np.float32)

        self.configp = configparser.ConfigParser()
        self.configp.read(os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'config.ini'))
        # Human preference weights
        self.C_v = self.config('velocity_weight', 'human_preferences')
        self.C_f = self.config('force_nontarget_weight', 'human_preferences')
        self.C_hf = self.config('high_forces_weight', 'human_preferences')
        self.C_fd = self.config('food_hit_weight', 'human_preferences')
        self.C_fdv = self.config('food_velocities_weight', 'human_preferences')
        self.C_d = self.config('dressing_force_weight', 'human_preferences')
        self.C_p = self.config('high_pressures_weight', 'human_preferences')

        # Execute actions at 10 Hz by default. A new action every 0.1 seconds
        self.frame_skip = frame_skip
        self.time_step = time_step

        self.setup_timing()
        self.seed(1001)

        self.world_creation = WorldCreation(self.id, robot_type=robot_type, task=task, time_step=self.time_step, np_random=self.np_random, config=self.config)
        self.util = Util(self.id, self.np_random)

        self.record_video = False
        self.video_writer = None
        # self.width = 1920
        # self.height = 1080
        self.width = 3840
        self.height = 2160

        self.human_limits_model = load_model(os.path.join(self.world_creation.directory, 'realistic_arm_limits_model.h5'))
        self.right_arm_previous_valid_pose = None
        self.left_arm_previous_valid_pose = None
        self.human_joint_lower_limits = None
        self.human_joint_upper_limits = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        raise NotImplementedError('Implement observations')

    def _get_obs(self, forces):
        raise NotImplementedError('Implement observations')

    def reset(self):
        raise NotImplementedError('Implement reset')

    def config(self, tag, section=None):
        return float(self.configp[self.task if section is None else section][tag])

    def take_step(self, action, robot_arm='left', gains=0.05, forces=1, human_gains=0.1, human_forces=1, step_sim=True):
        action = np.clip(action, a_min=self.action_space.low, a_max=self.action_space.high)
        # print('cameraYaw=%.2f, cameraPitch=%.2f, distance=%.2f' % p.getDebugVisualizerCamera(physicsClientId=self.id)[-4:-1])

        # print('Total time:', self.total_time)
        # self.total_time += 0.1
        self.iteration += 1
        if self.last_sim_time is None:
            self.last_sim_time = time.time()

        action *= 0.05
        action_robot = action
        indices = self.robot_left_arm_joint_indices if robot_arm == 'left' else self.robot_right_arm_joint_indices if robot_arm == 'right' else self.robot_both_arm_joint_indices

        if self.human_control or (self.world_creation.human_impairment == 'tremor' and self.human_controllable_joint_indices):
            human_len = len(self.human_controllable_joint_indices)
            if self.human_control:
                action_robot = action[:self.action_robot_len]
                action_human = action[self.action_robot_len:]
            else:
                action_human = np.zeros(human_len)
            if len(action_human) != human_len:
                print('Received human actions of length %d does not match expected action length of %d' % (len(action_human), human_len))
                exit()
            human_joint_states = p.getJointStates(self.human, jointIndices=self.human_controllable_joint_indices, physicsClientId=self.id)
            human_joint_positions = np.array([x[0] for x in human_joint_states])

        robot_joint_states = p.getJointStates(self.robot, jointIndices=indices, physicsClientId=self.id)
        robot_joint_positions = np.array([x[0] for x in robot_joint_states])

        for _ in range(self.frame_skip):
            action_robot[robot_joint_positions + action_robot < self.robot_lower_limits] = 0
            action_robot[robot_joint_positions + action_robot > self.robot_upper_limits] = 0
            robot_joint_positions += action_robot
            if self.human_control or (self.world_creation.human_impairment == 'tremor' and self.human_controllable_joint_indices):
                action_human[human_joint_positions + action_human < self.human_lower_limits] = 0
                action_human[human_joint_positions + action_human > self.human_upper_limits] = 0
                if self.world_creation.human_impairment == 'tremor':
                    human_joint_positions = self.target_human_joint_positions + self.world_creation.human_tremors * (1 if self.iteration % 2 == 0 else -1)
                    self.target_human_joint_positions += action_human
                human_joint_positions += action_human

        p.setJointMotorControlArray(self.robot, jointIndices=indices, controlMode=p.POSITION_CONTROL, targetPositions=robot_joint_positions, positionGains=np.array([gains]*self.action_robot_len), forces=[forces]*self.action_robot_len, physicsClientId=self.id)
        if self.human_control or (self.world_creation.human_impairment == 'tremor' and self.human_controllable_joint_indices):
            p.setJointMotorControlArray(self.human, jointIndices=self.human_controllable_joint_indices, controlMode=p.POSITION_CONTROL, targetPositions=human_joint_positions, positionGains=np.array([human_gains]*human_len), forces=[human_forces*self.world_creation.human_strength]*human_len, physicsClientId=self.id)

        if step_sim:
            # Update robot position
            for _ in range(self.frame_skip):
                p.stepSimulation(physicsClientId=self.id)
                if self.human_control:
                    self.enforce_realistic_human_joint_limits()
                self.enforce_hard_human_joint_limits()
                self.update_targets()
                if self.gui:
                    # Slow down time so that the simulation matches real time
                    self.slow_time()
            self.record_video_frame()

    def enforce_realistic_human_joint_limits(self):
        # Only enforce limits for the human arm that is moveable (if either arm is even moveable)
        if 3 in self.human_controllable_joint_indices:
            # Right human arm
            tz, tx, ty, qe = [j[0] for j in p.getJointStates(self.human, jointIndices=[3, 4, 5, 6], physicsClientId=self.id)]
            # Transform joint angles to match those from the Matlab data
            tz2 = (-tz + 2*np.pi) % (2*np.pi)
            tx2 = (tx + 2*np.pi) % (2*np.pi)
            ty2 = -ty
            qe2 = (-qe + 2*np.pi) % (2*np.pi)
            result = self.human_limits_model.predict_classes(np.array([[tz2, tx2, ty2, qe2]]))
            if result == 1:
                # This is a valid pose for the person
                self.right_arm_previous_valid_pose = [tz, tx, ty, qe]
            elif result == 0 and self.right_arm_previous_valid_pose is not None:
                # The person is in an invalid pose. Move them back to the most recent valid pose.
                for i, j in enumerate([3, 4, 5, 6]):
                    p.resetJointState(self.human, jointIndex=j, targetValue=self.right_arm_previous_valid_pose[i], targetVelocity=0, physicsClientId=self.id)
        if 13 in self.human_controllable_joint_indices:
            # Left human arm
            tz, tx, ty, qe = [j[0] for j in p.getJointStates(self.human, jointIndices=[13, 14, 15, 16], physicsClientId=self.id)]
            # Transform joint angles to match those from the Matlab data
            tz2 = (tz + 2*np.pi) % (2*np.pi)
            tx2 = (tx + 2*np.pi) % (2*np.pi)
            ty2 = ty
            qe2 = (-qe + 2*np.pi) % (2*np.pi)
            result = self.human_limits_model.predict_classes(np.array([[tz2, tx2, ty2, qe2]]))
            if result == 1:
                # This is a valid pose for the person
                self.left_arm_previous_valid_pose = [tz, tx, ty, qe]
            elif result == 0 and self.left_arm_previous_valid_pose is not None:
                # The person is in an invalid pose. Move them back to the most recent valid pose.
                for i, j in enumerate([13, 14, 15, 16]):
                    p.resetJointState(self.human, jointIndex=j, targetValue=self.left_arm_previous_valid_pose[i], targetVelocity=0, physicsClientId=self.id)

    def enforce_hard_human_joint_limits(self):
        if not self.human_controllable_joint_indices:
            return
        # Enforce joint limits. Sometimes, external forces and break the person's hard joint limits.
        joint_states = p.getJointStates(self.human, jointIndices=self.human_controllable_joint_indices, physicsClientId=self.id)
        joint_positions = np.array([x[0] for x in joint_states])
        if self.human_joint_lower_limits is None:
            self.human_joint_lower_limits = []
            self.human_joint_upper_limits = []
            for i, j in enumerate(self.human_controllable_joint_indices):
                joint_info = p.getJointInfo(self.human, j, physicsClientId=self.id)
                joint_name = joint_info[1]
                joint_pos = joint_positions[i]
                lower_limit = joint_info[8]
                upper_limit = joint_info[9]
                self.human_joint_lower_limits.append(lower_limit)
                self.human_joint_upper_limits.append(upper_limit)
                # print(joint_name, joint_pos, lower_limit, upper_limit)
        for i, j in enumerate(self.human_controllable_joint_indices):
            if joint_positions[i] < self.human_joint_lower_limits[i]:
                p.resetJointState(self.human, jointIndex=j, targetValue=self.human_joint_lower_limits[i], targetVelocity=0, physicsClientId=self.id)
            elif joint_positions[i] > self.human_joint_upper_limits[i]:
                p.resetJointState(self.human, jointIndex=j, targetValue=self.human_joint_upper_limits[i], targetVelocity=0, physicsClientId=self.id)

    def human_preferences(self, end_effector_velocity=0, total_force_on_human=0, tool_force_at_target=0, food_hit_human_reward=0, food_mouth_velocities=[], dressing_forces=[[]], arm_manipulation_tool_forces_on_human=[0, 0], arm_manipulation_total_force_on_human=0):
        # Slow end effector velocities
        reward_velocity = -end_effector_velocity

        # < 10 N force at target
        reward_high_target_forces = 0 if tool_force_at_target < 10 else -tool_force_at_target

        # --- Scratching, Wiping ---
        # Any force away from target is low
        reward_force_nontarget = -(total_force_on_human - tool_force_at_target)

        # --- Scooping, Feeding, Drinking ---
        if self.task in ['scooping', 'feeding', 'drinking']:
            # Penalty when robot's body applies force onto a person
            reward_force_nontarget = -total_force_on_human
        # Penalty when robot spills food on the person
        reward_food_hit_human = food_hit_human_reward
        # Human prefers food entering mouth at low velocities
        reward_food_velocities = 0 if len(food_mouth_velocities) == 0 else -np.sum(food_mouth_velocities)

        # --- Dressing ---
        # Penalty when cloth applies force onto a person
        reward_dressing_force = -np.sum(np.linalg.norm(dressing_forces, axis=-1))

        # --- Arm Manipulation ---
        # Penalty for applying large pressure to the person (high forces over small surface areas)
        if self.task in ['arm_manipulation']:
            tool_left_contact_points = len(p.getClosestPoints(bodyA=self.robot, bodyB=self.human, linkIndexA=(78 if self.robot_type=='pr2' else 24 if self.robot_type=='sawyer' else 54 if self.robot_type=='baxter' else 9 if self.robot_type=='jaco' else 7), distance=0.01, physicsClientId=self.id))
            tool_right_contact_points = len(p.getClosestPoints(bodyA=self.robot, bodyB=self.human, linkIndexA=(55 if self.robot_type=='pr2' else 24 if self.robot_type=='sawyer' else 31 if self.robot_type=='baxter' else 9 if self.robot_type=='jaco' else 7), distance=0.01, physicsClientId=self.id))
            tool_left_pressure = 0 if tool_left_contact_points <= 0 else (arm_manipulation_tool_forces_on_human[0] / tool_left_contact_points)
            tool_right_pressure = 0 if tool_right_contact_points <= 0 else (arm_manipulation_tool_forces_on_human[1] / tool_right_contact_points)
            reward_arm_manipulation_tool_pressures = -(tool_left_pressure + tool_right_pressure)
            reward_force_nontarget = -(arm_manipulation_total_force_on_human - np.sum(arm_manipulation_tool_forces_on_human))
        else:
            reward_arm_manipulation_tool_pressures = 0.0

        return self.C_v*reward_velocity + self.C_f*reward_force_nontarget + self.C_hf*reward_high_target_forces + self.C_fd*reward_food_hit_human + self.C_fdv*reward_food_velocities + self.C_d*reward_dressing_force + self.C_p*reward_arm_manipulation_tool_pressures

    def reset_robot_joints(self):
        # Reset all robot joints
        for rj in range(p.getNumJoints(self.robot, physicsClientId=self.id)):
            p.resetJointState(self.robot, jointIndex=rj, targetValue=0, targetVelocity=0, physicsClientId=self.id)
        # Position end effectors whith dual arm robots
        if self.robot_type == 'pr2':
            for i, j in enumerate(self.robot_left_arm_joint_indices):
                p.resetJointState(self.robot, jointIndex=j, targetValue=[1.75, 1.25, 1.5, -0.5, 1, 0, 1][i], targetVelocity=0, physicsClientId=self.id)
            for i, j in enumerate(self.robot_right_arm_joint_indices):
                p.resetJointState(self.robot, jointIndex=j, targetValue=[-1.75, 1.25, -1.5, -0.5, -1, 0, -1][i], targetVelocity=0, physicsClientId=self.id)
        elif self.robot_type == 'baxter':
            for i, j in enumerate(self.robot_left_arm_joint_indices):
                p.resetJointState(self.robot, jointIndex=j, targetValue=[0.75, 1, 0.5, 0.5, 1, -0.5, 0][i], targetVelocity=0, physicsClientId=self.id)
            for i, j in enumerate(self.robot_right_arm_joint_indices):
                p.resetJointState(self.robot, jointIndex=j, targetValue=[-0.75, 1, -0.5, 0.5, -1, -0.5, 0][i], targetVelocity=0, physicsClientId=self.id)

    def joint_limited_weighting(self, q, lower_limits, upper_limits):
        phi = 0.5
        lam = 0.05
        weights = []
        for qi, l, u in zip(q, lower_limits, upper_limits):
            qr = 0.5*(u - l)
            weights.append(1.0 - np.power(phi, (qr - np.abs(qr - qi + l)) / (lam*qr) + 1))
            if weights[-1] < 0.001:
                weights[-1] = 0.001
        # Joint-limited-weighting
        joint_limit_weight = np.diag(weights)
        return joint_limit_weight

    def get_motor_joint_states(self, robot):
        num_joints = p.getNumJoints(robot, physicsClientId=self.id)
        joint_states = p.getJointStates(robot, range(num_joints), physicsClientId=self.id)
        joint_infos = [p.getJointInfo(robot, i, physicsClientId=self.id) for i in range(num_joints)]
        joint_states = [j for j, i in zip(joint_states, joint_infos) if i[2] != p.JOINT_FIXED]
        joint_positions = [state[0] for state in joint_states]
        joint_velocities = [state[1] for state in joint_states]
        joint_torques = [state[3] for state in joint_states]
        return joint_positions, joint_velocities, joint_torques

    def position_robot_toc(self, robot, joints, start_pos_orient, target_pos_orients, joint_indices, lower_limits, upper_limits, ik_indices, pos_offset=np.zeros(3), base_euler_orient=np.zeros(3), max_ik_iterations=500, attempts=100, ik_random_restarts=1, step_sim=False, check_env_collisions=False, right_side=True, random_rotation=30, random_position=0.5, human_joint_indices=None, human_joint_positions=None):
        # Continually randomize the robot base position and orientation
        # Select best base pose according to number of goals reached and manipulability
        if type(joints) == int:
            joints = [joints]
            start_pos_orient = [start_pos_orient]
            target_pos_orients = [target_pos_orients]
            joint_indices = [joint_indices]
            lower_limits = [lower_limits]
            upper_limits = [upper_limits]
            ik_indices = [ik_indices]
        a = 6 # Order of the robot space. 6D (3D position, 3D orientation)
        best_position = None
        best_orientation = None
        best_num_goals_reached = None
        best_manipulability = None
        best_start_joint_poses = [None]*len(joints)
        start_fails = 0
        iteration = 0
        best_pose_count = 0
        while iteration < attempts or best_position is None:
            iteration += 1
            random_pos = np.array([self.np_random.uniform(-random_position if right_side else 0, 0 if right_side else random_position), self.np_random.uniform(-random_position, random_position), 0])
            random_orientation = p.getQuaternionFromEuler([base_euler_orient[0], base_euler_orient[1], base_euler_orient[2] + np.deg2rad(self.np_random.uniform(-random_rotation, random_rotation))], physicsClientId=self.id)
            p.resetBasePositionAndOrientation(robot, np.array([-0.85, -0.4, 0]) + pos_offset + random_pos, random_orientation, physicsClientId=self.id)
            # Check if the robot can reach all target locations from this base pose
            num_goals_reached = 0
            manipulability = 0.0
            start_joint_poses = [None]*len(joints)
            for i, joint in enumerate(joints):
                for j, (target_pos, target_orient) in enumerate(start_pos_orient[i] + target_pos_orients[i]):
                    best_jlwki = None
                    goal_success = False
                    orient = target_orient
                    for k in range(ik_random_restarts):
                        # Reset human joints in case they got perturbed by previous iterations
                        if human_joint_positions is not None:
                            for h, pos in zip(human_joint_indices, human_joint_positions):
                                p.resetJointState(self.human, jointIndex=h, targetValue=pos, targetVelocity=0, physicsClientId=self.id)
                        # Reset all robot joints
                        self.reset_robot_joints()
                        # Find IK solution
                        success, joint_positions_q_star = self.util.ik_jlwki(robot, joint, target_pos, orient, self.world_creation, joint_indices[i], lower_limits[i], upper_limits[i], ik_indices=ik_indices[i], max_iterations=max_ik_iterations, success_threshold=0.03, half_range=(self.robot_type=='baxter'), step_sim=step_sim, check_env_collisions=check_env_collisions)
                        if success:
                            goal_success = True
                        else:
                            goal_success = False
                            break
                        joint_positions, _, _ = self.get_motor_joint_states(robot)
                        joint_velocities = [0.0] * len(joint_positions)
                        joint_accelerations = [0.0] * len(joint_positions)
                        center_of_mass = p.getLinkState(robot, joint, computeLinkVelocity=True, computeForwardKinematics=True, physicsClientId=self.id)[2]
                        J_linear, J_angular = p.calculateJacobian(robot, joint, localPosition=center_of_mass, objPositions=joint_positions, objVelocities=joint_velocities, objAccelerations=joint_accelerations, physicsClientId=self.id)
                        J_linear = np.array(J_linear)[:, ik_indices[i]]
                        J_angular = np.array(J_angular)[:, ik_indices[i]]
                        J = np.concatenate([J_linear, J_angular], axis=0)
                        # Joint-limited-weighting
                        joint_limit_weight = self.joint_limited_weighting(joint_positions_q_star, lower_limits[i], upper_limits[i])
                        # Joint-limited-weighted kinematic isotropy (JLWKI)
                        det = np.linalg.det(np.matmul(np.matmul(J, joint_limit_weight), J.T))
                        if det < 0:
                            det = 0
                        jlwki = np.power(det, 1.0/a) / (np.trace(np.matmul(np.matmul(J, joint_limit_weight), J.T))/a)
                        if best_jlwki is None or jlwki > best_jlwki:
                            best_jlwki = jlwki
                    if goal_success:
                        num_goals_reached += 1
                        manipulability += best_jlwki
                        if j == 0:
                            start_joint_poses[i] = joint_positions_q_star
                    if j < len(start_pos_orient[i]) and not goal_success:
                        # Not able to find an IK solution to a start goal. We cannot use this base pose
                        start_fails += 1
                        num_goals_reached = -1
                        manipulability = None
                        break
                if num_goals_reached == -1:
                    break

            if num_goals_reached == 4:
                best_pose_count += 1
            if num_goals_reached > 0:
                if best_position is None or num_goals_reached > best_num_goals_reached or (num_goals_reached == best_num_goals_reached and manipulability > best_manipulability):
                    best_position = random_pos
                    best_orientation = random_orientation
                    best_num_goals_reached = num_goals_reached
                    best_manipulability = manipulability
                    best_start_joint_poses = start_joint_poses

        p.resetBasePositionAndOrientation(robot, np.array([-0.85, -0.4, 0]) + pos_offset + best_position, best_orientation, physicsClientId=self.id)
        for i, joint in enumerate(joints):
            self.world_creation.setup_robot_joints(robot, joint_indices[i], lower_limits[i], upper_limits[i], randomize_joint_positions=False, default_positions=np.array(best_start_joint_poses[i]), tool=None)
        # Reset human joints in case they got perturbed by previous iterations
        if human_joint_positions is not None:
            for h, pos in zip(human_joint_indices, human_joint_positions):
                p.resetJointState(self.human, jointIndex=h, targetValue=pos, targetVelocity=0, physicsClientId=self.id)
        return best_position, best_orientation, best_start_joint_poses

    def slow_time(self):
        # Slow down time so that the simulation matches real time
        t = time.time() - self.last_sim_time
        if t < self.time_step:
            time.sleep(self.time_step - t)
        self.last_sim_time = time.time()

    def setup_timing(self):
        self.total_time = 0
        self.last_sim_time = None
        self.iteration = 0

    def setup_record_video(self, task='scratch_itch_pr2'):
        if self.record_video and self.gui:
            if self.video_writer is not None:
                self.video_writer.release()
            now = datetime.datetime.now()
            date = now.strftime('%Y-%m-%d_%H-%M-%S')
            # self.video_writer = cv2.VideoWriter('%s_%s.avi' % (task, date), cv2.VideoWriter_fourcc(*'MJPG'), 10, (self.width, self.height))

    def record_video_frame(self):
        if self.record_video and self.gui:
            frame = np.reshape(p.getCameraImage(width=self.width, height=self.height, renderer=p.ER_BULLET_HARDWARE_OPENGL, physicsClientId=self.id)[2], (self.height, self.width, 4))[:, :, :3]
            # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            # self.video_writer.write(frame)

    def update_targets(self):
        pass

    def render(self, mode='human'):
        if not self.gui:
            self.gui = True
            p.disconnect(self.id)
            self.id = p.connect(p.GUI, options='--background_color_red=0.8 --background_color_green=0.9 --background_color_blue=1.0 --width=%d --height=%d' % (self.width, self.height))

            self.world_creation = WorldCreation(self.id, robot_type=self.robot_type, task=self.task, time_step=self.time_step, np_random=self.np_random, config=self.config)
            self.util = Util(self.id, self.np_random)
            # print('Physics server ID:', self.id)

