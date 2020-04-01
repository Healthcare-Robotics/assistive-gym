import os, time, datetime, configparser
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import pybullet as p
# import cv2
from keras.models import load_model
from screeninfo import get_monitors

from .util import Util
from .world_creation import WorldCreation
from .agents import tool
from .agents.tool import Tool

class AssistiveEnv(gym.Env):
    def __init__(self, robot, human, task='scratch_itch', human_control=False, frame_skip=5, time_step=0.02, obs_robot_len=30, obs_human_len=0):
        # Start the bullet physics server
        self.id = p.connect(p.DIRECT)
        self.gui = False

        self.robot = robot
        self.human = human
        self.tool = Tool()

        self.task = task
        self.human_control = human_control
        self.action_robot_len = len(self.robot.controllable_joint_indices)
        self.action_human_len = len(self.human.controllable_joint_indices) if human_control else 0
        self.obs_robot_len = obs_robot_len# + self.action_robot_len
        self.obs_human_len = obs_human_len# + self.action_human_len
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

        self.world_creation = WorldCreation(self.id, self.robot, self.human, task=task, time_step=self.time_step, np_random=self.np_random, config=self.config)
        self.util = Util(self.id, self.np_random)

        self.record_video = False
        self.video_writer = None
        try:
            self.width = get_monitors()[0].width
            self.height = get_monitors()[0].height
        except Exception as e:
            self.width = 1920
            self.height = 1080
            # self.width = 3840
            # self.height = 2160

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

        # If the human is controllable, split the action into action_robot and action_human
        human_control = self.human_control or (self.human.impairment == 'tremor' and self.human.controllable_joint_indices)
        if human_control:
            human_len = len(self.human.controllable_joint_indices)
            if self.human_control:
                action_robot = action[:self.action_robot_len]
                action_human = action[self.action_robot_len:]
            else:
                action_human = np.zeros(human_len)
            if len(action_human) != human_len:
                print('Received human actions of length %d does not match expected action length of %d' % (len(action_human), human_len))
                exit()
            human_joint_angles = self.human.get_joint_angles(self.human.controllable_joint_indices)

        robot_joint_angles = self.robot.get_joint_angles(self.robot.controllable_joint_indices)

        # Update the target robot/human joint angles based on the proposed action and joint limits
        for _ in range(self.frame_skip):
            action_robot[robot_joint_angles + action_robot < self.robot.controllable_joint_lower_limits] = 0
            action_robot[robot_joint_angles + action_robot > self.robot.controllable_joint_upper_limits] = 0
            robot_joint_angles += action_robot
            if human_control:
                action_human[human_joint_angles + action_human < self.human.controllable_joint_lower_limits] = 0
                action_human[human_joint_angles + action_human > self.human.controllable_joint_upper_limits] = 0
                if self.human.impairment == 'tremor':
                    human_joint_angles = self.human.target_joint_angles + self.human.tremors * (1 if self.iteration % 2 == 0 else -1)
                    self.human.target_joint_angles += action_human
                human_joint_angles += action_human

        self.robot.control(self.robot.controllable_joint_indices, robot_joint_angles, gains, forces)
        if human_control:
            self.human.control(self.human.controllable_joint_indices, human_joint_angles, human_gains, human_forces*self.human.strength)

        if step_sim:
            # Update robot position
            for _ in range(self.frame_skip):
                p.stepSimulation(physicsClientId=self.id)
                if self.human_control:
                    self.human.enforce_realistic_joint_limits()
                self.human.enforce_joint_limits()
                self.update_targets()
                if self.gui:
                    # Slow down time so that the simulation matches real time
                    self.slow_time()
            self.record_video_frame()

    def human_preferences(self, end_effector_velocity=0, total_force_on_human=0, tool_force_at_target=0, food_hit_human_reward=0, food_mouth_velocities=[], dressing_forces=[[]], arm_manipulation_tool_forces_on_human=[0, 0], arm_manipulation_total_force_on_human=0):
        # Slow end effector velocities
        reward_velocity = -end_effector_velocity

        # < 10 N force at target
        reward_high_target_forces = 0 if tool_force_at_target < 10 else -tool_force_at_target

        # --- Scratching, Wiping ---
        # Any force away from target is low
        reward_force_nontarget = -(total_force_on_human - tool_force_at_target)

        # --- Scooping, Feeding, Drinking ---
        if self.task in ['feeding', 'drinking']:
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

            self.world_creation = WorldCreation(self.id, self.robot, self.human, task=self.task, time_step=self.time_step, np_random=self.np_random, config=self.config)
            self.util = Util(self.id, self.np_random)

