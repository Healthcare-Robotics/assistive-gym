import os, time, configparser
import numpy as np
import gym
from gym import spaces
from gym.utils import seeding
from screeninfo import get_monitors
import pybullet as p
from keras.models import load_model

from .util import Util
from .human_creation import HumanCreation
from .agents import agent, human, robot, tool, furniture
from .agents.agent import Agent
from .agents.human import Human
from .agents.robot import Robot
from .agents.tool import Tool
from .agents.furniture import Furniture

class AssistiveEnv(gym.Env):
    def __init__(self, robot=None, human=None, task='', obs_robot_len=0, obs_human_len=0, time_step=0.02, frame_skip=5, render=False, gravity=-9.81, seed=1001):
        self.task = task
        self.time_step = time_step
        self.frame_skip = frame_skip
        self.gravity = gravity
        self.id = None
        self.gui = False
        self.gpu = False
        self.seed(seed)
        if render:
            self.render()
        else:
            self.id = p.connect(p.DIRECT)
            self.util = Util(self.id, self.np_random)

        self.directory = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'assets')
        self.human_creation = HumanCreation(self.id, np_random=self.np_random, cloth=('dressing' in task))
        self.human_limits_model = load_model(os.path.join(self.directory, 'realistic_arm_limits_model.h5'))
        self.action_robot_len = len(robot.controllable_joint_indices) if robot is not None else 0
        self.action_human_len = len(human.controllable_joint_indices) if human is not None and human.controllable else 0
        self.action_space = spaces.Box(low=np.array([-1.0]*(self.action_robot_len+self.action_human_len)), high=np.array([1.0]*(self.action_robot_len+self.action_human_len)), dtype=np.float32)
        self.obs_robot_len = obs_robot_len
        self.obs_human_len = obs_human_len if human is not None and human.controllable else 0
        self.observation_space = spaces.Box(low=np.array([-1.0]*(self.obs_robot_len+self.obs_human_len)), high=np.array([1.0]*(self.obs_robot_len+self.obs_human_len)), dtype=np.float32)

        self.agents = []
        self.plane = Agent()
        self.robot = robot
        self.human = human
        self.tool = Tool()
        self.furniture = Furniture()

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

    def step(self, action):
        raise NotImplementedError('Implement observations')

    def _get_obs(self, agent=None):
        raise NotImplementedError('Implement observations')

    def config(self, tag, section=None):
        return float(self.configp[self.task if section is None else section][tag])

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def enable_gpu(self):
        self.gpu = True

    def reset(self):
        p.resetSimulation(physicsClientId=self.id)
        if not self.gui:
            # Reconnect the physics engine to forcefully clear memory when running long training scripts
            p.disconnect(self.id)
            self.id = p.connect(p.DIRECT)
            self.util = Util(self.id, self.np_random)
        if self.gpu:
            self.util.enable_gpu()
        # Configure camera position
        p.resetDebugVisualizerCamera(cameraDistance=1.75, cameraYaw=-25, cameraPitch=-45, cameraTargetPosition=[-0.2, 0, 0.4], physicsClientId=self.id)
        p.configureDebugVisualizer(p.COV_ENABLE_MOUSE_PICKING, 0, physicsClientId=self.id)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=self.id)
        p.setTimeStep(self.time_step, physicsClientId=self.id)
        # Disable real time simulation so that the simulation only advances when we call stepSimulation
        p.setRealTimeSimulation(0, physicsClientId=self.id)
        p.setGravity(0, 0, self.gravity, physicsClientId=self.id)
        self.agents = []
        self.last_sim_time = None
        self.iteration = 0
        self.forces = []
        self.task_success = 0

    def build_assistive_env(self, furniture_type=None, fixed_robot_base=True, fixed_human_base=True, human_impairment='random', gender='random'):
        # Build plane, furniture, robot, human, etc. (just like world creation)
        # Load the ground plane
        plane = p.loadURDF(os.path.join(self.directory, 'plane', 'plane.urdf'), physicsClientId=self.id)
        self.plane.init(plane, self.id, self.np_random, indices=-1)
        # Disable rendering during creation
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0, physicsClientId=self.id)
        # Create robot
        if self.robot is not None:
            self.robot.init(self.directory, self.id, self.np_random, fixed_base=fixed_robot_base)
            self.agents.append(self.robot)
        # Create human
        if self.human is not None and isinstance(self.human, Human):
            self.human.init(self.human_creation, self.human_limits_model, fixed_human_base, human_impairment, gender, self.config, self.id, self.np_random)
            if self.human.controllable or self.human.impairment == 'tremor':
                self.agents.append(self.human)
        # Create furniture (wheelchair, bed, or table)
        if furniture_type is not None:
            self.furniture.init(furniture_type, self.directory, self.id, self.np_random, wheelchair_mounted=self.robot.wheelchair_mounted if self.robot is not None else False)

    def init_env_variables(self):
        if len(self.action_space.low) == 1:
            obs_len = len(self._get_obs())
            self.observation_space.__init__(low=-np.ones(obs_len), high=np.ones(obs_len), dtype=np.float32)
            self.update_action_space()
            # Define action/obs lengths
            self.action_robot_len = len(self.robot.controllable_joint_indices)
            self.action_human_len = len(self.human.controllable_joint_indices) if self.human.controllable else 0
            self.obs_robot_len = len(self._get_obs('robot'))
            self.obs_human_len = len(self._get_obs('human'))

    def update_action_space(self):
        action_len = np.sum([len(a.controllable_joint_indices) for a in self.agents if not isinstance(a, Human) or a.controllable])
        self.action_space.__init__(low=-np.ones(action_len), high=np.ones(action_len), dtype=np.float32)

    def create_human(self, controllable=False, controllable_joint_indices=[], fixed_base=False, human_impairment='random', gender='random', mass=None, radius_scale=1.0, height_scale=1.0):
        '''
        human_impairement in ['none', 'limits', 'weakness', 'tremor']
        gender in ['male', 'female']
        '''
        self.human = Human(controllable_joint_indices, controllable=controllable)
        self.human.init(self.human_creation, self.human_limits_model, fixed_base, human_impairment, gender, None, self.id, self.np_random, mass=mass, radius_scale=radius_scale, height_scale=height_scale)
        if controllable or self.human.impairment == 'tremor':
            self.agents.append(self.human)
            self.update_action_space()
        return self.human

    def create_robot(self, robot_class, controllable_joints='right', fixed_base=True):
        self.robot = robot_class(controllable_joints)
        self.robot.init(self.directory, self.id, self.np_random, fixed_base=fixed_base)
        self.agents.append(self.robot)
        self.update_action_space()
        return self.robot

    def take_step(self, actions, gains=0.05, forces=1, action_multiplier=0.05, step_sim=True):
        if type(gains) not in (list, tuple):
            gains = [gains]*len(self.agents)
        if type(forces) not in (list, tuple):
            forces = [forces]*len(self.agents)
        if self.last_sim_time is None:
            self.last_sim_time = time.time()
        self.iteration += 1
        self.forces = []
        actions = np.clip(actions, a_min=self.action_space.low, a_max=self.action_space.high)
        actions *= action_multiplier
        action_index = 0
        for i, agent in enumerate(self.agents):
            needs_action = not isinstance(agent, Human) or agent.controllable
            if needs_action:
                agent_action_len = len(agent.controllable_joint_indices)
                action = np.copy(actions[action_index:action_index+agent_action_len])
                action_index += agent_action_len
                if isinstance(agent, Robot):
                    action *= agent.action_multiplier
                if len(action) != agent_action_len:
                    print('Received agent actions of length %d does not match expected action length of %d' % (len(action), agent_action_len))
                    exit()
            # Append the new action to the current measured joint angles
            agent_joint_angles = agent.get_joint_angles(agent.controllable_joint_indices)
            # Update the target robot/human joint angles based on the proposed action and joint limits
            for _ in range(self.frame_skip):
                if needs_action:
                    below_lower_limits = agent_joint_angles + action < agent.controllable_joint_lower_limits
                    above_upper_limits = agent_joint_angles + action > agent.controllable_joint_upper_limits
                    action[below_lower_limits] = 0
                    action[above_upper_limits] = 0
                    agent_joint_angles[below_lower_limits] = agent.controllable_joint_lower_limits[below_lower_limits]
                    agent_joint_angles[above_upper_limits] = agent.controllable_joint_upper_limits[above_upper_limits]
                if isinstance(agent, Human) and agent.impairment == 'tremor':
                    if needs_action:
                        agent.target_joint_angles += action
                    agent_joint_angles = agent.target_joint_angles + agent.tremors * (1 if self.iteration % 2 == 0 else -1)
                else:
                    agent_joint_angles += action
            if isinstance(agent, Robot) and agent.action_duplication is not None:
                agent_joint_angles = np.concatenate([[a]*d for a, d in zip(agent_joint_angles, self.robot.action_duplication)])
                agent.control(agent.all_controllable_joints, agent_joint_angles, agent.gains, agent.forces)
            else:
                agent.control(agent.controllable_joint_indices, agent_joint_angles, gains[i], forces[i])
        if step_sim:
            # Update all agent positions
            for _ in range(self.frame_skip):
                p.stepSimulation(physicsClientId=self.id)
                for agent in self.agents:
                    if isinstance(agent, Human):
                        agent.enforce_joint_limits()
                        if agent.controllable:
                            agent.enforce_realistic_joint_limits()
                self.update_targets()
                if self.gui:
                    # Slow down time so that the simulation matches real time
                    self.slow_time()

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
        if self.task == 'arm_manipulation':
            tool_right_contact_points = len(self.tool_right.get_closest_points(self.human, distance=0.01)[-1])
            tool_left_contact_points = len(self.tool_left.get_closest_points(self.human, distance=0.01)[-1])
            tool_right_pressure = 0 if tool_right_contact_points <= 0 else (arm_manipulation_tool_forces_on_human[0] / tool_right_contact_points)
            tool_left_pressure = 0 if tool_left_contact_points <= 0 else (arm_manipulation_tool_forces_on_human[1] / tool_left_contact_points)

            reward_arm_manipulation_tool_pressures = -(tool_right_pressure + tool_left_pressure)
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

    def update_targets(self):
        pass

    def render(self, mode='human'):
        if not self.gui:
            self.gui = True
            if self.id is not None:
                p.disconnect(self.id)
            try:
                self.width = get_monitors()[0].width
                self.height = get_monitors()[0].height
            except Exception as e:
                self.width = 1920
                self.height = 1080
            self.id = p.connect(p.GUI, options='--background_color_red=0.8 --background_color_green=0.9 --background_color_blue=1.0 --width=%d --height=%d' % (self.width, self.height))
            self.util = Util(self.id, self.np_random)

    def get_euler(self, quaternion):
        return np.array(p.getEulerFromQuaternion(np.array(quaternion), physicsClientId=self.id))

    def get_quaternion(self, euler):
        return np.array(p.getQuaternionFromEuler(np.array(euler), physicsClientId=self.id))

    def create_sphere(self, radius=0.01, mass=0.0, pos=[0, 0, 0], visual=True, collision=True, rgba=[0, 1, 1, 1], maximal_coordinates=False, return_collision_visual=False):
        sphere_collision = p.createCollisionShape(shapeType=p.GEOM_SPHERE, radius=radius, physicsClientId=self.id) if collision else -1
        sphere_visual = p.createVisualShape(shapeType=p.GEOM_SPHERE, radius=radius, rgbaColor=rgba, physicsClientId=self.id) if visual else -1
        if return_collision_visual:
            return sphere_collision, sphere_visual
        body = p.createMultiBody(baseMass=mass, baseCollisionShapeIndex=sphere_collision, baseVisualShapeIndex=sphere_visual, basePosition=pos, useMaximalCoordinates=maximal_coordinates, physicsClientId=self.id)
        sphere = Agent()
        sphere.init(body, self.id, self.np_random, indices=-1)
        return sphere

    def create_spheres(self, radius=0.01, mass=0.0, batch_positions=[[0, 0, 0]], visual=True, collision=True, rgba=[0, 1, 1, 1]):
        sphere_collision = p.createCollisionShape(shapeType=p.GEOM_SPHERE, radius=radius, physicsClientId=self.id) if collision else -1
        sphere_visual = p.createVisualShape(shapeType=p.GEOM_SPHERE, radius=radius, rgbaColor=rgba, physicsClientId=self.id) if visual else -1
        spheres_ids = p.createMultiBody(baseMass=mass, baseCollisionShapeIndex=sphere_collision, baseVisualShapeIndex=sphere_visual, basePosition=[0, 0, 0], useMaximalCoordinates=False, batchPositions=batch_positions, physicsClientId=self.id)
        spheres = []
        for body in spheres_ids:
            sphere = Agent()
            sphere.init(body, self.id, self.np_random, indices=-1)
            spheres.append(sphere)
        return spheres

