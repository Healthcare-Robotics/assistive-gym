import os, time
import numpy as np
from gym import spaces
from screeninfo import get_monitors
import pybullet as p
from keras.models import load_model
from gym.utils import seeding
from .human_creation import HumanCreation
from .agents import agent, human
from .agents.agent import Agent
from .agents.human import Human

class BaseEnv:
    def __init__(self, time_step=0.02, frame_skip=5, render=True, gravity=-9.81, seed=1001):
        self.time_step = time_step
        self.frame_skip = frame_skip
        self.gravity = gravity
        if render:
            self.gui = True
            try:
                self.width = get_monitors()[0].width
                self.height = get_monitors()[0].height
            except Exception as e:
                self.width = 1920
                self.height = 1080
            self.id = p.connect(p.GUI, options='--background_color_red=0.8 --background_color_green=0.9 --background_color_blue=1.0 --width=%d --height=%d' % (self.width, self.height))
        else:
            self.gui = False
            self.id = p.connect(p.DIRECT)

        self.np_random, seed = seeding.np_random(seed)
        self.directory = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'assets')
        self.human_creation = HumanCreation(self.id, np_random=self.np_random, cloth=False)
        self.human_limits_model = load_model(os.path.join(self.directory, 'realistic_arm_limits_model.h5'))

        self.agents = []

    def reset(self):
        p.resetSimulation(physicsClientId=self.id)
        p.configureDebugVisualizer(p.COV_ENABLE_MOUSE_PICKING, 0, physicsClientId=self.id)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=self.id)
        p.setTimeStep(self.time_step, physicsClientId=self.id)
        # Disable real time simulation so that the simulation only advances when we call stepSimulation
        p.setRealTimeSimulation(0, physicsClientId=self.id)
        p.setGravity(0, 0, self.gravity, physicsClientId=self.id)
        self.agents = []
        self.last_sim_time = None
        self.iteration = 0
        self.action_space = None

    def create_human(self, controllable=False, controllable_joint_indices=[], fixed_base=False, human_impairment='random', gender='random', mass=None, radius_scale=1.0, height_scale=1.0):
        '''
        human_impairement in ['none', 'limits', 'weakness', 'tremor']
        gender in ['male', 'female']
        '''
        human = Human(controllable_joint_indices, controllable=controllable)
        human.init(self.human_creation, self.human_limits_model, fixed_base, human_impairment, gender, None, self.id, self.np_random, mass=mass, radius_scale=radius_scale, height_scale=height_scale)
        if controllable:
            self.agents.append(human)
        return human

    def create_robot(self, robot_class, controllable_joints='right', fixed_base=True):
        robot = robot_class(controllable_joints)
        robot.init(self.directory, self.id, self.np_random, fixed_base=fixed_base)
        self.agents.append(robot)
        return robot

    def take_step(self, actions, gains=0.05, forces=1, action_multiplier=0.05):
        if type(gains) not in (list, tuple):
            gains = [gains]*len(self.agents)
        if type(forces) not in (list, tuple):
            forces = [forces]*len(self.agents)
        if self.action_space is None:
            self.action_space = spaces.Box(low=-np.ones(np.sum([len(a.controllable_joint_indices) for a in self.agents])), high=np.ones(np.sum([len(a.controllable_joint_indices) for a in self.agents])), dtype=np.float32)
        if self.last_sim_time is None:
            self.last_sim_time = time.time()
        self.iteration += 1
        actions = np.clip(actions, a_min=self.action_space.low, a_max=self.action_space.high)
        actions *= action_multiplier
        action_index = 0
        for i, agent in enumerate(self.agents):
            agent_action_len = len(agent.controllable_joint_indices)
            action = np.copy(actions[action_index:action_index+agent_action_len])
            if len(action) != agent_action_len:
                print('Received agent actions of length %d does not match expected action length of %d' % (len(action), agent_action_len))
                exit()
            # Append the new action to the current measured joint angles
            agent_joint_angles = agent.get_joint_angles(agent.controllable_joint_indices)
            # Update the target robot/human joint angles based on the proposed action and joint limits
            for _ in range(self.frame_skip):
                action[agent_joint_angles + action < agent.controllable_joint_lower_limits] = 0
                action[agent_joint_angles + action > agent.controllable_joint_upper_limits] = 0
                if type(agent) == Human and agent.impairment == 'tremor':
                    agent.target_joint_angles += action
                    agent_joint_angles = agent.target_joint_angles + agent.tremors * (1 if self.iteration % 2 == 0 else -1)
                else:
                    agent_joint_angles += action
            agent.control(agent.controllable_joint_indices, agent_joint_angles, gains[i], forces[i])
            # Update robot position
            for _ in range(self.frame_skip):
                p.stepSimulation(physicsClientId=self.id)
                if type(agent) == Human:
                    if agent.controllable:
                        agent.enforce_realistic_joint_limits()
                    agent.enforce_joint_limits()
                if self.gui:
                    # Slow down time so that the simulation matches real time
                    self.slow_time()

    def slow_time(self):
        # Slow down time so that the simulation matches real time
        t = time.time() - self.last_sim_time
        if t < self.time_step:
            time.sleep(self.time_step - t)
        self.last_sim_time = time.time()

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
        last_sphere_id = p.createMultiBody(baseMass=mass, baseCollisionShapeIndex=sphere_collision, baseVisualShapeIndex=sphere_visual, basePosition=[0, 0, 0], useMaximalCoordinates=False, batchPositions=batch_positions, physicsClientId=self.id)
        spheres = []
        for body in list(range(last_sphere_id-len(batch_positions)+1, last_sphere_id+1)):
            sphere = Agent()
            sphere.init(body, self.id, self.np_random, indices=-1)
            spheres.append(sphere)
        return spheres

