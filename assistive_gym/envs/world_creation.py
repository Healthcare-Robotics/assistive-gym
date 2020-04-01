import os
import numpy as np
import pybullet as p
from keras.models import load_model
from .human_creation import HumanCreation
from .agents import furniture, agent
from .agents.furniture import Furniture
from .agents.agent import Agent

class WorldCreation:
    def __init__(self, id, robot, human, task='scratch_itch', time_step=0.02, np_random=None, config=None):
        self.id = id
        self.robot = robot
        self.human = human
        self.furniture = Furniture()
        self.task = task
        self.time_step = time_step
        self.np_random = np_random
        self.config = config
        self.directory = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'assets')
        self.human_creation = HumanCreation(self.id, np_random=np_random, cloth=(task=='dressing'))

    def create_new_world(self, furniture_type='wheelchair', static_human_base=False, human_impairment='random', print_joints=False, gender='random'):
        p.resetSimulation(physicsClientId=self.id)

        # Configure camera position
        p.resetDebugVisualizerCamera(cameraDistance=1.75, cameraYaw=-25, cameraPitch=-45, cameraTargetPosition=[-0.2, 0, 0.4], physicsClientId=self.id)

        p.configureDebugVisualizer(p.COV_ENABLE_MOUSE_PICKING, 0, physicsClientId=self.id)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=self.id)
        p.setTimeStep(self.time_step, physicsClientId=self.id)
        # Disable real time simulation so that the simulation only advances when we call stepSimulation
        p.setRealTimeSimulation(0, physicsClientId=self.id)

        # Load the ground plane
        p.loadURDF(os.path.join(self.directory, 'plane', 'plane.urdf'), physicsClientId=self.id)

        # Disable rendering during creation
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0, physicsClientId=self.id)

        # Create furniture (wheelchair, bed, or table)
        self.furniture.init(furniture_type, self.directory, self.id, self.np_random)

        human_limits_model = load_model(os.path.join(self.directory, 'realistic_arm_limits_model.h5'))
        self.human.init(self.human_creation, human_limits_model, static_human_base, human_impairment, gender, self.config, self.id, self.np_random)

        # Initialize robot
        self.robot.init(self.directory, self.id, self.np_random)

        return self.furniture, gender

    def create_sphere(self, radius=0.01, mass=0.0, pos=[0, 0, 0], collision=None, rgba=[0, 1, 1, 1]):
        sphere_collision = -1 if collision is None else p.createCollisionShape(shapeType=p.GEOM_SPHERE, radius=radius, physicsClientId=self.id)
        sphere_visual = p.createVisualShape(shapeType=p.GEOM_SPHERE, radius=radius, rgbaColor=rgba, physicsClientId=self.id)
        body = p.createMultiBody(baseMass=mass, baseCollisionShapeIndex=sphere_collision, baseVisualShapeIndex=sphere_visual, basePosition=pos, useMaximalCoordinates=False, physicsClientId=self.id)
        sphere = Agent()
        sphere.init(body, self.id, self.np_random, indices=-1)
        return sphere

    def print_joint_info(self, body, show_fixed=True):
        joint_names = []
        for j in range(p.getNumJoints(body, physicsClientId=self.id)):
            if show_fixed or p.getJointInfo(body, j, physicsClientId=self.id)[2] != p.JOINT_FIXED:
                print(p.getJointInfo(body, j, physicsClientId=self.id))
                joint_names.append((j, p.getJointInfo(body, j, physicsClientId=self.id)[1]))
        print(joint_names)
