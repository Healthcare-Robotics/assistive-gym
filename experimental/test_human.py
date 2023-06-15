import colorsys
import os
import pickle
import time

import numpy as np
import pybullet as p
import pybullet_data
import smplx
import trimesh
from gym.utils import seeding
from smplx import lbs

from assistive_gym.envs.agents.agent import Agent
from assistive_gym.envs.agents.furniture import Furniture
from assistive_gym.envs.utils.human_urdf_dict import HumanUrdfDict
from assistive_gym.envs.utils.human_utils import set_joint_angles, set_self_collisions, change_dynamic_properties, \
    check_collision, set_global_orientation
from assistive_gym.envs.utils.smpl_dict import SMPLDict

from assistive_gym.envs.utils.smpl_geom import generate_geom
from assistive_gym.envs.utils.urdf_utils import convert_aa_to_euler_quat, load_smpl, generate_urdf

SMPL_PATH = os.path.join(os.getcwd(), "examples/data/smpl_bp_ros_smpl_9.pkl")
class HumanUrdfTest(Agent):
    def __init__(self):
        super(HumanUrdfTest, self).__init__()
        self.smpl_dict = SMPLDict()
        self.human_pip_dict = HumanUrdfDict()
        self.controllable_joint_indices = [1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19, 21, 22, 23, 25, 26, 27,
                                       29, 30, 31, 33, 34, 35, 37, 38, 39, 41, 42, 43, 45, 46, 47, 49, 50, 51, 53,
                                       54, 55, 57, 58, 59, 61, 62, 63, 65, 66, 67, 69, 70, 71, 73, 74, 75, 77, 78,
                                       79, 81, 82, 83, 85, 86, 87, 89, 90, 91]

    def init(self, id, np_random):
        # TODO: no hard coding
        # self.human_id = p.loadURDF("assistive_gym/envs/assets/human/human_pip.urdf")
        self.id= id
        self.human_id = p.loadURDF("test_mesh.urdf", [0, 0, 0.2], flags = p.URDF_USE_SELF_COLLISION, useFixedBase=False)
        change_dynamic_properties(human.human_id, list(range(0, 93)))
        super(HumanUrdfTest, self).init(self.human_id, id, np_random)

    def step_forward(self):
        x0 = [1.0] * len(self.controllable_joint_indices)
        p.setJointMotorControlArray(self.human_id, jointIndices=self.controllable_joint_indices,
                                    controlMode=p.POSITION_CONTROL,
                                    forces=[1000] * len(self.controllable_joint_indices),
                                    positionGains=[0.01] * len(self.controllable_joint_indices),
                                    targetPositions=x0,
                                    physicsClientId=self.id)
        # for _ in range(1000):
        #     p.stepSimulation(physicsClientId=self.id)
        #     # time.sleep(0.1)
        p.setRealTimeSimulation(1)


if __name__ == "__main__":
    # Start the simulation engine
    physic_client_id = p.connect(p.GUI)

    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    np_random, seed = seeding.np_random(1001)

    #plane
    planeId = p.loadURDF("assistive_gym/envs/assets/plane/plane.urdf", [0,0,0])

    #bed
    # bed_id = p.loadURDF("assistive_gym/envs/assets/bed/hospital_bed.urdf", [0,0,0], useFixedBase=False)

    # robot
    # robotId = p.loadURDF("assistive_gym/envs/assets/stretch/stretch_uncalibrated.urdf", [1,0,0], useFixedBase=True)

    # human
    human = HumanUrdfTest()
    human.init(physic_client_id, np_random)

    # print all the joints
    for j in range(p.getNumJoints(human.human_id)):
        print (p.getJointInfo(human.human_id, j))
        # check if the joint is revolute
        if p.getJointInfo(human.human_id, j)[2] == p.JOINT_REVOLUTE:
            print ("joint is revolute")
            # p.setJointMotorControl2(human.human_id, jointIndex=j, controlMode=p.VELOCITY_CONTROL, force=0.0,
            #                         physicsClientId=physic_client_id)
    # Set the simulation parameters
    p.setGravity(0,0,-9.81)


    # Set the simulation parameters
    smpl_path = os.path.join(os.getcwd(), SMPL_PATH)
    smpl_data = load_smpl(smpl_path)
    set_joint_angles(human.human_id, smpl_data.body_pose)
    set_global_orientation(human.human_id, smpl_data.global_orient, [0, 0, 0.2])

    # Set the camera view
    cameraDistance = 3
    cameraYaw = 0
    cameraPitch = -30
    cameraTargetPosition = [0,0,1]
    p.resetDebugVisualizerCamera(cameraDistance, cameraYaw, cameraPitch, cameraTargetPosition)
    ########################   print all joints and draw lines at the joint locations ##########################
    # for j in range(1, num_joints):
    #     joint_info = p.getJointInfo(robotId, j)
    #
    #     link_index = int(joint_info[-1])
    #     link_info = p.getLinkState(robotId, link_index)
    #     link_pos = link_info[0]
    #     link_d_pos = tuple([link_pos[0], link_pos[1], link_pos[2] + 0.1])
    #
    #     joint_pos = tuple(
    #         [joint_info[14][0] + link_pos[0], joint_info[14][1] + link_pos[1], joint_info[14][2] + link_pos[2]])
    #     joint_d_pos = tuple([joint_pos[0], joint_pos[1], joint_pos[2] + 0.1])
    #
    #     p.addUserDebugLine(joint_pos, joint_d_pos, [1, 0, 0], 2)
    #     p.addUserDebugLine(link_pos, link_d_pos, [0, 0, 1], 2)

    #####################################  set joint angles for debugging ########################################
    # human.set_joint_angles([81,82,83], [0, np.pi/2, 0], use_limits=False) # right ankle
    # human.set_joint_angles([61, 62, 63], [0, -np.pi / 2, 0], use_limits=False)  # left ankle
    # human.set_joint_angles([77, 78, 79], [np.pi/2, 0, 0], use_limits=False)  # right shoulder
    # human.set_joint_angles([21,22,23], [np.pi/2, 0, 0], use_limits=False) # right knee
    # human.set_joint_angles([81,82,83], [np.pi/2, 0, 0])


    while True:
        p.stepSimulation()
        # check_collision(human.human_id, human.human_id)
        # human.step_forward()
    # Disconnect from the simulation
    p.disconnect()
