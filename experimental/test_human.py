import colorsys
import os
import pickle

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
    check_collision
from assistive_gym.envs.utils.smpl_dict import SMPLDict

from assistive_gym.envs.smpl.serialization import load_model
from assistive_gym.envs.utils.smpl_geom import generate_geom
from assistive_gym.envs.utils.urdf_utils import convert_aa_to_euler_quat, load_smpl, generate_urdf


class HumanUrdfTest(Agent):
    def __init__(self):
        super(HumanUrdfTest, self).__init__()
        self.smpl_dict = SMPLDict()
        self.human_pip_dict = HumanUrdfDict()
        self.controllable_joint_indices = list(range(0, 93)) #94 joints


    def init(self, id, np_random):
        # TODO: no hard coding
        # self.human_id = p.loadURDF("assistive_gym/envs/assets/human/human_pip.urdf")
        self.human_id = p.loadURDF("test_mesh.urdf", [0, 0, 0], flags=p.URDF_USE_SELF_COLLISION, useFixedBase=False)
        set_self_collisions(self.human_id, id)
        change_dynamic_properties(human.human_id, list(range(0, 93)))
        super(HumanUrdfTest, self).init(self.human_id, id, np_random)

if __name__ == "__main__":
    # Start the simulation engine
    physic_client_id = p.connect(p.GUI)

    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    np_random, seed = seeding.np_random(1001)

    #plane
    planeId = p.loadURDF("assistive_gym/envs/assets/plane/plane.urdf", [0,0,0])

    # human
    human = HumanUrdfTest()
    human.init(physic_client_id, np_random)

    # print all the joints
    for j in range(p.getNumJoints(human.human_id)):
        print (p.getJointInfo(human.human_id, j))
    # Set the simulation parameters
    p.setGravity(0,0,-9.81)

    # Set the simulation parameters

    # bed_height, bed_base_height = bed.get_heights(set_on_ground=True)
    smpl_path = os.path.join(os.getcwd(), "examples/data/smpl_bp_ros_smpl_8.pkl")
    smpl_data = load_smpl(smpl_path)
    set_joint_angles(human.human_id, smpl_data)

    # Set the camera view
    cameraDistance = 3
    cameraYaw = 0
    cameraPitch = -30
    cameraTargetPosition = [0,0,1]
    p.resetDebugVisualizerCamera(cameraDistance, cameraYaw, cameraPitch, cameraTargetPosition)
    # print all the joints
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

    while True:
        p.stepSimulation()
        check_collision(human.human_id, human.human_id)
    # Disconnect from the simulation
    p.disconnect()
