# TODO: clean up this file
import pickle

import numpy as np
import pybullet as p
import pybullet_data

from pytorch3d import transforms as t3d
import torch

from assistive_gym.envs.utils.human_urdf_dict import HumanUrdfDict
from assistive_gym.envs.utils.smpl_dict import SMPLDict

def mul_tuple(t, multiplier):

    return tuple(multiplier * elem for elem in t)

physicsClient = p.connect(p.GUI)

# Load the URDF file
p.setAdditionalSearchPath(pybullet_data.getDataPath())
planeId = p.loadURDF("assistive_gym/envs/assets/plane/plane.urdf", [0,0,0])
# robotId = p.loadURDF("assistive_gym/envs/assets/human/human_pip_original.urdf", [0, 0, 0], [0, 0, 0,1])
# robotId = p.loadURDF("test_mesh.urdf", [0, 0, 0], [0, 0, 0,1])
caneId = p.loadURDF("assistive_gym/envs/assets/cane/cane.urdf", [0, 0, 0], [0, 0, 0,1])

# num_joints = p.getNumJoints(robotId)
# joint_states = p.getJointStates(robotId, range(num_joints))
# # print all the joints
# print (p.getJointInfo(robotId, 0))
# for j in range(1, num_joints):
#     joint_info = p.getJointInfo(robotId, j)
#
#     print ("parent link: ", p.getLinkState(robotId, joint_info[-1]))
#     print (joint_info)
#     print("------------------")
#     joint_state = joint_states[j]
#     joint_pos = p.getJointInfo(robotId, j)[14]
#     link_index = int(joint_state[1])
#     link_info = p.getLinkState(robotId, link_index)
#     #
#     # link_pos = link_info[0]
#     # debug_joints = [2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35, 39, 42, 45, 48, 51, 54, 57, 60, 63, 66,
#     #                 69, 72, 75, 78, 81, 84, 87, 90, 93]
#     # # debug_joints = [2, 3,4, 5]
#     # if int(joint_info[0]) in debug_joints:
#     #     # p.addUserDebugLine(link_pos, joint_pos, [1, 0, 0], 2)
#     #     # random color
#     #     color = np.random.rand(3)
#     #     p.addUserDebugText(str(joint_info[1]), mul_tuple(joint_pos, -1), color, 2)
#     #     # p.addUserDebugText(str(joint_info[1]), mul_tuple(joint_pos, -1), [0, 1, 0], 2)
# # Set the simulation parameters
p.setGravity(0,0,-9.81)
p.setTimeStep(1./240.)

# Set the camera view
cameraDistance = 3
cameraYaw = 0
cameraPitch = -30
cameraTargetPosition = [0,0,1]
p.resetDebugVisualizerCamera(cameraDistance, cameraYaw, cameraPitch, cameraTargetPosition)
while True:
    p.stepSimulation()

# Disconnect from the simulation
p.disconnect()

