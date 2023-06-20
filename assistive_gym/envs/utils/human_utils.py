import colorsys

import numpy as np
import pybullet as p
import torch
import pytorch3d.transforms as t3d

from assistive_gym.envs.utils.human_urdf_dict import HumanUrdfDict
from assistive_gym.envs.utils.smpl_dict import SMPLDict
from assistive_gym.envs.utils.urdf_utils import convert_aa_to_euler_quat, SMPLData

"""
Collection of helper functions to change human properties by PyBullet API 
"""

smpl_dict = SMPLDict()
human_dict = HumanUrdfDict()


#######################################  Change human visual properites  ##########################################
def get_skin_color():
    hsv = list(colorsys.rgb_to_hsv(0.8, 0.6, 0.4))
    hsv[-1] = np.random.uniform(0.4, 0.8)
    skin_color = list(colorsys.hsv_to_rgb(*hsv)) + [1]
    return skin_color


def change_color(human_id, color):
    """
    Change the color of a robot.
    :param color: Vector4 for rgba.
    """
    for j in range(p.getNumJoints(human_id)):
        p.changeVisualShape(human_id, j, rgbaColor=color, specularColor=[0.1, 0.1, 0.1])


def set_global_orientation(human_id, axis_angle, pos):
    euler, quat = convert_aa_to_euler_quat(axis_angle)

    # due to Henry's implementation, we need flip current angle by 180 degree around x axis
    q_r = torch.tensor([0.0, 1.0, 0.0, 0.0])
    quat = t3d.quaternion_multiply(q_r, quat)

    # convert [w x y z] to [x y z w]
    q = np.array(list(quat[1:]) + [quat[0]])
    p.resetBasePositionAndOrientation(human_id, pos, q)


def set_joint_angle(human_id, pose, smpl_joint_name, robot_joint_name):
    smpl_angles, _ = convert_aa_to_euler_quat(pose[smpl_dict.get_pose_ids(smpl_joint_name)])
    print ("joint name: ", robot_joint_name, np.array(smpl_angles)*180.0/np.pi)
    # smpl_angles = pose[smpl_dict.get_pose_ids(smpl_joint_name)]
    robot_joints = human_dict.get_joint_ids(robot_joint_name)
    for i in range(0, 3):
        p.resetJointState(human_id, robot_joints[i], smpl_angles[i])


def set_joint_angles(human_id, pose):
    r"""
      Set the joint angles of the robot.
      :param pose: 75 dimensional vector of joint angles. (0:3 - root, 3:75 - joint angles)
      :return:
      """

    set_joint_angle(human_id, pose, "Spine1", "spine_2")
    set_joint_angle(human_id, pose, "Spine2", "spine_3")
    set_joint_angle(human_id, pose, "Spine3", "spine_4")

    set_joint_angle(human_id, pose, "L_Hip", "left_hip")
    set_joint_angle(human_id, pose, "L_Knee", "left_knee")
    set_joint_angle(human_id, pose, "L_Ankle", "left_ankle")
    set_joint_angle(human_id, pose, "L_Foot", "left_foot")

    set_joint_angle(human_id, pose, "R_Hip", "right_hip")
    set_joint_angle(human_id, pose, "R_Knee", "right_knee")
    set_joint_angle(human_id, pose, "R_Ankle", "right_ankle")
    set_joint_angle(human_id, pose, "R_Foot", "right_foot")

    set_joint_angle(human_id, pose, "R_Collar", "right_clavicle")
    set_joint_angle(human_id, pose, "R_Shoulder", "right_shoulder")
    set_joint_angle(human_id, pose, "R_Elbow", "right_elbow")
    set_joint_angle(human_id, pose, "R_Wrist", "right_lowarm")
    set_joint_angle(human_id, pose, "R_Hand", "right_hand")

    set_joint_angle(human_id, pose, "L_Collar", "left_clavicle")
    set_joint_angle(human_id, pose, "L_Shoulder", "left_shoulder")
    set_joint_angle(human_id, pose, "L_Elbow", "left_elbow")
    set_joint_angle(human_id, pose, "L_Wrist", "left_lowarm")
    set_joint_angle(human_id, pose, "L_Hand", "left_hand")

    set_joint_angle(human_id, pose, "Neck", "neck")
    set_joint_angle(human_id, pose, "Head", "head")

# TODO: review the parameters
def change_dynamic_properties(human_id, link_ids):
    for link_id in link_ids:
        p.changeDynamics(human_id, link_id,
                         lateralFriction=10.0,
                         spinningFriction=0.1,
                         rollingFriction=0.1,
                         restitution=0.9,
                         linearDamping=0.01,
                         angularDamping=0.01,
                         contactStiffness=1e3,
                         # contact stiffness need to be large otherwise the body will penetrate the ground
                         contactDamping=1e6)  # contact damping need to be much larger than contact stiffness so that no bounciness


def check_collision(body_id, other_body_id):
    """
    Check if two bodies are in collision and print out the link ids of the colliding links
    Can be used to check self collision if body_id == other_body_id
    :param body_id:
    :param other_body_id:
    :return:
    """
    contact_points = p.getContactPoints(bodyA=body_id, bodyB=other_body_id)
    contact_pais = set()
    for contact in contact_points:
        link_id_A = contact[3]
        link_id_B = contact[4]
        # print(f"Link {link_id_A} of body {body_id} collided with link {link_id_B} of body {other_body_id}.")
        contact_pais.add((link_id_A, link_id_B))
    return contact_pais


def set_self_collision(human_id, physic_client_id, num_joints, joint_names, joint_to_ignore=[]):
    """
    Set self collision for joints in joint_names with the rest of the body
    Ignore collision with the joints in joint_to_ignore
    :param human_id:
    :param physic_client_id:
    :param num_joints: number of joints
    :param joint_names: list of joint names
    :return:
    """
    fake_limb_ids = []
    real_limb_ids = []
    joint_to_ignore_ids = []
    for name in joint_names:
        fake_limb_ids.extend(human_dict.get_joint_ids(name))
        real_limb_ids.append(human_dict.get_dammy_joint_id(name))
    for name in joint_to_ignore:
        if name == "pelvis":
            joint_to_ignore_ids.append(human_dict.get_fixed_joint_id(name))
        else:
            joint_to_ignore_ids.extend(human_dict.get_joint_ids(name))
            joint_to_ignore_ids.append(human_dict.get_dammy_joint_id(name))

    # merge 3 lists
    joint_chain = fake_limb_ids + real_limb_ids + joint_to_ignore_ids
    # enable collision between the fake limbs and the rest of the body (except link in fake limbs and real limbs list)
    # for i in fake_limb_ids:
    #     for j in range(0, num_joints):
    #         if j not in joint_chain:
    #             p.setCollisionFilterPair(human_id, human_id, i, j, is_collision, physicsClientId=physic_client_id)
    # enable collision between the real limbs and the rest of the body (except link in fake limbs and real limbs list)
    for i in real_limb_ids:
        for j in range(0, num_joints):
            if j not in joint_chain:
                p.setCollisionFilterPair(human_id, human_id, i, j, 1, physicsClientId=physic_client_id)


def set_self_collision2(human_id, physic_client_id, joint_chain, joint_to_ignore=[]):
    all_real_limb_ids = human_dict.limb_index_dict.keys()
    for joint_name in joint_chain:
        # add parent
        parent = human_dict.joint_to_parent_joint_dict[joint_name]
        parent_limb_id = human_dict.get_dammy_joint_id(parent)

        # add child
        if not joint_name in ['left_foot', 'right_foot', 'left_hand', 'right_hand']: # no child
            child = human_dict.joint_to_child_joint_dict[joint_name]
            child_limb_id = human_dict.get_dammy_joint_id(child)

        limb_id = human_dict.get_dammy_joint_id(joint_name)

        ignore_ids = [parent_limb_id, child_limb_id, limb_id]
        for j_name in joint_to_ignore:
            if j_name == "pelvis":
                ignore_ids.append(human_dict.get_fixed_joint_id(j_name))
            # else:
            #     if joint_name ==
            #     ignore_ids.append(human_dict.get_dammy_joint_id(j_name))

        # print (f"ignore_ids: {ignore_ids}")
        for j in all_real_limb_ids:
            if j not in ignore_ids:
                # print (f"enable collision between {j} and {limb_id}")
                p.setCollisionFilterPair(human_id, human_id, limb_id, j, 1, physicsClientId=physic_client_id)


def disable_self_collisions(human_id, num_joints, physic_client_id):
    # disable all self collision
    for i in range(0, num_joints):
        for j in range(0, num_joints):
            p.setCollisionFilterPair(human_id, human_id, i, j, 0, physicsClientId=physic_client_id)


def set_self_collisions(human_id, physic_client_id):
    num_joints = p.getNumJoints(human_id, physicsClientId=physic_client_id)

    disable_self_collisions(human_id, num_joints, physic_client_id)

    # # only enable self collision for arms and legs with the rest of the body
    set_self_collision2(human_id, physic_client_id, human_dict.joint_chain_dict["right_hand"],
                        human_dict.joint_collision_ignore_dict["right_hand"])
    set_self_collision2(human_id, physic_client_id, human_dict.joint_chain_dict["left_hand"],
                        human_dict.joint_collision_ignore_dict["left_hand"])
    set_self_collision2(human_id, physic_client_id, human_dict.joint_chain_dict["right_foot"],
                        human_dict.joint_collision_ignore_dict["right_foot"])
    set_self_collision2(human_id, physic_client_id, human_dict.joint_chain_dict["left_foot"],
                        human_dict.joint_collision_ignore_dict["left_foot"])
