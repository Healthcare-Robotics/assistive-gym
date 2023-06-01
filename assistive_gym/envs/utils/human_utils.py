import colorsys

import numpy as np
import pybullet as p

from assistive_gym.envs.utils.human_urdf_dict import HumanUrdfDict
from assistive_gym.envs.utils.smpl_dict import SMPLDict
from assistive_gym.envs.utils.urdf_utils import convert_aa_to_euler_quat, SMPLData

"""
Collection of helper functions to change human properties by PyBullet API 
"""

smpl_dict = SMPLDict()
human_dict = HumanUrdfDict()


def change_color(human_id, color):
    """
    Change the color of a robot.
    :param color: Vector4 for rgba.
    """
    for j in range(p.getNumJoints(human_id)):
        p.changeVisualShape(human_id, j, rgbaColor=color, specularColor=[0.1, 0.1, 0.1])


def get_skin_color(self):
    hsv = list(colorsys.rgb_to_hsv(0.8, 0.6, 0.4))
    hsv[-1] = np.random.uniform(0.4, 0.8)
    skin_color = list(colorsys.hsv_to_rgb(*hsv)) + [1]
    return skin_color


def set_global_angle(human_id, pose):
    _, quat = convert_aa_to_euler_quat(pose[smpl_dict.get_pose_ids("Pelvis")])
    # quat = np.array(p.getQuaternionFromEuler(np.array(euler)))
    p.resetBasePositionAndOrientation(human_id, [0, 0, 1], quat)


def set_joint_angle(human_id, pose, smpl_joint_name, robot_joint_name):
    smpl_angles, _ = convert_aa_to_euler_quat(pose[smpl_dict.get_pose_ids(smpl_joint_name)])

    # smpl_angles = pose[smpl_dict.get_pose_ids(smpl_joint_name)]
    robot_joints = human_dict.get_joint_ids(robot_joint_name)
    for i in range(0, 3):
        p.resetJointState(human_id, robot_joints[i], smpl_angles[i])


def set_joint_angles(human_id, smpl_data: SMPLData):
    print("global_orient", smpl_data.global_orient)
    print("pelvis", smpl_data.body_pose[0:3])
    pose = smpl_data.body_pose

    set_global_angle(human_id, pose)

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
        pass
        # p.changeDynamics(human_id, link_id,
        #                  lateralFriction=1.0,
        #                  spinningFriction=0.1,
        #                  rollingFriction=0.1,
        #                  restitution=0.9,
        #                  linearDamping=0.01,
        #                  angularDamping=0.01,
        #                  contactStiffness=1e6,
        #                  contactDamping=1e3)

        # p.changeDynamics(human_id, link_id,
        #                  lateralFriction=0.2,
        #                  spinningFriction=0.2,
        #                  rollingFriction=0.2,
        #                  restitution=0.8,
        #                  linearDamping=0.01,
        #                  angularDamping=0.01,
        #                  contactStiffness=1e6,
        #                  contactDamping=1e3)


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
    # right arm vs the rest
    fake_limb_ids = []
    real_limb_ids = []
    joint_to_ignore_ids = []
    for name in joint_names:
        fake_limb_ids.extend(human_dict.get_joint_ids(name))
        real_limb_ids.append(human_dict.get_dammy_joint_id(name))
    for name in joint_to_ignore:
        if name == "pelvis":
            joint_to_ignore_ids.append(human_dict.get_joint_id(name))
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


def set_self_collisions(human_id, physic_client_id):
    num_joints = p.getNumJoints(human_id, physicsClientId=physic_client_id)

    # disable all self collision
    for i in range(0, num_joints):
        for j in range(0, num_joints):
            p.setCollisionFilterPair(human_id, human_id, i, j, 0, physicsClientId=physic_client_id)

    # # only enable self collision for arms and legs with the rest of the body
    set_self_collision(human_id, physic_client_id, num_joints, human_dict.joint_chain_dict["right_arm"],
                       human_dict.joint_collision_ignore_dict["right_arm"])
    set_self_collision(human_id, physic_client_id, num_joints, human_dict.joint_chain_dict["left_arm"],
                       human_dict.joint_collision_ignore_dict["left_arm"])
    set_self_collision(human_id, physic_client_id, num_joints, human_dict.joint_chain_dict["right_leg"],
                       human_dict.joint_collision_ignore_dict["right_leg"])
    set_self_collision(human_id, physic_client_id, num_joints, human_dict.joint_chain_dict["left_leg"],
                       human_dict.joint_collision_ignore_dict["left_leg"])
