import os
import pickle
from typing import List

import numpy as np
import pybullet as p
import torch
from pytorch3d import transforms as t3d
from trimesh import Trimesh

from assistive_gym.envs.utils.human_urdf_dict import HumanUrdfDict
from assistive_gym.envs.utils.smpl_geom import generate_geom
from assistive_gym.envs.utils.urdf_editor import UrdfEditor, UrdfJoint, UrdfLink
from experimental.urdf_name_resolver import get_urdf_filepath, get_urdf_mesh_folderpath


"""
Collection of helper functions to generate human URDF file from SMPL model
"""

################################################## SMPL setting #######################################################
class SMPLData:
    def __init__(self, body_pose, betas, global_orient, transl = None):  # TODO: add static typing
        self.body_pose = body_pose
        self.betas = betas
        self.global_orient = global_orient
        self.transl = transl

def get_template_smpl_path(gender):
    if not gender:
        return os.path.join(os.getcwd(), "examples/data/SMPL_NEUTRAL.pkl")
    else:
        return os.path.join(os.getcwd(), "examples/data/SMPL_FEMALE.pkl") if gender == 'female' else os.path.join(os.getcwd(), "examples/data/SMPL_MALE.pkl")

def load_smpl(filepath) -> SMPLData:
    with open(filepath, "rb") as handle:
        data = pickle.load(handle)

        if len(data["body_pose"]) == 69: # we need to extends the dimension of the smpl_data['pose'] from 69 to 72 to match the urdf
            data["body_pose"] = np.concatenate((np.array([0.0, 0.0, 0.0]), data["body_pose"]))
    smpl_data: SMPLData = SMPLData(data["body_pose"], data["betas"], data["global_orient"], data["transl"])
    return smpl_data


################################################## Static setting #####################################################
human_dict = HumanUrdfDict()
# BM_FRACTION = body_mass/70.45
BM_FRACTION = 1.0

JOINT_SETTING = {
    "pelvis": {
        "joint_limits":[[-60, 60], [-60, 60], [-60, 60]],  # deg
        "joint_damping": 0.0,
        "joint_stiffness": 0.0,
    },
    "left_hip": {
        "joint_limits": [[-90.0, 17.8], [-33.7, 32.6], [-30.5, 38.6]],
        "joint_damping": [15 * 10.0] * 3,
        "joint_stiffness": [10.0] * 3
    },
    "left_knee": {
        "joint_limits": [[-1.3, 139.9], [0, 0], [0, 0]],  # TODO: check
        "joint_damping": 0.0,
        "joint_stiffness": 0.0,
    },
    "left_ankle": {
        "joint_limits": [[-30, 30], [-30, 30], [-30, 30]],
        "joint_stiffness": 0.0,
    },
    "left_foot": {
        "joint_limits": [[-30, 30], [-30, 30], [-30, 30]],  # TODO: no value from Henry, set same as ankle
        "joint_damping": 0.0,
        "joint_stiffness": 0.0,
    },
    "right_hip":
        {
            "joint_limits": [[-90.0, 17.8], [-32.6, 33.7], [-38.6, 30.5]],
            "joint_damping": [15 * 10.0] * 3,
            "joint_stiffness": [10.0] * 3
        },
    "right_knee":
        {
            "joint_limits": [[-1.3, 139.9], [0, 0], [0, 0]],  # TODO: check
            "joint_damping": 0.0,
            "joint_stiffness": 0.0,
        },
    "right_ankle":
        {
            "joint_limits": [[-30, 30], [-30, 30], [-30, 30]],
            "joint_damping": 0.0,
            "joint_stiffness": 0.0,
        },
    "right_foot":
        {
            "joint_limits": [[-30, 30], [-30, 30], [-30, 30]],
            "joint_damping": 0.0,
            "joint_stiffness": 0.0,
        },
    "spine_2":
        {
            "joint_limits": [[-30, 30], [-30, 30], [-30, 30]],
            "joint_damping": 0.0,
            "joint_stiffness": 0.0,
        },
    "spine_3":
        {
            "joint_limits": [[-30, 30], [-30, 30], [-30, 30]],
            "joint_damping": 0.0,
            "joint_stiffness": 0.0,
        },
    "spine_4":
        {
            "joint_limits": [[-30, 30], [-30, 30], [-30, 30]],
            "joint_damping": 0.0,
            "joint_stiffness": 0.0,
        },
    "neck":
        {
            "joint_limits": [[-30, 30], [-5, 5], [-5, 5]],
            "joint_damping": 0.0,
            "joint_stiffness": 0.0,
        },
    "head":
        {
            "joint_limits": [[-30, 30], [-5, 5], [-5, 5]],
            "joint_damping": 0.0,
            "joint_stiffness": 0.0,
        },
    "left_clavicle":
        {
            "joint_limits": [[-88.9 / 3.0, 81.4 / 3.0], [-140.7 / 3.0, 43.7 / 3.0], [-90.0 / 3.0, 80.4 / 3.0]],
            "joint_damping": 0.0,
            "joint_stiffness": 0.0,
        },
    "left_shoulder":
        {
            "joint_limits": [[-88.9 * 2 / 3.0, 81.4 * 2 / 3.0], [-140.7 * 2 / 3.0, 43.7 * 2 / 3.0],
                             [-90.0 * 2 / 3.0, 80.4 * 2 / 3.0]],
            "joint_damping": 0.0,
            "joint_stiffness": 0.0,

        },
    "left_elbow":
        {
            "joint_limits": [[0, 0], [-147.3, 2.8], [0, 0]],
            "joint_damping": 0.0,
            "joint_stiffness": 0.0,
        },
    "left_lowarm":
        {
            "joint_limits": [[-30, 30], [-30, 30], [-30, 30]],
            "joint_damping": 0.0,
            "joint_stiffness": 0.0,
        },
    "left_hand":
        {
            "joint_limits": [[-30, 30], [-30, 30], [-30, 30]],
            "joint_damping": 0.0,
            "joint_stiffness": 0.0,
        },
    "right_clavicle":
        {
            "joint_limits": [[-88.9 / 3.0, 81.4 / 3.0], [-43.7 / 3.0, 140.7 / 3.0], [-90.0 / 3.0, 80.4 / 3.0]],
            "joint_damping": 0.0,
            "joint_stiffness": 0.0,
        },
    "right_shoulder":
        {
            "joint_limits": [[-88.9 * 2 / 3.0, 81.4 * 2 / 3.0], [-43.7 * 2 / 3.0, 140.7 * 2 / 3.0],
                             [-90.0 * 2 / 3.0, 80.4 * 2 / 3.0]],
            "joint_damping": 0.0,
            "joint_stiffness": 0.0,
        },
    "right_elbow":
        {
            "joint_limits": [[0, 0], [-2.8, 147.3], [0, 0]],
            "joint_damping": 0.0,
            "joint_stiffness": 0.0,
        },
    "right_lowarm":
        {
            "joint_limits": [[-30, 30], [-30, 30], [-30, 30]],
            "joint_damping": 0.0,
            "joint_stiffness": 0.0,
        },
    "right_hand":
        {
            "joint_limits": [[-30, 30], [-30, 30], [-30, 30]],  # TODO: no limits from Henry
            "joint_damping": 0.0,
            "joint_stiffness": 0.0,
        },
}


#################################### URDF Generation ##################################################################
def generate_human_mesh(physic_id, gender, ref_urdf_path, out_urdf_folder, smpl_path):
    smpl_data = load_smpl(smpl_path)


    out_geom_folder = get_urdf_mesh_folderpath(out_urdf_folder)
    template_smpl_path = get_template_smpl_path(gender)
    hull_dict, joint_pos_dict, _ = generate_geom(template_smpl_path, smpl_data, out_geom_folder)
    out_urdf_file = get_urdf_filepath(out_urdf_folder)
    # now trying to scale the urdf file
    body = p.loadURDF(ref_urdf_path, [0, 0, 0],
                      flags=p.URDF_USE_SELF_COLLISION,
                      useFixedBase=False)
    generate_urdf(body, physic_id, hull_dict, joint_pos_dict, out_urdf_file)


def generate_urdf(human_id, physic_client_id, hull_dict, pos_dict, out_path):
    editor = UrdfEditor()
    editor.initializeFromBulletBody(human_id, physic_client_id)  # load all properties to editor

    config_links(editor.urdfLinks, hull_dict)
    config_joints(editor.urdfJoints, pos_dict)

    editor.saveUrdf(out_path, saveVisuals=True)


#################################### Angle Conversion ##################################################################
def convert_aa_to_euler_quat(aa, seq="XYZ"):
    aa = np.array(aa)
    mat = t3d.axis_angle_to_matrix(torch.from_numpy(aa))
    # print ("mat", mat)
    quat = t3d.matrix_to_quaternion(mat)
    euler = t3d.matrix_to_euler_angles(mat, seq)
    return euler, quat


def euler_convert_np(q, from_seq='XYZ', to_seq='XYZ'):
    r"""
    Convert euler angles into different axis orders. (numpy, single/batch)

    :param q: An ndarray of euler angles (radians) in from_seq order. Shape [3] or [N, 3].
    :param from_seq: The source(input) axis order. See scipy for details.
    :param to_seq: The target(output) axis order. See scipy for details.
    :return: An ndarray with the same size but in to_seq order.
    """
    from scipy.spatial.transform import Rotation
    return Rotation.from_euler(from_seq, q).as_euler(to_seq)


def deg_to_rad(deg):
    return deg * np.pi / 180.0

#################################### Joint & Link Properties ###########################################################
def mul_tuple(t, multiplier):
    return tuple(multiplier * elem for elem in t)


def scale_body_part(human_id, physic_client_id, scale=10):
    editor = UrdfEditor()
    editor.initializeFromBulletBody(human_id, physic_client_id)  # load all properties to editor
    # scaling the robot
    for link in editor.urdfLinks:
        for v in link.urdf_visual_shapes:
            if v.geom_type == p.GEOM_BOX:
                v.geom_extents = mul_tuple(v.geom_extents, 10)
            if v.geom_type == p.GEOM_SPHERE:
                v.geom_radius *= 10
            if v.geom_type == p.GEOM_CAPSULE:
                v.geom_radius *= 10
                v.geom_length *= 10
            v.origin_xyz = mul_tuple(v.origin_xyz, 10)

        for c in link.urdf_collision_shapes:
            if c.geom_type == p.GEOM_BOX:
                c.geom_extents = mul_tuple(c.geom_extents, 10)
            if c.geom_type == p.GEOM_SPHERE:
                c.geom_radius *= 10
            if c.geom_type == p.GEOM_CAPSULE:
                c.geom_radius *= 10
                c.geom_length *= 10
            c.origin_xyz = mul_tuple(c.origin_xyz, 10)
    for j in editor.urdfJoints:
        j.joint_origin_xyz = mul_tuple(j.joint_origin_xyz, 10)
    editor.saveUrdf("test10.urdf", True)


def get_bodypart_name(urdf_name):
    """
    :param urdf_name: joint or link name in urdf
    :return:
    """
    # TODO: check for _
    last_underscore_idx = urdf_name.rfind("_")
    name = urdf_name[:last_underscore_idx]
    suffix = urdf_name[last_underscore_idx + 1:]
    return name, suffix


def simple_inertia(inertia):
    """
    :param inertia: np array 3x3
    :return: inertia tensor in tuple (ixx, iyy, izz)
    """
    return tuple([inertia[0][0], inertia[1][1], inertia[2][2]])


def config_joints(joints: List[UrdfJoint], pos_dict):
    # smpl joint is spherical.
    # due to the limitation of pybullet (no joint limit for spherical joint),
    # we need to decompose it to 3 revolute joints (rx, ry, rz) and 1 fixed joint (rzdammy) rx -> ry -> rz -> rzdammy
    # the human body part is a link (_limb) that attached to the fixed joint
    # we need to move the joint origin to correct position as follow
    # - set rx joint origin = urdf_joint pos - parent urdf joint pos
    # - set ry, rz, rxdammy origin = 0 (superimpose on the rx joint)

    for j in joints:
        name, suffix = get_bodypart_name(j.joint_name)
        if name in human_dict.urdf_to_smpl_dict.keys():
            smpl_name = human_dict.urdf_to_smpl_dict[name]
            if suffix == 'rx':
                pos = pos_dict[smpl_name]
                parent = human_dict.joint_to_parent_joint_dict[name]
                pos_parent = pos_dict[human_dict.urdf_to_smpl_dict[parent]]
                xyz = pos - pos_parent
            else:
                xyz = (0, 0, 0)
            j.joint_origin_xyz = xyz
        else:
            j.joint_origin_xyz = (0, 0, 0)
        # set joint limit
        if j.joint_type == p.JOINT_REVOLUTE:
            joint_limit = JOINT_SETTING[name]['joint_limits']
            if suffix == 'rx':
                idx = 0
            elif suffix == 'ry':
                idx = 1
            elif suffix == 'rz':
                idx = 2

            j.joint_lower_limit = deg_to_rad(joint_limit[idx][0])
            j.joint_upper_limit = deg_to_rad(joint_limit[idx][1])


def config_links(links: List[UrdfLink], hull_dict):
    xyz = (0, 0, 0)  # default
    for link in links:
        name, suffix = get_bodypart_name(link.link_name)
        if name in human_dict.urdf_to_smpl_dict.keys():
            for v in link.urdf_visual_shapes:
                v.origin_xyz = xyz
            for c in link.urdf_collision_shapes:
                c.origin_xyz = xyz

            smpl_name = human_dict.urdf_to_smpl_dict[name]
            hull: Trimesh = hull_dict[smpl_name].hull
            # inertia
            inertia = link.urdf_inertial
            inertia.origin_xyz = xyz
            if suffix == 'limb':
                inertia.mass = hull.mass
                # due to limitation of pybullet urdfeditor, we can only set ixx, iyy, izz
                inertia.inertia_xxyyzz = simple_inertia(hull.moment_inertia)
            else:
                # fake link
                inertia.mass = 0.0
                inertia.inertia_xxyyzz = tuple([0, 0, 0])
            link.urdf_inertial = inertia
