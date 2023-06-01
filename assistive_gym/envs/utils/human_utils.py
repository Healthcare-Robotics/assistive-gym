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
        print(f"Link {link_id_A} of body {body_id} collided with link {link_id_B} of body {other_body_id}.")
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

# def position_robot_toc(self, task, arms, start_pos_orient, target_pos_orients, human, base_euler_orient=np.zeros(3),
#                        max_ik_iterations=200, max_ik_random_restarts=1, randomize_limits=False, attempts=100,
#                        jlwki_restarts=1, step_sim=False, check_env_collisions=False, right_side=True,
#                        random_rotation=30, random_position=0.5):
#     # Continually randomize the robot base position and orientation
#     # Select best base pose according to number of goals reached and manipulability
#     if type(arms) == str:
#         arms = [arms]
#         start_pos_orient = [start_pos_orient]
#         target_pos_orients = [target_pos_orients]
#     a = 6  # Order of the robot space. 6D (3D position, 3D orientation)
#     best_position = None
#     best_orientation = None
#     best_num_goals_reached = None
#     best_manipulability = None
#     best_start_joint_poses = [None] * len(arms)
#     iteration = 0
#     # Save human joint states for later restoring
#     human_angles = human.get_joint_angles(human.controllable_joint_indices)
#     while iteration < attempts or best_position is None:
#         iteration += 1
#         # Randomize base position and orientation
#         random_pos = np.array(
#             [self.np_random.uniform(-random_position if right_side else 0, 0 if right_side else random_position),
#              self.np_random.uniform(-random_position, random_position), 0])
#         random_orientation = self.get_quaternion([base_euler_orient[0], base_euler_orient[1],
#                                                   base_euler_orient[2] + np.deg2rad(
#                                                       self.np_random.uniform(-random_rotation, random_rotation))])
#         self.set_base_pos_orient(np.array([-0.85, -0.4, 0]) + self.toc_base_pos_offset[task] + random_pos,
#                                  random_orientation)
#         # Reset all robot joints to their defaults
#         self.reset_joints()
#         # Reset human joints in case they got perturbed by previous iterations
#         human.set_joint_angles(human.controllable_joint_indices, human_angles)
#         num_goals_reached = 0
#         manipulability = 0.0
#         start_joint_poses = [None] * len(arms)
#         # Check if the robot can reach all target locations from this base pose
#         for i, arm in enumerate(arms):
#             right = (arm == 'right')
#             ee = self.right_end_effector if right else self.left_end_effector
#             ik_indices = self.right_arm_ik_indices if right else self.left_arm_ik_indices
#             lower_limits = self.right_arm_lower_limits if right else self.left_arm_lower_limits
#             upper_limits = self.right_arm_upper_limits if right else self.left_arm_upper_limits
#             for j, (target_pos, target_orient) in enumerate(start_pos_orient[i] + target_pos_orients[i]):
#                 best_jlwki = None
#                 best_joint_positions = None
#                 for k in range(jlwki_restarts):
#                     # Reset state in case anything was perturbed from the last iteration
#                     human.set_joint_angles(human.controllable_joint_indices, human_angles)
#                     # Find IK solution
#                     success, joint_positions_q_star = self.ik_random_restarts(right, target_pos, target_orient,
#                                                                               max_iterations=max_ik_iterations,
#                                                                               max_ik_random_restarts=max_ik_random_restarts,
#                                                                               success_threshold=0.03,
#                                                                               step_sim=step_sim,
#                                                                               check_env_collisions=check_env_collisions,
#                                                                               randomize_limits=randomize_limits)
#                     if not success:
#                         continue
#                     _, motor_positions, _, _ = self.get_motor_joint_states()
#                     joint_velocities = [0.0] * len(motor_positions)
#                     joint_accelerations = [0.0] * len(motor_positions)
#                     center_of_mass = \
#                         p.getLinkState(self.body, ee, computeLinkVelocity=True, computeForwardKinematics=True,
#                                        physicsClientId=self.id)[2]
#                     J_linear, J_angular = p.calculateJacobian(self.body, ee, localPosition=center_of_mass,
#                                                               objPositions=motor_positions,
#                                                               objVelocities=joint_velocities,
#                                                               objAccelerations=joint_accelerations,
#                                                               physicsClientId=self.id)
#                     J_linear = np.array(J_linear)[:, ik_indices]
#                     J_angular = np.array(J_angular)[:, ik_indices]
#                     J = np.concatenate([J_linear, J_angular], axis=0)
#                     # Joint-limited-weighting
#                     joint_limit_weight = self.joint_limited_weighting(joint_positions_q_star, lower_limits,
#                                                                       upper_limits)
#                     # Joint-limited-weighted kinematic isotropy (JLWKI)
#                     det = max(np.linalg.det(np.matmul(np.matmul(J, joint_limit_weight), J.T)), 0)
#                     jlwki = np.power(det, 1.0 / a) / (
#                             np.trace(np.matmul(np.matmul(J, joint_limit_weight), J.T)) / a)
#                     if best_jlwki is None or jlwki > best_jlwki:
#                         best_jlwki = jlwki
#                         best_joint_positions = joint_positions_q_star
#                 if best_jlwki is not None:
#                     num_goals_reached += 1
#                     manipulability += best_jlwki
#                     if j == 0:
#                         start_joint_poses[i] = best_joint_positions
#                 if j < len(start_pos_orient[i]) and best_jlwki is None:
#                     # Not able to find an IK solution to a start goal. We cannot use this base pose
#                     num_goals_reached = -1
#                     manipulability = None
#                     break
#             if num_goals_reached == -1:
#                 break
#
#         if num_goals_reached > 0:
#             if best_position is None or num_goals_reached > best_num_goals_reached or (
#                     num_goals_reached == best_num_goals_reached and manipulability > best_manipulability):
#                 best_position = random_pos
#                 best_orientation = random_orientation
#                 best_num_goals_reached = num_goals_reached
#                 best_manipulability = manipulability
#                 best_start_joint_poses = start_joint_poses
#
#         human.set_joint_angles(human.controllable_joint_indices, human_angles)
#
#     # Reset state in case anything was perturbed
#     human.set_joint_angles(human.controllable_joint_indices, human_angles)
#
#     # Set the robot base position/orientation and joint angles based on the best pose found
#     p.resetBasePositionAndOrientation(self.body, np.array([-0.85, -0.4, 0]) + np.array(
#         self.toc_base_pos_offset[task]) + best_position, best_orientation, physicsClientId=self.id)
#     for i, arm in enumerate(arms):
#         set_joint_angles(self.right_arm_joint_indices if arm == 'right' else self.left_arm_joint_indices,
#                               best_start_joint_poses[i])
#     return best_position, best_orientation, best_start_joint_poses
#
#
# def ik_random_restarts(self, right, target_pos, target_orient, max_iterations=1000, max_ik_random_restarts=40,
#                        success_threshold=0.03, step_sim=False, check_env_collisions=False, randomize_limits=True,
#                        collision_objects=[]):
#     if target_orient is not None and len(target_orient) < 4:
#         target_orient = self.get_quaternion(target_orient)
#     orient_orig = target_orient
#     best_ik_angles = None
#     best_ik_distance = 0
#     for r in range(max_ik_random_restarts):
#         target_joint_angles = self.ik(self.right_end_effector if right else self.left_end_effector, target_pos,
#                                       target_orient,
#                                       ik_indices=self.right_arm_ik_indices if right else self.left_arm_ik_indices,
#                                       max_iterations=max_iterations, half_range=self.half_range,
#                                       randomize_limits=(randomize_limits and r >= 10))
#         set_joint_angles(self.right_arm_joint_indices if right else self.left_arm_joint_indices,
#                               target_joint_angles)
#         gripper_pos, gripper_orient = self.get_pos_orient(self.right_end_effector if right else self.left_end_effector)
#         if np.linalg.norm(target_pos - np.array(gripper_pos)) < success_threshold and (
#                 target_orient is None or np.linalg.norm(
#                 target_orient - np.array(gripper_orient)) < success_threshold or np.isclose(
#                 np.linalg.norm(target_orient - np.array(gripper_orient)), 2, atol=success_threshold)):
#             # if step_sim:
#             #     # TODO: Replace this with getClosestPoints, see: https://github.gatech.edu/zerickson3/assistive-gym/blob/vr3/assistive_gym/envs/feeding.py#L156
#             #     for _ in range(5):
#             #         p.stepSimulation(physicsClientId=self.id)
#             #     # if len(p.getContactPoints(bodyA=self.body, bodyB=self.body, physicsClientId=self.id)) > 0 and orient_orig is not None:
#             #     #     # The robot's arm is in contact with itself. Continually randomize end effector orientation until a solution is found
#             #     #     target_orient = self.get_quaternion(self.get_euler(orient_orig) + np.deg2rad(self.np_random.uniform(-45, 45, size=3)))
#             # if check_env_collisions:
#             #     for _ in range(25):
#             #         p.stepSimulation(physicsClientId=self.id)
#
#             # Check if the robot is colliding with objects in the environment. If so, then continue sampling.
#             if len(collision_objects) > 0:
#                 dists_list = []
#                 for obj in collision_objects:
#                     dists_list.append(self.get_closest_points(obj, distance=0)[-1])
#                 if not all(not d for d in dists_list):
#                     continue
#             gripper_pos, gripper_orient = self.get_pos_orient(
#                 self.right_end_effector if right else self.left_end_effector)
#             if np.linalg.norm(target_pos - np.array(gripper_pos)) < success_threshold and (
#                     target_orient is None or np.linalg.norm(
#                     target_orient - np.array(gripper_orient)) < success_threshold or np.isclose(
#                     np.linalg.norm(target_orient - np.array(gripper_orient)), 2, atol=success_threshold)):
#                 set_joint_angles(self.right_arm_joint_indices if right else self.left_arm_joint_indices,
#                                       target_joint_angles)
#                 return True, np.array(target_joint_angles)
#         if best_ik_angles is None or np.linalg.norm(target_pos - np.array(gripper_pos)) < best_ik_distance:
#             best_ik_angles = target_joint_angles
#             best_ik_distance = np.linalg.norm(target_pos - np.array(gripper_pos))
#     set_joint_angles(self.right_arm_joint_indices if right else self.left_arm_joint_indices, best_ik_angles)
#     return False, np.array(best_ik_angles)
