import importlib
import json
import os
import pickle
from datetime import datetime
from typing import Set, Optional

import gym
import numpy as np
import pybullet as p
import yaml
from cma import CMA, CMAEvolutionStrategy
from cmaes import CMA
from deprecation import deprecated
from torch.utils.hipify.hipify_python import bcolors

from assistive_gym.envs.utils.dto import HandoverObject, HandoverObjectConfig, MaximumHumanDynamics, OriginalHumanInfo
from assistive_gym.envs.utils.log_utils import get_logger
from assistive_gym.envs.utils.plot_utils import plot_cmaes_metrics, plot_mean_evolution
from assistive_gym.envs.utils.point_utils import fibonacci_evenly_sampling_range_sphere, eulidean_distance
from experimental.urdf_name_resolver import get_urdf_filepath, get_urdf_folderpath

LOG = get_logger()

COLLISION_PENETRATION_THRESHOLD = {
    "self_collision": {
        "old": 0.015,  # 1.5cm
        "new": 0.005
    },
    "env_collision": {
        "old": 0.005,  # 1.5cm
        "new": 0.005
    }
}

COLLISION_OBJECT_RADIUS = {
    "pill": 0.0,
    "cup": 0.05,
    "cane": 0.1
}

OBJECT_PALM_OFFSET = {
    "pill": 0.05,
    "cup": 0.1,
    "cane": 0.08
}

GRIPPER_Z_ANGLE_LIMIT = {
    "pill": None,
    "cup": [-10, 10],
    "cane": None
}

GRIPPER_BEDSIDE_OFFSET = {
    "pill": None,
    "cup": None,
    "cane": 0.05
}

objectTaskMapping = {
    HandoverObject.PILL: "comfort_taking_medicine",
    HandoverObject.CUP: "comfort_drinking",
    HandoverObject.CANE: "comfort_standing_up"
}

YAML_FILE= os.path.join(os.getcwd(), 'params/param_1509.yaml')
def load_yaml(yaml_file):
    with open(yaml_file) as f:
        params = yaml.safe_load(f)
    # print ("yaml params: ", params)
    return params

PARAMS = load_yaml(YAML_FILE)

def create_point(point, size=0.01, color=[1, 0, 0, 1]):
    sphere = p.createCollisionShape(p.GEOM_SPHERE, radius=size)
    id = p.createMultiBody(baseMass=0,
                           baseCollisionShapeIndex=sphere,
                           basePosition=np.array(point))
    p.setGravity(0, 0, 0)
    return id


def draw_point(point, size=0.01, color=[1, 0, 0, 1]):
    sphere = p.createVisualShape(p.GEOM_SPHERE, radius=size, rgbaColor=color)
    p.createMultiBody(baseMass=0,
                      baseVisualShapeIndex=sphere,
                      basePosition=np.array(point))


def get_task_from_handover_object(object_name):
    if not object_name:
        return None
    object_type = HandoverObject.from_string(object_name)
    task = objectTaskMapping[object_type]
    return task


def find_ee_ik_goal(human, end_effector, handover_obj):
    ee_pos, ee_orient = human.get_ee_pos_orient(end_effector)
    ee_norm_vec = human.get_ee_normal_vector(end_effector)
    target_pos = np.array(ee_pos) + ee_norm_vec * OBJECT_PALM_OFFSET[
        handover_obj]  # need to depends on the size of the object as well
    return ee_pos, target_pos


def find_robot_ik_solution(env, end_effector: str, handover_obj: str, init_robot_setting=None):
    """
    Find robot ik solution with TOC. Place the robot in best base position and orientation.
    :param env:
    :param end_effector: str
    :param human_link_robot_collision: dict(agent, [link1, link2, ...]) to check for collision with robot
    :return:
    """

    human, robot, furniture, tool = env.human, env.robot, env.furniture, env.tool
    if not init_robot_setting:
        robot_base_pos, robot_base_orient, side = find_robot_start_pos_orient(env, end_effector)
    else:
        robot_base_pos, robot_base_orient, side = init_robot_setting.base_pos, init_robot_setting.base_orient, init_robot_setting.robot_side

    ee_pos, target_pos = find_ee_ik_goal(human, end_effector, handover_obj)
    # p.addUserDebugLine(ee_pos, target_pos, [1, 0, 0], 5, 0.1)

    best_position, best_orientation, best_joint_angles = robot.position_robot_toc2(robot_base_pos, side,
                                                                                   [(target_pos, None)],
                                                                                   [(target_pos, None)], human,
                                                                                   base_euler_orient=robot_base_orient,
                                                                                   attempts=5,
                                                                                   random_position=0.3,
                                                                                   max_ik_iterations=50,
                                                                                   collision_objects={
                                                                                       furniture: None,
                                                                                       human: None},
                                                                                   tool=tool)

    # TODO: reuse best_poses (ik solution) from toc instead of resolving ik
    is_success, robot_joint_angles, penetrations, dist_to_target, gripper_orient = robot.ik_random_restarts2(right=True,
                                                                                                             target_pos=target_pos,
                                                                                                             target_orient=None,
                                                                                                             max_iterations=100,
                                                                                                             randomize_limits=False,
                                                                                                             collision_objects={
                                                                                                                 furniture: None,
                                                                                                                 human: None},
                                                                                                             tool=tool)
    if is_success:
        # print("robot ik solution found")
        robot.set_joint_angles(robot.right_arm_joint_indices, robot_joint_angles, use_limits=True)
        tool.reset_pos_orient()

    return is_success, robot_joint_angles, best_position, best_orientation, side, penetrations, dist_to_target, gripper_orient


def find_max_val(human, cost_fn, original_joint_angles, original_link_positions, end_effector="right_hand"):
    x0 = np.array(original_joint_angles)
    optimizer = init_optimizer(x0, 0.1, human.controllable_joint_lower_limits, human.controllable_joint_upper_limits)
    timestep = 0
    while not optimizer.stop():
        fitness_values = []
        timestep += 1
        solutions = optimizer.ask()

        for s in solutions:
            if cost_fn == max_torque_cost_fn:
                human.set_joint_angles(human.controllable_joint_indices, s)
                cost = max_torque_cost_fn(human, end_effector)
                human.set_joint_angles(human.controllable_joint_indices, original_joint_angles)

            elif cost_fn == max_manipulibity_cost_fn:
                cost = max_manipulibity_cost_fn(human, end_effector, s)

            elif cost_fn == max_energy_cost_fn:
                human.set_joint_angles(human.controllable_joint_indices, s)
                cost = max_energy_cost_fn(human, end_effector, original_link_positions)
                human.set_joint_angles(human.controllable_joint_indices, original_joint_angles)

            fitness_values.append(cost)
        optimizer.tell(solutions, fitness_values)

    human.set_joint_angles(human.controllable_joint_indices, optimizer.best.x)
    return optimizer.best.x, 1.0 / optimizer.best.f


def find_robot_start_pos_orient(env, end_effector="right_hand", initial_side = None):
    # find bed bb
    bed = env.furniture
    bed_bb = p.getAABB(bed.body, physicsClientId=env.id)
    bed_pos = p.getBasePositionAndOrientation(bed.body, physicsClientId=env.id)[0]

    # find ee pos
    ee_pos, _ = env.human.get_ee_pos_orient(end_effector)
    # print ("ee real pos: ", ee_real_pos)
    if initial_side is not None:
        side = initial_side
    else:
        eyeline_side = get_eyeline_side(env.human)
        if eyeline_side is None:
            # find the side of the bed
            side = "right" if ee_pos[0] > bed_pos[0] else "left"
        else:
            side = eyeline_side
        bed_xx, bed_yy, bed_zz = bed_bb[1] if side == "right" else bed_bb[0]

        # find robot base and bb
        robot_bb = p.getAABB(env.robot.body, physicsClientId=env.id)
        robot_x_size, robot_y_size, robot_z_size = np.subtract(robot_bb[1], robot_bb[0])
        # print("robot: ", robot_bb)
        base_pos = p.getBasePositionAndOrientation(env.robot.body, physicsClientId=env.id)[0]

    # new pos: side of the bed, near end effector, with z axis unchanged
    if side == "right":
        pos = (
            bed_xx + robot_x_size / 2 + 0.1, ee_pos[1] ,
            base_pos[2])  # TODO: change back to original 0.3
        orient = env.robot.get_quaternion([0, 0, -np.pi / 2])
    else:  # left
        pos = ( bed_xx - robot_x_size / 2 - 0.1, ee_pos[1], base_pos[2])
        orient = env.robot.get_quaternion([0, 0, np.pi / 2])
    return pos, orient, side

def find_angle(vec_a, vec_b):
    # normal_vec = np.cross(vec_a, vec_b)/np.linalg.norm(np.cross(vec_a, vec_b))
    # return np.arctan2(np.dot(np.cross(vec_a, vec_b), normal_vec), np.dot(vec_a, vec_b))
    angle = np.arccos(np.dot(vec_a, vec_b) / (np.linalg.norm(vec_a) * np.linalg.norm(vec_b)))
    normal_vec = np.cross(vec_a, vec_b) / np.linalg.norm(np.cross(vec_a, vec_b))
    return angle if np.dot(normal_vec, np.array([0, 0, 1])) > 0 else -angle

def get_eyeline_side(human):
    '''
    Get eyeline side of the human - return None if it is not heavy left or right
    :param human:
    :return:
    '''
    head_pos, head_orient = human.get_ee_pos_orient("head")
    rotation_matrix = np.array(p.getMatrixFromQuaternion(head_orient)).reshape(3, 3)
    normal_vec = rotation_matrix[:, 2]/np.linalg.norm(rotation_matrix[:, 2])

    # draw for debug
    # p.addUserDebugLine(head_pos, head_pos + normal_vec*10, [1, 0, 0], 5, 10)
    # z_axis = np.array([0, 0, 1])
    # z_angle= find_angle(z_axis, normal_vec)
    # norm_vec = np.cross(z_axis, normal_vec)
    # p.addUserDebugLine(head_pos, head_pos + norm_vec*10, [0, 1, 0], 5, 10)
    # unsigned_z_angle = np.arccos(np.dot(z_axis, normal_vec))
    # z_angle = unsigned_z_angle if np.dot(z_axis, normal_vec)>0 else -unsigned_z_angle
    z_angle = get_angle_with_z_axis(rotation_matrix)
    # print ("normal vec", normal_vec, "z angle: ", z_angle)
    return "right" if z_angle > np.pi/6 else "left" if z_angle < -np.pi/6 else None

def get_handover_object_config(object_name, env) -> Optional[HandoverObjectConfig]:
    human = env.human
    eyeline_side = get_eyeline_side(human)
    if eyeline_side == None: # head look upward, undecided. return default
        ee = 'right_hand'
    else: # head look left or right
        upper_hand = choose_upper_hand(env.human) # choose the upper hand if it is heavy 1 higher than the other
        ee = upper_hand if upper_hand is not None else 'right_hand' # fall back to right hand if it is not heavy
    if object_name is None:  # case: no handover object
        return HandoverObjectConfig(None, weights=[0], limits=[0], end_effector=ee)  # original = 6
    # TODO: revise the hand choice
    # print ("object name: ", object_name)
    object_type = HandoverObject.from_string(object_name)
    if object_name == "pill":
        return HandoverObjectConfig(object_type, weights=[0], limits=[0.27], end_effector=ee)  # original = 6
    elif object_name == "cup":
        return HandoverObjectConfig(object_type, weights=[0], limits=[0.23], end_effector=ee)  # original = 6
    elif object_name == "cane":
        return HandoverObjectConfig(object_type, weights=[0], limits=[0.23], end_effector=ee)  # original = 6


def solve_ik(env, target_pos, end_effector="right_hand"):
    human = env.human
    ee_idx = human.human_dict.get_dammy_joint_id(end_effector)
    ik_joint_indices = human.find_ik_joint_indices()
    solution = human.ik(ee_idx, target_pos, None, ik_joint_indices, max_iterations=1000)  # TODO: Fix index
    # print ("ik solution: ", solution)
    return solution


def build_max_human_dynamics(env, end_effector, original_info: OriginalHumanInfo) -> MaximumHumanDynamics:
    """
    build maximum human dynamics by doing CMAES search
    will reset the env after all searches are done

    :param env:
    :param end_effector:
    :param original_info:
    :return:
    """
    human = env.human
    _, max_torque = find_max_val(human, max_torque_cost_fn, original_info.angles, original_info.link_positions,
                                 end_effector)
    _, max_manipubility = find_max_val(human, max_manipulibity_cost_fn, original_info.angles,
                                       original_info.link_positions,
                                       end_effector)
    _, max_energy = find_max_val(human, max_energy_cost_fn, original_info.angles, original_info.link_positions,
                                 end_effector)
    # max_torque, max_manipubility, max_energy = 10, 1, 100
    print("max torque: ", max_torque, "max manipubility: ", max_manipubility, "max energy: ", max_energy)
    max_dynamics = MaximumHumanDynamics(max_torque, max_manipubility, max_energy)


    return max_dynamics


def detect_collisions(original_info: OriginalHumanInfo, self_collisions, env_collisions, human, end_effector):
    # check collision
    new_self_penetrations = find_new_penetrations(original_info.self_collisions, self_collisions, human, end_effector, COLLISION_PENETRATION_THRESHOLD["self_collision"])
    new_env_penetrations = find_new_penetrations(original_info.env_collisions, env_collisions, human, end_effector, COLLISION_PENETRATION_THRESHOLD["env_collision"])
    LOG.info(f"self penetration: {new_self_penetrations}, env penetration: {new_env_penetrations}")
    # print(f"self penetration: {new_self_penetrations}, env penetration: {new_env_penetrations}")
    return new_self_penetrations, new_env_penetrations



"""
TODO: 
1. step forward
2. check collision and stop on collsion 
3. check if the target angle is reached. break
"""


def step_forward(env, x0, env_object_ids, end_effector="right_hand"):
    human = env.human
    # p.setJointMotorControlArray(human.body, jointIndices=human.controllable_joint_indices,
    #                             controlMode=p.POSITION_CONTROL,
    #                             forces=[1000] * len(human.controllable_joint_indices),
    #                             positionGains=[0.01] * len(human.controllable_joint_indices),
    #                             targetPositions=x0,
    #                             physicsClientId=human.id)
    human.control(human.controllable_joint_indices, x0, 0.01, 100)

    # for _ in range(5):
    #     p.stepSimulation(physicsClientId=env.human.id)
    # p.setRealTimeSimulation(1)
    original_self_collisions = human.check_self_collision(end_effector=end_effector)
    original_env_collisions = human.check_env_collision(env_object_ids, end_effector=end_effector)

    # print ("target: ", x0)

    prev_angle = [0] * len(human.controllable_joint_indices)
    count = 0
    while True:
        p.stepSimulation(physicsClientId=human.id)  # step simulation forward

        self_collision = human.check_self_collision(end_effector=end_effector)
        env_collision = human.check_env_collision(env_object_ids, end_effector=end_effector)
        cur_joint_angles = human.get_joint_angles(human.controllable_joint_indices)
        # print ("cur_joint_angles: ", cur_joint_angles)
        angle_dist = cal_angle_diff(cur_joint_angles, x0)
        count += 1
        if count_new_collision(original_self_collisions, self_collision, human, end_effector,
                               COLLISION_PENETRATION_THRESHOLD["self_collision"]) or count_new_collision(
            original_env_collisions,
            env_collision, human, end_effector, COLLISION_PENETRATION_THRESHOLD["env_collision"]):
            LOG.info(f"{bcolors.FAIL}sim step: {count}, collision{bcolors.ENDC}")
            return angle_dist, self_collision, env_collision, True

        if cal_angle_diff(cur_joint_angles, x0) < 0.05 or cal_angle_diff(cur_joint_angles, prev_angle) < 0.001:
            LOG.info(f"sim step: {count}, angle diff to prev: {cal_angle_diff(cur_joint_angles, prev_angle)}")
            return angle_dist, self_collision, env_collision, False
        prev_angle = cur_joint_angles


def make_env(env_name, person_id, smpl_file, object_name, coop=False, seed=1001):
    if not coop:
        env = gym.make('assistive_gym:' + env_name)
    else:
        module = importlib.import_module('assistive_gym.envs')
        env_class = getattr(module, env_name.split('-')[0] + 'Env')
        env = env_class()
    env.seed(seed)
    env.set_smpl_file(smpl_file)

    human_urdf_path = get_urdf_filepath(get_urdf_folderpath(person_id))
    env.set_human_urdf(human_urdf_path)

    task = get_task_from_handover_object(object_name)
    env.set_task(task)
    return env

def object_type_to_name(object_type: HandoverObject):
    return object_type.name.lower()

# TODO: better refactoring for seperating robot-ik/ non robot ik mode
def cost_func(human, ee_name: str, angle_config: np.ndarray, ee_target_pos: np.ndarray, original_info: OriginalHumanInfo,
              max_dynamics: MaximumHumanDynamics, new_self_penetrations, new_env_penetrations, has_valid_robot_ik,
              robot_penetrations, robot_dist_to_target, angle_dist,
              object_config: Optional[HandoverObjectConfig], robot_ik_mode: bool, object_specific_cost: float):
    # cal energy
    energy_change, energy_original, energy_final = cal_energy_change(human, original_info.link_positions, ee_name)

    # cal dist
    ee_pos, _ = human.get_ee_pos_orient(ee_name)
    dist = eulidean_distance(ee_pos, ee_target_pos)

    # cal torque
    torque = cal_torque_magnitude(human, ee_name)

    # cal manipulibility
    manipulibility = human.cal_chain_manipulibility(angle_config, ee_name)

    # cal angle displacement from mid angle
    mid_angle = cal_mid_angle(human.controllable_joint_lower_limits, human.controllable_joint_upper_limits)
    mid_angle_displacement = cal_angle_diff(angle_config, mid_angle)
    # print("mid_angle_displacement: ", mid_angle_displacement)

    object = 'default' if not object_config else  object_type_to_name(object_config.object_type)
    w= PARAMS[object]['weights']
    cost = 0
    o_specific_cost , self_penetration_cost, env_penetration_cost, ik_cost, robot_penetration_cost = 0, 0, 0, 0, 0

    cost += (w['dist'] * dist + w['manipulibility'] * 1 / (manipulibility / max_dynamics.manipulibility) + w[
        'energy'] * energy_final / max_dynamics.energy \
            + w['torque'] * torque / max_dynamics.torque + w['mid_angle'] * mid_angle_displacement)
    if object != 'default':
        o_specific_cost = w['special_cost'] * object_specific_cost
        cost+= o_specific_cost

    if new_self_penetrations:
        self_penetration_cost = w['self_penetration'] * sum(new_self_penetrations)
        cost += self_penetration_cost
        # cost += 10*len(new_self_penetrations)
    if new_env_penetrations:
        env_penetration_cost = w['env_penetration']  * sum(new_env_penetrations)
        cost += env_penetration_cost
    if robot_ik_mode:
        # if not has_valid_robot_ik:
            # cost += 1000
            # print('No valid ik solution found ', robot_dist_to_target)
        ik_cost = w['ik_dist'] *robot_dist_to_target
        cost += ik_cost
        if robot_penetrations:
            # flatten list
            robot_penetrations = [abs(item) for sublist in robot_penetrations for item in sublist]
            # print(robot_penetrations)]
            robot_penetration_cost = w['robot_penetration'] * sum(robot_penetrations)
            cost += robot_penetration_cost
    print('cost: ', cost/100, 'object specific cost: ', o_specific_cost/100,  'self_penetration_cost: ', self_penetration_cost/100, 'env_penetration_cost: ',
    env_penetration_cost/100, 'ik_cost: ', ik_cost/100, 'robot_penetration_cost: ', robot_penetration_cost/100)

    return cost/100, manipulibility, dist, energy_final, torque


def cal_energy_change(human, original_link_positions, end_effector):
    g = 9.81  # gravitational acceleration
    potential_energy_initial = 0
    potential_energy_final = 0

    link_indices = human.human_dict.get_real_link_indices(end_effector)
    current_link_positions = human.get_link_positions(True, end_effector_name=end_effector)

    # Iterate over all links
    for idx, j in enumerate(link_indices):
        mass = p.getDynamicsInfo(human.body, j)[0]
        LOG.debug(f"{bcolors.WARNING}link {j}, mass: {mass}{bcolors.ENDC}")

        # Calculate initial potential energy
        potential_energy_initial += mass * g * original_link_positions[idx][2]  # z axis
        potential_energy_final += mass * g * current_link_positions[idx][2]
        # Add changes to the total energy change
    total_energy_change = potential_energy_final - potential_energy_initial
    LOG.debug(
        f"Total energy change: {total_energy_change}, Potential energy initial: {potential_energy_initial}, Potential energy final: {potential_energy_final}")

    return total_energy_change, potential_energy_initial, potential_energy_final


def generate_target_points(env, num_points, ee_pos):
    human = env.human
    ee_pos, _ = human.get_ee_pos_orient("right_hand")
    points = fibonacci_evenly_sampling_range_sphere(ee_pos, [0.25, 0.5], num_points)
    return get_valid_points(env, points)


def get_initial_guess(env, target=None):
    if target is None:
        return np.zeros(len(env.human.controllable_joint_indices))  # no of joints
    else:
        # x0 = env.human.ik_chain(target)
        x0 = solve_ik(env, target, end_effector="right_hand")
        print(f"{bcolors.BOLD}x0: {x0}{bcolors.ENDC}")
        return x0


def debug_solution():
    # ee_pos, _, _= env.human.fk(["right_hand_limb"], x0)
    # ee_pos = env.human.fk_chain(x0)
    # print("ik error 2: ", eulidean_distance(ee_pos, target))
    # env.human.set_joint_angles(env.human.controllable_joint_indices, x0)

    # right_hand_ee = env.human.human_dict.get_dammy_joint_id("right_hand")
    # ee_positions, _ = env.human.forward_kinematic([right_hand_ee], x0)
    # print("ik error: ", eulidean_distance(ee_positions[0], target))
    #
    # for _ in range(1000):
    #     p.stepSimulation()
    # time.sleep(100)
    pass


def test_collision():
    # print("collision1: ", human.check_self_collision())
    # # print ("collision1: ", perform_collision_check(human))
    # x1 = np.random.uniform(-1, 1, len(human.controllable_joint_indices))
    # human.set_joint_angles(human.controllable_joint_indices, x1)
    # # for i in range (100):
    # #     p.stepSimulation(physicsClientId=human.id)
    # #     print("collision2: ", human.check_self_collision())
    # p.performCollisionDetection(physicsClientId=human.id)
    # # print("collision2: ", perform_collision_check(human))
    # print("collision2: ", human.check_self_collision())
    # time.sleep(100)
    pass


def cal_mid_angle(lower_bounds, upper_bounds):
    return (np.array(lower_bounds) + np.array(upper_bounds)) / 2


def count_new_collision(old_collisions: Set, new_collisions: Set, human, end_effector, penetration_threshold) -> int:
    # TODO: remove magic number (might need to check why self colllision happen in such case)
    # TODO: return number of collisions instead and use that to scale the cost
    link_ids = set(human.human_dict.get_real_link_indices(end_effector))
    # print ("link ids", link_ids)

    # convert old collision to set of tuples (link1, link2), remove penetration
    initial_collision_map = dict()
    for o in old_collisions:
        initial_collision_map[(o[0], o[1])] = o[2]

    collision_set = set()  # list of collision that is new or has deep penetration
    for collision in new_collisions:
        link1, link2, penetration = collision
        if not link1 in link_ids and not link2 in link_ids:
            continue  # not end effector chain collision, skip
        # TODO: fix it, since link1 and link2 in collision from different object, so there is a slim chance of collision
        if (link1, link2) not in initial_collision_map or (link2, link1) not in initial_collision_map:  # new collision:
            if abs(penetration) > penetration_threshold[
                "new"]:  # magic number. we have penetration between spine4 and shoulder in pose 5
                # print("new collision: ", collision)
                collision_set.add((collision[0], collision[1]))
        else:
            # collision in old collision
            initial_depth = initial_collision_map[(link1, link2)] if (link1, link2) in initial_collision_map else \
                initial_collision_map[(link2, link1)]
            if abs(penetration) > max(penetration_threshold["old"],
                                      initial_depth):  # magic number. we have penetration between spine4 and shoulder in pose 5
                # print("old collision with deep penetration: ", collision)
                collision_set.add((link1, link2))

    return len(collision_set)

def cal_angle_diff(cur, target):
    # print ("cur: ", len(cur), 'target: ', len(target))
    diff = np.sqrt(np.sum(np.square(np.array(cur) - np.array(target)))/len(cur))
    # print ("diff: ", diff)
    return diff


def get_valid_points(env, points):
    point_ids = []
    for point in points:
        id = create_point(point, size=0.01)
        point_ids.append(id)
    p.performCollisionDetection(physicsClientId=env.id)

    valid_points = []
    valid_ids = []
    for idx, point in enumerate(points):
        id = point_ids[idx]
        contact_points = p.getContactPoints(bodyA=id, physicsClientId=env.id)
        if len(contact_points) == 0:
            valid_points.append(point)
            # valid_ids.append(id)
        # else:
        #     p.removeBody(id)
        p.removeBody(id)
    return valid_points, valid_ids


def cal_torque_magnitude(human, end_effector):
    def pretty_print_torque(human, torques, end_effector):
        link_names = human.human_dict.joint_chain_dict[end_effector]
        # print ("link_names: ", link_names)
        # print ("torques: ", len(torques))
        for i in range(0, len(torques), 3):
            LOG.debug(f"{link_names[i // 3]}: {torques[i: i + 3]}")

    # torques = human.inverse_dynamic(end_effector)
    # print ("torques ee: ", len(torques), torques)
    # torques = human.inverse_dynamic()
    # print ("torques: ", len(torques), torques)
    torques = human.inverse_dynamic(end_effector)
    # print("torques ee: ", torques)
    # print ("----------------------------------")
    pretty_print_torque(human, torques, end_effector)

    torque_magnitude = 0
    for i in range(0, len(torques), 3):
        torque = np.sqrt(np.sum(np.square(torques[i:i + 3])))
        torque_magnitude += torque
    # LOG.debug(f"torques: {torques}, torque magnitude: {torque_magnitude}")
    return torque_magnitude


def get_actions_dict_key(handover_obj, robot_ik):
    return handover_obj + "-robot_ik" if robot_ik else handover_obj + "-no_robot_ik"


def get_save_dir(save_dir, env_name, person_id, smpl_file, timestamp=False):
    smpl_name = smpl_file.split('/')[-1].split('.')[0]
    time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    if timestamp:
        return os.path.join(save_dir, env_name, person_id, smpl_name, time)
    else:
        return os.path.join(save_dir, env_name, person_id, smpl_name)


def get_max_torque(env, end_effector="right_hand"):
    human = env.human
    human.set_joint_angles(human.controllable_joint_indices, len(human.controllable_joint_indices) * [0])
    torque = cal_torque_magnitude(human, end_effector)
    # print("max torque: ", torque)
    return torque


def max_torque_cost_fn(human, end_effector):
    torque = cal_torque_magnitude(human, end_effector)
    return 1.0 / torque


def max_manipulibity_cost_fn(human, end_effector, joint_angles):
    manipulibility = human.cal_chain_manipulibility(joint_angles, end_effector)
    return 1.0 / manipulibility


def max_energy_cost_fn(human, end_effector, original_link_positions):
    _, _, energy_final = cal_energy_change(human, original_link_positions, end_effector)
    return 1.0 / energy_final


def move_robot(env):  # for debugging purpose
    human, robot, furniture, tool = env.human, env.robot, env.furniture, env.tool
    target_joint_angles = np.random.uniform(-1, 1, len(robot.right_arm_joint_indices)) * np.pi

    for i in range(100):
        # random_val = np.random.uniform(-1, 1, len(robot.controllable_joint_indices))
        robot.control(robot.right_arm_joint_indices, np.array(target_joint_angles), 0.1, 100)
        p.stepSimulation()

    # print("tool mass: ", p.getDynamicsInfo(tool.body, -1)[0])


def get_human_link_robot_collision(human, end_effector):
    human_link_robot_collision = []
    for ee in human.human_dict.end_effectors:
        human_link_robot_collision.extend([link for link in human.human_dict.get_real_link_indices(ee)])
    # ignore collision with end effector and end effector's parent link
    parent_ee = human.human_dict.joint_to_parent_joint_dict[end_effector]
    link_to_ignores = [human.human_dict.get_dammy_joint_id(end_effector),
                       human.human_dict.get_dammy_joint_id(parent_ee)]
    human_link_robot_collision = [link for link in human_link_robot_collision if link not in link_to_ignores]
    # print("human_link: ", human_link_robot_collision)
    return human_link_robot_collision


def choose_upward_hand(human):
    right_offset = abs(-np.pi / 2 - human.get_roll_wrist_orientation(end_effector="right_hand"))
    left_offset = abs(-np.pi / 2 - human.get_roll_wrist_orientation(end_effector="left_hand"))

    if right_offset > np.pi / 2 and left_offset < np.pi / 2:
        return "left_hand"
    elif right_offset < np.pi / 2 and left_offset > np.pi / 2:
        return "right_hand"
    else:
        return None


def choose_upper_hand(human):
    right_pos = human.get_link_positions(True, end_effector_name="right_hand")
    left_pos = human.get_link_positions(True, end_effector_name="left_hand")
    right_shoulder_z = right_pos[1][2]
    left_shoulder_z = left_pos[1][2]
    # print("right_shoulder_z: ", right_shoulder_z, "\nleft_shoudler_z: ", left_shoulder_z)
    diff = right_shoulder_z - left_shoulder_z
    if diff > 0.2:
        return "right_hand"
    elif diff < -0.2:
        return "left_hand"
    else:
        return None


def choose_closer_bedside_hand(env):
    right_dist = cal_dist_to_bedside(env, "right_hand")
    left_dist = cal_dist_to_bedside(env, "left_hand")
    # print("right_dist: ", right_dist, "\nleft_dist: ", left_dist)
    return "right_hand" if right_dist < left_dist else "left_hand"


def build_original_human_info(human, env_object_ids, end_effector) -> OriginalHumanInfo:
    # original value
    original_joint_angles = human.get_joint_angles(human.controllable_joint_indices)
    original_link_positions = human.get_link_positions(center_of_mass=True, end_effector_name=end_effector)
    original_self_collisions = human.check_self_collision()
    original_env_collisions = human.check_env_collision(env_object_ids)
    original_info = OriginalHumanInfo(original_joint_angles, original_link_positions, original_self_collisions,
                                      original_env_collisions)
    return original_info


def translate_wrt_human_pelvis(human, pos, orient):
    # print("pos: ", pos, "orient: ", orient)
    pelvis_pos, pelvis_orient = human.get_pos_orient(human.human_dict.get_fixed_joint_id("pelvis"), center_of_mass=True)
    # print("pelvis_pos: ", pelvis_pos, "pelvis_orient: ", pelvis_orient)
    pelvis_pos_inv, pelvis_orient_inv = p.invertTransform(pelvis_pos, pelvis_orient, physicsClientId=human.id)
    if len(orient) == 0:
        orient = [0, 0, 0, 1]
    else:
        orient = orient if len(orient) == 4 else human.get_quaternion(orient)

    new_pos, new_orient = p.multiplyTransforms(pelvis_pos_inv, pelvis_orient_inv, pos, orient)
    return new_pos, new_orient


def init_optimizer(x0, sigma, lower_bounds, upper_bounds):  # for cmaes library
    opts = {}
    opts['tolfun'] = 1e-2
    opts['tolx'] = 1e-2

    for i in range(x0.size):
        if x0[i] < lower_bounds[i]:
            x0[i] = lower_bounds[i]
        if x0[i] > upper_bounds[i]:
            x0[i] = upper_bounds[i]
    for i in range(len(lower_bounds)):
        if lower_bounds[i] == 0:
            lower_bounds[i] = -1e-9
        if upper_bounds[i] == 0:
            upper_bounds[i] = 1e-9
    # bounds = [lower_bounds, upper_bounds]
    # opts['bounds'] = bounds
    es = CMAEvolutionStrategy(x0, sigma, opts)
    return es


def init_optimizer2(x0, sigma, lower_bounds, upper_bounds):  # for cma library
    # opts = {}
    # opts['tolfun'] = 1e-9
    # opts['tolx'] = 1e-9
    bounds = [[l, u] for l, u in zip(lower_bounds, upper_bounds)]
    bounds = np.array(bounds)
    # print ("bounds: ", bounds.shape, x0.shape, x0.size)
    # print("bounds: ", bounds)
    # print("x0: ", x0)
    for i in range(x0.size):
        if x0[i] < bounds[i][0]:
            x0[i] = bounds[i][0]
        if x0[i] > bounds[i][1]:
            x0[i] = bounds[i][1]
    es = CMA(x0, sigma, bounds=np.array(bounds))
    return es


def render(env_name, person_id, smpl_file, save_dir, handover_obj, robot_ik: bool):
    print("rendering person {} and smpl file {}".format(person_id, smpl_file))

    save_dir = get_save_dir(save_dir, env_name, person_id, smpl_file)
    actions = pickle.load(open(os.path.join(save_dir, "actions.pkl"), "rb"))
    if not actions:
        raise Exception("no actions found for person {} and smpl file {}".format(person_id, smpl_file))
    if handover_obj == "all":
        for key in actions.keys():
            action = actions[key]
            # print("key: ", key)
            handover_obj = key.split("-")[0]
            robot_ik = key.split("-")[1] == "robot_ik"
            # print("handover obj: ", handover_obj, "robot_ik: ", robot_ik)
            robot_pose, robot_joint_angles = None, None
            try:
                robot_pose = action["wrt_pelvis"]["robot"]['original']
                robot_joint_angles = action["wrt_pelvis"]["robot_joint_angles"]
            except Exception as e:
                print("no robot pose found")

            render_result(env_name, action, person_id, smpl_file, handover_obj, robot_ik, robot_pose,
                          robot_joint_angles)
    else:
        key = get_actions_dict_key(handover_obj, robot_ik)
        if key not in actions:
            raise Exception("no action found for ", key)
        action = actions[key]
        robot_pose, robot_joint_angles = None, None

        try:
            robot_pose = action["wrt_pelvis"]["robot"]['original']
            robot_joint_angles = action["wrt_pelvis"]["robot_joint_angles"]
        except Exception as e:
            print("no robot pose found")
        render_result(env_name, action, person_id, smpl_file, handover_obj, robot_ik, robot_pose, robot_joint_angles)


def render_result(env_name, action, person_id, smpl_file, handover_obj, robot_ik: bool, robot_pose=None,
                  robot_joint_angles=None):
    env = make_env(env_name, coop=True, smpl_file=smpl_file, object_name=handover_obj, person_id=person_id)
    env.render()  # need to call reset after render
    env.reset()

    smpl_name = os.path.basename(smpl_file).split(".")[0]
    p.addUserDebugText("person: {}, smpl: {}".format(person_id, smpl_name), [0, 0, 1], textColorRGB=[1, 0, 0])

    env.human.reset_controllable_joints(action["end_effector"])
    env.human.set_joint_angles(env.human.controllable_joint_indices, action["solution"])
    if robot_ik:
        # print("robot pose: ", robot_pose, "robot_joint_angles: ", robot_joint_angles)
        if robot_pose is None or robot_joint_angles is None:
            find_robot_ik_solution(env, action["end_effector"], handover_obj)
        else:
            # TODO: refactor - render_robot in mprocess_train
            # find_robot_ik_solution(env, action["end_effector"], handover_obj)
            base_pos, base_orient, side = find_robot_start_pos_orient(env, action["end_effector"])
            env.robot.set_base_pos_orient(robot_pose[0], robot_pose[1])
            env.robot.set_joint_angles(
                env.robot.right_arm_joint_indices if side == 'right' else env.robot.left_arm_joint_indices,
                robot_joint_angles)
            env.tool.reset_pos_orient()
            get_eyeline_side(env.human)
    # plot_cmaes_metrics(action['mean_cost'], action['mean_dist'], action['mean_m'], action['mean_energy'],
    #                    action['mean_torque'])
    # plot_mean_evolution(action['mean_evolution'])

    while True:
        keys = p.getKeyboardEvents()
        if ord('q') in keys:
            break
    env.disconnect()


def render_pose(env_name, person_id, smpl_file):
    env = make_env(env_name, coop=True, smpl_file=smpl_file, object_name=None, person_id=person_id)
    env.render()  # need to call reset after render
    env.reset()
    eyeline_side = get_eyeline_side(env.human)
    while True:
        keys = p.getKeyboardEvents()
        if ord('q') in keys:
            break
    env.disconnect()


def save_train_result(save_dir, env_name, person_id, smpl_file, actions):
    save_dir = get_save_dir(save_dir, env_name, person_id, smpl_file)
    os.makedirs(save_dir, exist_ok=True)

    if os.path.exists(os.path.join(save_dir, "actions.pkl")):
        old_actions = pickle.load(open(os.path.join(save_dir, "actions.pkl"), "rb"))
        # merge
        if old_actions:
            for key in old_actions.keys():
                if key not in actions.keys():
                    actions[key] = old_actions[key]
    pickle.dump(actions, open(os.path.join(save_dir, "actions.pkl"), "wb"))

def find_new_penetrations(old_collisions: Set, new_collisions: Set, human, end_effector, penetration_threshold) -> int:
    # TODO: remove magic number (might need to check why self colllision happen in such case)
    # TODO: return number of collisions instead and use that to scale the cost
    link_ids = set(human.human_dict.get_real_link_indices(end_effector))
    # print ("link ids", link_ids)

    # convert old collision to set of tuples (link1, link2), remove penetration
    initial_collision_map = dict()
    for o in old_collisions:
        initial_collision_map[(o[0], o[1])]= o[2]

    collision_set = set() # list of collision that is new or has deep penetration
    adjusted_penetrations = []
    for collision in new_collisions:
        link1, link2, penetration = collision
        if not link1 in link_ids and not link2 in link_ids:
            continue # not end effector chain collision, skip
        # TODO: fix it, since link1 and link2 in collision from different object, so there is a slim chance of collision
        if (link1, link2) not in initial_collision_map or (link2, link1) not in initial_collision_map: #new collision:
            if abs(penetration) > penetration_threshold["new"]:  # magic number. we have penetration between spine4 and shoulder in pose 5
                # print ("new collision: ", collision)
                collision_set.add((collision[0], collision[1]))
                adjusted_penetrations.append(abs(penetration) - penetration_threshold["new"])
        else:
            # collision in old collision
            initial_depth = initial_collision_map[(link1, link2)] if (link1, link2) in initial_collision_map else initial_collision_map[(link2, link1)]
            if abs(penetration) > max(penetration_threshold["old"], initial_depth): # magic number. we have penetration between spine4 and shoulder in pose 5
                # print ("old collision with deep penetration: ", collision)
                collision_set.add((link1, link2))
                adjusted_penetrations.append(abs(penetration) - max(penetration_threshold["old"], initial_depth))

    return adjusted_penetrations

def get_angle_with_z_axis(rot_matrix):
    vec_z = rot_matrix[:, 2] / np.linalg.norm(rot_matrix[:, 2])
    rad_angle =  np.arccos(np.dot(vec_z, np.array([0, 0, 1])))
    # determine the sign
    vec_z_proj_x = np.dot(vec_z,  np.array([1, 0, 0]))
    return rad_angle if vec_z_proj_x >= 0 else -rad_angle

def get_gripper_z_angle_cost(env, side, offset):
    '''
    get the z angle of the gripper in radian
    :param robot:
    :param side:
    :return:
    '''
    robot = env.robot
    right = True if side == 'right' else False
    # find gripper orientation
    gripper_pos, gripper_orient = robot.get_pos_orient( robot.right_end_effector if right else robot.left_end_effector)
    rot_matrix = np.array(p.getMatrixFromQuaternion(np.array(gripper_orient))).reshape(3, 3)
    # target_pos = np.array(gripper_pos) + z_axis * 2
    # p.addUserDebugLine(gripper_pos, target_pos, [1, 0, 0], 5)
    # p.addUserDebugLine(gripper_pos, gripper_pos + np.array([0, 0, 2]), [0, 1, 0], 5)
    angle = get_angle_with_z_axis(rot_matrix)

    # print("z_axis", z_axis)
    # print ("angle", angle)
    lower_limit, upper_limit =  offset[0]* np.pi/180.0, offset[1]* np.pi/180.0
    if lower_limit<=angle<=upper_limit:
        return abs(angle) * 0.1
    else:
        return abs(angle - lower_limit) if angle<0 else abs(angle - upper_limit)

def cal_ee_bedside_dist_cost(env, side, end_effector, offset): # TODO: duplicate code with cal_distance_to_bed
    human, bed = env.human, env.furniture

    bed_bb = p.getAABB(bed.body, physicsClientId=env.id)
    right = True if side == 'right' else False
    ee_pos, _ = human.get_ee_pos_orient(end_effector)
    bed_xx, _, _ = bed_bb[1] if right else bed_bb[0]
    bed_xx = bed_xx + offset if right else bed_xx - offset
    # print ('bed size: ', np.array(bed_bb[1]) - np.array(bed_bb[0]))
    # print ("bed_xx: ", bed_xx, "ee_pos: ", ee_pos, "side: ", side)
    if right:
        return abs(ee_pos[0] - bed_xx)*0.1 if ee_pos[0] > bed_xx else abs(ee_pos[0] - bed_xx)
    else:
        return abs(ee_pos[0] - bed_xx)*0.1 if ee_pos[0] < bed_xx else abs(ee_pos[0] - bed_xx)


@deprecated
def translate_bed_to_realworld(env, cord):
    def find_corner(env):
        bed = env.furniture
        bed_pos, bed_orient = p.getBasePositionAndOrientation(bed.body, physicsClientId=env.id)
        # get aabb
        bed_aabb = p.getAABB(bed.body, physicsClientId=env.id)
        # print('bed_aabb: ', bed_aabb)
        p.setPhysicsEngineParameter(contactBreakingThreshold=0.00001)
        bed_size = np.array(bed_aabb[1]) - np.array(bed_aabb[0])
        # draw aabb box
        p.addUserDebugLine(bed_aabb[0], bed_aabb[0] + np.array([bed_size[0], 0, 0]), [1, 0, 0], 5,
                           physicsClientId=env.id)
        p.addUserDebugLine(bed_aabb[0], bed_aabb[0] + np.array([0, bed_size[1], 0]), [0, 1, 0], 5,
                           physicsClientId=env.id)
        p.addUserDebugLine(bed_aabb[0], bed_aabb[0] + np.array([0, 0, bed_size[2]]), [0, 0, 1], 5,
                           physicsClientId=env.id)

        # some hardcode offset - to find the corner
        corner = np.array(bed_aabb[0]) + np.array([0.5, 0.5, 0])  # TODO: change this
        # draw corner
        p.addUserDebugLine(corner, corner + np.array([0, 0, 1]), [1, 0, 0], 5, physicsClientId=env.id)
        return corner

    corner = find_corner(env)
    return np.array(cord) - corner

class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
