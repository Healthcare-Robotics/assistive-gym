import os, sys, multiprocessing, gym, ray, shutil, argparse, importlib, glob
import pickle
import time
from enum import Enum

from torch.utils.hipify.hipify_python import bcolors

from assistive_gym.envs.utils.log_utils import get_logger
from typing import Set, Optional

import numpy as np
from cma import CMA, CMAEvolutionStrategy
from cmaes import CMA
import pybullet as p

from assistive_gym.envs.utils.plot_utils import plot_cmaes_metrics, plot_mean_evolution
from assistive_gym.envs.utils.point_utils import fibonacci_evenly_sampling_range_sphere, eulidean_distance

LOG = get_logger()


class OriginalHumanInfo:
    def __init__(self, original_angles: np.ndarray, original_link_positions: np.ndarray, original_self_collisions,
                 original_env_collisions):
        self.link_positions = original_link_positions  # should be array of tuples that are the link positions
        self.angles = original_angles
        self.self_collisions = original_self_collisions
        self.env_collisions = original_env_collisions


class MaximumHumanDynamics:
    def __init__(self, max_torque, max_manipulibility, max_energy):
        self.torque = max_torque
        self.manipulibility = max_manipulibility
        self.energy = max_energy


class HandoverObject(Enum):
    PILL = "pill"
    CUP = "cup"
    CANE = "cane"


    @staticmethod
    def from_string(label):
        if label == "pill":
            return HandoverObject.PILL
        elif label == "cup":
            return HandoverObject.CUP
        elif label == "cane":
            return HandoverObject.CANE
        else:
            raise ValueError(f"Invalid handover object label: {label}")


objectTaskMapping = {
        HandoverObject.PILL: "comfort_taking_medicine",
        HandoverObject.CUP: "comfort_drinking",
        HandoverObject.CANE: "comfort_standing_up"
}

class HandoverObjectConfig:
    def __init__(self, object_type: HandoverObject, weights: list, limits: list, end_effector: Optional[str]):
        self.object_type = object_type
        self.weights = weights
        self.limits = limits
        self.end_effector = end_effector

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


def make_env(env_name, smpl_file, object_name, coop=False, seed=1001):
    if not coop:
        env = gym.make('assistive_gym:' + env_name)
    else:
        module = importlib.import_module('assistive_gym.envs')
        env_class = getattr(module, env_name.split('-')[0] + 'Env')
        env = env_class()
    env.seed(seed)
    env.set_smpl_file(smpl_file)
    task = get_task_from_handover_object(object_name)
    env.set_task(task)
    return env


def solve_ik(env, target_pos, end_effector="right_hand"):
    human = env.human
    ee_idx = human.human_dict.get_dammy_joint_id(end_effector)
    ik_joint_indices = human.find_ik_joint_indices()
    solution = human.ik(ee_idx, target_pos, None, ik_joint_indices, max_iterations=1000)  # TODO: Fix index
    # print ("ik solution: ", solution)
    return solution


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
        if has_new_collision(original_self_collisions, self_collision) or has_new_collision(original_env_collisions,
                                                                                            env_collision, human, end_effector):
            LOG.info(f"{bcolors.FAIL}sim step: {count}, collision{bcolors.ENDC}")
            return angle_dist, self_collision, env_collision, True

        if cal_angle_diff(cur_joint_angles, x0) < 0.05 or cal_angle_diff(cur_joint_angles, prev_angle) < 0.001:
            LOG.info(f"sim step: {count}, angle diff to prev: {cal_angle_diff(cur_joint_angles, prev_angle)}")
            return angle_dist, self_collision, env_collision, False
        prev_angle = cur_joint_angles


def cal_angle_diff(cur, target):
    # print ("cur: ", len(cur), 'target: ', len(target))
    diff = np.sqrt(np.sum(np.square(np.array(cur) - np.array(target)))) / len(cur)
    # print ("diff: ", diff)
    return diff


def cal_mid_angle(lower_bounds, upper_bounds):
    return (np.array(lower_bounds) + np.array(upper_bounds)) / 2

def has_new_collision(old_collisions: Set, new_collisions: Set, human, end_effector="right_hand") -> bool:
    # TODO: remove magic number (might need to check why self colllision happen in such case)
    # TODO: return number of collisions instead and use that to scale the cost
    link_ids = set(human.human_dict.get_real_link_indices(end_effector))
    # print ("link ids", link_ids)

    # convert old collision to set of tuples (link1, link2), remove penetration
    collisions = set()
    for o in old_collisions:
        collisions.add((o[0], o[1]))

    collision_arr = [] # list of collision that is new or has deep penetration
    for collision in new_collisions:
        if not collision[0] in link_ids and not collision[1] in link_ids:
            continue # not end effector chain collision, skip
        # TODO: fix it, since link1 and link2 in collision from different object, so there is a slim chance of collision
        if (collision[0], collision[1]) not in collisions or (collision[1], collision[0]) not in collisions: #new collision:
            print ("new collision: ", collision)
            if abs(collision[2]) > 0.005:  # magic number. we have penetration between spine4 and shoulder in pose 5
                collision_arr.append(collision)
        else:
            # collision in old collision
            link1, link2, penetration = collision
            print("old collision with deep: ", collision)
            if abs(penetration) > 0.015: # magic number. we have penetration between spine4 and shoulder in pose 5
                print ("old collision with deep penetration: ", collision)
                collision_arr.append(collision)

    return True if len(collision_arr) > 0 else False


# TODO: better refactoring for seperating robot-ik/ non robot ik mode
def cost_fn(human, ee_name: str, angle_config: np.ndarray, ee_target_pos: np.ndarray, original_info: OriginalHumanInfo,
            max_dynamics: MaximumHumanDynamics,  has_self_collision,has_env_collision, has_valid_robot_ik, angle_dist, 
            object_config: Optional[HandoverObjectConfig], robot_ik_mode: bool):

    # cal energy
    energy_change, energy_original, energy_final = cal_energy_change(human, original_info.link_positions, ee_name)

    # cal dist
    ee_pos, _= human.get_ee_pos_orient(ee_name)
    dist = eulidean_distance(ee_pos, ee_target_pos)

    # cal torque
    torque = cal_torque_magnitude(human, ee_name)

    # cal manipulibility
    manipulibility = human.cal_chain_manipulibility(angle_config, ee_name)

    # cal angle displacement from mid angle
    mid_angle = cal_mid_angle(human.controllable_joint_lower_limits, human.controllable_joint_upper_limits)
    mid_angle_displacement = cal_angle_diff(angle_config, mid_angle)
    print("mid_angle_displacement: ", mid_angle_displacement)

    # cal reba
    reba = human.get_reba_score(end_effector=ee_name)
    max_reba = 9.0

    w = [1, 1, 4, 1, 1, 2]
    cost = None

    if not object_config: # no object handover case
        cost = (w[0] * dist + w[1] * 1 / (manipulibility / max_dynamics.manipulibility) + w[
            2] * energy_final / max_dynamics.energy \
                + w[3] * torque / max_dynamics.torque + w[4] * mid_angle_displacement + w[
                    5] * reba / max_reba) / np.sum(w)
    else:
        if object_config.object_type == HandoverObject.PILL:
            # cal wrist orient (pill)
            wr_offset = human.get_roll_wrist_orientation(end_effector=ee_name)
            max_wr_offset = 1
            w = w + object_config.weights
            # cal cost
            cost = (w[0] * dist + w[1] * 1 / (manipulibility / max_dynamics.manipulibility) + w[2] * energy_final / max_dynamics.energy \
                + w[3] * torque / max_dynamics.torque + w[4] * mid_angle_displacement + w[5] * reba/max_reba
                + w[6] * wr_offset/max_wr_offset) / np.sum(w)
            # check angle
            cost += 100 * (wr_offset > object_config.limits[0])

        elif object_config.object_type in [HandoverObject.CUP, HandoverObject.CANE]:
            # cal wrist orient (cup and cane)
            cup_wr_offset = abs(human.get_pitch_wrist_orientation(end_effector=ee_name) - 1)
            max_cup = 1
            w = w + object_config.weights
            # cal cost
            cost = (w[0] * dist + w[1] * 1 / (manipulibility / max_dynamics.manipulibility) + w[2] * energy_final / max_dynamics.energy \
                + w[3] * torque / max_dynamics.torque + w[4] * mid_angle_displacement + w[5] * reba/max_reba
                + w[6] * cup_wr_offset/max_cup) / np.sum(w)
            if not robot_ik_mode: # using raycast to calculate cost
                # check angles and raycasts
                cost += 100 * cup_wr_offset > object_config.limits[0]
                cost += 100 * human.ray_cast_perpendicular(end_effector=ee_name, ray_length=0.1)
                if object_config.object_type == HandoverObject.CANE:
                    cost += 100 * human.ray_cast_parallel(end_effector=ee_name)

    if has_self_collision:
        cost += 1000
    if has_env_collision:
        cost += 1000

    if robot_ik_mode:
        if not has_valid_robot_ik:
            cost += 1000

    return cost, manipulibility, dist, energy_final, torque, reba


def get_save_dir(save_dir, env_name, smpl_file):
    smpl_name = smpl_file.split('/')[-1].split('.')[0]
    return os.path.join(save_dir, env_name, smpl_name)


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
    LOG.debug(f"torques: {torques}, torque magnitude: {torque_magnitude}")
    return torque_magnitude


def get_max_torque(env, end_effector="right_hand"):
    human = env.human
    human.set_joint_angles(human.controllable_joint_indices, len(human.controllable_joint_indices) * [0])
    torque = cal_torque_magnitude(human, end_effector)
    print("max torque: ", torque)
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

def detect_collisions(original_info: OriginalHumanInfo, self_collisions, env_collisions, human, end_effector):
    # check collision
    has_new_self_collision = has_new_collision(original_info.self_collisions, self_collisions, human, end_effector)
    has_new_env_collision = has_new_collision(original_info.env_collisions, env_collisions, human, end_effector)
    LOG.debug(f"self collision: {has_new_self_collision}, env collision: {has_new_env_collision}")

    return has_new_self_collision, has_new_env_collision


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


def find_robot_start_pos_orient(env, end_effector="right_hand"):
    # find bed bb
    bed = env.furniture
    bed_bb = p.getAABB(bed.body, physicsClientId=env.id)
    bed_pos = p.getBasePositionAndOrientation(bed.body, physicsClientId=env.id)[0]

    # find ee pos
    ee_pos, _ = env.human.get_ee_pos_orient(end_effector)
    # print ("ee real pos: ", ee_real_pos)

    # find the side of the bed
    side = "right" if ee_pos[0] > bed_pos[0] else "left"
    bed_xx, bed_yy, bed_zz = bed_bb[1] if side == "right" else bed_bb[0]

    # find robot base and bb
    robot_bb = p.getAABB(env.robot.body, physicsClientId=env.id)
    robot_x_size, robot_y_size, robot_z_size = np.subtract(robot_bb[1], robot_bb[0])
    # print("robot: ", robot_bb)
    base_pos = p.getBasePositionAndOrientation(env.robot.body, physicsClientId=env.id)[0]

    # new pos: side of the bed, near end effector, with z axis unchanged
    if side == "right":
        pos = (bed_xx + robot_x_size / 2 + 0.3, ee_pos[1] + robot_y_size / 2, base_pos[2])
        orient = env.robot.get_quaternion([0, 0, -np.pi / 2])
    else:  # left
        pos = (bed_xx - robot_x_size / 2 - 0.3, ee_pos[1], base_pos[2])
        orient = env.robot.get_quaternion([0, 0, np.pi / 2])
    return pos, orient, side


def move_robot(env): # for debugging purpose
    human, robot, furniture, tool = env.human, env.robot, env.furniture, env.tool
    target_joint_angles = np.random.uniform(-1, 1, len(robot.right_arm_joint_indices)) * np.pi

    for i in range(100):
        # random_val = np.random.uniform(-1, 1, len(robot.controllable_joint_indices))
        robot.control(robot.right_arm_joint_indices, np.array(target_joint_angles), 0.1, 100)
        p.stepSimulation()

    print ("tool mass: ", p.getDynamicsInfo(tool.body, -1)[0])


def find_robot_ik_solution(env, end_effector:str, human_link_robot_collision, tool=None):
    """
    Find robot ik solution with TOC. Place the robot in best base position and orientation.
    :param env:
    :param end_effector: str
    :param human_link_robot_collision: dict(agent, [link1, link2, ...]) to check for collision with robot
    :return:
    """

    # # naive solution without toc
    # # robot_base_pos, robot_base_orient, side = find_robot_start_pos_orient(env)
    # # p.resetBasePositionAndOrientation(robot.body, robot_base_pos, robot_base_orient)
    # # ee_pos, _ = human.get_ee_pos_orient(end_effector)
    # # has_valid_robot_ik, robot_angles = robot.ik_random_restarts2(right=True, target_pos=ee_pos,
    # #                                                target_orient=None, max_iterations=500,
    # #                                                randomize_limits=False,
    # #                                                collision_objects={furniture: None,
    # #                                                                   human: human_link_robot_collision})
    # # if has_valid_robot_ik:
    # #     robot.set_joint_angles(robot.controllable_joint_indices, robot_angles)
    #
    # find robot base pos

    human, robot, furniture, tool = env.human, env.robot, env.furniture, env.tool

    robot_base_pos, robot_base_orient, side = find_robot_start_pos_orient(env)
    ee_pos, ee_orient = human.get_ee_pos_orient(end_effector)
    _, _, best_poses = robot.position_robot_toc2(robot_base_pos, side, [(ee_pos, None)],
                                                 [(ee_pos, None)], human,
                                                 base_euler_orient=robot_base_orient, attempts=100,
                                                 random_position=0.5, max_ik_iterations=100,
                                                 collision_objects={
                                                     furniture: None,
                                                     human: human_link_robot_collision},
                                                 tool = tool)

    # TODO: reuse best_poses (ik solution) from toc instead of resolving ik
    is_success, robot_joint_angles = robot.ik_random_restarts2(right=True, target_pos=ee_pos,
                                                               target_orient=None, max_iterations=500,
                                                               randomize_limits=False,
                                                               collision_objects={furniture: None,
                                                                                  human: human_link_robot_collision},
                                                               tool = tool)

    if is_success: # TODO: what if we can't find a solution?
        robot.set_joint_angles(robot.right_arm_joint_indices, robot_joint_angles, use_limits=True)
        tool.reset_pos_orient()

    return is_success


def get_human_link_robot_collision(human, end_effector):
    human_link_robot_collision = []
    for ee in human.human_dict.end_effectors:
        human_link_robot_collision.extend([link for link in human.human_dict.get_real_link_indices(ee)])
    # ignore collision with end effector and end effector's parent link
    parent_ee = human.human_dict.joint_to_parent_joint_dict[end_effector]
    link_to_ignores = [human.human_dict.get_dammy_joint_id(end_effector),
                       human.human_dict.get_dammy_joint_id(parent_ee)]
    human_link_robot_collision = [link for link in human_link_robot_collision if link not in link_to_ignores]
    print("human_link: ", human_link_robot_collision)
    return human_link_robot_collision


def choose_upward_hand(human):
    right_offset = abs(-np.pi/2 - human.get_roll_wrist_orientation(end_effector="right_hand"))
    left_offset = abs(-np.pi/2 - human.get_roll_wrist_orientation(end_effector="left_hand"))

    if right_offset > np.pi/2 and left_offset < np.pi/2:
        return "left_hand"
    elif right_offset < np.pi/2 and left_offset > np.pi/2:
        return "right_hand"
    else:
        return None


def choose_upper_hand(human):
    right_pos = human.get_link_positions(True, end_effector_name="right_hand")
    left_pos = human.get_link_positions(True, end_effector_name="left_hand")
    right_shoulder_z = right_pos[1][2]
    left_shoulder_z = left_pos[1][2]
    print("right_shoulder_z: ", right_shoulder_z, "\nleft_shoudler_z: ", left_shoulder_z)
    diff = right_shoulder_z - left_shoulder_z
    if diff > 0.1:
        return "right_hand"
    elif diff < -0.1:
        return "left_hand"
    else:
        return None


def get_handover_object_config(object_name, human) -> Optional[HandoverObjectConfig]:
    if object_name == None: # case: no handover object
        return None

    object_type = HandoverObject.from_string(object_name)
    if object_name == "pill":
        ee = choose_upward_hand(human)
        return HandoverObjectConfig(object_type, weights=[6], limits=[0.27], end_effector=ee)
    elif object_name == "cup":
        ee = choose_upper_hand(human)
        return HandoverObjectConfig(object_type, weights=[6], limits=[0.23], end_effector=ee)
    elif object_name == "cane":
        return HandoverObjectConfig(object_type, weights=[6], limits=[0.23], end_effector=None)


def get_task_from_handover_object(object_name):
    if not object_name:
        return None
    object_type = HandoverObject.from_string(object_name)
    task = objectTaskMapping[object_type]
    return task

def train(env_name, seed=0, num_points=50, smpl_file='examples/data/smpl_bp_ros_smpl_re2.pkl',
          end_effector='right_hand', save_dir='./trained_models/', render=False, simulate_collision=False, robot_ik=False, handover_obj=None):
    start_time = time.time()

    env = make_env(env_name, smpl_file, handover_obj, coop=True)
    if render:
        env.render()
    env.reset()

    human, robot, furniture, plane = env.human, env.robot, env.furniture, env.plane
    # switch controllable indicies to left arm if end_effector does not equal right > this will likely be removed/can be overwritten by an object
    handover_obj_config = get_handover_object_config(handover_obj, human)
    if handover_obj_config and handover_obj_config.end_effector: # reset the end effector based on the object
        human.reset_controllable_joints(handover_obj_config.end_effector)

    env_object_ids = [furniture.body, plane.body]  # set env object for collision check
    human_link_robot_collision = get_human_link_robot_collision(human, end_effector)

    # original value
    original_joint_angles = human.get_joint_angles(human.controllable_joint_indices)
    original_link_positions = human.get_link_positions(center_of_mass=True, end_effector_name=end_effector)
    original_self_collisions = human.check_self_collision()
    original_env_collisions = human.check_env_collision(env_object_ids)
    original_info = OriginalHumanInfo(original_joint_angles, original_link_positions, original_self_collisions,
                                        original_env_collisions)
    # draw original ee pos
    original_ee_pos = human.get_pos_orient(human.human_dict.get_dammy_joint_id(end_effector), center_of_mass=True)[0]
    draw_point(original_ee_pos, size=0.01, color=[0, 1, 0, 1])

    timestep = 0
    mean_cost, mean_dist, mean_m, mean_energy, mean_torque, mean_evolution, mean_reba = [], [], [], [], [], [], []
    actions = []

    _, max_torque = find_max_val(human, max_torque_cost_fn, original_joint_angles, original_link_positions,
                                 end_effector)
    _, max_manipubility = find_max_val(human, max_manipulibity_cost_fn, original_joint_angles, original_link_positions,
                                       end_effector)
    _, max_energy = find_max_val(human, max_energy_cost_fn, original_joint_angles, original_link_positions,
                                 end_effector)
    # max_torque, max_manipubility, max_energy = 10, 1, 100
    print("max torque: ", max_torque, "max manipubility: ", max_manipubility, "max energy: ", max_energy)
    max_dynamics = MaximumHumanDynamics(max_torque, max_manipubility, max_energy)

    env.reset()

    # init optimizer
    x0 = np.array(original_joint_angles)
    optimizer = init_optimizer(x0, 0.1, human.controllable_joint_lower_limits, human.controllable_joint_upper_limits)

    if not robot_ik: # simulate collision
        ee_link_idx = human.human_dict.get_dammy_joint_id(end_effector)
        ee_collision_radius = 0.05 # 20cm range
        ee_collision_body = human.add_collision_object_around_link(ee_link_idx, radius=ee_collision_radius) # TODO: ignore collision with hand

    while not optimizer.stop():
        timestep += 1
        solutions = optimizer.ask()
        fitness_values, dists, manipus, energy_changes, torques = [], [], [], [], []
        for s in solutions:

            if simulate_collision:
                # step forward env
                angle_dist, _, env_collisions, _ = step_forward(env, s, env_object_ids, end_effector)
                self_collisions = human.check_self_collision()
                has_self_collision, has_env_collision= detect_collisions(original_info, self_collisions, env_collisions, human, end_effector)
                cost, m, dist, energy, torque = cost_fn(human, end_effector, s, original_ee_pos, original_info,
                                                        max_dynamics, has_self_collision, has_env_collision, has_valid_robot_ik, 
                                                        angle_dist, handover_obj_config)
                env.reset_human(is_collision=True)
                LOG.info(
                    f"{bcolors.OKGREEN}timestep: {timestep}, cost: {cost}, angle_dist: {angle_dist} , dist: {dist}, manipulibility: {m}, energy: {energy}, torque: {torque}{bcolors.ENDC}")
            else:
                # set angle directly
                human.set_joint_angles(human.controllable_joint_indices, s)  # force set joint angle

                # check collision
                env_collisions, self_collisions = human.check_env_collision(env_object_ids), human.check_self_collision()
                has_self_collision, has_env_collision = detect_collisions(original_info, self_collisions, env_collisions, human, end_effector)
                # move_robot(env)
                if robot_ik: # solve robot ik when doing training
                    has_valid_robot_ik = False if has_env_collision or has_self_collision else find_robot_ik_solution(env, end_effector, human_link_robot_collision )
                else:
                    ee_collision_body_pos, ee_collision_body_orient = human.get_ee_collision_shape_pos_orient(end_effector, ee_collision_radius)
                    p.resetBasePositionAndOrientation(ee_collision_body, ee_collision_body_pos, ee_collision_body_orient, physicsClientId=env.id)
                    has_valid_robot_ik = True


                cost, m, dist, energy, torque, reba = cost_fn(human, end_effector, s, original_ee_pos, original_info,
                                                        max_dynamics, has_self_collision, has_env_collision, has_valid_robot_ik, 
                                                        0, handover_obj_config, robot_ik_mode=robot_ik)
                # restore joint angle
                human.set_joint_angles(human.controllable_joint_indices, original_joint_angles)


                LOG.info(
                    f"{bcolors.OKGREEN}timestep: {timestep}, cost: {cost}, dist: {dist}, manipulibility: {m}, energy: {energy}, torque: {torque}{bcolors.ENDC}")
            # for ray_id in ray_ids:
            #     p.removeUserDebugItem(ray_id)
            fitness_values.append(cost)
            dists.append(dist)
            manipus.append(m)
            energy_changes.append(energy)
            torques.append(torque)

        optimizer.tell(solutions, fitness_values)
        # optimizer.result_pretty()
        # LOG.info(
        #     f"{bcolors.OKGREEN}timestep: {timestep}, cost: {cost}, dist: {dist}, manipulibility: {m}, energy: {energy}, torque: {torque}{bcolors.ENDC}")

        mean_evolution.append(np.mean(solutions, axis=0))
        mean_cost.append(np.mean(fitness_values, axis=0))
        mean_dist.append(np.mean(dists, axis=0))
        mean_m.append(np.mean(manipus, axis=0))
        mean_energy.append(np.mean(energy_changes, axis=0))
        mean_torque.append(np.mean(torques, axis=0))

    if simulate_collision:
        angle_dist, self_collision, env_collisions, is_collision = step_forward(env, optimizer.best.x, env_object_ids)
    else:
        # get the best solution value
        env.human.set_joint_angles(env.human.controllable_joint_indices, optimizer.best.x)
        self_collisions, env_collisions = human.check_self_collision(), human.check_env_collision(env_object_ids)
        has_self_collision, has_env_collision = detect_collisions(original_info, self_collisions, env_collisions, human, end_effector)
        if robot_ik:  # solve robot ik when doing training
            has_valid_robot_ik = False if has_env_collision or has_self_collision else find_robot_ik_solution(env,end_effector, human_link_robot_collision)
        else:
            has_valid_robot_ik = True
            ee_collision_body_pos, ee_collision_body_offset = human.get_ee_collision_shape_pos_orient(end_effector, ee_collision_radius)
            p.resetBasePositionAndOrientation(ee_collision_body, ee_collision_body_pos,ee_collision_body_offset, physicsClientId=env.id)
        angle_dist = 0
    cost, m, dist, energy, torque, reba = cost_fn(human, end_effector, optimizer.best.x, original_ee_pos, original_info,
                                            max_dynamics, has_self_collision, has_env_collision, has_valid_robot_ik, angle_dist, 
                                            handover_obj_config, robot_ik_mode=robot_ik)
    LOG.info(
        f"{bcolors.OKBLUE} Best cost: {cost}, dist: {dist}, manipulibility: {m}, energy: {energy}, torque: {torque}{bcolors.ENDC}")
    action = {
        "solution": optimizer.best.x,
        "cost": cost,
        "m": m,
        "dist": dist,
        "mean_energy": mean_energy,
        "target": original_ee_pos,
        "mean_cost": mean_cost,
        "mean_dist": mean_dist,
        "mean_m": mean_m,
        "mean_evolution": mean_evolution,
        "mean_torque": mean_torque,
        "mean_reba": mean_reba
    }
    actions.append(action)
    # plot_cmaes_metrics(mean_cost, mean_dist, mean_m, mean_energy, mean_torque)
    # plot_mean_evolution(mean_evolution)

    env.disconnect()

    # save action to replay
    # print("actions: ", len(actions))
    save_dir = get_save_dir(save_dir, env_name, smpl_file)
    os.makedirs(save_dir, exist_ok=True)
    pickle.dump(actions, open(os.path.join(save_dir, "actions.pkl"), "wb"))

    print("training time (s): ", time.time() - start_time)
    return env, actions


def init_optimizer(x0, sigma, lower_bounds, upper_bounds):
    opts = {}
    opts['tolfun'] = 1e-3
    opts['tolx'] = 1e-3

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


def init_optimizer2(x0, sigma, lower_bounds, upper_bounds): # for cma library
    # opts = {}
    # opts['tolfun'] = 1e-9
    # opts['tolx'] = 1e-9
    bounds = [[l, u] for l, u in zip(lower_bounds, upper_bounds)]
    bounds = np.array(bounds)
    # print ("bounds: ", bounds.shape, x0.shape, x0.size)
    print("bounds: ", bounds)
    print("x0: ", x0)
    for i in range(x0.size):
        if x0[i] < bounds[i][0]:
            x0[i] = bounds[i][0]
        if x0[i] > bounds[i][1]:
            x0[i] = bounds[i][1]
    es = CMA(x0, sigma, bounds=np.array(bounds))
    return es


def render(env_name, smpl_file, save_dir, end_effector, object):
    save_dir = get_save_dir(save_dir, env_name, smpl_file)
    actions = pickle.load(open(os.path.join(save_dir, "actions.pkl"), "rb"))
    env = make_env(env_name, coop=True, smpl_file=smpl_file, object_name=object)
    env.render()  # need to call reset after render
    env.reset()
    best_idx = 0

    env.human.reset_controllable_joints(end_effector)
    for (idx, action) in enumerate(actions):
        env.human.set_joint_angles(env.human.controllable_joint_indices, action["solution"])
        human_link_robot_collision = get_human_link_robot_collision(env.human, end_effector)
        # find_robot_ik_solution(env, "right_hand", human_link_robot_collision)
        time.sleep(2)

        if idx == best_idx:
            plot_cmaes_metrics(action['mean_cost'], action['mean_dist'], action['mean_m'], action['mean_energy'],
                               action['mean_torque'])
            plot_mean_evolution(action['mean_evolution'])
    # for i in range (1000):
    #     p.stepSimulation(env.id)
    # time.sleep(100)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RL for Assistive Gym')
    # env
    parser.add_argument('--env', default='ScratchItchJaco-v0',
                        help='Environment to train.py on (default: ScratchItchJaco-v0)')
    parser.add_argument('--seed', type=int, default=1,
                        help='Random seed (default: 1)')
    # mode
    parser.add_argument('--train', action='store_true', default=False,
                        help='Whether to train.py a new policy')
    parser.add_argument('--render', action='store_true', default=False,
                        help='Whether to render a single rollout of a trained policy')
    parser.add_argument('--evaluate', action='store_true', default=False,
                        help='Whether to evaluate a trained policy over n_episodes')
    # train details
    parser.add_argument('--num-points', type=int, default=100, help="Number of points to sample")
    parser.add_argument('--smpl-file', default='examples/data/smpl_bp_ros_smpl_re2.pkl', help='smpl or smplx')
    parser.add_argument('--save-dir', default='./trained_models/',
                        help='Directory to save trained policy in (default ./trained_models/)')
    parser.add_argument('--render-gui', action='store_true', default=False,
                        help='Whether to render during training')
    parser.add_argument('--end-effector', default='right_hand', help='end effector name')
    parser.add_argument("--simulate-collision", action='store_true', default=False, help="simulate collision")
    parser.add_argument("--robot-ik", action='store_true', default=False, help="solve robot ik during training")
    parser.add_argument("--handover-obj", default=None, help="define if the handover object is default, pill, bottle, or cane")

    # replay
    parser.add_argument('--load-policy-path', default='./trained_models/',
                        help='Path name to saved policy checkpoint (NOTE: Use this to continue training an existing policy, or to evaluate a trained policy)')
    # verbosity
    parser.add_argument('--verbose', action='store_true', default=False,
                        help='Whether to output more verbose prints')
    args = parser.parse_args()

    coop = ('Human' in args.env)
    checkpoint_path = None

    if args.train:
        _, actions = train(args.env, args.seed, args.num_points, args.smpl_file, args.end_effector, args.save_dir,
                           args.render_gui, args.simulate_collision, args.robot_ik, args.handover_obj)

    if args.render:
        render(args.env, args.smpl_file, args.save_dir, args.end_effector, args.handover_obj)
