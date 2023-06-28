import os, sys, multiprocessing, gym, ray, shutil, argparse, importlib, glob
import pickle
import time

from torch.utils.hipify.hipify_python import bcolors

from assistive_gym.envs.utils.log_utils import get_logger
from typing import Set

import numpy as np
from cma import CMA, CMAEvolutionStrategy
from cmaes import CMA
import pybullet as p

from assistive_gym.envs.utils.plot_utils import plot_cmaes_metrics, plot_mean_evolution
from assistive_gym.envs.utils.point_utils import fibonacci_evenly_sampling_range_sphere, eulidean_distance

LOG = get_logger()

class OriginalHumanInfo:
    def __init__(self, original_angles, original_link_positions, original_self_collisions, original_env_collisions):
        self.link_positions = original_link_positions
        self.angles = original_angles
        self.self_collisions = original_self_collisions
        self.env_collisions = original_env_collisions

class MaximumHumanDynamics:
    def __init__(self, max_torque, max_manipulibility, max_energy):
        self.torque = max_torque
        self.manipulibility = max_manipulibility
        self.energy = max_energy

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


def make_env(env_name, smpl_file, coop=False, seed=1001):
    if not coop:
        env = gym.make('assistive_gym:' + env_name)
    else:
        module = importlib.import_module('assistive_gym.envs')
        env_class = getattr(module, env_name.split('-')[0] + 'Env')
        env = env_class()
    env.seed(seed)
    env.set_smpl_file(smpl_file)
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
    right_hand_pos = p.getLinkState(human.body, human.human_dict.get_dammy_joint_id("right_hand"))[0]
    points = fibonacci_evenly_sampling_range_sphere(right_hand_pos, [0.25, 0.5], num_points)
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
                                                                                            env_collision):
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


def has_new_collision(old_collisions: Set, new_collisions: Set) -> bool:
    for collision in new_collisions:
        if collision not in old_collisions:
            return True
    return False


def cost_fn(human, ee_name:str, angle_config: np.ndarray, ee_target_pos: np.ndarray, original_info: OriginalHumanInfo,
            max_dynamics: MaximumHumanDynamics, env_collisions, angle_dist):
    # check collision
    has_new_self_collision = has_new_collision(original_info.self_collisions, human.check_self_collision())
    has_new_env_collision = has_new_collision(original_info.env_collisions, env_collisions)
    LOG.debug(f"self collision: {has_new_self_collision}, env collision: {has_new_env_collision}")

    # cal energy
    energy_change, energy_original, energy_final = cal_energy_change(human, original_info.link_positions, ee_name)

    # cal dist
    ee_real_pos = p.getLinkState(human.body, human.human_dict.get_dammy_joint_id(ee_name))[0]
    dist = eulidean_distance(ee_real_pos, ee_target_pos)

    # cal torque
    torque = cal_torque_magnitude(human, ee_name)

    # cal manipulibility
    manipulibility = human.cal_chain_manipulibility(angle_config, ee_name)

    # cal angle displacement from mid angle
    mid_angle = cal_mid_angle(human.controllable_joint_lower_limits, human.controllable_joint_upper_limits)
    mid_angle_displacement = cal_angle_diff(angle_config, mid_angle)
    print ("mid_angle_displacement: ", mid_angle_displacement)

    w = [1, 1, 3, 1, 2]
    # cost without simulate collision
    # cost = dist + 1.0/m + np.abs(energy_final)/1000.0
    # cost = 1.0/m + (energy_final-49)/5
    # cost = dist + 1 / manipulibility + energy_final / 100 + torque / 10
    cost = (w[0]*dist + w[1]* 1 / (manipulibility / max_dynamics.manipulibility) + w[2] * energy_final / max_dynamics.energy \
           + w[3]* torque / max_dynamics.torque + w[4] * mid_angle_displacement)/np.sum(w)

    # cost with simulate collision
    # cost = angle_dist + 1 / manipulibility + energy_final / 50 + torque / 10

    if has_new_self_collision:
        cost += 100
    if has_new_env_collision:
        cost += 100

    return cost, manipulibility, dist, energy_final, torque


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
    print ("max torque: ", torque)
    return torque


def max_torque_cost_fn(human, end_effector):
    torque = cal_torque_magnitude(human, end_effector)
    return 1.0/torque


def max_manipulibity_cost_fn(human, end_effector, joint_angles):
    manipulibility = human.cal_chain_manipulibility(joint_angles, end_effector)
    return 1.0/ manipulibility


def max_energy_cost_fn(human, end_effector, original_link_positions):
    _, _, energy_final = cal_energy_change(human, original_link_positions, end_effector)
    return 1.0/energy_final


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
    return optimizer.best.x, 1.0/optimizer.best.f


def train(env_name, seed=0, num_points=50, smpl_file='examples/data/smpl_bp_ros_smpl_re2.pkl',
          end_effector='right_hand', save_dir='./trained_models/', render=False, simulate_collision=False):
    start_time = time.time()
    env = make_env(env_name, smpl_file, coop=True)
    if render:
        env.render()
    env.reset()

    human = env.human
    env_object_ids = [env.robot.body, env.furniture.body, env.plane.body]  # set env object for collision check

    # original value
    original_joint_angles = human.get_joint_angles(human.controllable_joint_indices)
    original_link_positions = human.get_link_positions(True, end_effector_name=end_effector)
    original_self_collisions = human.check_self_collision()
    original_env_collisions = human.check_env_collision(env_object_ids)
    original_info = OriginalHumanInfo(original_joint_angles, original_link_positions, original_self_collisions,
                                        original_env_collisions)
    # draw original ee pos
    original_ee_pos = human.get_pos_orient(human.human_dict.get_dammy_joint_id(end_effector), center_of_mass=True)[0]
    draw_point(original_ee_pos, size=0.01, color=[0, 1, 0, 1])

    timestep = 0
    mean_cost, mean_dist, mean_m, mean_energy, mean_torque, mean_evolution = [], [], [], [], [], []
    actions = []

    _, max_torque = find_max_val(human, max_torque_cost_fn, original_joint_angles, original_link_positions, end_effector)
    _, max_manipubility = find_max_val(human, max_manipulibity_cost_fn, original_joint_angles, original_link_positions, end_effector)
    _, max_energy = find_max_val(human, max_energy_cost_fn, original_joint_angles, original_link_positions, end_effector)
    print ("max torque: ", max_torque, "max manipubility: ", max_manipubility, "max energy: ", max_energy)
    max_dynamics = MaximumHumanDynamics(max_torque, max_manipubility, max_energy)
    env.reset()

    # init optimizer
    x0 = np.array(original_joint_angles)
    optimizer = init_optimizer(x0, 0.1, human.controllable_joint_lower_limits, human.controllable_joint_upper_limits)

    while not optimizer.stop():
        timestep += 1
        solutions = optimizer.ask()
        fitness_values, dists, manipus, energy_changes, torques = [], [], [], [], []
        for s in solutions:
            if simulate_collision:
                # step forward env
                angle_dist, _, env_collisions, _ = step_forward(env, s, env_object_ids, end_effector)
                cost, m, dist, energy, torque = cost_fn(human, end_effector, s, original_ee_pos, original_info, max_dynamics, env_collisions, angle_dist)
                env.reset_human(is_collision=True)
                LOG.info(
                    f"{bcolors.OKGREEN}timestep: {timestep}, cost: {cost}, angle_dist: {angle_dist} , dist: {dist}, manipulibility: {m}, energy: {energy}, torque: {torque}{bcolors.ENDC}")
            else:
                # set angle directly
                human.set_joint_angles(human.controllable_joint_indices, s)  # force set joint angle
                env_collisions = human.check_env_collision(env_object_ids)
                cost, m, dist, energy, torque = cost_fn(human, end_effector,s, original_ee_pos,  original_info, max_dynamics, env_collisions, angle_dist=0)
                # restore joint angle
                human.set_joint_angles(human.controllable_joint_indices, original_joint_angles)
                LOG.info(
                    f"{bcolors.OKGREEN}timestep: {timestep}, cost: {cost}, dist: {dist}, manipulibility: {m}, energy: {energy}, torque: {torque}{bcolors.ENDC}")
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
        env_collisions = env.human.check_env_collision(env_object_ids)
        angle_dist = 0
    cost, m, dist, energy, torque = cost_fn(human, end_effector, optimizer.best.x, original_ee_pos,  original_info, max_dynamics, env_collisions, angle_dist)
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
        "mean_torque": mean_torque
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


def init_optimizer2(x0, sigma, lower_bounds, upper_bounds):
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


def render(env_name, smpl_file, save_dir):
    save_dir = get_save_dir(save_dir, env_name, smpl_file)
    actions = pickle.load(open(os.path.join(save_dir, "actions.pkl"), "rb"))
    env = make_env(env_name, coop=True, smpl_file=smpl_file)
    env.render()  # need to call reset after render
    env.reset()
    best_idx = 0

    for (idx, action) in enumerate(actions):
        env.human.set_joint_angles(env.human.controllable_joint_indices, action["solution"])
        time.sleep(0.5)
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
                           args.render_gui, args.simulate_collision)

    if args.render:
        render(args.env, args.smpl_file, args.save_dir)
