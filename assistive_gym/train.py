import math
import os, sys, multiprocessing, gym, ray, shutil, argparse, importlib, glob
import pickle
import time

import numpy as np
from cma import CMA, CMAEvolutionStrategy
from numpngw import write_apng
import pybullet as p
from matplotlib import pyplot as plt

from assistive_gym.envs.utils.point_utils import fibonacci_evenly_sampling_range_sphere, eulidean_distance


def inverse_dynamic(human):
    pos = []
    default_vel = 0.1
    for j in human.all_joint_indices:
        if p.getJointInfo(human.body, j, physicsClientId=human.id)[2] != p.JOINT_FIXED:
            joint_state = p.getJointState(human.body, j)
            pos.append(joint_state[0])

    # need to pass flags=1 to overcome inverseDynamics error for floating base
    # see https://github.com/bulletphysics/bullet3/issues/3188
    ivd = p.calculateInverseDynamics(human.body, objPositions= pos, objVelocities=[default_vel]*len(pos),
                                                 objAccelerations=[0] * len(pos), physicsClientId=human.id, flags=1)
    print ("inverse_dynamic: ", ivd, np.array(ivd).sum())
    return ivd


def draw_point(point, size=0.01, color=[1, 0, 0, 1]):
    sphere = p.createVisualShape(p.GEOM_SPHERE, radius=size, rgbaColor=color)
    multiBody = p.createMultiBody(baseMass=0,
                                  baseVisualShapeIndex=sphere,
                                  basePosition=np.array(point))
    p.setGravity(0, 0, 0)


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
    # print ("ik_joint_indices: ", ik_joint_indices)
    solution = human.ik(ee_idx, target_pos, None, ik_joint_indices,  max_iterations=1000)  # TODO: Fix index
    # print ("ik solution: ", solution)
    return solution


def get_link_positions(human):
    link_pos = []
    for i in range(-1, p.getNumJoints(human.body)): # include base
        pos, orient = human.get_pos_orient(i)
        link_pos.append(pos)
    return link_pos

def cal_energy_change(env, original_link_positions, current_link_positions):
    g = -9.81  # gravitational acceleration
    human_id = env.human.body
    total_energy_change = 0

    # Get the number of joints
    num_joints = p.getNumJoints(human_id)

    # Iterate over all links
    for i in range(-1, num_joints):

        # Get link state
        if i == -1:
            # The base case
            # state = p.getBasePositionAndOrientation(human_id)
            # velocity = p.getBaseVelocity(human_id)
            mass = p.getDynamicsInfo(human_id, -1)[0]
        else:
            # state = p.getLinkState(human_id, i)
            # velocity = p.getLinkState(human_id, i, computeLinkVelocity=1)[6:8]
            mass = p.getDynamicsInfo(human_id, i)[0]
        # Calculate initial potential energy
        potential_energy_initial = mass * g * original_link_positions[i][2] # z axis
        potential_energy_final = mass * g * current_link_positions[i][2]
        # Add changes to the total energy change
        total_energy_change += potential_energy_final - potential_energy_initial
    print(f"Total energy change: {total_energy_change}")

    return total_energy_change

def cost_fn(env, solution, target_pos, end_effector="right_hand", is_self_collision = False, is_env_collision= False,  energy_change = 0):
    human = env.human

    real_pos = p.getLinkState(human.body, human.human_dict.get_dammy_joint_id(end_effector))[0]
    dist = eulidean_distance(real_pos, target_pos)
    m = human.cal_chain_manipulibility(solution, end_effector)

    cost = dist + 1.0/m + np.abs(energy_change)/10.0
    if is_self_collision:
        cost+=10
    if is_env_collision:
        cost+=10
    print("euclidean distance: ", dist, "manipubility: ", m, "cost: ", cost)

    return cost, m, dist

def generate_target_points(env, num_points):
    # init points
    # human_pos = p.getBasePositionAndOrientation(env.human.body, env.human.id)[0]
    # points = uniform_sample(human_pos, 0.5, 20)
    human = env.human
    right_hand_pos = p.getLinkState(human.body, human.human_dict.get_dammy_joint_id("right_hand"))[0]
    # points = uniform_sample(right_hand_pos, 0.5, num_points)
    points = fibonacci_evenly_sampling_range_sphere(right_hand_pos, [0.25, 0.5], 50)
    # points = fibonacci_sphere(right_hand_pos, 0.5, num_points)
    return points


def get_initial_guess(env, target=None):
    if target is None:
        return np.zeros(len(env.human.controllable_joint_indices))  # no of joints
    else:
        # x0 = env.human.ik_chain(target)
        x0 = solve_ik(env, target, end_effector="right_hand")
        print("x0: ", x0)
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

def plot(vals, title, xlabel, ylabel):
    plt.figure()
    plt.plot(vals)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()


def plot_CMAES_metrics(mean_cost, mean_dist, mean_m):
    # Plot the fitness values
    plot(mean_cost, "Cost Function", "Iteration", "Cost")

    # Plot the distance values
    plot(mean_dist, "Distance Values", "Iteration", "Distance")

    # Plot the manipubility values
    plot(mean_m, "Manipubility Values", "Iteration", "Manipubility")

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

def plot_mean_evolution(mean_evolution):
    # Plot the mean vector evolution
    mean_evolution = np.array(mean_evolution)
    plt.figure()
    for i in range(mean_evolution.shape[1]):
        plt.plot(mean_evolution[:, i], label=f"Dimension {i + 1}")
    plt.xlabel("Iteration")
    plt.ylabel("Mean Value")
    plt.title("Mean Vector Evolution")
    plt.legend()
    plt.show()


def step_forward(env, x0):
    p.setJointMotorControlArray(env.human.body, jointIndices=env.human.controllable_joint_indices, controlMode=p.POSITION_CONTROL,
                                forces=[1000] * len(env.human.controllable_joint_indices),
                                positionGains = [0.01] * len(env.human.controllable_joint_indices),
                                targetPositions=x0,
                                physicsClientId=env.human.id)
    # for _ in range(5):
    #     p.stepSimulation(physicsClientId=env.human.id)
    p.setRealTimeSimulation(1)


def cal_cost(env, solution, target,  original_self_collisions, original_collisions, env_object_ids, original_link_positions):
    human = env.human
    cur_link_positions = get_link_positions(human)
    # inverse_dynamic(human)

    self_collisions = human.check_self_collision()
    is_self_collision = len(self_collisions) > len(original_self_collisions)
    env_collisions = human.check_env_collision(env_object_ids)
    is_env_collision = len(env_collisions) > len(original_collisions)
    print("self collision: ", is_self_collision, "env collision: ", env_collisions, "is env collision: ",
          is_env_collision)

    energy_change = cal_energy_change(env, original_link_positions, cur_link_positions)
    cost, m, dist = cost_fn(env, solution, target, is_self_collision=is_self_collision, is_env_collision=is_env_collision,
                            energy_change=energy_change)
    return cost, m, dist, energy_change


def get_save_dir(save_dir, env_name, smpl_file):
    smpl_name = smpl_file.split('/')[-1].split('.')[0]
    return os.path.join(save_dir, env_name, smpl_name)


# args.env, args.seed, args.num_points, args.smpl_file, args.save_dir, args.render_train
def train(env_name, seed=0,  num_points = 50, smpl_file = 'examples/data/smpl_bp_ros_smpl_re2.pkl', save_dir='./trained_models/', render=False):
    start_time = time.time()
    env = make_env(env_name, smpl_file, coop=True)

    if render:
        env.render()
    env.reset()

    # init points
    save_dir = get_save_dir(save_dir, env_name, smpl_file)
    os.makedirs(save_dir, exist_ok=True)

    points = generate_target_points(env, num_points)
    pickle.dump(points, open(os.path.join(save_dir, "points.pkl"), "wb"))

    actions = []
    best_action_idx = 0
    best_cost = 10 ^ 9
    cost = 0
    env_object_ids= [env.robot.body, env.furniture.body, env.plane.body] # set env object for collision check
    human = env.human
    for (idx, target) in enumerate(points):
        draw_point(target, size=0.01)
        # original value
        original_joint_angles = human.get_joint_angles(human.controllable_joint_indices)
        original_link_positions = get_link_positions(human)
        original_self_collisions = human.check_self_collision()
        original_collisions = human.check_env_collision(env_object_ids)

        x0 = get_initial_guess(env, None)
        optimizer = init_optimizer(x0, sigma=0.1)
        timestep = 0
        mean_cost, mean_dist, mean_m, mean_evolution  =[], [], [], []

        while not optimizer.stop():
            timestep += 1
            solutions = optimizer.ask()
            fitness_values, dists, manipus, energy_changes = [], [], [], []
            for s in solutions:
                human.set_joint_angles(human.controllable_joint_indices, s)  # force set joint angle
                cost, m, dist, energy_change = cal_cost(env, s, target, original_self_collisions, original_collisions, env_object_ids, original_link_positions)
                # restore joint angle
                # human.set_joint_angles(human.controllable_joint_indices, original_joint_angles)

                fitness_values.append(cost)
                dists.append(dist)
                manipus.append(m)
                energy_changes.append(energy_change)
            optimizer.tell(solutions, fitness_values)
            print("timestep: ", timestep, "cost: ", cost)
            optimizer.result_pretty()

            mean_evolution.append(np.mean(solutions, axis=0))
            mean_cost.append(np.mean(fitness_values, axis=0))
            mean_dist.append(np.mean(dists, axis=0))
            mean_m.append(np.mean(manipus, axis=0))

        # get the best solution value
        env.human.set_joint_angles(env.human.controllable_joint_indices, optimizer.best.x)

        cost, m, dist, energy_change = cal_cost(env, optimizer.best.x, target, original_self_collisions,
                                                original_collisions, env_object_ids, original_link_positions)
        action = {
            "solution": optimizer.best.x,
            "cost": cost,
            "m": m,
            "dist": dist,
            "energy_change": energy_change,
            "target": target,
            "mean_cost": mean_cost,
            "mean_dist": mean_dist,
            "mean_m": mean_m,
            "mean_evolution": mean_evolution
        }
        actions.append(action)
        # plot_CMAES_metrics(mean_cost, mean_dist, mean_m)
        # plot_mean_evolution(mean_evolution)

        if cost < best_cost:
            best_cost = cost
            best_action_idx = idx
    env.disconnect()

    #save action to replay
    print("actions: ", len(actions))
    pickle.dump(actions, open(os.path.join(save_dir, "actions.pkl"), "wb"))
    pickle.dump(best_action_idx, open(os.path.join(save_dir,"best_action_idx.pkl"), "wb"))

    print ("training time (s): ", time.time() - start_time)
    return env, actions


def init_optimizer(x0, sigma):
    opts = {}
    opts['tolfun']=  1e-4
    opts['tolx'] = 1e-4
    es = CMAEvolutionStrategy(x0, sigma, opts)
    return es


def render(env_name, smpl_file, save_dir):
    save_dir = get_save_dir(save_dir, env_name, smpl_file)
    actions = pickle.load(open(os.path.join(save_dir, "actions.pkl"), "rb"))
    env = make_env(env_name, coop=True, smpl_file=smpl_file)
    env.render()  # need to call reset after render
    env.reset()

    # init points
    points = pickle.load(open(os.path.join(save_dir, "points.pkl"), "rb"))
    print ("points: ", len(points))
    best_idx = pickle.load(open(os.path.join(save_dir,"best_action_idx.pkl"), "rb"))
    for (idx, point) in enumerate(points):
        # print(idx, point)
        if idx == best_idx:
            draw_point(point, color=[0, 0, 1, 1])
        else:
            draw_point(point)
    for (idx, action) in enumerate(actions):
        env.human.set_joint_angles(env.human.controllable_joint_indices, action["solution"])
        time.sleep(0.5)
        if idx == best_idx:
            plot_CMAES_metrics(action['mean_cost'], action['mean_dist'], action['mean_m'])
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
        _, actions = train(args.env, args.seed, args.num_points, args.smpl_file, args.save_dir, args.render_gui)

    if args.render:

        render(args.env, args.smpl_file, args.save_dir)
