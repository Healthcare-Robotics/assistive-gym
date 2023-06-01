import os, sys, multiprocessing, gym, ray, shutil, argparse, importlib, glob
import pickle
import time

import numpy as np
from cma import CMA, CMAEvolutionStrategy
from numpngw import write_apng
import pybullet as p
from matplotlib import pyplot as plt

def uniform_sample(pos, radius, num_samples):
    """
    Sample points uniformly from the given space
    :param pos: (x, y, z)
    :return:
    """
    # pos = np.array(pos)
    # points = np.random.uniform(low=pos-radius, high=pos + radius, size=(num_samples, 3))
    points = []
    for i in range(num_samples):
        r = np.random.uniform(radius / 2, radius)
        theta = np.random.uniform(0, np.pi / 2)
        phi = np.random.uniform(0, np.pi / 2)  # Only sample from 0 to pi/2

        # Convert from spherical to cartesian coordinates
        dx = r * np.sin(phi) * np.cos(theta)
        dy = r * np.sin(phi) * np.sin(theta)
        dz = r * np.cos(phi)

        # Add to original point
        x_new = pos[0] + dx
        y_new = pos[1] + dy
        z_new = pos[2] + dz
        points.append([x_new, y_new, z_new])
    return points


def draw_point(point, size=0.01):
    sphere = p.createCollisionShape(p.GEOM_SPHERE, radius=size)
    multiBody = p.createMultiBody(baseMass=0,
                                  baseCollisionShapeIndex=sphere,
                                  basePosition=np.array(point))
    p.setGravity(0, 0, 0, multiBody)


def make_env(env_name, coop=False, seed=1001):
    if not coop:
        env = gym.make('assistive_gym:' + env_name)
    else:
        module = importlib.import_module('assistive_gym.envs')
        env_class = getattr(module, env_name.split('-')[0] + 'Env')
        env = env_class()
    env.seed(seed)
    return env


def solve_ik(env, target_pos, end_effector="right_hand"):
    human = env.human
    ee_idx = human.human_dict.get_dammy_joint_id(end_effector)
    ik_joint_indices = human.find_ik_joint_indices()
    # print ("ik_joint_indices: ", ik_joint_indices)
    solution = human.ik(ee_idx, target_pos, None, ik_joint_indices,  max_iterations=1000)  # TODO: Fix index
    # print ("ik solution: ", solution)
    return solution


def cost_fn(env, solution, target_pos, end_effector="right_hand"):
    human = env.human

    real_pos = p.getLinkState(human.body, human.human_dict.get_dammy_joint_id(end_effector))[0]
    dist = eulidean_distance(real_pos, target_pos)
    m = human.cal_manipulibility_chain(solution)
    # torque = human.cal_torque()
    cost = dist + 1/m
    print("euclidean distance: ", dist, "manipubility: ", m, "cost: ", cost)

    return cost, m, dist


def eulidean_distance(cur, target):
    print("current: ", cur, "target: ", target)
    # convert tuple to np array
    cur = np.array(cur)
    return np.sqrt(np.sum(np.square(cur - target)))

# for debugging
def get_single_target(ee_pos):
    point = np.array(list(ee_pos))
    point[1] -= 0.2
    point[0] += 0.2
    point[2] += 0.2
    return point


def generate_target_points(env, num_points=10):
    # init points
    # human_pos = p.getBasePositionAndOrientation(env.human.body, env.human.id)[0]
    # points = uniform_sample(human_pos, 0.5, 20)
    human = env.human
    right_hand_pos = p.getLinkState(human.body, human.human_dict.get_dammy_joint_id("right_hand"))[0]
    points = uniform_sample(right_hand_pos, 0.3, num_points)
    return points


def get_initial_guess(env, target=None):
    if target is None:
        return np.zeros(len(env.human.controllable_joint_indices))  # no of joints
    else:
        # x0 = env.human.ik_chain(target)
        x0 = solve_ik(env, target, end_effector="right_hand")
        # print("x0: ", x0)
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
    plot(mean_cost, "Fitness Values", "Iteration", "Fitness Value")

    # Plot the distance values
    plot(mean_dist, "Distance Values", "Iteration", "Distance Value")

    # Plot the manipubility values
    plot(mean_m, "Manipubility Values", "Iteration", "Manipubility Value")


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
                                forces=[10000] * len(env.human.controllable_joint_indices),
                                positionGains = [0.01] * len(env.human.controllable_joint_indices),
                                targetPositions=x0,
                                physicsClientId=env.human.id)
    # for _ in range(5):
    #     p.stepSimulation(physicsClientId=env.human.id)
    p.setRealTimeSimulation(1)


def train(env_name, algo, timesteps_total=10, save_dir='./trained_models/', load_policy_path='', coop=False, seed=0,
          extra_configs={}):
    env = make_env(env_name, coop=True)
    env.render()
    env.reset()

    # init points
    points = generate_target_points(env)
    pickle.dump(points, open("points.pkl", "wb"))

    actions = {}
    best_action_idx = 0
    best_cost = 10 ^ 9
    cost = 0

    human = env.human
    for (idx, target) in enumerate(points):
        original_joint_angles = human.get_joint_angles(human.controllable_joint_indices)
        draw_point(target, size=0.01)

        x0 = get_initial_guess(env, None)
        optimizer = init_optimizer(x0, sigma=0.1)

        timestep = 0
        mean_evolution = []
        dists = []
        manipus = []
        mean_cost = []
        mean_dist = []
        mean_m = []

        while not optimizer.stop():
            timestep += 1
            solutions = optimizer.ask()
            # print("solutions: ", solutions)
            fitness_values = []
            for s in solutions:
                human.set_joint_angles(human.controllable_joint_indices, s)  # force set joint angle

                cost, m, dist = cost_fn(env, s, target)
                # restore joint angle
                human.set_joint_angles(human.controllable_joint_indices, original_joint_angles)

                fitness_values.append(cost)
                dists.append(dist)
                manipus.append(m)
            optimizer.tell(solutions, fitness_values)

            # step forward
            # action = {'robot': env.action_space_robot.sample(), 'human': np.mean(solutions, axis=0)}
            # actions[idx].append(action)
            mean_evolution.append(np.mean(solutions, axis=0))

            # env.step(action)
            # cost = cost_function(env, action['human'], target)
            print("timestep: ", timestep, "cost: ", cost)
            # optimizer.disp()

            optimizer.result_pretty()
            mean_cost.append(np.mean(fitness_values))
            mean_dist.append(np.mean(dists))
            mean_m.append(np.mean(manipus))
        env.human.set_joint_angles(env.human.controllable_joint_indices, optimizer.best.x)
        actions[idx] = optimizer.best.x

        plot_CMAES_metrics(mean_cost, mean_dist, mean_m)
        plot_mean_evolution(mean_evolution)

        if cost < best_cost:
            best_cost = cost
            best_action_idx = idx

    env.disconnect()
    # save action to replay
    # print("actions: ", len(actions))
    # pickle.dump(actions, open("actions.pkl", "wb"))
    # pickle.dump(best_action_idx, open("best_action_idx.pkl", "wb"))

    return env, actions


def init_optimizer(x0, sigma):
    opts = {}
    opts['tolfun']=  1e-3
    opts['tolx'] = 1e-3
    es = CMAEvolutionStrategy(x0, sigma, opts)
    return es


def render(env, actions):
    # print("actions: ", actions)
    env.render()  # need to call reset after render
    env.reset()

    # init points
    points = pickle.load(open("points.pkl", "rb"))
    best_idx = pickle.load(open("best_action_idx.pkl", "rb"))
    for (idx, point) in enumerate(points):
        print(idx, point)

        if idx == best_idx:
            draw_point(point, size=0.05)
        else:
            draw_point(point)
    for a in actions[best_idx]:
        env.step(a)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RL for Assistive Gym')
    parser.add_argument('--env', default='ScratchItchJaco-v0',
                        help='Environment to train.py on (default: ScratchItchJaco-v0)')
    parser.add_argument('--algo', default='ppo',
                        help='Reinforcement learning algorithm')
    parser.add_argument('--seed', type=int, default=1,
                        help='Random seed (default: 1)')
    parser.add_argument('--train', action='store_true', default=False,
                        help='Whether to train.py a new policy')
    parser.add_argument('--render', action='store_true', default=False,
                        help='Whether to render a single rollout of a trained policy')
    parser.add_argument('--evaluate', action='store_true', default=False,
                        help='Whether to evaluate a trained policy over n_episodes')
    parser.add_argument('--train-timesteps', type=int, default=10,
                        help='Number of simulation timesteps to train.py a policy (default: 1000000)')
    parser.add_argument('--save-dir', default='./trained_models/',
                        help='Directory to save trained policy in (default ./trained_models/)')
    parser.add_argument('--load-policy-path', default='./trained_models/',
                        help='Path name to saved policy checkpoint (NOTE: Use this to continue training an existing policy, or to evaluate a trained policy)')
    parser.add_argument('--render-episodes', type=int, default=1,
                        help='Number of rendering episodes (default: 1)')
    parser.add_argument('--eval-episodes', type=int, default=100,
                        help='Number of evaluation episodes (default: 100)')
    parser.add_argument('--colab', action='store_true', default=False,
                        help='Whether rendering should generate an animated png rather than open a window (e.g. when using Google Colab)')
    parser.add_argument('--verbose', action='store_true', default=False,
                        help='Whether to output more verbose prints')
    args = parser.parse_args()

    coop = ('Human' in args.env)
    checkpoint_path = None

    if args.train:
        _, actions = train(args.env, args.algo, timesteps_total=args.train_timesteps, save_dir=args.save_dir,
                           load_policy_path=args.load_policy_path, coop=coop, seed=args.seed)

    if args.render:
        actions = pickle.load(open("actions.pkl", "rb"))
        env = make_env(args.env, coop=True)
        render(env, actions)
