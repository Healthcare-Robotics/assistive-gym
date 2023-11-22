import pickle
import time

import argparse
import gym
import importlib

from assistive_gym.envs.utils.dto import RobotSetting
from assistive_gym.envs.utils.log_utils import get_logger
from assistive_gym.envs.utils.point_utils import eulidean_distance
from assistive_gym.envs.utils.train_utils import *
from experimental.urdf_name_resolver import get_urdf_filepath, get_urdf_folderpath

LOG = get_logger()
MAX_ITERATIONS = 100
def train(env_name, seed=0, smpl_file='examples/data/smpl_bp_ros_smpl_re2.pkl', person_id='p001',
          end_effector='right_hand', save_dir='./trained_models/', render=False, simulate_collision=False,
          robot_ik=False, handover_obj=None):
    start_time = time.time()
    # init
    env = make_env(env_name, person_id, smpl_file, handover_obj, coop=True)
    print("person_id: ", person_id, smpl_file)
    if render:
        env.render()
    env.reset()
    p.addUserDebugText("person: {}, smpl: {}".format(person_id, smpl_file), [0, 0, 1], textColorRGB=[1, 0, 0])

    human, robot, furniture, plane = env.human, env.robot, env.furniture, env.plane

    # choose end effector
    handover_obj_config = get_handover_object_config(handover_obj, env)
    if handover_obj_config and handover_obj_config.end_effector:  # reset the end effector based on the object
        human.reset_controllable_joints(handover_obj_config.end_effector)
        end_effector = handover_obj_config.end_effector

    # init collision check
    env_object_ids = [furniture.body, plane.body]  # set env object for collision check
    human_link_robot_collision = get_human_link_robot_collision(human, end_effector)

    # init original info and max dynamics
    original_info = build_original_human_info(human, env_object_ids, end_effector)
    max_dynamics = build_max_human_dynamics(env, end_effector, original_info)

    # draw original ee pos
    original_ee_pos = human.get_pos_orient(human.human_dict.get_dammy_joint_id(end_effector), center_of_mass=True)[0]
    draw_point(original_ee_pos, size=0.01, color=[0, 1, 0, 1])

    timestep = 0
    mean_cost, mean_dist, mean_m, mean_energy, mean_torque, mean_evolution, mean_reba = [], [], [], [], [], [], []

    # init optimizer
    x0 = np.array(original_info.angles)
    optimizer = init_optimizer(x0, 0.1, human.controllable_joint_lower_limits, human.controllable_joint_upper_limits)

    if not robot_ik:  # simulate collision
        ee_link_idx = human.human_dict.get_dammy_joint_id(end_effector)
        ee_collision_radius = COLLISION_OBJECT_RADIUS[handover_obj]  # 20cm range
        ee_collision_body = human.add_collision_object_around_link(ee_link_idx,
                                                                   radius=ee_collision_radius)  # TODO: ignore collision with hand

    smpl_name = os.path.basename(smpl_file)
    p.addUserDebugText("person: {}, smpl: {}".format(person_id, smpl_name), [0, 0, 1], textColorRGB=[1, 0, 0])

    while timestep < MAX_ITERATIONS and not optimizer.stop():
        timestep += 1
        solutions = optimizer.ask()
        best_cost, best_angle, best_robot_setting = float('inf'), None, None
        fitness_values, dists, manipus, energy_changes, torques = [], [], [], [], []
        for s in solutions:

            if simulate_collision:
                # step forward env
                angle_dist, _, env_collisions, _ = step_forward(env, s, env_object_ids, end_effector)
                self_collisions = human.check_self_collision()
                new_self_collision, new_env_collision = detect_collisions(original_info, self_collisions,
                                                                          env_collisions, human, end_effector)
                # cal dist to bedside
                dist_to_bedside = cal_dist_to_bedside(env, end_effector)
                cost, m, dist, energy, torque = cost_func(human, end_effector, s, original_ee_pos, original_info,
                                                          max_dynamics, new_self_collision, new_env_collision,
                                                          has_valid_robot_ik,
                                                          angle_dist, handover_obj_config, robot_ik, dist_to_bedside)
                env.reset_human(is_collision=True)
                LOG.info(
                    f"{bcolors.OKGREEN}timestep: {timestep}, cost: {cost}, angle_dist: {angle_dist} , dist: {dist}, manipulibility: {m}, energy: {energy}, torque: {torque}{bcolors.ENDC}")
            else:
                # set angle directly
                human.set_joint_angles(human.controllable_joint_indices, s)  # force set joint angle

                # check collision
                env_collisions, self_collisions = human.check_env_collision(
                    env_object_ids), human.check_self_collision()
                new_self_collision, new_env_collision = detect_collisions(original_info, self_collisions,
                                                                          env_collisions, human, end_effector)
                # move_robot(env)
                # cal dist to bedside
                dist_to_bedside = cal_dist_to_bedside(env, end_effector)
                if robot_ik:  # solve robot ik when doing training
                    has_valid_robot_ik, robot_joint_angles, robot_base_pos, robot_base_orient, robot_side, robot_penetrations, robot_dist_to_target, gripper_orient = find_robot_ik_solution(
                        env, end_effector, handover_obj)
                else:
                    ee_collision_body_pos, ee_collision_body_orient = human.get_ee_collision_shape_pos_orient(
                        end_effector, ee_collision_radius)
                    p.resetBasePositionAndOrientation(ee_collision_body, ee_collision_body_pos,
                                                      ee_collision_body_orient, physicsClientId=env.id)
                    has_valid_robot_ik = True

                cost, m, dist, energy, torque, reba = cost_func(human, end_effector, s, original_ee_pos, original_info,
                                                                max_dynamics, new_self_collision, new_env_collision,
                                                                has_valid_robot_ik, robot_penetrations,
                                                                robot_dist_to_target,
                                                                0, handover_obj_config, robot_ik, dist_to_bedside)
                LOG.info(
                    f"{bcolors.OKGREEN}timestep: {timestep}, cost: {cost}, dist: {dist}, manipulibility: {m}, energy: {energy}, torque: {torque}{bcolors.ENDC}")
                if cost < best_cost:
                    best_cost = cost
                    best_angle = s
                    best_robot_setting = RobotSetting(robot_base_pos, robot_base_orient, robot_joint_angles, robot_side,
                                                      gripper_orient)
                # restore joint angle
                # human.set_joint_angles(human.controllable_joint_indices, original_info.angles)

                # robot_ee = robot.get_pos_orient(robot.right_end_effector, center_of_mass=True)
                # robot_ee_transform = translate_wrt_human_pelvis(human, robot_ee[0], robot_ee[1])
                #
                # add_debug_line_wrt_parent_frame(robot_ee_transform[0], robot_ee_transform[1], human.body,
                #                           human.human_dict.get_fixed_joint_id("pelvis"))

            fitness_values.append(cost)
            dists.append(dist)
            manipus.append(m)
            energy_changes.append(energy)
            torques.append(torque)

        optimizer.tell(solutions, fitness_values)

        mean_evolution.append(np.mean(solutions, axis=0))
        mean_cost.append(np.mean(fitness_values, axis=0))
        mean_dist.append(np.mean(dists, axis=0))
        mean_m.append(np.mean(manipus, axis=0))
        mean_energy.append(np.mean(energy_changes, axis=0))
        mean_torque.append(np.mean(torques, axis=0))

    LOG.info(
        f"{bcolors.OKBLUE} Best cost: {best_cost}, best cost 2: {optimizer.best.f}, best angle: {best_angle}, best angle2: {optimizer.best.x}, best robot setting: {best_robot_setting}{bcolors.ENDC}, ")
    human.set_joint_angles(env.human.controllable_joint_indices, best_angle)

    robot.set_base_pos_orient(best_robot_setting.base_pos, best_robot_setting.base_orient)
    env.robot.set_joint_angles(
        env.robot.right_arm_joint_indices if best_robot_setting.robot_side == 'right' else env.robot.left_arm_joint_indices,
        best_robot_setting.robot_joint_angles)
    env.tool.reset_pos_orient()
    ee_pos, ik_target_pos = find_ee_ik_goal(human, end_effector, handover_obj)

    action = {
        "solution": optimizer.best.x,
        "cost": cost,
        "end_effector": end_effector,
        "m": m,
        "dist": dist,
        "mean_energy": mean_energy,
        "target": original_ee_pos,
        "mean_cost": mean_cost,
        "mean_dist": mean_dist,
        "mean_m": mean_m,
        "mean_evolution": mean_evolution,
        "mean_torque": mean_torque,
        "mean_reba": mean_reba,
        "robot_settings": {
            "joint_angles": robot_joint_angles,
            "base_pos": robot_base_pos,
            "base_orient": robot_base_orient,
            "side": robot_side
        },
        "wrt_pelvis": {
            'pelvis': human.get_pos_orient(human.human_dict.get_fixed_joint_id("pelvis"), center_of_mass=True),
            "ee": {
                'original': human.get_ee_pos_orient(end_effector),
                'transform': translate_wrt_human_pelvis(human, np.array(human.get_ee_pos_orient(end_effector)[0]),
                                                        np.array(human.get_ee_pos_orient(end_effector)[1])),
            },
            "ik_target": {
                'original': [np.array(ik_target_pos), np.array(gripper_orient)],  # [pos, orient
                'transform': translate_wrt_human_pelvis(human, np.array(ik_target_pos), np.array(gripper_orient)),
            },
            'robot': {
                'original': [np.array(robot_base_pos), np.array(robot_base_orient)],
                'transform': translate_wrt_human_pelvis(human, np.array(robot_base_pos), np.array(robot_base_orient)),
            },
            'robot_joint_angles': robot_joint_angles
        }
    }

    actions = {}
    key = get_actions_dict_key(handover_obj, robot_ik)
    actions[key] = action
    # plot_cmaes_metrics(mean_cost, mean_dist, mean_m, mean_energy, mean_torque)
    # plot_mean_evolution(mean_evolution)

    save_train_result(save_dir, env_name, person_id, smpl_file, actions)

    print("training time (s): ", time.time() - start_time)
    env.disconnect()
    return env, actions, action




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RL for Assistive Gym')
    # env
    parser.add_argument('--env', default='',
                        help='Environment to train.py on (default: HumanComfort-v1)')
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
    parser.add_argument('--smpl-file', default='examples/data/slp3d/p002/s01.pkl', help='smpl file')
    parser.add_argument('--person-id', default='p002', help='person id')
    parser.add_argument('--save-dir', default='./trained_models/',
                        help='Directory to save trained policy in (default ./trained_models/)')
    parser.add_argument('--render-gui', action='store_true', default=False,
                        help='Whether to render during training')
    parser.add_argument('--end-effector', default='right_hand', help='end effector name')
    parser.add_argument("--simulate-collision", action='store_true', default=False, help="simulate collision")
    parser.add_argument("--robot-ik", action='store_true', default=False, help="solve robot ik during training")
    parser.add_argument("--handover-obj", default=None,
                        help="define if the handover object is default, pill, bottle, or cane")

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
        if args.handover_obj == 'all':  # train all at once
            handover_objs = ['pill', 'cup', 'cane']
            for handover_obj in handover_objs:
                train(args.env, args.seed, args.smpl_file, args.person_id, args.end_effector, args.save_dir,
                      args.render_gui, args.simulate_collision, args.robot_ik, handover_obj)
        else:
            _, actions = train(args.env, args.seed, args.smpl_file, args.person_id, args.end_effector, args.save_dir,
                               args.render_gui, args.simulate_collision, args.robot_ik, args.handover_obj)

    if args.render:
        render(args.env, args.person_id, args.smpl_file, args.save_dir, args.handover_obj, args.robot_ik)