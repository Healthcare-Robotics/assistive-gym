import pybullet as p
import numpy as np
from assistive_gym.envs.env import AssistiveEnv
from assistive_gym.envs.agents.pr2 import PR2
from assistive_gym.envs.agents.agent import Agent
import assistive_gym.envs.agents.human as h
from gibson2.core.physics.scene import BuildingScene

env = AssistiveEnv(render=True)
env.reset()

# Load iGibson environment
scene = BuildingScene('Placida', is_interactive=True, build_graph=True, pybullet_load_texture=True)
scene.load()

# Change position of a chair (the 7th object)
chair = Agent()
chair.init(7, env.id, env.np_random, indices=-1)
chair.set_base_pos_orient([-2, 1, 0.36], [0, 0, np.pi/2.0])

# Create human
human = env.create_human(controllable=False, controllable_joint_indices=h.right_arm_joints, fixed_base=False, human_impairment='none', gender='random', mass=None, radius_scale=1.0, height_scale=1.0)
joints_positions = [(human.j_right_elbow, -90), (human.j_left_elbow, -90), (human.j_right_hip_x, -80), (human.j_right_knee, 80), (human.j_left_hip_x, -80), (human.j_left_knee, 80)]
human.setup_joints(joints_positions, use_static_joints=False, reactive_force=None, reactive_gain=0.05)
# Set human base and increase body friction
human.set_base_pos_orient([-2, 1, 0.89 if human.gender == 'male' else 0.86], [-0.1, 0, -np.pi/3.5])
human.set_whole_body_frictions(lateral_friction=10, spinning_friction=10, rolling_friction=10)
# Stiffen the joints so they do not fall limp so easily
human.set_all_joints_stiffness(0.01)
# target_joint_angles = human.get_joint_angles(human.all_joint_indices)
# human.control(human.all_joint_indices, target_joint_angles, 0.05, 1)

# Create robot
robot = env.create_robot(PR2, controllable_joints='wheel_right', fixed_base=False)
# robot.print_joint_info()
pos = np.array([-1.5, 0.25, 0])
orient = np.array([0, 0, np.pi/1.5])
robot.set_base_pos_orient(pos, orient)
# Position robot end effectors / arms
robot.ik_random_restarts(right=True, target_pos=pos + np.array([-0.1, 0.25, 0.6]), target_orient=orient + np.array([0, 0, np.pi/2.0]), step_sim=False, check_env_collisions=False)
robot.ik_random_restarts(right=False, target_pos=pos + np.array([-0.15, 0.2, 0.6]), target_orient=orient + np.array([0, 0, -np.pi/2.0]), step_sim=False, check_env_collisions=False)

for _ in range(200):
    # Drive forward
    actions = np.zeros(len(robot.controllable_joint_indices))
    actions[[1, 2, 4, 5, 7, 8, 10, 11]] = 1.0
    env.take_step(actions, gains=0.05, forces=50, action_multiplier=0.05)

p.disconnect(env.id)

