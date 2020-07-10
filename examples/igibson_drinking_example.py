import pybullet as p
import numpy as np
from assistive_gym.envs.env import AssistiveEnv
from assistive_gym.envs.agents.pr2 import PR2
from assistive_gym.envs.agents.agent import Agent
from assistive_gym.envs.agents.tool import Tool
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
chair.set_base_pos_orient([-2.8, 1.15, 0.36], [0, 0, -np.pi/6.0])
p.removeBody(4, physicsClientId=env.id)

# Create human
human = env.create_human(controllable=False, controllable_joint_indices=h.head_joints, fixed_base=False, human_impairment='none', gender='random', mass=None, radius_scale=1.0, height_scale=1.0)
joints_positions = [(human.j_right_elbow, -90), (human.j_left_elbow, -90), (human.j_right_hip_x, -80), (human.j_right_knee, 80), (human.j_left_hip_x, -80), (human.j_left_knee, 80)]
human.setup_joints(joints_positions, use_static_joints=False, reactive_force=10, reactive_gain=0.05)
# Set human base and increase body friction
human.set_base_pos_orient([-2.8, 1.125, 0.89 if human.gender == 'male' else 0.86], [-0.1, 0, -np.pi/3.5 - np.pi/2.0 - np.pi/6.0])
human.set_whole_body_frictions(lateral_friction=10, spinning_friction=10, rolling_friction=10)
# Stiffen the joints so they do not fall limp so easily
human.set_all_joints_stiffness(0.01)

# Create robot
robot = env.create_robot(PR2, controllable_joints='wheel_right', fixed_base=False)
# robot.print_joint_info()
pos = np.array([-2.25, 1.25, 0])
orient = np.array([0, 0, np.pi])
robot.set_base_pos_orient(pos, orient)
# Position robot end effectors / arms
robot.ik_random_restarts(right=True, target_pos=pos + np.array([-0.4, 0.55, 0.85]), target_orient=orient, step_sim=False, check_env_collisions=False)
robot.ik_random_restarts(right=False, target_pos=pos + np.array([-0.2, -0.05, 0.6]), target_orient=orient + np.array([0, 0, -np.pi/2.0]), step_sim=False, check_env_collisions=False)
robot.set_gripper_open_position(robot.right_gripper_indices, [0.5]*4, set_instantly=True)
# Increase friction of the gripper for grasping objects
robot.set_friction([57, 58, 59, 60, 61], 10)

# Create a cup on the table
cup = Tool()
cup.init(None, 'drinking', env.directory, env.id, env.np_random, mesh_scale=[0.035, 0.05, 0.035], alpha=0.75)
cup.set_base_pos_orient([-2.8, 1.8, 0.75], [np.pi/2.0, 0, 0])
cup.set_gravity(0, 0, 0)
# Generate water
cup_pos, cup_orient = cup.get_base_pos_orient()
water_radius = 0.005
water_mass = 0.001
batch_positions = []
for i in range(4):
    for j in range(4):
        for k in range(4):
            batch_positions.append(np.array([i*2*water_radius-0.02, j*2*water_radius-0.02, k*2*water_radius+0.075]) + cup_pos)
waters = env.create_spheres(radius=water_radius, mass=water_mass, batch_positions=batch_positions, visual=False, collision=True)
for w in waters:
    p.changeVisualShape(w.body, -1, rgbaColor=[0.25, 0.5, 1, 1], physicsClientId=env.id)
    w.set_friction(w.base, 0)

p.setPhysicsEngineParameter(numSubSteps=4, numSolverIterations=10, physicsClientId=env.id)

# Overwrite update function to update mouth location point
# def update_targets():
#     # Update position of mouth point
#     head_pos, head_orient = human.get_pos_orient(human.head)
#     target_pos, target_orient = p.multiplyTransforms(head_pos, head_orient, mouth_pos, [0, 0, 0, 1], physicsClientId=env.id)
#     target.set_base_pos_orient(target_pos, [0, 0, 0, 1])
# env.update_targets = update_targets

# Set target on mouth
mouth_pos = [0, -0.11, 0.03] if human.gender == 'male' else [0, -0.1, 0.03]
head_pos, head_orient = human.get_pos_orient(human.head)
target_pos, target_orient = p.multiplyTransforms(head_pos, head_orient, mouth_pos, [0, 0, 0, 1], physicsClientId=env.id)
# target = env.create_sphere(radius=0.01, mass=0.0, pos=target_pos, collision=False, rgba=[0, 1, 0, 1])

target_joint_angles_cup = robot.ik(robot.right_end_effector, pos + np.array([-0.58, 0.55, 0.8]), orient, ik_indices=robot.right_arm_ik_indices, max_iterations=200)
target_joint_angles_lift = robot.ik(robot.right_end_effector, pos + np.array([-0.58, 0.55, 1.0]), orient, ik_indices=robot.right_arm_ik_indices, max_iterations=200)
mouth_offset = np.array([0, 0.09, 0.02])
for iteration in range(200):
    print(iteration)
    # Define robot action
    actions = np.zeros(len(robot.controllable_joint_indices))
    current_joint_angles = robot.get_joint_angles(robot.right_arm_joint_indices)
    head_pos, head_orient = human.get_pos_orient(human.head)
    target_pos, target_orient = p.multiplyTransforms(head_pos, head_orient, mouth_pos, [0, 0, 0, 1], physicsClientId=env.id)
    gains = 0.05
    if iteration < 20:
        actions[len(robot.wheel_joint_indices):] = target_joint_angles_cup - current_joint_angles
    elif iteration < 25:
        robot.set_gripper_open_position(robot.right_gripper_indices, [0.4]*4, set_instantly=False)
    elif iteration < 40:
        actions[len(robot.wheel_joint_indices):] = target_joint_angles_lift - current_joint_angles
    elif iteration < 100:
        target_joint_angles_near_mouth = robot.ik(robot.right_end_effector, target_pos + mouth_offset, orient, ik_indices=robot.right_arm_ik_indices, max_iterations=200)
        actions[len(robot.wheel_joint_indices):] = target_joint_angles_near_mouth - current_joint_angles
        gains = 0.01
    else:
        target_joint_angles_tilt = robot.ik(robot.right_end_effector, target_pos + mouth_offset, orient + np.array([-np.pi/1.8, 0, 0]), ik_indices=robot.right_arm_ik_indices, max_iterations=200)
        actions[len(robot.wheel_joint_indices):] = target_joint_angles_tilt - current_joint_angles
        gains = 0.01
        # Remove water particles that enter the mouth
        waters_to_remove = []
        for w in waters:
            water_pos, water_orient = w.get_base_pos_orient()
            distance_to_mouth = np.linalg.norm(target_pos - water_pos)
            if distance_to_mouth < 0.04:
                waters_to_remove.append(w)
                w.set_base_pos_orient(env.np_random.uniform(1000, 2000, size=3), [0, 0, 0, 1])
        waters = [w for w in waters if w not in waters_to_remove]
    # This is a hack to keep the end effector closed
    if iteration >= 25:
        robot.set_gripper_open_position(robot.right_gripper_indices, [0.4]*4, set_instantly=True)
    env.take_step(actions, gains=gains, forces=50, action_multiplier=0.05)

p.disconnect(env.id)

