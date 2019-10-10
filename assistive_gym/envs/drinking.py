import os
from gym import spaces
import numpy as np
import pybullet as p

from .env import AssistiveEnv

class DrinkingEnv(AssistiveEnv):
    def __init__(self, robot_type='pr2', human_control=False):
        super(DrinkingEnv, self).__init__(robot_type=robot_type, task='drinking', human_control=human_control, frame_skip=25, time_step=0.004, action_robot_len=7, action_human_len=(4 if human_control else 0), obs_robot_len=25, obs_human_len=(23 if human_control else 0))

    def step(self, action):
        self.take_step(action, robot_arm='right', gains=self.config('robot_gains'), forces=self.config('robot_forces'), human_gains=0.0005)

        robot_force_on_human, cup_force_on_human = self.get_total_force()
        total_force_on_human = robot_force_on_human + cup_force_on_human
        reward_water, water_mouth_velocities, water_hit_human_reward = self.get_water_rewards()
        end_effector_velocity = np.linalg.norm(p.getBaseVelocity(self.cup, physicsClientId=self.id)[0])
        obs = self._get_obs([cup_force_on_human], [robot_force_on_human, cup_force_on_human])

        # Get human preferences
        preferences_score = self.human_preferences(end_effector_velocity=end_effector_velocity, total_force_on_human=robot_force_on_human, tool_force_at_target=cup_force_on_human, food_hit_human_reward=water_hit_human_reward, food_mouth_velocities=water_mouth_velocities)

        cup_pos, cup_orient = p.getBasePositionAndOrientation(self.cup, physicsClientId=self.id)
        cup_pos, cup_orient = p.multiplyTransforms(cup_pos, cup_orient, [0, 0.06, 0], p.getQuaternionFromEuler([np.pi/2.0, 0, 0], physicsClientId=self.id), physicsClientId=self.id)
        cup_top_center_pos, _ = p.multiplyTransforms(cup_pos, cup_orient, self.cup_top_center_offset, [0, 0, 0, 1], physicsClientId=self.id)
        reward_distance = -np.linalg.norm(self.target_pos - np.array(cup_top_center_pos)) # Penalize distances between top of cup and mouth
        reward_action = -np.sum(np.square(action)) # Penalize actions
        # Encourage robot to have a tilted end effector / cup
        cup_euler = p.getEulerFromQuaternion(cup_orient, physicsClientId=self.id)
        reward_tilt = -abs(cup_euler[0] + np.pi/2) if self.robot_type == 'jaco' else -abs(cup_euler[0] - np.pi/2)

        reward = self.config('distance_weight')*reward_distance + self.config('action_weight')*reward_action + self.config('cup_tilt_weight')*reward_tilt + self.config('drinking_reward_weight')*reward_water + preferences_score

        if self.gui and reward_water != 0:
            print('Task success:', self.task_success, 'Water reward:', reward_water)

        info = {'total_force_on_human': total_force_on_human, 'task_success': int(self.task_success >= self.total_water_count*self.config('task_success_threshold')), 'action_robot_len': self.action_robot_len, 'action_human_len': self.action_human_len, 'obs_robot_len': self.obs_robot_len, 'obs_human_len': self.obs_human_len}
        done = False

        return obs, reward, done, info

    def get_total_force(self):
        robot_force_on_human = 0
        cup_force_on_human = 0
        for c in p.getContactPoints(bodyA=self.robot, bodyB=self.human, physicsClientId=self.id):
            robot_force_on_human += c[9]
        for c in p.getContactPoints(bodyA=self.cup, bodyB=self.human, physicsClientId=self.id):
            cup_force_on_human += c[9]
        return robot_force_on_human, cup_force_on_human

    def get_water_rewards(self):
        # Check all water particles to see if they have entered the person's mouth or have left the scene
        # Delete such particles and give the robot a reward or penalty depending on particle status
        cup_pos, cup_orient = p.getBasePositionAndOrientation(self.cup, physicsClientId=self.id)
        cup_pos, cup_orient = p.multiplyTransforms(cup_pos, cup_orient, [0, 0.06, 0], p.getQuaternionFromEuler([np.pi/2.0, 0, 0], physicsClientId=self.id), physicsClientId=self.id)
        top_center_pos, _ = p.multiplyTransforms(cup_pos, cup_orient, self.cup_top_center_offset, [0, 0, 0, 1], physicsClientId=self.id)
        bottom_center_pos, _ = p.multiplyTransforms(cup_pos, cup_orient, self.cup_bottom_center_offset, [0, 0, 0, 1], physicsClientId=self.id)
        top_center_pos = np.array(top_center_pos)
        bottom_center_pos = np.array(bottom_center_pos)
        if self.cup_top_center is not None:
            p.resetBasePositionAndOrientation(self.cup_top_center, top_center_pos, [0, 0, 0, 1], physicsClientId=self.id)
            p.resetBasePositionAndOrientation(self.cup_bottom_center, bottom_center_pos, [0, 0, 0, 1], physicsClientId=self.id)
            p.resetBasePositionAndOrientation(self.cup_cylinder, cup_pos, cup_orient, physicsClientId=self.id)
        water_reward = 0
        water_hit_human_reward = 0
        water_mouth_velocities = []
        waters_to_remove = []
        for w in self.waters:
            water_pos, water_orient = p.getBasePositionAndOrientation(w, physicsClientId=self.id)
            if not self.util.points_in_cylinder(top_center_pos, bottom_center_pos, 0.05, np.array(water_pos)):
                distance_to_mouth = np.linalg.norm(self.target_pos - water_pos)
                if distance_to_mouth < 0.03: # hard
                # if distance_to_mouth < 0.05: # easy
                    # Delete particle and give robot a reward
                    water_reward += 10
                    self.task_success += 1
                    p.resetBasePositionAndOrientation(w, self.np_random.uniform(1000, 2000, size=3), [0, 0, 0, 1], physicsClientId=self.id)
                    water_velocity = np.linalg.norm(p.getBaseVelocity(w, physicsClientId=self.id)[0])
                    water_mouth_velocities.append(water_velocity)
                    waters_to_remove.append(w)
                    continue
                elif water_pos[-1] < 0.5:
                    # Delete particle and give robot a penalty for spilling water
                    water_reward -= 1
                    waters_to_remove.append(w)
                    continue
                if len(p.getContactPoints(bodyA=w, bodyB=self.human, physicsClientId=self.id)) > 0:
                    # Record that this water particle just hit the person, so that we can penalize the robot
                    waters_to_remove.append(w)
                    water_hit_human_reward -= 1
        self.waters = [w for w in self.waters if w not in waters_to_remove]
        return water_reward, water_mouth_velocities, water_hit_human_reward

    def _get_obs(self, forces, forces_human):
        torso_pos = np.array(p.getLinkState(self.robot, 15 if self.robot_type == 'pr2' else 0, computeForwardKinematics=True, physicsClientId=self.id)[0])
        tool_pos, tool_orient = p.getBasePositionAndOrientation(self.cup, physicsClientId=self.id)
        robot_joint_states = p.getJointStates(self.robot, jointIndices=self.robot_right_arm_joint_indices, physicsClientId=self.id)
        robot_joint_positions = np.array([x[0] for x in robot_joint_states])
        robot_pos, robot_orient = p.getBasePositionAndOrientation(self.robot, physicsClientId=self.id)
        if self.human_control:
            human_pos = np.array(p.getBasePositionAndOrientation(self.human, physicsClientId=self.id)[0])
            human_joint_states = p.getJointStates(self.human, jointIndices=self.human_controllable_joint_indices, physicsClientId=self.id)
            human_joint_positions = np.array([x[0] for x in human_joint_states])

        head_pos, head_orient = p.getLinkState(self.human, 23, computeForwardKinematics=True, physicsClientId=self.id)[:2]

        robot_obs = np.concatenate([tool_pos-torso_pos, tool_orient, tool_pos - self.target_pos, robot_joint_positions, head_pos-torso_pos, head_orient, forces]).ravel()
        if self.human_control:
            human_obs = np.concatenate([tool_pos-human_pos, tool_orient, tool_pos - self.target_pos, human_joint_positions, head_pos-human_pos, head_orient, forces_human]).ravel()
        else:
            human_obs = []

        return np.concatenate([robot_obs, human_obs]).ravel()

    def reset(self):
        self.setup_timing()
        self.task_success = 0
        self.human, self.wheelchair, self.robot, self.robot_lower_limits, self.robot_upper_limits, self.human_lower_limits, self.human_upper_limits, self.robot_right_arm_joint_indices, self.robot_left_arm_joint_indices, self.gender = self.world_creation.create_new_world(furniture_type='wheelchair', static_human_base=True, human_impairment='random', print_joints=False, gender='random')
        self.robot_lower_limits = self.robot_lower_limits[self.robot_right_arm_joint_indices]
        self.robot_upper_limits = self.robot_upper_limits[self.robot_right_arm_joint_indices]
        self.reset_robot_joints()
        if self.robot_type == 'jaco':
            wheelchair_pos, wheelchair_orient = p.getBasePositionAndOrientation(self.wheelchair, physicsClientId=self.id)
            p.resetBasePositionAndOrientation(self.robot, np.array(wheelchair_pos) + np.array([-0.35, -0.3, 0.3]), p.getQuaternionFromEuler([0, 0, -np.pi/2.0], physicsClientId=self.id), physicsClientId=self.id)
            base_pos, base_orient = p.getBasePositionAndOrientation(self.robot, physicsClientId=self.id)

        joints_positions = [(6, np.deg2rad(-90)), (16, np.deg2rad(-90)), (28, np.deg2rad(-90)), (31, np.deg2rad(80)), (35, np.deg2rad(-90)), (38, np.deg2rad(80))]
        joints_positions += [(21, self.np_random.uniform(np.deg2rad(-30), np.deg2rad(30))), (22, self.np_random.uniform(np.deg2rad(-30), np.deg2rad(30))), (23, self.np_random.uniform(np.deg2rad(-30), np.deg2rad(30)))]
        self.human_controllable_joint_indices = [20, 21, 22, 23]
        self.world_creation.setup_human_joints(self.human, joints_positions, self.human_controllable_joint_indices if (self.human_control or self.world_creation.human_impairment == 'tremor') else [], use_static_joints=True, human_reactive_force=None)
        p.resetBasePositionAndOrientation(self.human, [0, 0.03, 0.89 if self.gender == 'male' else 0.86], [0, 0, 0, 1], physicsClientId=self.id)
        human_joint_states = p.getJointStates(self.human, jointIndices=self.human_controllable_joint_indices, physicsClientId=self.id)
        self.target_human_joint_positions = np.array([x[0] for x in human_joint_states])
        self.human_lower_limits = self.human_lower_limits[self.human_controllable_joint_indices]
        self.human_upper_limits = self.human_upper_limits[self.human_controllable_joint_indices]

        shoulder_pos, shoulder_orient = p.getLinkState(self.human, 5, computeForwardKinematics=True, physicsClientId=self.id)[:2]
        elbow_pos, elbow_orient = p.getLinkState(self.human, 7, computeForwardKinematics=True, physicsClientId=self.id)[:2]
        wrist_pos, wrist_orient = p.getLinkState(self.human, 9, computeForwardKinematics=True, physicsClientId=self.id)[:2]
        head_pos, head_orient = p.getLinkState(self.human, 23, computeForwardKinematics=True, physicsClientId=self.id)[:2]

        # Set target on mouth
        self.mouth_pos = [0, -0.11, 0.03] if self.gender == 'male' else [0, -0.1, 0.03]
        head_pos, head_orient = p.getLinkState(self.human, 23, computeForwardKinematics=True, physicsClientId=self.id)[:2]
        target_pos, target_orient = p.multiplyTransforms(head_pos, head_orient, self.mouth_pos, [0, 0, 0, 1], physicsClientId=self.id)
        self.target_pos = np.array(target_pos)
        sphere_collision = -1
        sphere_visual = p.createVisualShape(shapeType=p.GEOM_SPHERE, radius=0.01, rgbaColor=[0, 1, 0, 1], physicsClientId=self.id)
        self.target = p.createMultiBody(baseMass=0.0, baseCollisionShapeIndex=sphere_collision, baseVisualShapeIndex=sphere_visual, basePosition=self.target_pos, useMaximalCoordinates=False, physicsClientId=self.id)

        target_pos = np.array([-0.2, -0.5, 1]) + self.np_random.uniform(-0.05, 0.05, size=3)
        if self.robot_type == 'pr2':
            target_orient = p.getQuaternionFromEuler([0, 0, 0], physicsClientId=self.id)
            self.position_robot_toc(self.robot, 54, [(target_pos, target_orient), (self.target_pos, None)], [(self.target_pos, target_orient)], self.robot_right_arm_joint_indices, self.robot_lower_limits, self.robot_upper_limits, ik_indices=range(15, 15+7), pos_offset=np.array([0.2, 0.2, 0]), max_ik_iterations=200, step_sim=True, check_env_collisions=False, human_joint_indices=self.human_controllable_joint_indices, human_joint_positions=self.target_human_joint_positions)
            self.world_creation.set_gripper_open_position(self.robot, position=0.45, left=False, set_instantly=True)
            self.cup = self.world_creation.init_tool(self.robot, mesh_scale=[0.045]*3, pos_offset=[-0.01, 0, -0.05], orient_offset=p.getQuaternionFromEuler([np.pi/2.0, 0, 0], physicsClientId=self.id), left=False, maximal=False, alpha=0.75)
        elif self.robot_type == 'jaco':
            target_orient = p.getQuaternionFromEuler([0, np.pi/2.0, 0], physicsClientId=self.id)
            self.util.ik_random_restarts(self.robot, 8, target_pos, target_orient, self.world_creation, self.robot_right_arm_joint_indices, self.robot_lower_limits, self.robot_upper_limits, ik_indices=[0, 1, 2, 3, 4, 5, 6], max_iterations=1000, max_ik_random_restarts=40, random_restart_threshold=0.01, step_sim=True)
            self.world_creation.set_gripper_open_position(self.robot, position=0.63, left=False, set_instantly=True)
            self.cup = self.world_creation.init_tool(self.robot, mesh_scale=[0.045]*3, pos_offset=[0.05, -0.005, 0], orient_offset=p.getQuaternionFromEuler([0, 0, np.pi/2.0], physicsClientId=self.id), left=False, maximal=False, alpha=0.75)
        else:
            target_orient = p.getQuaternionFromEuler(np.array([0, -np.pi/2.0, np.pi]), physicsClientId=self.id)
            if self.robot_type == 'baxter':
                self.position_robot_toc(self.robot, 26, [(target_pos, target_orient), (self.target_pos, None)], [(self.target_pos, target_orient)], self.robot_right_arm_joint_indices, self.robot_lower_limits, self.robot_upper_limits, ik_indices=range(1, 8), pos_offset=np.array([0, 0.2, 0.975]), max_ik_iterations=200, step_sim=True, check_env_collisions=False, human_joint_indices=self.human_controllable_joint_indices, human_joint_positions=self.target_human_joint_positions)
            else:
                self.position_robot_toc(self.robot, 19, [(target_pos, target_orient), (self.target_pos, None)], [(self.target_pos, target_orient)], self.robot_right_arm_joint_indices, self.robot_lower_limits, self.robot_upper_limits, ik_indices=[0, 2, 3, 4, 5, 6, 7], pos_offset=np.array([-0.1, 0.2, 0.975]), max_ik_iterations=200, step_sim=True, check_env_collisions=False, human_joint_indices=self.human_controllable_joint_indices, human_joint_positions=self.target_human_joint_positions)
            self.world_creation.set_gripper_open_position(self.robot, position=0.025, left=False, set_instantly=True)
            self.cup = self.world_creation.init_tool(self.robot, mesh_scale=[0.045]*3, pos_offset=[0.05, 0.125, 0], orient_offset=p.getQuaternionFromEuler([0, 0, np.pi/2.0], physicsClientId=self.id), left=False, maximal=False, alpha=0.75)

        self.cup_top_center_offset = np.array([0, 0, -0.055])
        self.cup_bottom_center_offset = np.array([0, 0, 0.07])
        self.cup_top_center = None
        # self.display_cup_points()

        p.setGravity(0, 0, -9.81, physicsClientId=self.id)
        p.setGravity(0, 0, 0, body=self.robot, physicsClientId=self.id)
        p.setGravity(0, 0, 0, body=self.human, physicsClientId=self.id)

        # p.setPhysicsEngineParameter(contactBreakingThreshold=0.001, numSolverIterations=10, numSubSteps=2, physicsClientId=self.id)

        p.resetDebugVisualizerCamera(cameraDistance=1.10, cameraYaw=55, cameraPitch=-45, cameraTargetPosition=[-0.2, 0, 0.75], physicsClientId=self.id)

        # Generate water
        cup_pos, cup_orient = p.getBasePositionAndOrientation(self.cup, physicsClientId=self.id)
        cup_pos = np.array(cup_pos)
        water_radius = 0.005
        water_collision = p.createCollisionShape(p.GEOM_SPHERE, radius=water_radius, physicsClientId=self.id)
        water_visual = -1
        water_mass = 0.001
        water_count = 4*4*8
        water_count = 4*4*4
        batch_positions = []
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    batch_positions.append(np.array([i*2*water_radius-0.02, j*2*water_radius-0.02, k*2*water_radius+0.075]) + cup_pos)
        last_water_id = p.createMultiBody(baseMass=water_mass, baseCollisionShapeIndex=water_collision, baseVisualShapeIndex=water_visual, basePosition=[0, 0, 0], useMaximalCoordinates=False, batchPositions=batch_positions, physicsClientId=self.id)
        self.waters = list(range(last_water_id-water_count+1, last_water_id+1))
        for w in self.waters:
            p.changeVisualShape(w, -1, rgbaColor=[0.25, 0.5, 1, 1], physicsClientId=self.id)
        self.total_water_count = len(self.waters)

        # Enable rendering
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1, physicsClientId=self.id)

        # Drop water in the cup
        for _ in range(100):
            p.stepSimulation(physicsClientId=self.id)

        return self._get_obs([0], [0, 0])

    def display_cup_points(self):
        sphere_collision = -1
        sphere_visual = p.createVisualShape(shapeType=p.GEOM_SPHERE, radius=0.01, rgbaColor=[0, 1, 1, 1], physicsClientId=self.id)
        cup_pos, cup_orient = p.getBasePositionAndOrientation(self.cup, physicsClientId=self.id)
        cup_pos, cup_orient = p.multiplyTransforms(cup_pos, cup_orient, [0, 0.06, 0], p.getQuaternionFromEuler([np.pi/2.0, 0, 0], physicsClientId=self.id), physicsClientId=self.id)
        top_center_pos, _ = p.multiplyTransforms(cup_pos, cup_orient, self.cup_top_center_offset, [0, 0, 0, 1], physicsClientId=self.id)
        bottom_center_pos, _ = p.multiplyTransforms(cup_pos, cup_orient, self.cup_bottom_center_offset, [0, 0, 0, 1], physicsClientId=self.id)
        self.cup_top_center = p.createMultiBody(baseMass=0.0, baseCollisionShapeIndex=sphere_collision, baseVisualShapeIndex=sphere_visual, basePosition=top_center_pos, useMaximalCoordinates=False, physicsClientId=self.id)
        self.cup_bottom_center = p.createMultiBody(baseMass=0.0, baseCollisionShapeIndex=sphere_collision, baseVisualShapeIndex=sphere_visual, basePosition=bottom_center_pos, useMaximalCoordinates=False, physicsClientId=self.id)

        cylinder_collision = -1
        cylinder_visual = p.createVisualShape(shapeType=p.GEOM_CYLINDER, radius=0.05, length=0.14, rgbaColor=[0, 1, 1, 0.25], physicsClientId=self.id)
        self.cup_cylinder = p.createMultiBody(baseMass=0.0, baseCollisionShapeIndex=cylinder_collision, baseVisualShapeIndex=cylinder_visual, basePosition=cup_pos, baseOrientation=cup_orient, useMaximalCoordinates=False, physicsClientId=self.id)

    def update_targets(self):
        head_pos, head_orient = p.getLinkState(self.human, 23, computeForwardKinematics=True, physicsClientId=self.id)[:2]
        target_pos, target_orient = p.multiplyTransforms(head_pos, head_orient, self.mouth_pos, [0, 0, 0, 1], physicsClientId=self.id)
        self.target_pos = np.array(target_pos)
        p.resetBasePositionAndOrientation(self.target, self.target_pos, [0, 0, 0, 1], physicsClientId=self.id)

