import os
from gym import spaces
import numpy as np
import pybullet as p

from .env import AssistiveEnv

class ArmManipulationEnv(AssistiveEnv):
    def __init__(self, robot_type='pr2', human_control=False):
        super(ArmManipulationEnv, self).__init__(robot_type=robot_type, task='arm_manipulation', human_control=human_control, frame_skip=5, time_step=0.02, action_robot_len=14, action_human_len=(10 if human_control else 0), obs_robot_len=45, obs_human_len=(42 if human_control else 0))

    def step(self, action):
        self.take_step(action, robot_arm='both', gains=self.config('robot_gains'), forces=self.config('robot_forces'), human_gains=0.05, human_forces=2)

        tool_left_force, tool_right_force, total_force_on_human, tool_left_force_on_human, tool_right_force_on_human = self.get_total_force()
        end_effector_velocity = np.linalg.norm(p.getLinkState(self.robot, 78 if self.robot_type=='pr2' else 24 if self.robot_type=='sawyer' else 54 if self.robot_type=='baxter' else 9 if self.robot_type=='jaco' else 7, computeForwardKinematics=True, computeLinkVelocity=True, physicsClientId=self.id)[6])
        end_effector_velocity += np.linalg.norm(p.getLinkState(self.robot, 55 if self.robot_type=='pr2' else 24 if self.robot_type=='sawyer' else 31 if self.robot_type=='baxter' else 9 if self.robot_type=='jaco' else 7, computeForwardKinematics=True, computeLinkVelocity=True, physicsClientId=self.id)[6])
        obs = self._get_obs([tool_left_force, tool_right_force], [total_force_on_human, tool_left_force_on_human, tool_right_force_on_human])

        # Get human preferences
        preferences_score = self.human_preferences(end_effector_velocity=end_effector_velocity, arm_manipulation_tool_forces_on_human=[tool_left_force_on_human, tool_right_force_on_human], arm_manipulation_total_force_on_human=total_force_on_human)

        tool_left_pos = np.array(p.getLinkState(self.robot, 78 if self.robot_type=='pr2' else 24 if self.robot_type=='sawyer' else 54 if self.robot_type=='baxter' else 9 if self.robot_type=='jaco' else 7, computeForwardKinematics=True, physicsClientId=self.id)[0])
        tool_right_pos = np.array(p.getLinkState(self.robot, 55 if self.robot_type=='pr2' else 24 if self.robot_type=='sawyer' else 31 if self.robot_type=='baxter' else 9 if self.robot_type=='jaco' else 7, computeForwardKinematics=True, physicsClientId=self.id)[0])
        elbow_pos = np.array(p.getLinkState(self.human, 7, computeForwardKinematics=True, physicsClientId=self.id)[0])
        hand_pos = np.array(p.getLinkState(self.human, 9, computeForwardKinematics=True, physicsClientId=self.id)[0])
        waist_pos = np.array(p.getLinkState(self.human, 24, computeForwardKinematics=True, physicsClientId=self.id)[0])
        hips_pos = np.array(p.getLinkState(self.human, 27, computeForwardKinematics=True, physicsClientId=self.id)[0])
        reward_distance_robot_left = -np.linalg.norm(tool_left_pos - elbow_pos) # Penalize distances away from human hand
        reward_distance_robot_right = -np.linalg.norm(tool_right_pos - hand_pos) # Penalize distances away from human hand
        reward_distance_human = -np.linalg.norm(elbow_pos - waist_pos) - np.linalg.norm(hand_pos - hips_pos) # Penalize distances between human hand and waist
        reward_action = -np.sum(np.square(action)) # Penalize actions

        if self.robot_type in ['sawyer', 'jaco', 'kinova_gen3']:
            reward = self.config('distance_human_weight')*reward_distance_human + 2*self.config('distance_end_effector_weight')*reward_distance_robot_left + self.config('action_weight')*reward_action + preferences_score
        else:
            reward = self.config('distance_human_weight')*reward_distance_human + self.config('distance_end_effector_weight')*reward_distance_robot_left + self.config('distance_end_effector_weight')*reward_distance_robot_right + self.config('action_weight')*reward_action + preferences_score

        if self.task_success == 0 or reward_distance_human > self.task_success:
            self.task_success = reward_distance_human

        if self.gui and total_force_on_human > 0:
            print('Task success:', self.task_success, 'Total force on human:', total_force_on_human, 'Tool force on human:', tool_left_force_on_human, tool_right_force_on_human)

        info = {'total_force_on_human': total_force_on_human, 'task_success': int(self.task_success >= self.config('task_success_threshold')), 'action_robot_len': self.action_robot_len, 'action_human_len': self.action_human_len, 'obs_robot_len': self.obs_robot_len, 'obs_human_len': self.obs_human_len}
        done = False

        return obs, reward, done, info

    def get_total_force(self):
        tool_left_force = 0
        tool_right_force = 0
        total_force_on_human = 0
        tool_left_force_on_human = 0
        tool_right_force_on_human = 0
        for c in p.getContactPoints(bodyA=self.robot, physicsClientId=self.id):
            linkA = c[3]
            if linkA == (55 if self.robot_type=='pr2' else 24 if self.robot_type=='sawyer' else 31 if self.robot_type=='baxter' else 9 if self.robot_type=='jaco' else 7):
                tool_right_force += c[9]
            elif linkA == (78 if self.robot_type=='pr2' else 24 if self.robot_type=='sawyer' else 54 if self.robot_type=='baxter' else 9 if self.robot_type=='jaco' else 7):
                tool_left_force += c[9]
        for c in p.getContactPoints(bodyA=self.robot, bodyB=self.human, physicsClientId=self.id):
            total_force_on_human += c[9]
            linkA = c[3]
            linkB = c[4]
            if linkA == (55 if self.robot_type=='pr2' else 24 if self.robot_type=='sawyer' else 31 if self.robot_type=='baxter' else 9 if self.robot_type=='jaco' else 7):
                tool_right_force_on_human += c[9]
            elif linkA == (78 if self.robot_type=='pr2' else 24 if self.robot_type=='sawyer' else 54 if self.robot_type=='baxter' else 9 if self.robot_type=='jaco' else 7):
                tool_left_force_on_human += c[9]
        return tool_left_force, tool_right_force, total_force_on_human, tool_left_force_on_human, tool_right_force_on_human

    def _get_obs(self, forces, forces_human):
        torso_pos = np.array(p.getLinkState(self.robot, 15 if self.robot_type == 'pr2' else 0, computeForwardKinematics=True, physicsClientId=self.id)[0])
        tool_left_pos, tool_left_orient = p.getLinkState(self.robot, 78 if self.robot_type=='pr2' else 24 if self.robot_type=='sawyer' else 54 if self.robot_type=='baxter' else 9 if self.robot_type=='jaco' else 7, computeForwardKinematics=True, physicsClientId=self.id)[:2]
        tool_right_pos, tool_right_orient = p.getLinkState(self.robot, 55 if self.robot_type=='pr2' else 24 if self.robot_type=='sawyer' else 31 if self.robot_type=='baxter' else 9 if self.robot_type=='jaco' else 7, computeForwardKinematics=True, physicsClientId=self.id)[:2]
        robot_joint_states = p.getJointStates(self.robot, jointIndices=self.robot_both_arm_joint_indices, physicsClientId=self.id)
        robot_joint_positions = np.array([x[0] for x in robot_joint_states])
        robot_pos, robot_orient = p.getBasePositionAndOrientation(self.robot, physicsClientId=self.id)
        if self.human_control:
            human_pos = np.array(p.getBasePositionAndOrientation(self.human, physicsClientId=self.id)[0])
            human_joint_states = p.getJointStates(self.human, jointIndices=self.human_controllable_joint_indices, physicsClientId=self.id)
            human_joint_positions = np.array([x[0] for x in human_joint_states])

        # Human shoulder, elbow, and wrist joint locations
        shoulder_pos, shoulder_orient = p.getLinkState(self.human, 5, computeForwardKinematics=True, physicsClientId=self.id)[:2]
        elbow_pos, elbow_orient = p.getLinkState(self.human, 7, computeForwardKinematics=True, physicsClientId=self.id)[:2]
        wrist_pos, wrist_orient = p.getLinkState(self.human, 9, computeForwardKinematics=True, physicsClientId=self.id)[:2]
        waist_pos = np.array(p.getLinkState(self.human, 24, computeForwardKinematics=True, physicsClientId=self.id)[0])
        hips_pos = np.array(p.getLinkState(self.human, 27, computeForwardKinematics=True, physicsClientId=self.id)[0])
        shoulder_pos, elbow_pos, wrist_pos, waist_pos, hips_pos = np.array(shoulder_pos), np.array(elbow_pos), np.array(wrist_pos), np.array(waist_pos), np.array(hips_pos)

        robot_obs = np.concatenate([tool_left_pos-torso_pos, tool_left_orient, tool_right_pos-torso_pos, tool_right_orient, robot_joint_positions, shoulder_pos-torso_pos, elbow_pos-torso_pos, wrist_pos-torso_pos, waist_pos-torso_pos, hips_pos-torso_pos, forces]).ravel()
        if self.human_control:
            human_obs = np.concatenate([tool_left_pos-human_pos, tool_left_orient, tool_right_pos-human_pos, tool_right_orient, human_joint_positions, shoulder_pos-human_pos, elbow_pos-human_pos, wrist_pos-human_pos, waist_pos-human_pos, hips_pos-human_pos, forces_human]).ravel()
        else:
            human_obs = []

        return np.concatenate([robot_obs, human_obs]).ravel()

    def reset(self):
        self.setup_timing()
        self.task_success = 0
        self.human, self.bed, self.robot, self.robot_lower_limits, self.robot_upper_limits, self.human_lower_limits, self.human_upper_limits, self.robot_right_arm_joint_indices, self.robot_left_arm_joint_indices, self.gender = self.world_creation.create_new_world(furniture_type='bed', static_human_base=False, human_impairment='no_tremor', print_joints=False, gender='random')
        self.robot_both_arm_joint_indices = self.robot_left_arm_joint_indices + self.robot_right_arm_joint_indices
        self.robot_right_lower_limits = self.robot_lower_limits[self.robot_right_arm_joint_indices]
        self.robot_right_upper_limits = self.robot_upper_limits[self.robot_right_arm_joint_indices]
        self.robot_left_lower_limits = self.robot_lower_limits[self.robot_left_arm_joint_indices]
        self.robot_left_upper_limits = self.robot_upper_limits[self.robot_left_arm_joint_indices]
        self.robot_lower_limits = self.robot_lower_limits[self.robot_both_arm_joint_indices]
        self.robot_upper_limits = self.robot_upper_limits[self.robot_both_arm_joint_indices]
        self.reset_robot_joints()

        friction = 5
        p.changeDynamics(self.bed, -1, lateralFriction=friction, spinningFriction=friction, rollingFriction=friction, physicsClientId=self.id)

        # Setup human in the air and let them settle into a resting pose on the bed
        joints_positions = [(3, np.deg2rad(30))]
        controllable_joints = []
        self.world_creation.setup_human_joints(self.human, joints_positions, controllable_joints, use_static_joints=False, human_reactive_force=None)
        p.resetBasePositionAndOrientation(self.human, [-0.25, 0.2, 0.95], p.getQuaternionFromEuler([-np.pi/2.0, 0, 0], physicsClientId=self.id), physicsClientId=self.id)

        p.setGravity(0, 0, -1, physicsClientId=self.id)
        p.setGravity(0, 0, 0, body=self.robot, physicsClientId=self.id)

        # Add small variation in human joint positions
        for j in range(p.getNumJoints(self.human, physicsClientId=self.id)):
            if p.getJointInfo(self.human, j, physicsClientId=self.id)[2] != p.JOINT_FIXED:
                p.resetJointState(self.human, jointIndex=j, targetValue=self.np_random.uniform(-0.1, 0.1), targetVelocity=0, physicsClientId=self.id)

        # Let the person settle on the bed
        for _ in range(100):
            p.stepSimulation(physicsClientId=self.id)

        friction = 0.3
        p.changeDynamics(self.bed, -1, lateralFriction=friction, spinningFriction=friction, rollingFriction=friction, physicsClientId=self.id)

        # Lock human joints (except arm) and set velocities to 0
        joints_positions = [(3, np.deg2rad(60)), (4, np.deg2rad(-60)), (6, 0)]
        self.human_controllable_joint_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        self.world_creation.setup_human_joints(self.human, joints_positions, self.human_controllable_joint_indices, use_static_joints=True, human_reactive_force=None)
        p.changeDynamics(self.human, -1, mass=0, physicsClientId=self.id)
        p.resetBaseVelocity(self.human, linearVelocity=[0, 0, 0], angularVelocity=[0, 0, 0], physicsClientId=self.id)
        for i in self.human_controllable_joint_indices:
            p.resetJointState(self.human, i, targetValue=p.getJointState(self.human, i, physicsClientId=self.id)[0], targetVelocity=0, physicsClientId=self.id)

        for _ in range(100):
            p.stepSimulation(physicsClientId=self.id)

        human_joint_states = p.getJointStates(self.human, jointIndices=self.human_controllable_joint_indices, physicsClientId=self.id)
        self.target_human_joint_positions = np.array([x[0] for x in human_joint_states])
        self.human_lower_limits = self.human_lower_limits[self.human_controllable_joint_indices]
        self.human_upper_limits = self.human_upper_limits[self.human_controllable_joint_indices]

        shoulder_pos, shoulder_orient = p.getLinkState(self.human, 5, computeForwardKinematics=True, physicsClientId=self.id)[:2]
        elbow_pos, elbow_orient = p.getLinkState(self.human, 7, computeForwardKinematics=True, physicsClientId=self.id)[:2]
        wrist_pos, wrist_orient = p.getLinkState(self.human, 9, computeForwardKinematics=True, physicsClientId=self.id)[:2]
        waist_pos = p.getLinkState(self.human, 24, computeForwardKinematics=True, physicsClientId=self.id)[0]
        hips_pos = p.getLinkState(self.human, 27, computeForwardKinematics=True, physicsClientId=self.id)[0]

        if self.robot_type == 'jaco':
            bed_pos, bed_orient = p.getBasePositionAndOrientation(self.bed, physicsClientId=self.id)
            p.resetBasePositionAndOrientation(self.robot, np.array(bed_pos) + np.array([-0.7, 0.75, 0.6]), p.getQuaternionFromEuler([0, 0, -np.pi/2.0], physicsClientId=self.id), physicsClientId=self.id)
        elif self.robot_type == 'kinova_gen3':
            bed_pos, bed_orient = p.getBasePositionAndOrientation(self.bed, physicsClientId=self.id)
            p.resetBasePositionAndOrientation(self.robot, np.array(bed_pos) + np.array([-0.7, 0.75, 0.6]), p.getQuaternionFromEuler([0, 0, -np.pi/2.0], physicsClientId=self.id), physicsClientId=self.id)
        target_pos_right = np.array([-0.9, -0.3, 0.8]) + self.np_random.uniform(-0.05, 0.05, size=3)
        target_pos_left = np.array([-0.9, 0.7, 0.8]) + self.np_random.uniform(-0.05, 0.05, size=3)
        if self.robot_type == 'pr2':
            target_orient = np.array(p.getQuaternionFromEuler(np.array([0, 0, 0]), physicsClientId=self.id))
            self.position_robot_toc(self.robot, [54, 77], [[(target_pos_right, target_orient)], [(target_pos_left, target_orient)]], [[(wrist_pos, None), (hips_pos, None)], [(elbow_pos, None), (waist_pos, None)]], [self.robot_right_arm_joint_indices, self.robot_left_arm_joint_indices], [self.robot_right_lower_limits, self.robot_left_lower_limits], [self.robot_right_upper_limits, self.robot_left_upper_limits], ik_indices=[range(15, 15+7), range(29, 29+7)], pos_offset=np.array([-0.3, 0.7, 0]), max_ik_iterations=200, step_sim=True, check_env_collisions=False, human_joint_indices=self.human_controllable_joint_indices, human_joint_positions=self.target_human_joint_positions)
            self.world_creation.set_gripper_open_position(self.robot, position=0.15, left=True, set_instantly=True, indices=[81, 82, 83, 84])
            self.world_creation.set_gripper_open_position(self.robot, position=0.15, left=False, set_instantly=True, indices=[58, 59, 60, 61])
        elif self.robot_type in ['jaco', 'kinova_gen3']:
            if self.robot_type == 'jaco':
                target_pos_left = np.array([-0.9, 0.4, 1]) + self.np_random.uniform(-0.05, 0.05, size=3)
                target_orient = p.getQuaternionFromEuler(np.array([0, np.pi/2.0, 0]), physicsClientId=self.id)
                base_position, base_orientation, _ = self.position_robot_toc(self.robot, 8, [(target_pos_left, target_orient)], [(wrist_pos, None), (hips_pos, None), (elbow_pos, None), (waist_pos, None)], self.robot_left_arm_joint_indices, self.robot_left_lower_limits, self.robot_left_upper_limits, ik_indices=[0, 1, 2, 3, 4, 5, 6], pos_offset=np.array([-0.05, 1.15, 0.6]), max_ik_iterations=200, step_sim=True, random_position=0.1, check_env_collisions=False, human_joint_indices=self.human_controllable_joint_indices, human_joint_positions=self.target_human_joint_positions)
                self.world_creation.set_gripper_open_position(self.robot, position=1.05, left=False, set_instantly=True, indices=[10, 12, 14])
            else:
                base_position = np.array([-0.05, 0, 0])
                target_pos_left = np.array([-0.9, 0.5, 1]) + self.np_random.uniform(-0.05, 0.05, size=3)
                target_orient = p.getQuaternionFromEuler(np.array([0, np.pi/2.0, 0]), physicsClientId=self.id)
                self.util.ik_random_restarts(self.robot, 7, target_pos_left, target_orient, self.world_creation, self.robot_left_arm_joint_indices, self.robot_left_lower_limits, self.robot_left_upper_limits, ik_indices=[0, 1, 2, 3, 4, 5, 6], max_iterations=100, max_ik_random_restarts=40, random_restart_threshold=0.03, step_sim=True)
            # Load a nightstand in the environment for the jaco arm
            self.nightstand_scale = 0.275
            visual_filename = os.path.join(self.world_creation.directory, 'nightstand', 'nightstand.obj')
            collision_filename = os.path.join(self.world_creation.directory, 'nightstand', 'nightstand.obj')
            nightstand_visual = p.createVisualShape(shapeType=p.GEOM_MESH, fileName=visual_filename, meshScale=[self.nightstand_scale]*3, rgbaColor=[0.5, 0.5, 0.5, 1.0], physicsClientId=self.id)
            nightstand_collision = p.createCollisionShape(shapeType=p.GEOM_MESH, fileName=collision_filename, meshScale=[self.nightstand_scale]*3, physicsClientId=self.id)
            nightstand_pos = np.array([-0.9, 0.8, 0]) + base_position
            nightstand_orient = p.getQuaternionFromEuler(np.array([np.pi/2.0, 0, 0]), physicsClientId=self.id)
            self.nightstand = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=nightstand_collision, baseVisualShapeIndex=nightstand_visual, basePosition=nightstand_pos, baseOrientation=nightstand_orient, baseInertialFramePosition=[0, 0, 0], useMaximalCoordinates=False, physicsClientId=self.id)
        else:
            target_orient = p.getQuaternionFromEuler(np.array([0, -np.pi/2.0, np.pi]), physicsClientId=self.id)
            if self.robot_type == 'baxter':
                self.position_robot_toc(self.robot, [26, 49], [[(target_pos_right, target_orient)], [(target_pos_left, target_orient)]], [[(wrist_pos, None), (hips_pos, None)], [(elbow_pos, None), (waist_pos, None)]], [self.robot_right_arm_joint_indices, self.robot_left_arm_joint_indices], [self.robot_right_lower_limits, self.robot_left_lower_limits], [self.robot_right_upper_limits, self.robot_left_upper_limits], ik_indices=[range(1, 8), range(10, 17)], pos_offset=np.array([-0.3, 0.6, 0.975]), max_ik_iterations=200, step_sim=True, check_env_collisions=False, human_joint_indices=self.human_controllable_joint_indices, human_joint_positions=self.target_human_joint_positions)
                self.world_creation.set_gripper_open_position(self.robot, position=0.01, left=True, set_instantly=True, indices=[50, 52])
                self.world_creation.set_gripper_open_position(self.robot, position=0.01, left=False, set_instantly=True, indices=[27, 29])
            else:
                self.position_robot_toc(self.robot, 19, [(target_pos_left, target_orient)], [(wrist_pos, None), (hips_pos, None), (elbow_pos, None), (waist_pos, None)], self.robot_left_arm_joint_indices, self.robot_left_lower_limits, self.robot_left_upper_limits, ik_indices=[0, 2, 3, 4, 5, 6, 7], pos_offset=np.array([-0.3, 0.6, 0.975]), max_ik_iterations=200, step_sim=True, check_env_collisions=False, human_joint_indices=self.human_controllable_joint_indices, human_joint_positions=self.target_human_joint_positions)
                self.world_creation.set_gripper_open_position(self.robot, position=0.01, left=True, set_instantly=True)

        p.setGravity(0, 0, -9.81, physicsClientId=self.id)
        p.setGravity(0, 0, 0, body=self.robot, physicsClientId=self.id)

        # Enable rendering
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1, physicsClientId=self.id)

        return self._get_obs([0, 0], [0, 0, 0])

