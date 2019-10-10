import os
from gym import spaces
import numpy as np
import pybullet as p

from .env import AssistiveEnv

class BedBathingEnv(AssistiveEnv):
    def __init__(self, robot_type='pr2', human_control=False):
        super(BedBathingEnv, self).__init__(robot_type=robot_type, task='bed_bathing', human_control=human_control, frame_skip=5, time_step=0.02, action_robot_len=7, action_human_len=(10 if human_control else 0), obs_robot_len=24, obs_human_len=(28 if human_control else 0))

    def step(self, action):
        self.take_step(action, robot_arm='left', gains=self.config('robot_gains'), forces=self.config('robot_forces'), human_gains=0.05)

        total_force, tool_force, tool_force_on_human, total_force_on_human, new_contact_points = self.get_total_force()
        end_effector_velocity = np.linalg.norm(p.getLinkState(self.tool, 1, computeForwardKinematics=True, computeLinkVelocity=True, physicsClientId=self.id)[6])
        obs = self._get_obs([tool_force], [total_force_on_human, tool_force_on_human])

        # Get human preferences
        preferences_score = self.human_preferences(end_effector_velocity=end_effector_velocity, total_force_on_human=total_force_on_human, tool_force_at_target=tool_force_on_human)

        reward_distance = -min([c[8] for c in p.getClosestPoints(self.tool, self.human, distance=4.0, physicsClientId=self.id)])
        reward_action = -np.sum(np.square(action)) # Penalize actions
        reward_new_contact_points = new_contact_points # Reward new contact points on a person

        reward = self.config('distance_weight')*reward_distance + self.config('action_weight')*reward_action + self.config('wiping_reward_weight')*reward_new_contact_points + preferences_score

        if self.gui and tool_force_on_human > 0:
            print('Task success:', self.task_success, 'Force at tool on human:', tool_force_on_human, reward_new_contact_points)

        info = {'total_force_on_human': total_force_on_human, 'task_success': int(self.task_success >= (self.total_target_count*self.config('task_success_threshold'))), 'action_robot_len': self.action_robot_len, 'action_human_len': self.action_human_len, 'obs_robot_len': self.obs_robot_len, 'obs_human_len': self.obs_human_len}
        done = False

        return obs, reward, done, info

    def get_total_force(self):
        total_force = 0
        tool_force = 0
        tool_force_on_human = 0
        total_force_on_human = 0
        new_contact_points = 0
        for c in p.getContactPoints(bodyA=self.tool, physicsClientId=self.id):
            total_force += c[9]
            tool_force += c[9]
        for c in p.getContactPoints(bodyA=self.robot, physicsClientId=self.id):
            bodyB = c[2]
            if bodyB != self.tool:
                total_force += c[9]
        for c in p.getContactPoints(bodyA=self.robot, bodyB=self.human, physicsClientId=self.id):
            total_force_on_human += c[9]
        for c in p.getContactPoints(bodyA=self.tool, bodyB=self.human, physicsClientId=self.id):
            linkA = c[3]
            linkB = c[4]
            contact_position = np.array(c[6])
            total_force_on_human += c[9]
            if linkA in [1]:
                tool_force_on_human += c[9]
                # Contact with human upperarm, forearm, hand
                if linkB < 0 or linkB > p.getNumJoints(self.human, physicsClientId=self.id):
                    continue

                indices_to_delete = []
                for i, (target_pos_world, target) in enumerate(zip(self.targets_pos_upperarm_world, self.targets_upperarm)):
                    if np.linalg.norm(contact_position - target_pos_world) < 0.025:
                        # The robot made contact with a point on the person's arm
                        new_contact_points += 1
                        self.task_success += 1
                        p.resetBasePositionAndOrientation(target, [1000, 1000, 1000], [0, 0, 0, 1], physicsClientId=self.id)
                        indices_to_delete.append(i)
                self.targets_pos_on_upperarm = [t for i, t in enumerate(self.targets_pos_on_upperarm) if i not in indices_to_delete]
                self.targets_upperarm = [t for i, t in enumerate(self.targets_upperarm) if i not in indices_to_delete]
                self.targets_pos_upperarm_world = [t for i, t in enumerate(self.targets_pos_upperarm_world) if i not in indices_to_delete]

                indices_to_delete = []
                for i, (target_pos_world, target) in enumerate(zip(self.targets_pos_forearm_world, self.targets_forearm)):
                    if np.linalg.norm(contact_position - target_pos_world) < 0.025:
                        # The robot made contact with a point on the person's arm
                        new_contact_points += 1
                        self.task_success += 1
                        p.resetBasePositionAndOrientation(target, [1000, 1000, 1000], [0, 0, 0, 1], physicsClientId=self.id)
                        indices_to_delete.append(i)
                self.targets_pos_on_forearm = [t for i, t in enumerate(self.targets_pos_on_forearm) if i not in indices_to_delete]
                self.targets_forearm = [t for i, t in enumerate(self.targets_forearm) if i not in indices_to_delete]
                self.targets_pos_forearm_world = [t for i, t in enumerate(self.targets_pos_forearm_world) if i not in indices_to_delete]

        return total_force, tool_force, tool_force_on_human, total_force_on_human, new_contact_points

    def _get_obs(self, forces, forces_human):
        torso_pos = np.array(p.getLinkState(self.robot, 15 if self.robot_type == 'pr2' else 0, computeForwardKinematics=True, physicsClientId=self.id)[0])
        state = p.getLinkState(self.tool, 1, computeForwardKinematics=True, physicsClientId=self.id)
        tool_pos = np.array(state[0])
        tool_orient = np.array(state[1]) # Quaternions
        robot_joint_states = p.getJointStates(self.robot, jointIndices=self.robot_left_arm_joint_indices, physicsClientId=self.id)
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

        robot_obs = np.concatenate([tool_pos-torso_pos, tool_orient, robot_joint_positions, shoulder_pos-torso_pos, elbow_pos-torso_pos, wrist_pos-torso_pos, forces]).ravel()
        if self.human_control:
            human_obs = np.concatenate([tool_pos-human_pos, tool_orient, human_joint_positions, shoulder_pos-human_pos, elbow_pos-human_pos, wrist_pos-human_pos, forces_human]).ravel()
        else:
            human_obs = []

        return np.concatenate([robot_obs, human_obs]).ravel()

    def reset(self):
        self.setup_timing()
        self.task_success = 0
        self.contact_points_on_arm = {}
        self.human, self.bed, self.robot, self.robot_lower_limits, self.robot_upper_limits, self.human_lower_limits, self.human_upper_limits, self.robot_right_arm_joint_indices, self.robot_left_arm_joint_indices, self.gender = self.world_creation.create_new_world(furniture_type='bed', static_human_base=False, human_impairment='random', print_joints=False, gender='random')
        self.robot_lower_limits = self.robot_lower_limits[self.robot_left_arm_joint_indices]
        self.robot_upper_limits = self.robot_upper_limits[self.robot_left_arm_joint_indices]
        self.reset_robot_joints()

        friction = 5
        p.changeDynamics(self.bed, -1, lateralFriction=friction, spinningFriction=friction, rollingFriction=friction, physicsClientId=self.id)

        # Setup human in the air and let them settle into a resting pose on the bed
        joints_positions = [(3, np.deg2rad(30))]
        controllable_joints = []
        self.world_creation.setup_human_joints(self.human, joints_positions, controllable_joints, use_static_joints=False, human_reactive_force=None)
        p.resetBasePositionAndOrientation(self.human, [-0.15, 0.2, 0.95], p.getQuaternionFromEuler([-np.pi/2.0, 0, 0], physicsClientId=self.id), physicsClientId=self.id)

        p.setGravity(0, 0, -1, physicsClientId=self.id)

        # Add small variation in human joint positions
        for j in range(p.getNumJoints(self.human, physicsClientId=self.id)):
            if p.getJointInfo(self.human, j, physicsClientId=self.id)[2] != p.JOINT_FIXED:
                p.resetJointState(self.human, jointIndex=j, targetValue=self.np_random.uniform(-0.1, 0.1), targetVelocity=0, physicsClientId=self.id)

        # Let the person settle on the bed
        for _ in range(100):
            p.stepSimulation(physicsClientId=self.id)

        # Lock human joints and set velocities to 0
        joints_positions = []
        self.human_controllable_joint_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] if self.human_control else []
        self.world_creation.setup_human_joints(self.human, joints_positions, self.human_controllable_joint_indices, use_static_joints=True, human_reactive_force=None, human_reactive_gain=0.01)
        self.target_human_joint_positions = []
        if self.human_control:
            human_joint_states = p.getJointStates(self.human, jointIndices=self.human_controllable_joint_indices, physicsClientId=self.id)
            self.target_human_joint_positions = np.array([x[0] for x in human_joint_states])
        self.human_lower_limits = self.human_lower_limits[self.human_controllable_joint_indices]
        self.human_upper_limits = self.human_upper_limits[self.human_controllable_joint_indices]
        p.changeDynamics(self.human, -1, mass=0, physicsClientId=self.id)
        p.resetBaseVelocity(self.human, linearVelocity=[0, 0, 0], angularVelocity=[0, 0, 0], physicsClientId=self.id)

        p.setGravity(0, 0, 0, physicsClientId=self.id)
        p.setGravity(0, 0, -1, body=self.human, physicsClientId=self.id)

        # Find the base position and joint positions for a static person in bed
        # print(p.getBasePositionAndOrientation(self.human, physicsClientId=self.id))
        # joint_states = p.getJointStates(self.human, jointIndices=list(range(p.getNumJoints(self.human, physicsClientId=self.id))), physicsClientId=self.id)
        # joint_positions = np.array([x[0] for x in joint_states])
        # joint_string = '['
        # for i, jp in enumerate(joint_positions):
        #     joint_string += '(%d, %.4f), ' % (i, jp)
        # print(joint_string + ']')
        # exit()

        shoulder_pos, shoulder_orient = p.getLinkState(self.human, 5, computeForwardKinematics=True, physicsClientId=self.id)[:2]
        elbow_pos, elbow_orient = p.getLinkState(self.human, 7, computeForwardKinematics=True, physicsClientId=self.id)[:2]
        wrist_pos, wrist_orient = p.getLinkState(self.human, 9, computeForwardKinematics=True, physicsClientId=self.id)[:2]

        target_pos = np.array([-0.6, 0.2, 1]) + self.np_random.uniform(-0.05, 0.05, size=3)
        if self.robot_type == 'pr2':
            target_orient = np.array(p.getQuaternionFromEuler(np.array([0, 0, 0]), physicsClientId=self.id))
            self.position_robot_toc(self.robot, 76, [(target_pos, target_orient)], [(shoulder_pos, None), (elbow_pos, None), (wrist_pos, None)], self.robot_left_arm_joint_indices, self.robot_lower_limits, self.robot_upper_limits, ik_indices=range(29, 29+7), pos_offset=np.array([-0.1, 0, 0]), max_ik_iterations=200, step_sim=True, check_env_collisions=False, human_joint_indices=self.human_controllable_joint_indices, human_joint_positions=self.target_human_joint_positions)
            self.world_creation.set_gripper_open_position(self.robot, position=0.2, left=True, set_instantly=True)
            self.tool = self.world_creation.init_tool(self.robot, mesh_scale=[1]*3, pos_offset=[0, 0, 0], orient_offset=p.getQuaternionFromEuler([0, 0, 0], physicsClientId=self.id), maximal=False)
        elif self.robot_type == 'jaco':
            target_orient = p.getQuaternionFromEuler(np.array([0, np.pi/2.0, 0]), physicsClientId=self.id)
            base_position, base_orientation, _ = self.position_robot_toc(self.robot, 8, [(target_pos, target_orient)], [(shoulder_pos, None), (elbow_pos, None), (wrist_pos, None)], self.robot_left_arm_joint_indices, self.robot_lower_limits, self.robot_upper_limits, ik_indices=[0, 1, 2, 3, 4, 5, 6], pos_offset=np.array([-0.05, 1.05, 0.6]), max_ik_iterations=200, step_sim=True, random_position=0.1, check_env_collisions=False, human_joint_indices=self.human_controllable_joint_indices, human_joint_positions=self.target_human_joint_positions)
            self.world_creation.set_gripper_open_position(self.robot, position=1.1, left=True, set_instantly=True)
            self.tool = self.world_creation.init_tool(self.robot, mesh_scale=[1]*3, pos_offset=[-0.01, 0, 0.03], orient_offset=p.getQuaternionFromEuler([0, -np.pi/2.0, 0], physicsClientId=self.id), maximal=False)
            # Load a nightstand in the environment for the jaco arm
            self.nightstand_scale = 0.275
            visual_filename = os.path.join(self.world_creation.directory, 'nightstand', 'nightstand.obj')
            collision_filename = os.path.join(self.world_creation.directory, 'nightstand', 'nightstand.obj')
            nightstand_visual = p.createVisualShape(shapeType=p.GEOM_MESH, fileName=visual_filename, meshScale=[self.nightstand_scale]*3, rgbaColor=[0.5, 0.5, 0.5, 1.0], physicsClientId=self.id)
            nightstand_collision = p.createCollisionShape(shapeType=p.GEOM_MESH, fileName=collision_filename, meshScale=[self.nightstand_scale]*3, physicsClientId=self.id)
            nightstand_pos = np.array([-0.9, 0.7, 0]) + base_position
            nightstand_orient = p.getQuaternionFromEuler(np.array([np.pi/2.0, 0, 0]), physicsClientId=self.id)
            self.nightstand = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=nightstand_collision, baseVisualShapeIndex=nightstand_visual, basePosition=nightstand_pos, baseOrientation=nightstand_orient, baseInertialFramePosition=[0, 0, 0], useMaximalCoordinates=False, physicsClientId=self.id)
        else:
            target_orient = p.getQuaternionFromEuler(np.array([0, np.pi/2.0, 0]), physicsClientId=self.id)
            if self.robot_type == 'baxter':
                self.position_robot_toc(self.robot, 48, [(target_pos, target_orient)], [(shoulder_pos, None), (elbow_pos, None), (wrist_pos, None)], self.robot_left_arm_joint_indices, self.robot_lower_limits, self.robot_upper_limits, ik_indices=range(10, 17), pos_offset=np.array([-0.2, 0, 0.975]), max_ik_iterations=200, step_sim=True, check_env_collisions=False, human_joint_indices=self.human_controllable_joint_indices, human_joint_positions=self.target_human_joint_positions)
            else:
                self.position_robot_toc(self.robot, 19, [(target_pos, target_orient)], [(shoulder_pos, None), (elbow_pos, None), (wrist_pos, None)], self.robot_left_arm_joint_indices, self.robot_lower_limits, self.robot_upper_limits, ik_indices=[0, 2, 3, 4, 5, 6, 7], pos_offset=np.array([-0.2, 0, 0.975]), max_ik_iterations=200, step_sim=True, check_env_collisions=False, human_joint_indices=self.human_controllable_joint_indices, human_joint_positions=self.target_human_joint_positions)
            self.world_creation.set_gripper_open_position(self.robot, position=0.0125, left=True, set_instantly=True)
            self.tool = self.world_creation.init_tool(self.robot, mesh_scale=[0.001]*3, pos_offset=[0, 0.1175, 0], orient_offset=p.getQuaternionFromEuler([np.pi/2.0, 0, np.pi/2.0], physicsClientId=self.id), maximal=False)

        self.generate_targets()

        # Enable rendering
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1, physicsClientId=self.id)

        return self._get_obs([0], [0, 0])

    def generate_targets(self):
        self.target_indices_to_ignore = []
        if self.gender == 'male':
            self.upperarm, self.upperarm_length, self.upperarm_radius = 5, 0.279, 0.043
            self.forearm, self.forearm_length, self.forearm_radius = 7, 0.257, 0.033
        else:
            self.upperarm, self.upperarm_length, self.upperarm_radius = 5, 0.264, 0.0355
            self.forearm, self.forearm_length, self.forearm_radius = 7, 0.234, 0.027

        self.targets_pos_on_upperarm = self.util.capsule_points(p1=np.array([0, 0, 0]), p2=np.array([0, 0, -self.upperarm_length]), radius=self.upperarm_radius, distance_between_points=0.03)
        self.targets_pos_on_forearm = self.util.capsule_points(p1=np.array([0, 0, 0]), p2=np.array([0, 0, -self.forearm_length]), radius=self.forearm_radius, distance_between_points=0.03)

        sphere_collision = -1
        sphere_visual = p.createVisualShape(shapeType=p.GEOM_SPHERE, radius=0.01, rgbaColor=[0, 1, 1, 1], physicsClientId=self.id)
        self.targets_upperarm = []
        self.targets_forearm = []
        for _ in range(len(self.targets_pos_on_upperarm)):
            self.targets_upperarm.append(p.createMultiBody(baseMass=0.0, baseCollisionShapeIndex=sphere_collision, baseVisualShapeIndex=sphere_visual, basePosition=[0, 0, 0], useMaximalCoordinates=False, physicsClientId=self.id))
        for _ in range(len(self.targets_pos_on_forearm)):
            self.targets_forearm.append(p.createMultiBody(baseMass=0.0, baseCollisionShapeIndex=sphere_collision, baseVisualShapeIndex=sphere_visual, basePosition=[0, 0, 0], useMaximalCoordinates=False, physicsClientId=self.id))
        self.total_target_count = len(self.targets_upperarm) + len(self.targets_forearm)
        self.update_targets()

    def update_targets(self):
        upperarm_pos, upperarm_orient = p.getLinkState(self.human, self.upperarm, computeForwardKinematics=True, physicsClientId=self.id)[:2]
        self.targets_pos_upperarm_world = []
        for target_pos_on_arm, target in zip(self.targets_pos_on_upperarm, self.targets_upperarm):
            target_pos = np.array(p.multiplyTransforms(upperarm_pos, upperarm_orient, target_pos_on_arm, [0, 0, 0, 1], physicsClientId=self.id)[0])
            self.targets_pos_upperarm_world.append(target_pos)
            p.resetBasePositionAndOrientation(target, target_pos, [0, 0, 0, 1], physicsClientId=self.id)

        forearm_pos, forearm_orient = p.getLinkState(self.human, self.forearm, computeForwardKinematics=True, physicsClientId=self.id)[:2]
        self.targets_pos_forearm_world = []
        for target_pos_on_arm, target in zip(self.targets_pos_on_forearm, self.targets_forearm):
            target_pos = np.array(p.multiplyTransforms(forearm_pos, forearm_orient, target_pos_on_arm, [0, 0, 0, 1], physicsClientId=self.id)[0])
            self.targets_pos_forearm_world.append(target_pos)
            p.resetBasePositionAndOrientation(target, target_pos, [0, 0, 0, 1], physicsClientId=self.id)

