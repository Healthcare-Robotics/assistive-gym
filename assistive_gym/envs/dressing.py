import os
from gym import spaces
import numpy as np
import pybullet as p

from .env import AssistiveEnv

class DressingEnv(AssistiveEnv):
    def __init__(self, robot_type='pr2', human_control=False):
        super(DressingEnv, self).__init__(robot_type=robot_type, task='dressing', human_control=human_control, frame_skip=10, time_step=0.01, action_robot_len=7, action_human_len=(10 if human_control else 0), obs_robot_len=24, obs_human_len=(28 if human_control else 0))

    def step(self, action):
        self.take_step(action, robot_arm='left', gains=self.config('robot_gains'), forces=self.config('robot_forces'), human_gains=0.0025, step_sim=False)

        # Update robot position
        forces_torques = []
        for _ in range(self.frame_skip):
            # Force the cloth attachment to stay at the end effector
            state = p.getLinkState(self.robot, 76 if self.robot_type=='pr2' else 19 if self.robot_type=='sawyer' else 48 if self.robot_type=='baxter' else 8, computeForwardKinematics=True, physicsClientId=self.id)
            p.resetBasePositionAndOrientation(self.cloth_attachment, np.array(state[0]), [0, 0, 0, 1], physicsClientId=self.id)
            p.stepSimulation(physicsClientId=self.id)
        self.record_video_frame()

        x, y, z, cx, cy, cz, fx, fy, fz = p.getSoftBodyData(self.cloth, physicsClientId=self.id)
        mesh_points = np.concatenate([np.expand_dims(x, axis=-1), np.expand_dims(y, axis=-1), np.expand_dims(z, axis=-1)], axis=-1)
        triangle1_points = mesh_points[self.triangle1_point_indices]
        triangle2_points = mesh_points[self.triangle2_point_indices]
        forearm_in_sleeve, upperarm_in_sleeve, distance_along_forearm, distance_along_upperarm, distance_to_hand, distance_to_elbow, distance_to_shoulder, forearm_length, upperarm_length = self.util.sleeve_on_arm_reward(triangle1_points, triangle2_points, self.human, self.world_creation.human_creation.hand_radius, self.world_creation.human_creation.elbow_radius, self.world_creation.human_creation.shoulder_radius)
        if forearm_in_sleeve and not self.forearm_in_sleeve:
            self.forearm_in_sleeve = True
        if upperarm_in_sleeve and not self.upperarm_in_sleeve:
            self.upperarm_in_sleeve = True

        forces = np.concatenate([np.expand_dims(fx, axis=-1), np.expand_dims(fy, axis=-1), np.expand_dims(fz, axis=-1)], axis=-1) * 10
        contact_positions = np.concatenate([np.expand_dims(cx, axis=-1), np.expand_dims(cy, axis=-1), np.expand_dims(cz, axis=-1)], axis=-1)
        forces_temp = []
        contact_positions_temp = []
        for f, c in zip(forces, contact_positions):
            if c[-1] < 1.1 and np.linalg.norm(f) < 20:
                forces_temp.append(f)
                contact_positions_temp.append(c)
        forces = np.array(forces_temp)
        contact_positions = np.array(contact_positions_temp)
        end_effector_velocity = np.linalg.norm(p.getLinkState(self.robot, 76 if self.robot_type=='pr2' else 19 if self.robot_type=='sawyer' else 48 if self.robot_type=='baxter' else 8, computeForwardKinematics=True, computeLinkVelocity=True, physicsClientId=self.id)[6])

        reward_action = -np.sum(np.square(action)) # Penalize actions
        if self.upperarm_in_sleeve:
            reward_dressing = forearm_length
            if distance_along_upperarm < upperarm_length:
                reward_dressing += distance_along_upperarm
        elif self.forearm_in_sleeve and distance_along_forearm < forearm_length:
            reward_dressing = distance_along_forearm
        else:
            reward_dressing = -distance_to_hand
        # Get human preferences
        preferences_score = self.human_preferences(end_effector_velocity=end_effector_velocity, dressing_forces=forces)

        end_effector_pos = np.array(p.getLinkState(self.robot, 76 if self.robot_type=='pr2' else 19 if self.robot_type=='sawyer' else 48 if self.robot_type=='baxter' else 8, computeForwardKinematics=True, physicsClientId=self.id)[0])
        shoulder_pos = np.array(p.getLinkState(self.human, 15, computeForwardKinematics=True, physicsClientId=self.id)[0])
        elbow_pos, elbow_orient = p.getLinkState(self.human, 17, computeForwardKinematics=True, physicsClientId=self.id)[:2]

        reward = self.config('dressing_reward_weight')*reward_dressing + self.config('action_weight')*reward_action + preferences_score

        cloth_force_sum = np.sum(np.linalg.norm(forces, axis=-1))
        ft = [cloth_force_sum]
        robot_force_on_human = 0
        for c in p.getContactPoints(bodyA=self.robot, bodyB=self.human, physicsClientId=self.id):
            robot_force_on_human += c[9]
        obs = self._get_obs(ft, [cloth_force_sum, robot_force_on_human])

        if reward_dressing > self.task_success:
            self.task_success = reward_dressing

        if self.gui:
            print('Task success:', self.task_success, 'Average forces on arm:', cloth_force_sum)

        total_force_on_human = robot_force_on_human + cloth_force_sum
        info = {'total_force_on_human': total_force_on_human, 'task_success': int(self.task_success >= self.config('task_success_threshold')), 'action_robot_len': self.action_robot_len, 'action_human_len': self.action_human_len, 'obs_robot_len': self.obs_robot_len, 'obs_human_len': self.obs_human_len}
        done = False

        return obs, reward, done, info

    def _get_obs(self, forces, forces_human):
        torso_pos = np.array(p.getLinkState(self.robot, 15 if self.robot_type == 'pr2' else 0, computeForwardKinematics=True, physicsClientId=self.id)[0])
        state = p.getLinkState(self.robot, 76 if self.robot_type=='pr2' else 19 if self.robot_type=='sawyer' else 48 if self.robot_type=='baxter' else 8, computeForwardKinematics=True, physicsClientId=self.id)
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
        shoulder_pos, shoulder_orient = p.getLinkState(self.human, 15, computeForwardKinematics=True, physicsClientId=self.id)[:2]
        elbow_pos, elbow_orient = p.getLinkState(self.human, 17, computeForwardKinematics=True, physicsClientId=self.id)[:2]
        wrist_pos, wrist_orient = p.getLinkState(self.human, 19, computeForwardKinematics=True, physicsClientId=self.id)[:2]

        robot_obs = np.concatenate([tool_pos-torso_pos, tool_orient, robot_joint_positions, shoulder_pos-torso_pos, elbow_pos-torso_pos, wrist_pos-torso_pos, forces]).ravel()
        if self.human_control:
            human_obs = np.concatenate([tool_pos-human_pos, tool_orient, human_joint_positions, shoulder_pos-human_pos, elbow_pos-human_pos, wrist_pos-human_pos, forces_human]).ravel()
        else:
            human_obs = []

        return np.concatenate([robot_obs, human_obs]).ravel()

    def reset(self):
        self.setup_timing()
        self.task_success = 0
        self.forearm_in_sleeve = False
        self.upperarm_in_sleeve = False
        self.human, self.wheelchair, self.robot, self.robot_lower_limits, self.robot_upper_limits, self.human_lower_limits, self.human_upper_limits, self.robot_right_arm_joint_indices, self.robot_left_arm_joint_indices, self.gender = self.world_creation.create_new_world(furniture_type='wheelchair', static_human_base=True, human_impairment='random', print_joints=False, gender='random')
        self.robot_lower_limits = self.robot_lower_limits[self.robot_left_arm_joint_indices]
        self.robot_upper_limits = self.robot_upper_limits[self.robot_left_arm_joint_indices]
        self.reset_robot_joints()
        if self.robot_type == 'jaco':
            wheelchair_pos, wheelchair_orient = p.getBasePositionAndOrientation(self.wheelchair, physicsClientId=self.id)
            p.resetBasePositionAndOrientation(self.robot, np.array(wheelchair_pos) + np.array([0.35, -0.3, 0.3]), p.getQuaternionFromEuler([0, 0, 0], physicsClientId=self.id), physicsClientId=self.id)

        joints_positions = [(6, np.deg2rad(-90)), (13, np.deg2rad(-80)), (16, np.deg2rad(-90)), (28, np.deg2rad(-90)), (31, np.deg2rad(80)), (35, np.deg2rad(-90)), (38, np.deg2rad(80))]
        self.human_controllable_joint_indices = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
        self.world_creation.setup_human_joints(self.human, joints_positions, self.human_controllable_joint_indices if (self.human_control or self.world_creation.human_impairment == 'tremor') else [], use_static_joints=True, human_reactive_force=None)
        p.resetBasePositionAndOrientation(self.human, [0, 0.03, 0.89 if self.gender == 'male' else 0.86], [0, 0, 0, 1], physicsClientId=self.id)
        human_joint_states = p.getJointStates(self.human, jointIndices=self.human_controllable_joint_indices, physicsClientId=self.id)
        self.target_human_joint_positions = np.array([x[0] for x in human_joint_states])
        self.human_lower_limits = self.human_lower_limits[self.human_controllable_joint_indices]
        self.human_upper_limits = self.human_upper_limits[self.human_controllable_joint_indices]

        shoulder_pos, shoulder_orient = p.getLinkState(self.human, 15, computeForwardKinematics=True, physicsClientId=self.id)[:2]
        elbow_pos, elbow_orient = p.getLinkState(self.human, 17, computeForwardKinematics=True, physicsClientId=self.id)[:2]
        wrist_pos, wrist_orient = p.getLinkState(self.human, 19, computeForwardKinematics=True, physicsClientId=self.id)[:2]

        target_pos = np.array([-0.85, -0.4, 0]) + np.array([1.85, 0.6, 0]) + np.array([-0.55, -0.5, 1.2]) + self.np_random.uniform(-0.05, 0.05, size=3)
        offset = np.array([0, 0, 0.1])
        if self.robot_type == 'pr2':
            target_orient = p.getQuaternionFromEuler([0, 0, np.pi], physicsClientId=self.id)
            target_orient_shoulder = p.getQuaternionFromEuler([0, 0, np.pi*3/2.0], physicsClientId=self.id)
            self.position_robot_toc(self.robot, 76, [(target_pos, target_orient)], [(shoulder_pos+offset, target_orient_shoulder), (elbow_pos+offset, target_orient), (wrist_pos+offset, target_orient)], self.robot_left_arm_joint_indices, self.robot_lower_limits, self.robot_upper_limits, ik_indices=range(29, 29+7), pos_offset=np.array([1.7, 0.7, 0]), base_euler_orient=[0, 0, np.pi], right_side=False, max_ik_iterations=200, step_sim=True, check_env_collisions=False, human_joint_indices=self.human_controllable_joint_indices, human_joint_positions=self.target_human_joint_positions)
        elif self.robot_type == 'jaco':
            target_orient = p.getQuaternionFromEuler(np.array([0, -np.pi/2.0, 0]), physicsClientId=self.id)
            target_joint_positions = self.util.ik_random_restarts(self.robot, 8, target_pos, target_orient, self.world_creation, self.robot_left_arm_joint_indices, self.robot_lower_limits, self.robot_upper_limits, ik_indices=[0, 1, 2, 3, 4, 5, 6], max_iterations=1000, max_ik_random_restarts=40, random_restart_threshold=0.03, step_sim=True)
            self.world_creation.set_gripper_open_position(self.robot, position=1.33, left=False, set_instantly=True)
        else:
            target_orient = p.getQuaternionFromEuler([0, -np.pi/2.0, 0], physicsClientId=self.id)
            target_orient_shoulder = p.getQuaternionFromEuler([np.pi/2.0, -np.pi/2.0, 0], physicsClientId=self.id)
            if self.robot_type == 'baxter':
                self.position_robot_toc(self.robot, 48, [(target_pos, target_orient)], [(shoulder_pos+offset, target_orient_shoulder), (elbow_pos+offset, target_orient), (wrist_pos+offset, target_orient)], self.robot_left_arm_joint_indices, self.robot_lower_limits, self.robot_upper_limits, ik_indices=range(10, 17), pos_offset=np.array([1.7, 0.7, 0.975]), base_euler_orient=[0, 0, np.pi], right_side=False, max_ik_iterations=200, step_sim=True, check_env_collisions=False, human_joint_indices=self.human_controllable_joint_indices, human_joint_positions=self.target_human_joint_positions)
            else:
                self.position_robot_toc(self.robot, 19, [(target_pos, target_orient)], [(shoulder_pos+offset, target_orient_shoulder), (elbow_pos+offset, target_orient), (wrist_pos+offset, target_orient)], self.robot_left_arm_joint_indices, self.robot_lower_limits, self.robot_upper_limits, ik_indices=[0, 2, 3, 4, 5, 6, 7], pos_offset=np.array([1.8, 0.7, 0.975]), base_euler_orient=[0, 0, np.pi], right_side=False, max_ik_iterations=200, step_sim=True, check_env_collisions=False, human_joint_indices=self.human_controllable_joint_indices, human_joint_positions=self.target_human_joint_positions)
        if self.human_control or self.world_creation.human_impairment == 'tremor':
            human_len = len(self.human_controllable_joint_indices)
            human_joint_states = p.getJointStates(self.human, jointIndices=self.human_controllable_joint_indices, physicsClientId=self.id)
            human_joint_positions = np.array([x[0] for x in human_joint_states])
            p.setJointMotorControlArray(self.human, jointIndices=self.human_controllable_joint_indices, controlMode=p.POSITION_CONTROL, targetPositions=human_joint_positions, positionGains=np.array([0.005]*human_len), forces=[1]*human_len, physicsClientId=self.id)

        state = p.getLinkState(self.robot, 76 if self.robot_type=='pr2' else 19 if self.robot_type=='sawyer' else 48 if self.robot_type=='baxter' else 8, computeForwardKinematics=True, physicsClientId=self.id)
        self.start_ee_pos = np.array(state[0])
        self.start_ee_orient = np.array(state[1]) # Quaternions
        self.cloth_orig_pos = np.array([0.34658437, -0.30296362, 1.20023387])
        self.cloth_offset = self.start_ee_pos - self.cloth_orig_pos

        # Set up gripper - cloth constraint
        gripper_visual = p.createVisualShape(p.GEOM_SPHERE, radius=0.01, rgbaColor=[0, 0, 0, 0], physicsClientId=self.id)
        gripper_collision = p.createCollisionShape(p.GEOM_SPHERE, radius=0.0001, physicsClientId=self.id)
        self.cloth_attachment = p.createMultiBody(baseMass=0, baseVisualShapeIndex=gripper_visual, baseCollisionShapeIndex=gripper_collision, basePosition=self.start_ee_pos, useMaximalCoordinates=1, physicsClientId=self.id)

        # Load cloth
        self.cloth = p.loadCloth(os.path.join(self.world_creation.directory, 'clothing', 'hospitalgown_reduced.obj'), scale=1.4, mass=0.23, position=np.array([0.02, -0.38, 0.83]) + self.cloth_offset/1.4, orientation=p.getQuaternionFromEuler([0, 0, np.pi], physicsClientId=self.id), bodyAnchorId=self.cloth_attachment, anchors=[2087, 3879, 3681, 3682, 2086, 2041, 987, 2042, 2088, 1647, 2332], collisionMargin=0.04, rgbaColor=np.array([139./256., 195./256., 74./256., 0.6]), rgbaLineColor=np.array([197./256., 225./256., 165./256., 1]), physicsClientId=self.id)
        p.clothParams(self.cloth, kLST=0.05, kAST=1.0, kVST=1.0, kDP=0.001, kDG=10, kDF=0.25, kCHR=1.0, kKHR=1.0, kAHR=0.5, piterations=5, physicsClientId=self.id)
        self.triangle1_point_indices = [621, 37, 1008]
        self.triangle2_point_indices = [130, 3908, 2358]
        # double m_kLST;       // Material: Linear stiffness coefficient [0,1]
        # double m_kAST;       // Material: Area/Angular stiffness coefficient [0,1]
        # double m_kVST;       // Material: Volume stiffness coefficient [0,1]
        # double m_kVCF;       // Velocities correction factor (Baumgarte)
        # double m_kDP;        // Damping coefficient [0,1]
        # double m_kDG;        // Drag coefficient [0,+inf]
        # double m_kLF;        // Lift coefficient [0,+inf]
        # double m_kPR;        // Pressure coefficient [-inf,+inf]
        # double m_kVC;        // Volume conversation coefficient [0,+inf]
        # double m_kDF;        // Dynamic friction coefficient [0,1]
        # double m_kMT;        // Pose matching coefficient [0,1]
        # double m_kCHR;       // Rigid contacts hardness [0,1]
        # double m_kKHR;       // Kinetic contacts hardness [0,1]
        # double m_kSHR;       // Soft contacts hardness [0,1]
        # double m_kAHR;       // Anchors hardness [0,1]
        # int m_viterations;   // Velocities solver iterations
        # int m_piterations;   // Positions solver iterations
        # int m_diterations;   // Drift solver iterations

        p.setGravity(0, 0, -9.81/2, physicsClientId=self.id) # Let the cloth settle more gently
        p.setGravity(0, 0, 0, body=self.robot, physicsClientId=self.id)
        p.setGravity(0, 0, -1, body=self.human, physicsClientId=self.id)
        p.setGravity(0, 0, 0, body=self.cloth_attachment, physicsClientId=self.id)

        p.setPhysicsEngineParameter(numSubSteps=4, physicsClientId=self.id)

        # Enable rendering
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1, physicsClientId=self.id)

        # Wait for the cloth to settle
        for _ in range(200):
            # Force the cloth attachment to stay at the end effector
            p.resetBasePositionAndOrientation(self.cloth_attachment, self.start_ee_pos, [0, 0, 0, 1], physicsClientId=self.id)
            p.stepSimulation(physicsClientId=self.id)

        p.setGravity(0, 0, -9.81, physicsClientId=self.id)

        return self._get_obs([0], [0, 0])

