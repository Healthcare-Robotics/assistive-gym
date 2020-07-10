import os, time
from gym import spaces
import numpy as np
import pybullet as p

from .env import AssistiveEnv

class DressingEnv(AssistiveEnv):
    def __init__(self, robot, human):
        super(DressingEnv, self).__init__(robot=robot, human=human, task='dressing')
        self.tt = None

    def step(self, action):
        # if self.tt is not None:
        #     print('Time per iteration:', time.time() - self.tt)
        # self.tt = time.time()
        # action = np.zeros(7)
        # action[0] = -1
        # action[3] = -1
        self.take_step(action, gains=self.config('robot_gains'), forces=self.config('robot_forces'))
        # self.take_step(action, gains=self.config('robot_gains'), forces=self.config('robot_forces'), step_sim=False, action_multiplier=0.01)

        shoulder_pos = self.human.get_pos_orient(self.human.left_shoulder)[0]
        elbow_pos = self.human.get_pos_orient(self.human.left_elbow)[0]
        wrist_pos = self.human.get_pos_orient(self.human.left_wrist)[0]

        # Get cloth data
        x, y, z, cx, cy, cz, fx, fy, fz = p.getSoftBodyData(self.cloth, physicsClientId=self.id)
        mesh_points = np.concatenate([np.expand_dims(x, axis=-1), np.expand_dims(y, axis=-1), np.expand_dims(z, axis=-1)], axis=-1)
        # Get 3D points for two triangles around the sleeve to detect if the sleeve is around the arm
        triangle1_points = mesh_points[self.triangle1_point_indices]
        triangle2_points = mesh_points[self.triangle2_point_indices]
        forearm_in_sleeve, upperarm_in_sleeve, distance_along_forearm, distance_along_upperarm, distance_to_hand, distance_to_elbow, distance_to_shoulder, forearm_length, upperarm_length = self.util.sleeve_on_arm_reward(triangle1_points, triangle2_points, shoulder_pos, elbow_pos, wrist_pos, self.human.hand_radius, self.human.elbow_radius, self.human.shoulder_radius)
        self.forearm_in_sleeve = forearm_in_sleeve
        self.upperarm_in_sleeve = upperarm_in_sleeve

        # Get human preferences, exclude cloth forces due to collision with the end effector
        forces = np.concatenate([np.expand_dims(fx, axis=-1), np.expand_dims(fy, axis=-1), np.expand_dims(fz, axis=-1)], axis=-1) * 10
        contact_positions = np.concatenate([np.expand_dims(cx, axis=-1), np.expand_dims(cy, axis=-1), np.expand_dims(cz, axis=-1)], axis=-1)
        end_effector_pos = self.robot.get_pos_orient(self.robot.left_end_effector)[0]
        forces_temp = []
        contact_positions_temp = []
        for f, c in zip(forces, contact_positions):
            if c[-1] < end_effector_pos[-1] - 0.05 and np.linalg.norm(f) < 20:
                forces_temp.append(f)
                contact_positions_temp.append(c)
        self.cloth_forces = np.array(forces_temp)
        contact_positions = np.array(contact_positions_temp)
        end_effector_velocity = np.linalg.norm(self.robot.get_velocity(self.robot.left_end_effector))
        preferences_score = self.human_preferences(end_effector_velocity=end_effector_velocity, dressing_forces=self.cloth_forces)

        reward_action = -np.linalg.norm(action) # Penalize actions
        if self.upperarm_in_sleeve:
            reward_dressing = forearm_length
            if distance_along_upperarm < upperarm_length:
                reward_dressing += distance_along_upperarm
        elif self.forearm_in_sleeve and distance_along_forearm < forearm_length:
            reward_dressing = distance_along_forearm
        else:
            reward_dressing = -distance_to_hand

        reward = self.config('dressing_reward_weight')*reward_dressing + self.config('action_weight')*reward_action + preferences_score

        obs = self._get_obs()

        if reward_dressing > self.task_success:
            self.task_success = reward_dressing

        if self.gui:
            print('Task success:', self.task_success, 'Average forces on arm:', self.cloth_force_sum)

        info = {'total_force_on_human': self.total_force_on_human, 'task_success': int(self.task_success >= self.config('task_success_threshold')), 'action_robot_len': self.action_robot_len, 'action_human_len': self.action_human_len, 'obs_robot_len': self.obs_robot_len, 'obs_human_len': self.obs_human_len}
        done = False

        return obs, reward, done, info

    def _get_obs(self, agent=None):
        end_effector_pos, end_effector_orient = self.robot.get_pos_orient(self.robot.left_end_effector)
        end_effector_pos_real, end_effector_orient_real = self.robot.convert_to_realworld(end_effector_pos, end_effector_orient)
        robot_joint_angles = self.robot.get_joint_angles(self.robot.controllable_joint_indices)
        shoulder_pos = self.human.get_pos_orient(self.human.left_shoulder)[0]
        elbow_pos = self.human.get_pos_orient(self.human.left_elbow)[0]
        wrist_pos = self.human.get_pos_orient(self.human.left_wrist)[0]
        shoulder_pos_real, _ = self.robot.convert_to_realworld(shoulder_pos)
        elbow_pos_real, _ = self.robot.convert_to_realworld(elbow_pos)
        wrist_pos_real, _ = self.robot.convert_to_realworld(wrist_pos)
        self.cloth_force_sum = np.sum(np.linalg.norm(self.cloth_forces, axis=-1))
        self.robot_force_on_human = np.sum(self.robot.get_contact_points(self.human)[-1])
        self.total_force_on_human = self.robot_force_on_human + self.cloth_force_sum
        if self.human.controllable:
            human_joint_angles = self.human.get_joint_angles(self.human.controllable_joint_indices)
            end_effector_pos_human, end_effector_orient_human = self.human.convert_to_realworld(end_effector_pos, end_effector_orient)
            shoulder_pos_human, _ = self.human.convert_to_realworld(shoulder_pos)
            elbow_pos_human, _ = self.human.convert_to_realworld(elbow_pos)
            wrist_pos_human, _ = self.human.convert_to_realworld(wrist_pos)

        robot_obs = np.concatenate([end_effector_pos_real, end_effector_orient_real, robot_joint_angles, shoulder_pos_real, elbow_pos_real, wrist_pos_real, [self.cloth_force_sum]]).ravel()
        if self.human.controllable:
            human_obs = np.concatenate([end_effector_pos_human, end_effector_orient_human, human_joint_angles, shoulder_pos_human, elbow_pos_human, wrist_pos_human, [self.cloth_force_sum, self.robot_force_on_human]]).ravel()
        else:
            human_obs = []

        if agent == 'robot':
            return robot_obs
        elif agent == 'human':
            return human_obs
        return np.concatenate([robot_obs, human_obs]).ravel()

    def reset(self):
        super(DressingEnv, self).reset()
        self.build_assistive_env('wheelchair')
        self.cloth_forces = np.zeros((1, 1))
        if self.robot.wheelchair_mounted:
            wheelchair_pos, wheelchair_orient = self.furniture.get_base_pos_orient()
            self.robot.set_base_pos_orient(wheelchair_pos + np.array(self.robot.toc_base_pos_offset[self.task]), [0, 0, -np.pi/2.0])

        joints_positions = [(self.human.j_right_elbow, -90), (self.human.j_left_shoulder_x, -80), (self.human.j_left_elbow, -90), (self.human.j_right_hip_x, -90), (self.human.j_right_knee, 80), (self.human.j_left_hip_x, -90), (self.human.j_left_knee, 80)]
        self.human.setup_joints(joints_positions, use_static_joints=True, reactive_force=None if self.human.controllable else 1, reactive_gain=0.01)

        shoulder_pos = self.human.get_pos_orient(self.human.left_shoulder)[0]
        elbow_pos = self.human.get_pos_orient(self.human.left_elbow)[0]
        wrist_pos = self.human.get_pos_orient(self.human.left_wrist)[0]

        target_ee_pos = np.array([0.45, -0.3, 1.2]) + self.np_random.uniform(-0.05, 0.05, size=3)
        target_ee_orient = np.array(p.getQuaternionFromEuler(np.array(self.robot.toc_ee_orient_rpy[self.task][0]), physicsClientId=self.id))
        offset = np.array([0, 0, 0.1])
        if self.robot.wheelchair_mounted:
            # Use IK to find starting joint angles for mounted robots
            self.robot.ik_random_restarts(right=False, target_pos=target_ee_pos, target_orient=target_ee_orient, max_iterations=1000, max_ik_random_restarts=40, success_threshold=0.03, step_sim=True, check_env_collisions=False)
        else:
            # Use TOC with JLWKI to find an optimal base position for the robot near the person
            target_ee_orient_shoulder = np.array(p.getQuaternionFromEuler(np.array(self.robot.toc_ee_orient_rpy[self.task][1]), physicsClientId=self.id))
            self.robot.position_robot_toc(self.task, 'left', [(target_ee_pos, target_ee_orient)], [(shoulder_pos+offset, target_ee_orient_shoulder), (elbow_pos+offset, target_ee_orient), (wrist_pos+offset, target_ee_orient)], self.human, base_euler_orient=[0, 0, np.pi], step_sim=True, check_env_collisions=False, right_side=False)
        # Open gripper to hold the tool
        self.robot.set_gripper_open_position(self.robot.left_gripper_indices, self.robot.gripper_pos[self.task], set_instantly=True)
        if self.human.controllable or self.human.impairment == 'tremor':
            # Ensure the human arm remains stable while loading the cloth
            self.human.control(self.human.controllable_joint_indices, self.human.get_joint_angles(self.human.controllable_joint_indices), 0.05, 1)

        self.start_ee_pos, self.start_ee_orient = self.robot.get_pos_orient(self.robot.left_end_effector)
        self.cloth_orig_pos = np.array([0.34658437, -0.30296362, 1.20023387])
        self.cloth_offset = self.start_ee_pos - self.cloth_orig_pos

        self.cloth_attachment = self.create_sphere(radius=0.0001, mass=0, pos=self.start_ee_pos, visual=True, collision=False, rgba=[0, 0, 0, 0], maximal_coordinates=True)

        # Load cloth
        self.cloth = p.loadCloth(os.path.join(self.directory, 'clothing', 'hospitalgown_reduced.obj'), scale=1.4, mass=0.16, position=np.array([0.02, -0.38, 0.83]) + self.cloth_offset/1.4, orientation=p.getQuaternionFromEuler([0, 0, np.pi], physicsClientId=self.id), bodyAnchorId=self.cloth_attachment.body, anchors=[2087, 3879, 3681, 3682, 2086, 2041, 987, 2042, 2088, 1647, 2332, 719, 1569, 1528, 721], collisionMargin=0.04, rgbaColor=np.array([139./256., 195./256., 74./256., 0.6]), rgbaLineColor=np.array([197./256., 225./256., 165./256., 1]), physicsClientId=self.id)
        p.clothParams(self.cloth, kLST=0.055, kAST=1.0, kVST=0.5, kDP=0.01, kDG=10, kDF=0.39, kCHR=1.0, kKHR=1.0, kAHR=1.0, piterations=5, physicsClientId=self.id)
        # Points along the opening of sleeve
        self.triangle1_point_indices = [1180, 2819, 30]
        self.triangle2_point_indices = [1322, 13, 696]

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
        self.robot.set_gravity(0, 0, 0)
        self.human.set_gravity(0, 0, -1)
        self.cloth_attachment.set_gravity(0, 0, 0)

        p.setPhysicsEngineParameter(numSubSteps=8, physicsClientId=self.id)

        # Enable rendering
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1, physicsClientId=self.id)

        # Wait for the cloth to settle
        for _ in range(50):
            # Force the end effector attachment to stay at the end effector
            self.cloth_attachment.set_base_pos_orient(self.start_ee_pos, [0, 0, 0, 1])
            p.stepSimulation(physicsClientId=self.id)

        p.setGravity(0, 0, -9.81, physicsClientId=self.id)

        self.init_env_variables()
        return self._get_obs()

    def update_targets(self):
        # Force the end effector to move forward
        # action = np.array([0, 0.025, 0]) / 10.0
        # ee_pos = self.robot.get_pos_orient(self.robot.left_end_effector)[0] + action
        # ee_pos[-1] = self.start_ee_pos[-1]
        # ik_joint_poses = np.array(p.calculateInverseKinematics(self.robot.body, self.robot.left_end_effector, targetPosition=ee_pos, targetOrientation=self.start_ee_orient, maxNumIterations=100, physicsClientId=self.id))
        # target_joint_positions = ik_joint_poses[self.robot.left_arm_ik_indices]
        # self.robot.set_joint_angles(self.robot.left_arm_joint_indices, target_joint_positions, use_limits=False)

        # Force the end effector attachment to stay at the end effector
        self.cloth_attachment.set_base_pos_orient(self.robot.get_pos_orient(self.robot.left_end_effector)[0], [0, 0, 0, 1])

