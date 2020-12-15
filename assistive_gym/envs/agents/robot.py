import numpy as np
import pybullet as p
from .agent import Agent

class Robot(Agent):
    def __init__(self, controllable_joints, right_arm_joint_indices, left_arm_joint_indices, wheel_joint_indices, right_end_effector, left_end_effector, right_gripper_indices, left_gripper_indices, gripper_pos, right_tool_joint, left_tool_joint, tool_pos_offset, tool_orient_offset, right_gripper_collision_indices, left_gripper_collision_indices, toc_base_pos_offset, toc_ee_orient_rpy, wheelchair_mounted, half_range=False, controllable_joint_indices=None, action_duplication=None, action_multiplier=1, flags=None):
        self.controllable_joints = controllable_joints if controllable_joint_indices is None else ''
        self.right_arm_joint_indices = right_arm_joint_indices # Controllable arm joints
        self.left_arm_joint_indices = left_arm_joint_indices # Controllable arm joints
        self.wheel_joint_indices = wheel_joint_indices # Controllable wheel joints
        self.mobile = 'wheel' in controllable_joints
        if controllable_joint_indices is not None:
            self.controllable_joint_indices = controllable_joint_indices
        else:
            self.controllable_joint_indices = self.wheel_joint_indices if self.mobile else []
            self.controllable_joint_indices = self.controllable_joint_indices + (self.right_arm_joint_indices if 'right' in controllable_joints else self.left_arm_joint_indices if 'left' in controllable_joints else self.right_arm_joint_indices + self.left_arm_joint_indices)
        self.right_end_effector = right_end_effector # Used to get the pose of the end effector
        self.left_end_effector = left_end_effector # Used to get the pose of the end effector
        self.right_gripper_indices = right_gripper_indices # Gripper actuated joints
        self.left_gripper_indices = left_gripper_indices # Gripper actuated joints
        self.gripper_pos = gripper_pos # Gripper open position for holding tools
        self.right_tool_joint = right_tool_joint # Joint that tools are attached to
        self.left_tool_joint = left_tool_joint # Joint that tools are attached to
        self.tool_pos_offset = tool_pos_offset
        self.tool_orient_offset = tool_orient_offset
        self.right_gripper_collision_indices = right_gripper_collision_indices # Used to disable collision between gripper and tools
        self.left_gripper_collision_indices = left_gripper_collision_indices # Used to disable collision between gripper and tools
        self.toc_base_pos_offset = toc_base_pos_offset # Robot base offset before TOC base pose optimization
        self.toc_ee_orient_rpy = toc_ee_orient_rpy # Initial end effector orientation
        self.wheelchair_mounted = wheelchair_mounted
        self.half_range = half_range # Try setting this to true if the robot is struggling to find IK solutions
        self.action_duplication = action_duplication
        self.action_multiplier = action_multiplier
        self.flags = flags # Used to store any additional information for the robot
        self.has_single_arm = self.right_end_effector == self.left_end_effector
        self.motor_forces = 1.0
        self.motor_gains = 0.05
        self.skip_pose_optimization = False
        super(Robot, self).__init__()

    def enable_wheels(self):
        self.mobile = True
        self.skip_pose_optimization = True
        self.controllable_joint_indices = self.wheel_joint_indices + (self.right_arm_joint_indices if 'right' in self.controllable_joints else self.left_arm_joint_indices if 'left' in self.controllable_joints else self.right_arm_joint_indices + self.left_arm_joint_indices)

    def init(self, body, id, np_random):
        super(Robot, self).init(body, id, np_random)
        if self.mobile:
            self.controllable_joint_lower_limits[:len(self.wheel_joint_indices)] = -np.inf
            self.controllable_joint_upper_limits[:len(self.wheel_joint_indices)] = np.inf
        self.right_arm_lower_limits = [self.lower_limits[i] for i in self.right_arm_joint_indices]
        self.right_arm_upper_limits = [self.upper_limits[i] for i in self.right_arm_joint_indices]
        self.left_arm_lower_limits = [self.lower_limits[i] for i in self.left_arm_joint_indices]
        self.left_arm_upper_limits = [self.upper_limits[i] for i in self.left_arm_joint_indices]
        self.joint_max_forces = self.get_joint_max_force(self.controllable_joint_indices)
        # Determine ik indices for the right and left arms (indices differ since fixed joints are not counted)
        self.right_arm_ik_indices = []
        self.left_arm_ik_indices = []
        for i in self.right_arm_joint_indices:
            counter = 0
            for j in self.all_joint_indices:
                if i == j:
                    self.right_arm_ik_indices.append(counter)
                joint_type = p.getJointInfo(self.body, j, physicsClientId=self.id)[2]
                if joint_type != p.JOINT_FIXED:
                    counter += 1
        for i in self.left_arm_joint_indices:
            counter = 0
            for j in self.all_joint_indices:
                if i == j:
                    self.left_arm_ik_indices.append(counter)
                joint_type = p.getJointInfo(self.body, j, physicsClientId=self.id)[2]
                if joint_type != p.JOINT_FIXED:
                    counter += 1

    def set_gripper_open_position(self, indices, positions, set_instantly=False, force=500):
        p.setJointMotorControlArray(self.body, jointIndices=indices, controlMode=p.POSITION_CONTROL, targetPositions=positions, positionGains=np.array([0.05]*len(indices)), forces=[force]*len(indices), physicsClientId=self.id)
        if set_instantly:
            self.set_joint_angles(indices, positions, use_limits=True)

    def randomize_init_joint_angles(self, task, offset=0):
        return

    def ik_random_restarts(self, right, target_pos, target_orient, max_iterations=1000, max_ik_random_restarts=40, success_threshold=0.03, step_sim=False, check_env_collisions=False, randomize_limits=True):
        if target_orient is not None and len(target_orient) < 4:
            target_orient = self.get_quaternion(target_orient)
        orient_orig = target_orient
        best_ik_angles = None
        best_ik_distance = 0
        for r in range(max_ik_random_restarts):
            target_joint_angles = self.ik(self.right_end_effector if right else self.left_end_effector, target_pos, target_orient, ik_indices=self.right_arm_ik_indices if right else self.left_arm_ik_indices, max_iterations=max_iterations, half_range=self.half_range, randomize_limits=randomize_limits)
            self.set_joint_angles(self.right_arm_joint_indices if right else self.left_arm_joint_indices, target_joint_angles)
            if step_sim:
                # TODO: Replace this with getClosestPoints, see: https://github.gatech.edu/zerickson3/assistive-gym/blob/vr3/assistive_gym/envs/feeding.py#L156
                for _ in range(5):
                    p.stepSimulation(physicsClientId=self.id)
                # if len(p.getContactPoints(bodyA=self.body, bodyB=self.body, physicsClientId=self.id)) > 0 and orient_orig is not None:
                #     # The robot's arm is in contact with itself. Continually randomize end effector orientation until a solution is found
                #     target_orient = self.get_quaternion(self.get_euler(orient_orig) + np.deg2rad(self.np_random.uniform(-45, 45, size=3)))
            if check_env_collisions:
                for _ in range(25):
                    p.stepSimulation(physicsClientId=self.id)
            gripper_pos, gripper_orient = self.get_pos_orient(self.right_end_effector if right else self.left_end_effector)
            if np.linalg.norm(target_pos - np.array(gripper_pos)) < success_threshold and (target_orient is None or np.linalg.norm(target_orient - np.array(gripper_orient)) < success_threshold or np.isclose(np.linalg.norm(target_orient - np.array(gripper_orient)), 2, atol=success_threshold)):
                self.set_joint_angles(self.right_arm_joint_indices if right else self.left_arm_joint_indices, target_joint_angles)
                return True, np.array(target_joint_angles)
            if best_ik_angles is None or np.linalg.norm(target_pos - np.array(gripper_pos)) < best_ik_distance:
                best_ik_angles = target_joint_angles
                best_ik_distance = np.linalg.norm(target_pos - np.array(gripper_pos))
        self.set_joint_angles(self.right_arm_joint_indices if right else self.left_arm_joint_indices, best_ik_angles)
        return False, np.array(best_ik_angles)

    def position_robot_toc(self, task, arms, start_pos_orient, target_pos_orients, human, base_euler_orient=np.zeros(3), max_ik_iterations=200, max_ik_random_restarts=1, randomize_limits=False, attempts=100, jlwki_restarts=1, step_sim=False, check_env_collisions=False, right_side=True, random_rotation=30, random_position=0.5):
        # Continually randomize the robot base position and orientation
        # Select best base pose according to number of goals reached and manipulability
        if type(arms) == str:
            arms = [arms]
            start_pos_orient = [start_pos_orient]
            target_pos_orients = [target_pos_orients]
        a = 6 # Order of the robot space. 6D (3D position, 3D orientation)
        best_position = None
        best_orientation = None
        best_num_goals_reached = None
        best_manipulability = None
        best_start_joint_poses = [None]*len(arms)
        iteration = 0
        # Save human joint states for later restoring
        human_angles = human.get_joint_angles(human.controllable_joint_indices)
        while iteration < attempts or best_position is None:
            iteration += 1
            # Randomize base position and orientation
            random_pos = np.array([self.np_random.uniform(-random_position if right_side else 0, 0 if right_side else random_position), self.np_random.uniform(-random_position, random_position), 0])
            random_orientation = self.get_quaternion([base_euler_orient[0], base_euler_orient[1], base_euler_orient[2] + np.deg2rad(self.np_random.uniform(-random_rotation, random_rotation))])
            self.set_base_pos_orient(np.array([-0.85, -0.4, 0]) + self.toc_base_pos_offset[task] + random_pos, random_orientation)
            # Reset all robot joints to their defaults
            self.reset_joints()
            # Reset human joints in case they got perturbed by previous iterations
            human.set_joint_angles(human.controllable_joint_indices, human_angles)
            num_goals_reached = 0
            manipulability = 0.0
            start_joint_poses = [None]*len(arms)
            # Check if the robot can reach all target locations from this base pose
            for i, arm in enumerate(arms):
                right = (arm == 'right')
                ee = self.right_end_effector if right else self.left_end_effector
                ik_indices = self.right_arm_ik_indices if right else self.left_arm_ik_indices
                lower_limits = self.right_arm_lower_limits if right else self.left_arm_lower_limits
                upper_limits = self.right_arm_upper_limits if right else self.left_arm_upper_limits
                for j, (target_pos, target_orient) in enumerate(start_pos_orient[i] + target_pos_orients[i]):
                    best_jlwki = None
                    best_joint_positions = None
                    for k in range(jlwki_restarts):
                        # Reset state in case anything was perturbed from the last iteration
                        human.set_joint_angles(human.controllable_joint_indices, human_angles)
                        # Find IK solution
                        success, joint_positions_q_star = self.ik_random_restarts(right, target_pos, target_orient, max_iterations=max_ik_iterations, max_ik_random_restarts=max_ik_random_restarts, success_threshold=0.03, step_sim=step_sim, check_env_collisions=check_env_collisions, randomize_limits=randomize_limits)
                        if not success:
                            continue
                        _, motor_positions, _, _ = self.get_motor_joint_states()
                        joint_velocities = [0.0] * len(motor_positions)
                        joint_accelerations = [0.0] * len(motor_positions)
                        center_of_mass = p.getLinkState(self.body, ee, computeLinkVelocity=True, computeForwardKinematics=True, physicsClientId=self.id)[2]
                        J_linear, J_angular = p.calculateJacobian(self.body, ee, localPosition=center_of_mass, objPositions=motor_positions, objVelocities=joint_velocities, objAccelerations=joint_accelerations, physicsClientId=self.id)
                        J_linear = np.array(J_linear)[:, ik_indices]
                        J_angular = np.array(J_angular)[:, ik_indices]
                        J = np.concatenate([J_linear, J_angular], axis=0)
                        # Joint-limited-weighting
                        joint_limit_weight = self.joint_limited_weighting(joint_positions_q_star, lower_limits, upper_limits)
                        # Joint-limited-weighted kinematic isotropy (JLWKI)
                        det = max(np.linalg.det(np.matmul(np.matmul(J, joint_limit_weight), J.T)), 0)
                        jlwki = np.power(det, 1.0/a) / (np.trace(np.matmul(np.matmul(J, joint_limit_weight), J.T))/a)
                        if best_jlwki is None or jlwki > best_jlwki:
                            best_jlwki = jlwki
                            best_joint_positions = joint_positions_q_star
                    if best_jlwki is not None:
                        num_goals_reached += 1
                        manipulability += best_jlwki
                        if j == 0:
                            start_joint_poses[i] = best_joint_positions
                    if j < len(start_pos_orient[i]) and best_jlwki is None:
                        # Not able to find an IK solution to a start goal. We cannot use this base pose
                        num_goals_reached = -1
                        manipulability = None
                        break
                if num_goals_reached == -1:
                    break

            if num_goals_reached > 0:
                if best_position is None or num_goals_reached > best_num_goals_reached or (num_goals_reached == best_num_goals_reached and manipulability > best_manipulability):
                    best_position = random_pos
                    best_orientation = random_orientation
                    best_num_goals_reached = num_goals_reached
                    best_manipulability = manipulability
                    best_start_joint_poses = start_joint_poses

            human.set_joint_angles(human.controllable_joint_indices, human_angles)

        # Reset state in case anything was perturbed
        human.set_joint_angles(human.controllable_joint_indices, human_angles)

        # Set the robot base position/orientation and joint angles based on the best pose found
        p.resetBasePositionAndOrientation(self.body, np.array([-0.85, -0.4, 0]) + np.array(self.toc_base_pos_offset[task]) + best_position, best_orientation, physicsClientId=self.id)
        for i, arm in enumerate(arms):
            self.set_joint_angles(self.right_arm_joint_indices if arm == 'right' else self.left_arm_joint_indices, best_start_joint_poses[i])
        return best_position, best_orientation, best_start_joint_poses

    def joint_limited_weighting(self, q, lower_limits, upper_limits):
        phi = 0.5
        lam = 0.05
        weights = []
        for qi, l, u in zip(q, lower_limits, upper_limits):
            qr = 0.5*(u - l)
            weights.append(1.0 - np.power(phi, (qr - np.abs(qr - qi + l)) / (lam*qr) + 1))
            if weights[-1] < 0.001:
                weights[-1] = 0.001
        # Joint-limited-weighting
        joint_limit_weight = np.diag(weights)
        return joint_limit_weight

