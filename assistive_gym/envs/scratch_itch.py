import numpy as np
import pybullet as p

from .env import AssistiveEnv

class ScratchItchEnv(AssistiveEnv):
    def __init__(self, robot, human):
        super(ScratchItchEnv, self).__init__(robot=robot, human=human, task='scratch_itch', obs_robot_len=(23 + len(robot.controllable_joint_indices) - (len(robot.wheel_joint_indices) if robot.mobile else 0)), obs_human_len=(24 + len(human.controllable_joint_indices)))

    def step(self, action):
        if self.human.controllable:
            action = np.concatenate([action['robot'], action['human']])
        self.take_step(action)

        obs = self._get_obs()
        # print(np.array_str(obs, precision=3, suppress_small=True))

        # Get human preferences
        end_effector_velocity = np.linalg.norm(self.robot.get_velocity(self.robot.left_end_effector))
        preferences_score = self.human_preferences(end_effector_velocity=end_effector_velocity, total_force_on_human=self.total_force_on_human, tool_force_at_target=self.tool_force_at_target)

        tool_pos = self.tool.get_pos_orient(1)[0]
        reward_distance = -np.linalg.norm(self.target_pos - tool_pos) # Penalize distances away from target
        reward_action = -np.linalg.norm(action) # Penalize actions
        reward_force_scratch = 0.0 # Reward force near the target
        if self.target_contact_pos is not None and np.linalg.norm(self.target_contact_pos - self.prev_target_contact_pos) > 0.01 and self.tool_force_at_target < 10:
            # Encourage the robot to move around near the target to simulate scratching
            reward_force_scratch = 5
            self.prev_target_contact_pos = self.target_contact_pos
            self.task_success += 1

        reward = self.config('distance_weight')*reward_distance + self.config('action_weight')*reward_action + self.config('scratch_reward_weight')*reward_force_scratch + preferences_score

        if self.gui and self.tool_force_at_target > 0:
            print('Task success:', self.task_success, 'Tool force at target:', self.tool_force_at_target, reward_force_scratch)

        info = {'total_force_on_human': self.total_force_on_human, 'task_success': int(self.task_success >= self.config('task_success_threshold')), 'action_robot_len': self.action_robot_len, 'action_human_len': self.action_human_len, 'obs_robot_len': self.obs_robot_len, 'obs_human_len': self.obs_human_len}
        done = self.iteration >= 200

        if not self.human.controllable:
            return obs, reward, done, info
        else:
            # Co-optimization with both human and robot controllable
            return obs, {'robot': reward, 'human': reward}, {'robot': done, 'human': done, '__all__': done}, {'robot': info, 'human': info}

    def get_total_force(self):
        total_force_on_human = np.sum(self.robot.get_contact_points(self.human)[-1])
        tool_force = np.sum(self.tool.get_contact_points()[-1])
        tool_force_at_target = 0
        target_contact_pos = None
        for linkA, linkB, posA, posB, force in zip(*self.tool.get_contact_points(self.human)):
            total_force_on_human += force
            # Enforce that contact is close to the target location
            if linkA in [0, 1] and np.linalg.norm(posB - self.target_pos) < 0.025:
                tool_force_at_target += force
                target_contact_pos = posB
        return total_force_on_human, tool_force, tool_force_at_target, None if target_contact_pos is None else np.array(target_contact_pos)

    def _get_obs(self, agent=None):
        tool_pos, tool_orient = self.tool.get_pos_orient(1)
        tool_pos_real, tool_orient_real = self.robot.convert_to_realworld(tool_pos, tool_orient)
        robot_joint_angles = self.robot.get_joint_angles(self.robot.controllable_joint_indices)
        # Fix joint angles to be in [-pi, pi]
        robot_joint_angles = (np.array(robot_joint_angles) + np.pi) % (2*np.pi) - np.pi
        if self.robot.mobile:
            # Don't include joint angles for the wheels
            robot_joint_angles = robot_joint_angles[len(self.robot.wheel_joint_indices):]
        shoulder_pos = self.human.get_pos_orient(self.human.right_shoulder)[0]
        elbow_pos = self.human.get_pos_orient(self.human.right_elbow)[0]
        wrist_pos = self.human.get_pos_orient(self.human.right_wrist)[0]
        shoulder_pos_real, _ = self.robot.convert_to_realworld(shoulder_pos)
        elbow_pos_real, _ = self.robot.convert_to_realworld(elbow_pos)
        wrist_pos_real, _ = self.robot.convert_to_realworld(wrist_pos)
        target_pos_real, _ = self.robot.convert_to_realworld(self.target_pos)
        self.total_force_on_human, self.tool_force, self.tool_force_at_target, self.target_contact_pos = self.get_total_force()
        robot_obs = np.concatenate([tool_pos_real, tool_orient_real, tool_pos_real - target_pos_real, target_pos_real, robot_joint_angles, shoulder_pos_real, elbow_pos_real, wrist_pos_real, [self.tool_force]]).ravel()
        if agent == 'robot':
            return robot_obs
        if self.human.controllable:
            human_joint_angles = self.human.get_joint_angles(self.human.controllable_joint_indices)
            tool_pos_human, tool_orient_human = self.human.convert_to_realworld(tool_pos, tool_orient)
            shoulder_pos_human, _ = self.human.convert_to_realworld(shoulder_pos)
            elbow_pos_human, _ = self.human.convert_to_realworld(elbow_pos)
            wrist_pos_human, _ = self.human.convert_to_realworld(wrist_pos)
            target_pos_human, _ = self.human.convert_to_realworld(self.target_pos)
            human_obs = np.concatenate([tool_pos_human, tool_orient_human, tool_pos_human - target_pos_human, target_pos_human, human_joint_angles, shoulder_pos_human, elbow_pos_human, wrist_pos_human, [self.total_force_on_human, self.tool_force_at_target]]).ravel()
            if agent == 'human':
                return human_obs
            # Co-optimization with both human and robot controllable
            return {'robot': robot_obs, 'human': human_obs}
        return robot_obs

    def reset(self):
        super(ScratchItchEnv, self).reset()
        self.build_assistive_env('wheelchair')
        self.prev_target_contact_pos = np.zeros(3)
        if self.robot.wheelchair_mounted:
            wheelchair_pos, wheelchair_orient = self.furniture.get_base_pos_orient()
            self.robot.set_base_pos_orient(wheelchair_pos + np.array(self.robot.toc_base_pos_offset[self.task]), [0, 0, -np.pi/2.0])

        # self.robot.print_joint_info()

        # Set joint angles for human joints (in degrees)
        joints_positions = [(self.human.j_right_shoulder_x, 30), (self.human.j_right_elbow, -90), (self.human.j_left_elbow, -90), (self.human.j_right_hip_x, -90), (self.human.j_right_knee, 80), (self.human.j_left_hip_x, -90), (self.human.j_left_knee, 80)]
        self.human.setup_joints(joints_positions, use_static_joints=True, reactive_force=None if self.human.controllable else 1, reactive_gain=0.01)

        shoulder_pos = self.human.get_pos_orient(self.human.right_shoulder)[0]
        elbow_pos = self.human.get_pos_orient(self.human.right_elbow)[0]
        wrist_pos = self.human.get_pos_orient(self.human.right_wrist)[0]

        # Initialize the tool in the robot's gripper
        self.tool.init(self.robot, self.task, self.directory, self.id, self.np_random, right=False, mesh_scale=[0.001]*3)

        target_ee_pos = np.array([-0.6, 0, 0.8]) + self.np_random.uniform(-0.05, 0.05, size=3)
        target_ee_orient = self.get_quaternion(self.robot.toc_ee_orient_rpy[self.task])
        self.init_robot_pose(target_ee_pos, target_ee_orient, [(target_ee_pos, target_ee_orient)], [(shoulder_pos, None), (elbow_pos, None), (wrist_pos, None)], arm='left', tools=[self.tool], collision_objects=[self.human, self.furniture])

        # Open gripper to hold the tool
        self.robot.set_gripper_open_position(self.robot.left_gripper_indices, self.robot.gripper_pos[self.task], set_instantly=True)

        self.generate_target()

        if not self.robot.mobile:
            self.robot.set_gravity(0, 0, 0)
        self.human.set_gravity(0, 0, 0)
        self.tool.set_gravity(0, 0, 0)

        # Enable rendering
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1, physicsClientId=self.id)

        self.init_env_variables()
        return self._get_obs()

    def generate_target(self):
        # Randomly select either upper arm or forearm for the target limb to scratch
        if self.human.gender == 'male':
            self.limb, length, radius = [[self.human.right_shoulder, 0.279, 0.043], [self.human.right_elbow, 0.257, 0.033]][self.np_random.randint(2)]
        else:
            self.limb, length, radius = [[self.human.right_shoulder, 0.264, 0.0355], [self.human.right_elbow, 0.234, 0.027]][self.np_random.randint(2)]
        self.target_on_arm = self.util.point_on_capsule(p1=np.array([0, 0, 0]), p2=np.array([0, 0, -length]), radius=radius, theta_range=(0, np.pi*2))
        arm_pos, arm_orient = self.human.get_pos_orient(self.limb)
        target_pos, target_orient = p.multiplyTransforms(arm_pos, arm_orient, self.target_on_arm, [0, 0, 0, 1], physicsClientId=self.id)

        self.target = self.create_sphere(radius=0.01, mass=0.0, pos=target_pos, visual=True, collision=False, rgba=[0, 1, 1, 1])

        self.update_targets()

    def update_targets(self):
        arm_pos, arm_orient = self.human.get_pos_orient(self.limb)
        target_pos, target_orient = p.multiplyTransforms(arm_pos, arm_orient, self.target_on_arm, [0, 0, 0, 1], physicsClientId=self.id)
        self.target_pos = np.array(target_pos)
        self.target.set_base_pos_orient(self.target_pos, [0, 0, 0, 1])

