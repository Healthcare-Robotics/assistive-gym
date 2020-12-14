import numpy as np
import pybullet as p

from .env import AssistiveEnv
from .agents import furniture, tool
from .agents.furniture import Furniture
from .agents.tool import Tool

class ArmManipulationEnv(AssistiveEnv):
    def __init__(self, robot, human):
        super(ArmManipulationEnv, self).__init__(robot=robot, human=human, task='arm_manipulation', obs_robot_len=(31 + len(robot.controllable_joint_indices) - (len(robot.wheel_joint_indices) if robot.mobile else 0)), obs_human_len=(32 + len(human.controllable_joint_indices)))
        self.tool_right = self.tool
        if self.robot.has_single_arm:
            self.tool_left = self.tool_right
        else:
            self.tool_left = Tool()

    def step(self, action):
        if self.human.controllable:
            action = np.concatenate([action['robot'], action['human']])
        self.take_step(action)

        obs = self._get_obs()

        # Get human preferences
        end_effector_velocity = np.linalg.norm(self.robot.get_velocity(self.robot.right_end_effector))
        end_effector_velocity += np.linalg.norm(self.robot.get_velocity(self.robot.left_end_effector))
        preferences_score = self.human_preferences(end_effector_velocity=end_effector_velocity, arm_manipulation_tool_forces_on_human=[self.tool_right_force_on_human, self.tool_left_force_on_human], arm_manipulation_total_force_on_human=self.total_force_on_human)

        tool_right_pos = self.tool_right.get_base_pos_orient()[0]
        tool_left_pos = self.tool_left.get_base_pos_orient()[0]
        elbow_pos = self.human.get_pos_orient(self.human.right_elbow)[0]
        wrist_pos = self.human.get_pos_orient(self.human.right_wrist)[0]
        stomach_pos = self.human.get_pos_orient(self.human.stomach)[0]
        waist_pos = self.human.get_pos_orient(self.human.waist)[0]
        reward_distance_robot_left = -np.linalg.norm(tool_left_pos - elbow_pos) # Penalize distances away from human hand
        reward_distance_robot_right = -np.linalg.norm(tool_right_pos - wrist_pos) # Penalize distances away from human hand
        reward_distance_human = -np.linalg.norm(elbow_pos - stomach_pos) - np.linalg.norm(wrist_pos - waist_pos) # Penalize distances between human hand and waist
        reward_action = -np.linalg.norm(action) # Penalize actions

        if self.robot.has_single_arm:
            reward = self.config('distance_human_weight')*reward_distance_human + 2*self.config('distance_end_effector_weight')*reward_distance_robot_left + self.config('action_weight')*reward_action + preferences_score
        else:
            reward = self.config('distance_human_weight')*reward_distance_human + self.config('distance_end_effector_weight')*reward_distance_robot_left + self.config('distance_end_effector_weight')*reward_distance_robot_right + self.config('action_weight')*reward_action + preferences_score

        if self.task_success == 0 or reward_distance_human > self.task_success:
            self.task_success = reward_distance_human

        if self.gui and self.total_force_on_human > 0:
            print('Task success:', self.task_success, 'Total force on human:', self.total_force_on_human, 'Tool force on human:', self.tool_left_force_on_human, self.tool_right_force_on_human)

        info = {'total_force_on_human': self.total_force_on_human, 'task_success': int(self.task_success >= self.config('task_success_threshold')), 'action_robot_len': self.action_robot_len, 'action_human_len': self.action_human_len, 'obs_robot_len': self.obs_robot_len, 'obs_human_len': self.obs_human_len}
        done = self.iteration >= 200

        if not self.human.controllable:
            return obs, reward, done, info
        else:
            # Co-optimization with both human and robot controllable
            return obs, {'robot': reward, 'human': reward}, {'robot': done, 'human': done, '__all__': done}, {'robot': info, 'human': info}

    def get_total_force(self):
        tool_right_force = np.sum(self.tool_right.get_contact_points()[-1])
        tool_left_force = np.sum(self.tool_left.get_contact_points()[-1])
        tool_right_force_on_human = np.sum(self.tool_right.get_contact_points(self.human)[-1])
        tool_left_force_on_human = np.sum(self.tool_left.get_contact_points(self.human)[-1])
        total_force_on_human = np.sum(self.robot.get_contact_points(self.human)[-1]) + tool_right_force_on_human + tool_left_force_on_human
        return tool_right_force, tool_left_force, tool_right_force_on_human, tool_left_force_on_human, total_force_on_human

    def _get_obs(self, agent=None):
        tool_right_pos, tool_right_orient = self.tool_right.get_base_pos_orient()
        tool_left_pos, tool_left_orient = self.tool_left.get_base_pos_orient()
        tool_right_pos_real, tool_right_orient_real = self.robot.convert_to_realworld(tool_right_pos, tool_right_orient)
        tool_left_pos_real, tool_left_orient_real = self.robot.convert_to_realworld(tool_left_pos, tool_left_orient)
        robot_joint_angles = self.robot.get_joint_angles(self.robot.controllable_joint_indices)
        # Fix joint angles to be in [-pi, pi]
        robot_joint_angles = (np.array(robot_joint_angles) + np.pi) % (2*np.pi) - np.pi
        if self.robot.mobile:
            # Don't include joint angles for the wheels
            robot_joint_angles = robot_joint_angles[len(self.robot.wheel_joint_indices):]
        shoulder_pos = self.human.get_pos_orient(self.human.right_shoulder)[0]
        elbow_pos = self.human.get_pos_orient(self.human.right_elbow)[0]
        wrist_pos = self.human.get_pos_orient(self.human.right_wrist)[0]
        stomach_pos = self.human.get_pos_orient(self.human.stomach)[0]
        waist_pos = self.human.get_pos_orient(self.human.waist)[0]
        shoulder_pos_real, _ = self.robot.convert_to_realworld(shoulder_pos)
        elbow_pos_real, _ = self.robot.convert_to_realworld(elbow_pos)
        wrist_pos_real, _ = self.robot.convert_to_realworld(wrist_pos)
        stomach_pos_real, _ = self.robot.convert_to_realworld(stomach_pos)
        waist_pos_real, _ = self.robot.convert_to_realworld(waist_pos)
        self.tool_right_force, self.tool_left_force, self.tool_right_force_on_human, self.tool_left_force_on_human, self.total_force_on_human = self.get_total_force()
        robot_obs = np.concatenate([tool_right_pos_real, tool_right_orient_real, tool_left_pos_real, tool_left_orient_real, robot_joint_angles, shoulder_pos_real, elbow_pos_real, wrist_pos_real, stomach_pos_real, waist_pos_real, [self.tool_left_force, self.tool_right_force]]).ravel()
        if agent == 'robot':
            return robot_obs
        if self.human.controllable:
            human_joint_angles = self.human.get_joint_angles(self.human.controllable_joint_indices)
            tool_right_pos_human, tool_right_orient_human = self.human.convert_to_realworld(tool_right_pos, tool_right_orient)
            tool_left_pos_human, tool_left_orient_human = self.human.convert_to_realworld(tool_left_pos, tool_left_orient)
            shoulder_pos_human, _ = self.human.convert_to_realworld(shoulder_pos)
            elbow_pos_human, _ = self.human.convert_to_realworld(elbow_pos)
            wrist_pos_human, _ = self.human.convert_to_realworld(wrist_pos)
            stomach_pos_human, _ = self.human.convert_to_realworld(stomach_pos)
            waist_pos_human, _ = self.human.convert_to_realworld(waist_pos)
            human_obs = np.concatenate([tool_right_pos_human, tool_right_orient_human, tool_left_pos_human, tool_left_orient_human, human_joint_angles, shoulder_pos_human, elbow_pos_human, wrist_pos_human, stomach_pos_human, waist_pos_human, [self.total_force_on_human, self.tool_left_force_on_human, self.tool_right_force_on_human]]).ravel()
            if agent == 'human':
                return human_obs
            # Co-optimization with both human and robot controllable
            return {'robot': robot_obs, 'human': human_obs}
        return robot_obs

    def reset(self):
        super(ArmManipulationEnv, self).reset()
        self.build_assistive_env('bed', fixed_human_base=False, human_impairment='no_tremor')

        # Update robot and human motor gains
        self.robot.motor_forces = 10.0
        self.human.motor_forces = 2.0

        self.furniture.set_friction(self.furniture.base, friction=5)

        # Setup human in the air and let them settle into a resting pose on the bed
        joints_positions = [(self.human.j_right_shoulder_x, 30)]
        self.human.setup_joints(joints_positions, use_static_joints=False, reactive_force=None)
        self.human.set_base_pos_orient([-0.25, 0.2, 0.95], [-np.pi/2.0, 0, 0])

        p.setGravity(0, 0, -1, physicsClientId=self.id)
        if not self.robot.mobile:
            self.robot.set_gravity(0, 0, 0)

        # Add small variation in human joint positions
        motor_indices, motor_positions, motor_velocities, motor_torques = self.human.get_motor_joint_states()
        self.human.set_joint_angles(motor_indices, self.np_random.uniform(-0.1, 0.1, size=len(motor_indices)))

        # Let the person settle on the bed
        for _ in range(100):
            p.stepSimulation(physicsClientId=self.id)

        self.furniture.set_friction(self.furniture.base, friction=0.3)

        # Lock human joints and set velocities to 0
        joints_positions = [(self.human.j_right_shoulder_x, 60), (self.human.j_right_shoulder_y, -60), (self.human.j_right_elbow, 0)]
        self.human.setup_joints(joints_positions, use_static_joints=True, reactive_force=0.01)
        self.human.set_mass(self.human.base, mass=0)
        self.human.set_base_velocity(linear_velocity=[0, 0, 0], angular_velocity=[0, 0, 0])

        # Let the right arm fall to the ground
        for _ in range(100):
            p.stepSimulation(physicsClientId=self.id)

        elbow_pos = self.human.get_pos_orient(self.human.right_elbow)[0]
        wrist_pos = self.human.get_pos_orient(self.human.right_wrist)[0]
        stomach_pos = self.human.get_pos_orient(self.human.stomach)[0]
        waist_pos = self.human.get_pos_orient(self.human.waist)[0]

        # Initialize the tool in the robot's gripper
        self.tool_right.init(self.robot, self.task, self.directory, self.id, self.np_random, right=True, mesh_scale=[0.001]*3)
        if not self.robot.has_single_arm:
            self.tool_left.init(self.robot, self.task, self.directory, self.id, self.np_random, right=False, mesh_scale=[0.001]*3)

        target_ee_right_pos = np.array([-1, 0.4 if self.robot.has_single_arm else -0.3, 0.8]) + self.np_random.uniform(-0.05, 0.05, size=3)
        target_ee_left_pos = np.array([-1, 0.7, 0.8]) + self.np_random.uniform(-0.05, 0.05, size=3)
        target_ee_orient = self.get_quaternion(self.robot.toc_ee_orient_rpy[self.task])
        if self.robot.has_single_arm:
            base_position = self.init_robot_pose(target_ee_right_pos, target_ee_orient, [(target_ee_right_pos, target_ee_orient)], [(wrist_pos, None), (waist_pos, None), (elbow_pos, None), (stomach_pos, None)], arm='right', tools=[self.tool_right], collision_objects=[self.human, self.furniture], wheelchair_enabled=False)
        else:
            base_position = self.init_robot_pose(target_ee_right_pos, target_ee_orient, [[(target_ee_right_pos, target_ee_orient)], [(target_ee_left_pos, target_ee_orient)]], [[(wrist_pos, None), (waist_pos, None)], [(elbow_pos, None), (stomach_pos, None)]], arm=['right', 'left'], tools=[self.tool_right, self.tool_left], collision_objects=[self.human, self.furniture], wheelchair_enabled=False)

        if self.robot.wheelchair_mounted:
            # Load a nightstand in the environment for mounted arms
            self.nightstand = Furniture()
            self.nightstand.init('nightstand', self.directory, self.id, self.np_random)
            self.nightstand.set_base_pos_orient(np.array([-1.2, 0.7, 0]) + base_position, [0, 0, 0, 1])

        # Open gripper to hold the tools
        self.robot.set_gripper_open_position(self.robot.right_gripper_indices, self.robot.gripper_pos[self.task], set_instantly=True)
        if not self.robot.has_single_arm:
            self.robot.set_gripper_open_position(self.robot.left_gripper_indices, self.robot.gripper_pos[self.task], set_instantly=True)

        p.setGravity(0, 0, -9.81, physicsClientId=self.id)
        self.tool_right.set_gravity(0, 0, 0)
        if not self.robot.has_single_arm:
            self.tool_left.set_gravity(0, 0, 0)

        # Enable rendering
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1, physicsClientId=self.id)

        self.init_env_variables()
        return self._get_obs()

