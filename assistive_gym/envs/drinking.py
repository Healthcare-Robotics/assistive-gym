import numpy as np
import pybullet as p

from .env import AssistiveEnv

class DrinkingEnv(AssistiveEnv):
    def __init__(self, robot, human):
        super(DrinkingEnv, self).__init__(robot=robot, human=human, task='drinking', obs_robot_len=(18 + len(robot.controllable_joint_indices) - (len(robot.wheel_joint_indices) if robot.mobile else 0)), obs_human_len=(19 + len(human.controllable_joint_indices)))

    def step(self, action):
        if self.human.controllable:
            action = np.concatenate([action['robot'], action['human']])
        self.take_step(action)

        obs = self._get_obs()

        reward_water, water_mouth_velocities, water_hit_human_reward = self.get_water_rewards()

        # Get human preferences
        end_effector_velocity = np.linalg.norm(self.robot.get_velocity(self.robot.right_end_effector))
        preferences_score = self.human_preferences(end_effector_velocity=end_effector_velocity, total_force_on_human=self.total_force_on_human, tool_force_at_target=self.cup_force_on_human, food_hit_human_reward=water_hit_human_reward, food_mouth_velocities=water_mouth_velocities)

        cup_pos, cup_orient = self.tool.get_base_pos_orient()
        cup_pos, cup_orient = p.multiplyTransforms(cup_pos, cup_orient, [0, 0.06, 0], self.get_quaternion([np.pi/2.0, 0, 0]), physicsClientId=self.id)
        cup_top_center_pos, _ = p.multiplyTransforms(cup_pos, cup_orient, self.cup_top_center_offset, [0, 0, 0, 1], physicsClientId=self.id)
        reward_distance_mouth_target = -np.linalg.norm(self.target_pos - np.array(cup_top_center_pos)) # Penalize distances between top of cup and mouth
        reward_action = -np.linalg.norm(action) # Penalize actions

        # Encourage robot to have a tilted end effector / cup
        cup_euler = self.get_euler(cup_orient)
        reward_tilt = -abs(cup_euler[0] - np.pi/2)

        reward = self.config('distance_weight')*reward_distance_mouth_target + self.config('action_weight')*reward_action + self.config('cup_tilt_weight')*reward_tilt + self.config('drinking_reward_weight')*reward_water + preferences_score

        if self.gui and reward_water != 0:
            print('Task success:', self.task_success, 'Water reward:', reward_water)

        info = {'total_force_on_human': self.total_force_on_human, 'task_success': int(self.task_success >= self.total_water_count*self.config('task_success_threshold')), 'action_robot_len': self.action_robot_len, 'action_human_len': self.action_human_len, 'obs_robot_len': self.obs_robot_len, 'obs_human_len': self.obs_human_len}
        done = self.iteration >= 200

        if not self.human.controllable:
            return obs, reward, done, info
        else:
            # Co-optimization with both human and robot controllable
            return obs, {'robot': reward, 'human': reward}, {'robot': done, 'human': done, '__all__': done}, {'robot': info, 'human': info}

    def get_total_force(self):
        robot_force_on_human = np.sum(self.robot.get_contact_points(self.human)[-1])
        cup_force_on_human = np.sum(self.tool.get_contact_points(self.human)[-1])
        return robot_force_on_human, cup_force_on_human

    def get_water_rewards(self):
        # Check all water particles to see if they have entered the person's mouth or have left the scene
        # Delete such particles and give the robot a reward or penalty depending on particle status
        cup_pos, cup_orient = self.tool.get_base_pos_orient()
        cup_pos, cup_orient = p.multiplyTransforms(cup_pos, cup_orient, [0, 0.06, 0], self.get_quaternion([np.pi/2.0, 0, 0]), physicsClientId=self.id)
        top_center_pos = np.array(p.multiplyTransforms(cup_pos, cup_orient, self.cup_top_center_offset, [0, 0, 0, 1], physicsClientId=self.id)[0])
        bottom_center_pos = np.array(p.multiplyTransforms(cup_pos, cup_orient, self.cup_bottom_center_offset, [0, 0, 0, 1], physicsClientId=self.id)[0])
        water_reward = 0
        water_hit_human_reward = 0
        water_mouth_velocities = []
        waters_to_remove = []
        waters_active_to_remove = []
        for w in self.waters:
            water_pos, water_orient = w.get_base_pos_orient()
            if not self.util.points_in_cylinder(top_center_pos, bottom_center_pos, 0.05, np.array(water_pos)):
                distance_to_mouth = np.linalg.norm(self.target_pos - water_pos)
                if distance_to_mouth < 0.03: # hard
                # if distance_to_mouth < 0.05: # easy
                    # Delete particle and give robot a reward
                    water_reward += 10
                    self.task_success += 1
                    water_velocity = np.linalg.norm(w.get_velocity(w.base))
                    water_mouth_velocities.append(water_velocity)
                    waters_to_remove.append(w)
                    waters_active_to_remove.append(w)
                    w.set_base_pos_orient(self.np_random.uniform(1000, 2000, size=3), [0, 0, 0, 1])
                    continue
                elif len(w.get_closest_points(self.tool, distance=0.1)[-1]) == 0:
                    # Delete particle and give robot a penalty for spilling water
                    water_reward -= 1
                    waters_to_remove.append(w)
                    continue
        for w in self.waters_active:
            if len(w.get_contact_points(self.human)[-1]) > 0:
                # Record that this water particle just hit the person, so that we can penalize the robot
                water_hit_human_reward -= 1
                waters_active_to_remove.append(w)
        self.waters = [w for w in self.waters if w not in waters_to_remove]
        self.waters_active = [w for w in self.waters_active if w not in waters_active_to_remove]
        return water_reward, water_mouth_velocities, water_hit_human_reward

    def _get_obs(self, agent=None):
        cup_pos, cup_orient = self.tool.get_base_pos_orient()
        cup_pos_real, cup_orient_real = self.robot.convert_to_realworld(cup_pos, cup_orient)
        robot_joint_angles = self.robot.get_joint_angles(self.robot.controllable_joint_indices)
        # Fix joint angles to be in [-pi, pi]
        robot_joint_angles = (np.array(robot_joint_angles) + np.pi) % (2*np.pi) - np.pi
        if self.robot.mobile:
            # Don't include joint angles for the wheels
            robot_joint_angles = robot_joint_angles[len(self.robot.wheel_joint_indices):]
        head_pos, head_orient = self.human.get_pos_orient(self.human.head)
        head_pos_real, head_orient_real = self.robot.convert_to_realworld(head_pos, head_orient)
        target_pos_real, _ = self.robot.convert_to_realworld(self.target_pos)
        self.robot_force_on_human, self.cup_force_on_human = self.get_total_force()
        self.total_force_on_human = self.robot_force_on_human + self.cup_force_on_human
        robot_obs = np.concatenate([cup_pos_real, cup_orient_real, cup_pos_real - target_pos_real, robot_joint_angles, head_pos_real, head_orient_real, [self.cup_force_on_human]]).ravel()
        if agent == 'robot':
            return robot_obs
        if self.human.controllable:
            human_joint_angles = self.human.get_joint_angles(self.human.controllable_joint_indices)
            cup_pos_human, cup_orient_human = self.human.convert_to_realworld(cup_pos, cup_orient)
            head_pos_human, head_orient_human = self.human.convert_to_realworld(head_pos, head_orient)
            target_pos_human, _ = self.human.convert_to_realworld(self.target_pos)
            human_obs = np.concatenate([cup_pos_human, cup_orient_human, cup_pos_human - target_pos_human, human_joint_angles, head_pos_human, head_orient_human, [self.robot_force_on_human, self.cup_force_on_human]]).ravel()
            if agent == 'human':
                return human_obs
            # Co-optimization with both human and robot controllable
            return {'robot': robot_obs, 'human': human_obs}
        return robot_obs

    def reset(self):
        super(DrinkingEnv, self).reset()
        self.build_assistive_env('wheelchair')
        if self.robot.wheelchair_mounted:
            wheelchair_pos, wheelchair_orient = self.furniture.get_base_pos_orient()
            self.robot.set_base_pos_orient(wheelchair_pos + np.array(self.robot.toc_base_pos_offset[self.task]), [0, 0, -np.pi/2.0])

        # Update robot and human motor gains
        self.robot.motor_gains = self.human.motor_gains = 0.005

        joints_positions = [(self.human.j_right_elbow, -90), (self.human.j_left_elbow, -90), (self.human.j_right_hip_x, -90), (self.human.j_right_knee, 80), (self.human.j_left_hip_x, -90), (self.human.j_left_knee, 80)]
        joints_positions += [(self.human.j_head_x, self.np_random.uniform(-30, 30)), (self.human.j_head_y, self.np_random.uniform(-30, 30)), (self.human.j_head_z, self.np_random.uniform(-30, 30))]
        self.human.setup_joints(joints_positions, use_static_joints=True, reactive_force=None)

        self.generate_target()

        p.resetDebugVisualizerCamera(cameraDistance=1.10, cameraYaw=55, cameraPitch=-45, cameraTargetPosition=[-0.2, 0, 0.75], physicsClientId=self.id)

        # Initialize the tool in the robot's gripper
        self.tool.init(self.robot, self.task, self.directory, self.id, self.np_random, right=True, mesh_scale=[0.045]*3, alpha=0.75)
        self.cup_top_center_offset = np.array([0, 0, -0.055])
        self.cup_bottom_center_offset = np.array([0, 0, 0.07])

        target_ee_pos = np.array([-0.2, -0.5, 1]) + self.np_random.uniform(-0.05, 0.05, size=3)
        target_ee_orient = self.get_quaternion(self.robot.toc_ee_orient_rpy[self.task])
        self.init_robot_pose(target_ee_pos, target_ee_orient, [(target_ee_pos, target_ee_orient), (self.target_pos, None)], [(self.target_pos, target_ee_orient)], arm='right', tools=[self.tool], collision_objects=[self.human, self.furniture])

        # Open gripper to hold the tool
        self.robot.set_gripper_open_position(self.robot.right_gripper_indices, self.robot.gripper_pos[self.task], set_instantly=True)

        if not self.robot.mobile:
            self.robot.set_gravity(0, 0, 0)
        self.human.set_gravity(0, 0, 0)
        self.tool.set_gravity(0, 0, 0)

        p.setPhysicsEngineParameter(numSubSteps=4, numSolverIterations=10, physicsClientId=self.id)

        # Generate water
        cup_pos, cup_orient = self.tool.get_base_pos_orient()
        water_radius = 0.005
        water_mass = 0.001
        batch_positions = []
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    batch_positions.append(np.array([i*2*water_radius-0.02, j*2*water_radius-0.02, k*2*water_radius+0.075]) + cup_pos)
        self.waters = self.create_spheres(radius=water_radius, mass=water_mass, batch_positions=batch_positions, visual=False, collision=True)
        for w in self.waters:
            p.changeVisualShape(w.body, -1, rgbaColor=[0.25, 0.5, 1, 1], physicsClientId=self.id)
        self.total_water_count = len(self.waters)
        self.waters_active = [w for w in self.waters]

        # Enable rendering
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1, physicsClientId=self.id)

        # Drop water in the cup
        for _ in range(50):
            p.stepSimulation(physicsClientId=self.id)

        self.init_env_variables()
        return self._get_obs()

    def generate_target(self):
        # Set target on mouth
        self.mouth_pos = [0, -0.11, 0.03] if self.human.gender == 'male' else [0, -0.1, 0.03]
        head_pos, head_orient = self.human.get_pos_orient(self.human.head)
        target_pos, target_orient = p.multiplyTransforms(head_pos, head_orient, self.mouth_pos, [0, 0, 0, 1], physicsClientId=self.id)
        self.target = self.create_sphere(radius=0.01, mass=0.0, pos=target_pos, collision=False, rgba=[0, 1, 0, 1])
        self.update_targets()

    def update_targets(self):
        head_pos, head_orient = self.human.get_pos_orient(self.human.head)
        target_pos, target_orient = p.multiplyTransforms(head_pos, head_orient, self.mouth_pos, [0, 0, 0, 1], physicsClientId=self.id)
        self.target_pos = np.array(target_pos)
        self.target.set_base_pos_orient(self.target_pos, [0, 0, 0, 1])

