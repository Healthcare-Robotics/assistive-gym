import os

from assistive_gym.envs.agents.sawyer import Sawyer
from assistive_gym.envs.agents.stretch import Stretch
from assistive_gym.envs.env import AssistiveEnv
from assistive_gym.envs.utils.urdf_utils import load_smpl
from experimental.human_urdf import HumanUrdf
import numpy as np
import pybullet as p

SMPL_PATH = os.path.join(os.getcwd(), "examples/data/smpl_bp_ros_smpl_re2.pkl")
class HumanComfortEnv(AssistiveEnv):
    def __init__(self):
        self.robot = Stretch('wheel_right')
        self.human = HumanUrdf()
        super(HumanComfortEnv, self).__init__(robot=self.robot, human=self.human, task='', obs_robot_len=len(self.robot.controllable_joint_indices),
                                         obs_human_len=len(self.human.controllable_joint_indices)) #hardcoded
        self.target_pos = np.array([0, 0, 0])
    def get_comfort_score(self):
        return np.random.rand() #TODO: implement this
    # TODO: refactor train to move the score return to env.step

    def step(self, action):
        if self.human.controllable:
            # print("action", action)
            action = np.concatenate([action['robot'], action['human']])

        self.take_step(action)

        obs = self._get_obs()

        comfort_score = self.get_comfort_score()
        reward = comfort_score
        if self.gui and comfort_score != 0:
            # print('Task success:', self.task_success, 'Food reward:', comfort_score)
            pass

        # info = {'total_force_on_human': self.total_force_on_human,
        #         'task_success': int(self.task_success >= self.total_food_count * self.config('task_success_threshold')),
        #         'action_robot_len': self.action_robot_len, 'action_human_len': self.action_human_len,
        #         'obs_robot_len': self.obs_robot_len, 'obs_human_len': self.obs_human_len}
        info = {'comfort_score': comfort_score}
        done = self.iteration >= 200
        print (done, self.iteration)
        if not self.human.controllable:
            return obs, reward, done, info
        else:
            # Co-optimization with both human and robot controllable
            return obs, {'robot': reward, 'human': reward}, {'robot': done, 'human': done, '__all__': done}, {
                'robot': info, 'human': info}

    def _get_obs(self, agent=None):
        # robot_joint_angles = self.robot.get_joint_angles(self.robot.controllable_joint_indices)
        # Fix joint angles to be in [-pi, pi]
        # robot_joint_angles = (np.array(robot_joint_angles) + np.pi) % (2 * np.pi) - np.pi
        # if self.robot.mobile:
        #     # Don't include joint angles for the wheels
        #     robot_joint_angles = robot_joint_angles[len(self.robot.wheel_joint_indices):]

        target_pos_real, _ = self.robot.convert_to_realworld(self.target_pos)
        self.robot_force_on_human= self.get_total_force()
        self.total_force_on_human = self.robot_force_on_human
        # robot_obs = np.concatenate(
        #     [robot_joint_angles]).ravel()
        robot_obs = np.array([self.robot_force_on_human])
        if agent == 'robot':
            return robot_obs
        if self.human.controllable:
            human_joint_angles = self.human.get_joint_angles(self.human.controllable_joint_indices)
            target_pos_human, _ = self.human.convert_to_realworld(self.target_pos)
            human_obs = np.concatenate(
                [human_joint_angles, [self.robot_force_on_human]]).ravel()
            if agent == 'human':
                return human_obs

            # self.human.cal_manipulibility()
            # print ("manipulability", self.human.cal_manipulibility())
            # Co-optimization with both human and robot controllable
            return {'robot': robot_obs, 'human': human_obs}
        return robot_obs

    def get_total_force(self):
        total_force_on_human = np.sum(self.robot.get_contact_points(self.human)[-1])
        # tool_force = np.sum(self.tool.get_contact_points()[-1])
        # tool_force_at_target = 0
        # target_contact_pos = None
        # for linkA, linkB, posA, posB, force in zip(*self.tool.get_contact_points(self.human)):
        #     total_force_on_human += force
        #     # Enforce that contact is close to the target location
        #     if linkA in [0, 1] and np.linalg.norm(posB - self.target_pos) < 0.025:
        #         tool_force_at_target += force
        #         target_contact_pos = posB
        # return total_force_on_human, tool_force, tool_force_at_target, None if target_contact_pos is None else np.array(
        #     target_contact_pos)
        return total_force_on_human


    def reset_human(self):
        # reset human pose
        # smpl_data = load_smpl(SMPL_PATH)
        # self.human.set_joint_angles_with_smpl(smpl_data)

        # bed_height, bed_base_height = self.furniture.get_heights(set_on_ground=True)
        # self.human.set_on_ground(bed_height)
        # self.human.set_gravity(0, 0, -9.81)
        p.stepSimulation(physicsClientId=self.id)

    def reset(self):
        super(HumanComfortEnv, self).reset()

        self.build_assistive_env("hospital_bed")

        # Update robot and human motor gains
        self.robot.motor_gains = self.human.motor_gains = 0.005

        # reset human pose
        # smpl_data = load_smpl(SMPL_PATH)
        # self.human.set_joint_angles_with_smpl(smpl_data)

        bed_height, bed_base_height = self.furniture.get_heights(set_on_ground=True)
        self.human.set_on_ground(bed_height)
        # p.resetBasePositionAndOrientation(self.human.body, [0.0, 0.0, 2.0], [0.0, 0.0, 0.0, 1.0], physicsClientId=self.id)

        if not self.robot.mobile:
            self.robot.set_gravity(0, 0, -9.81)
        # self.human.set_gravity(0, 0, -9.81)
        self.human.set_gravity(0, 0, -9.81)

        # drop human on bed
        for _ in range(100):
            p.stepSimulation(physicsClientId=self.id)

        p.setPhysicsEngineParameter(numSubSteps=4, numSolverIterations=10, physicsClientId=self.id)

        # Enable rendering
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1, physicsClientId=self.id)
        # p.setTimeStep(1/240., physicsClientId=self.id)
        self.init_env_variables()
        return self._get_obs()

