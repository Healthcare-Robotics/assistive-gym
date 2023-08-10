import os
import time

from assistive_gym.envs.agents.pr2 import PR2
from assistive_gym.envs.agents.sawyer import Sawyer
from assistive_gym.envs.agents.stretch import Stretch
from assistive_gym.envs.agents.stretch_dex import StretchDex
from assistive_gym.envs.env import AssistiveEnv
from assistive_gym.envs.utils.human_utils import set_self_collisions, disable_self_collisions
from assistive_gym.envs.utils.urdf_utils import load_smpl
from assistive_gym.envs.utils.human_urdf_dict import HumanUrdfDict
from experimental.human_urdf import HumanUrdf
from ergonomics.reba import RebaScore
import numpy as np
import pybullet as p

class HumanComfortEnv(AssistiveEnv):
    def __init__(self):
        self.robot = StretchDex('wheel_right')
        self.human = HumanUrdf()

        super(HumanComfortEnv, self).__init__(robot=self.robot, human=self.human, task='', obs_robot_len=len(self.robot.controllable_joint_indices), 
                                         obs_human_len=len(self.human.controllable_joint_indices)) #hardcoded
        self.target_pos = np.array([0, 0, 0])
        self.smpl_file = None
        self.task = None # task = 'comfort_standing_up', 'comfort_taking_medicine',  'comfort_drinking'


    def get_comfort_score(self):
        return np.random.rand() #TODO: implement this
    # TODO: refactor train to move the score return to env.step

    def set_smpl_file(self, smpl_file):
        self.smpl_file = smpl_file

    def set_human_urdf(self, urdf_path):
        self.human.set_urdf_path(urdf_path)

    def set_task(self, task):
        if not task: # default task
            task = "comfort_taking_medicine"
        self.task = task  # task = 'comfort_standing_up', 'comfort_taking_medicine',  'comfort_drinking'

    def step(self, action):
        if self.human.controllable:
            # print("action", action)
            action = np.concatenate([action['robot'], action['human']])

        self.take_step(action)

        obs = self._get_obs()

        # comfort_score = self.get_comfort_score()
        # reward = comfort_score
        # if self.gui and comfort_score != 0:
        #     # print('Task success:', self.task_success, 'Food reward:', comfort_score)
        #     pass
        #
        # info = {'comfort_score': comfort_score}
        # done = self.iteration >= 200
        # print (done, self.iteration)
        # if not self.human.controllable:
        #     return obs, reward, done, info
        # else:
        #     # Co-optimization with both human and robot controllable
        #     return obs, {'robot': reward, 'human': reward}, {'robot': done, 'human': done, '__all__': done}, {
        #         'robot': info, 'human': info}
        return None

    def _get_obs(self, agent=None): # not needed
        target_pos_real, _ = self.robot.convert_to_realworld(self.target_pos) 

        robot_obs = np.array([]) # TODO: fix
        if agent == 'robot':
            return robot_obs
        if self.human.controllable:
            human_joint_angles = self.human.get_joint_angles(self.human.controllable_joint_indices)
            target_pos_human, _ = self.human.convert_to_realworld(self.target_pos)
            human_obs =np.array(human_joint_angles)
            if agent == 'human':
                return human_obs

            return {'robot': robot_obs, 'human': human_obs}
        return robot_obs

    def reset_human(self, is_collision):
        if not is_collision:
            self.human.set_joint_angles_with_smpl(load_smpl(self.smpl_file)) #TODO: fix
        else:
            bed_height, bed_base_height = self.furniture.get_heights(set_on_ground=True)
            smpl_data = load_smpl(self.smpl_file)
            self.human.set_joint_angles_with_smpl(smpl_data)
            height, base_height = self.human.get_heights()
            print("human height ", height, base_height, "bed height ", bed_height, bed_base_height)
            self.human.set_global_orientation(smpl_data, [0, 0, bed_height])
            self.human.set_gravity(0, 0, -9.81)
            p.setPhysicsEngineParameter(numSubSteps=4, numSolverIterations=10, physicsClientId=self.id)
            for _ in range(100):
                p.stepSimulation(physicsClientId=self.id)

    def reset(self):
        super(HumanComfortEnv, self).reset()

        # magic happen here - now call agent.init()
        self.build_assistive_env("hospital_bed")

        bed_height, bed_base_height = self.furniture.get_heights(set_on_ground=True)
        min_pos, max_pos = p.getAABB(self.furniture.body, physicsClientId=self.id)
        print("bed height ", bed_height, bed_base_height, "bed pos ", min_pos, max_pos)
        # reset human pose
        smpl_data = load_smpl(self.smpl_file)
        self.human.set_joint_angles_with_smpl(smpl_data)
        height, base_height = self.human.get_heights()
        print ("human height ", height, base_height, "bed height ", bed_height, bed_base_height)
        self.human.set_global_orientation(smpl_data, [0, 0,  bed_height+0.2])

        self.robot.set_gravity(0, 0, -9.81)
        self.human.set_gravity(0, 0, -9.81)

        self.robot.set_joint_angles([4], [0.5]) # for stretch_dex: move the gripper upward

        # init tool
        self.tool.init(self.robot, self.task, self.directory, self.id, self.np_random, right=True,
                       mesh_scale=[0.045] * 3, alpha=0.75)
        # Open gripper to hold the tool
        self.robot.set_gripper_open_position(self.robot.right_gripper_indices, self.robot.gripper_pos[self.task],
                                             set_instantly=True)


        # debug robot
        # for j in range(p.getNumJoints(self.robot.body, physicsClientId=self.id)):
        #     print(p.getJointInfo(self.robot.body, j, physicsClientId=self.id))

        # debug human links
        # for j in range(p.getNumJoints(self.human.body, physicsClientId=self.id)):
        #     print(p.getLinkState(self.human.body, j, physicsClientId=self.id))
        # for j in range(p.getNumJoints(self.human.body, physicsClientId=self.id)):
        #     print(p.getJointInfo(self.human.body, j, physicsClientId=self.id))

        # p.setPhysicsEngineParameter(numSubSteps=4, numSolverIterations=10, physicsClientId=self.id)
        p.setTimeStep(1/240., physicsClientId=self.id)

        # disable self collision before dropping on bed
        num_joints = p.getNumJoints(self.human.body, physicsClientId=self.id)
        disable_self_collisions(self.human.body, num_joints, self.id)

        # Enable rendering
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1, physicsClientId=self.id)
        # drop human on bed
        for i in range(300):
            p.stepSimulation(physicsClientId=self.id)

        # enable self collision and reset joint angle after dropping on bed
        human_pos = p.getBasePositionAndOrientation(self.human.body, physicsClientId=self.id)[0]

        self.human.set_global_orientation(smpl_data, human_pos)
        self.human.set_joint_angles_with_smpl2(smpl_data)

        set_self_collisions(self.human.body, self.id)
        self.human.initial_self_collisions= self.human.check_self_collision()

        self.init_env_variables()
        return self._get_obs()

