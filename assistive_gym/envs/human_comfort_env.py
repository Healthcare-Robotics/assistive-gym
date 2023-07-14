import os
import time

from assistive_gym.envs.agents.pr2 import PR2
from assistive_gym.envs.agents.sawyer import Sawyer
from assistive_gym.envs.agents.stretch import Stretch
from assistive_gym.envs.env import AssistiveEnv
from assistive_gym.envs.utils.urdf_utils import load_smpl
from assistive_gym.envs.utils.human_urdf_dict import HumanUrdfDict
from experimental.human_urdf import HumanUrdf
from ergonomics.reba import RebaScore
import numpy as np
import pybullet as p

SMPL_PATH = os.path.join(os.getcwd(), "examples/data/smpl_bp_ros_smpl_8.pkl")
class HumanComfortEnv(AssistiveEnv):
    def __init__(self):
        self.robot = Stretch('wheel_right') # ADD STRETCH
        self.human = HumanUrdf()
        super(HumanComfortEnv, self).__init__(robot=self.robot, human=self.human, task='', obs_robot_len=len(self.robot.controllable_joint_indices), # ADD STRETCH
                                         obs_human_len=len(self.human.controllable_joint_indices)) #hardcoded
        self.target_pos = np.array([0, 0, 0])
        self.smpl_file = SMPL_PATH

    def get_comfort_score(self):
        return np.random.rand() #TODO: implement this
    # TODO: refactor train to move the score return to env.step

    def set_smpl_file(self, smpl_file):
        self.smpl_file = smpl_file

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

        info = {'comfort_score': comfort_score}
        done = self.iteration >= 200
        print (done, self.iteration)
        if not self.human.controllable:
            return obs, reward, done, info
        else:
            # Co-optimization with both human and robot controllable
            return obs, {'robot': reward, 'human': reward}, {'robot': done, 'human': done, '__all__': done}, {
                'robot': info, 'human': info}

    def _get_obs(self, agent=None): # not needed
        target_pos_real, _ = self.robot.convert_to_realworld(self.target_pos) # ADD STRETCH

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


    
    def get_reba_score(self):
        human_dict = HumanUrdfDict()
        rebaScore = RebaScore()
        # list joints in the order required for a reba score
        joints = ["head", "neck", "left_shoulder", "left_elbow", "left_lowarm", "right_shoulder", "right_elbow", "right_lowarm", 
            "left_hip", "left_knee", "left_ankle", "right_hip", "right_knee", "right_ankle", "left_hand", "right_hand"]
        
        # obtain the links in the right order for the rebascore code
        dammy_ids = []
        for joint in joints:
            dammy_ids.append(human_dict.get_dammy_joint_id(joint))

        # use dammy ids to obtain the right link, use the right knee as the root joint
        pose = []
        root = p.getLinkState(self.human.body, dammy_ids[12])[0] # root joint

        for i in dammy_ids:
            # get the location of each dammy joint and append to the pose list
            loc = p.getLinkState(self.human.body, i)[0]
            norm_loc = [loc[0] - root[0], loc[1] - root[1], loc[2] - root[2]]
            pose.append(norm_loc)
        
        # convert to numpy array
        pose = np.array(pose)
        # print("pose: ", pose)

        # print("angles: ", angles, "\nlength: ", len(angles))
        sample_pose = np.array([[ 0.08533354,  1.03611605,  0.09013124],
                      [ 0.15391247,  0.91162637, -0.00353906],
                      [ 0.22379057,  0.87361878,  0.11541229],
                      [ 0.4084777 ,  0.69462843,  0.1775224 ],
                      [ 0.31665226,  0.46389668,  0.16556387],
                      [ 0.1239769 ,  0.82994377, -0.11715403],
                      [ 0.08302169,  0.58146328, -0.19830338],
                      [-0.06767788,  0.53928527, -0.00511249],
                      [ 0.11368726,  0.49372503,  0.21275574],
                      [ 0.069179  ,  0.07140968,  0.26841402],
                      [ 0.10831762, -0.36339359,  0.34032449],
                      [ 0.11368726,  0.41275504, -0.01171348],
                      [ 0.        ,  0.        ,  0.        ],
                      [ 0.02535541, -0.43954643,  0.04373671],
                      [ 0.26709431,  0.33643749,  0.17985192],
                      [-0.15117603,  0.49462711,  0.02703403]])

        # following code is from the ergonomic repo (https://github.com/rs9000/ergonomics/blob/master/ergonomics/reba.py)
        body_params = rebaScore.get_body_angles_from_pose_right(pose)
        arms_params = rebaScore.get_arms_angles_from_pose_right(pose)

        # calculate scores
        rebaScore.set_body(body_params)
        score_a, partial_a = rebaScore.compute_score_a()
        rebaScore.set_arms(arms_params)
        score_b, partial_b = rebaScore.compute_score_b()

        # get score breakdowns
        neck_score = partial_a[0]
        trunk_score = partial_a[1]
        leg_score = partial_a[2]
        upper_arm_score = partial_b[0]
        lower_arm_score = partial_b[1]
        wrist_score = partial_b[2]

        score_c, caption = rebaScore.compute_score_c(score_a, score_b)

        print("-----Reba Score for my pose-----")
        print("Score A: ", score_a, "Partial: ", partial_a)
        print("Score A: ", score_b, "Partial: ", partial_b)
        print("Score C: ", score_c, caption)
        # print score breakdowns
        print("---score breakdown---")
        print("neck: ", neck_score, "\ntrunk: ", trunk_score, "\nleg: ", leg_score, "\nupper arm: ", upper_arm_score, "\nlower arm: ",
            lower_arm_score, "\nwrist: ", wrist_score)
        
        # return all info
        ret = {"score_a": score_a, "partial_a": partial_a, "score_b": score_b, "partial_b": partial_b, "score_c": score_c, "caption": caption}
        return ret



    def reset(self):
        super(HumanComfortEnv, self).reset()

        self.build_assistive_env("hospital_bed")

        bed_height, bed_base_height = self.furniture.get_heights(set_on_ground=True)

        # reset human pose
        smpl_data = load_smpl(self.smpl_file)
        self.human.set_joint_angles_with_smpl(smpl_data)
        height, base_height = self.human.get_heights()
        print ("human height ", height, base_height, "bed height ", bed_height, bed_base_height)
        self.human.set_global_orientation(smpl_data, [0, 0,  bed_height])

        self.robot.set_gravity(0, 0, -9.81) # ADD STRETCH
        self.human.set_gravity(0, 0, -9.81)
        # debug robot
        for j in range(p.getNumJoints(self.robot.body, physicsClientId=self.id)):
            print(p.getJointInfo(self.robot.body, j, physicsClientId=self.id))

        # calculating the rebaScore
        self.get_reba_score()

        p.setPhysicsEngineParameter(numSubSteps=4, numSolverIterations=10, physicsClientId=self.id)

        # Enable rendering
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1, physicsClientId=self.id)
        # drop human on bed
        for i in range(100):
            p.stepSimulation(physicsClientId=self.id)

        # p.setTimeStep(1/240., physicsClientId=self.id)
        self.init_env_variables()
        return self._get_obs()

