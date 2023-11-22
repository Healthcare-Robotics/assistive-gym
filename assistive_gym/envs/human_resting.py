import os, time
import numpy as np
import pybullet as p
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import smplx
import pickle
import torch
from scipy import optimize

from .env import AssistiveEnv
from .agents import human
from .agents.human import Human
from scipy.spatial.transform import Rotation as R
from pytorch3d import transforms as t3d


# human_controllable_joint_indices = human.motion_right_arm_joints

# join all lists
human_controllable_joint_indices = (
    human.right_arm_joints
    + human.left_arm_joints
    + human.right_leg_joints
    + human.left_leg_joints
    + human.head_joints
)


class HumanRestingEnv(AssistiveEnv):
    def __init__(self, use_mesh=False):
        super(HumanRestingEnv, self).__init__(
            robot=None,
            human=Human(human_controllable_joint_indices, controllable=True),
            task="human_resting",
            obs_robot_len=0,
            obs_human_len=0,
            frame_skip=5,
            time_step=0.02,
        )
        self.use_mesh = use_mesh
        self.f_name = os.path.join(os.getcwd(), "examples/data/smpl_bp_ros_smpl_re2.pkl")
        print(self.f_name)
        self.save_fname = "human_joints_simulation_realtime_1.pkl"
        self.save_fname_png = "human_joints_simulation_realtime_1.png"
        self.count = 0

    def set_file_name(self, pkl_file_name):
        self.f_name = pkl_file_name

    def step(self, action):
        # self.take_step(action, action_multiplier=0.003)
        # self.count += 1
        # print('-----------------------step--------------------------', self.count)
        # self.convert_smpl_body_to_gym()
        if self.count == 30:
            # save the human pose
            self.save_human_model()

        return np.zeros(1), 0, False, {}

    def save_human_model(self):
        # save the 3D human pose
        joints_3d_h = self.get_human_joint_position()
        dict_item = {"human_joint_3d": joints_3d_h}
        f = open(self.save_fname, "wb")
        print("-----------------------Saved------Human3djoints--------------------")
        pickle.dump(dict_item, f)
        img, depth = self.get_camera_image_depth()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # save the image
        cv2.imwrite(self.save_fname_png, img)

    def _get_obs(self, agent=None):
        # self.human.set_joint_angles([self.human.j_left_elbow], [0.0])
        self.convert_smpl_body_to_gym()
        return np.zeros(1)

    def change_human_pose(self):
        self.load_smpl_model()
        for _ in range(12):
            p.stepSimulation(physicsClientId=self.id)
            self.convert_smpl_body_to_gym()

        for _ in range(32):
            p.stepSimulation(physicsClientId=self.id)

        for _ in range(100):
            st_flag = self.human_bed_collision()
            p.stepSimulation(physicsClientId=self.id)
            if st_flag:
                self.human.control(
                    self.human.all_joint_indices,
                    self.human.get_joint_angles(),
                    1.125,
                    1.1,
                )
                break

    def reset(self):
        super(HumanRestingEnv, self).reset()

        with open(self.f_name, "rb") as handle:
            data = pickle.load(handle)

        # print('--data', data)
        # set the human body shape by beta
        self.human.body_shape = torch.from_numpy(np.array(data["betas"])) #TODO: clean

        self.build_assistive_env(
            furniture_type="hospital_bed",
            fixed_human_base=True,
            gender="male",
            human_impairment="none",
        )  # fixedhumanbas-True works well not realistic

        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1, physicsClientId=self.id)
        p.resetDebugVisualizerCamera(
            cameraDistance=1.5,
            cameraYaw=0,
            cameraPitch=-80,
            cameraTargetPosition=[0.12, 0, 1.5],
            physicsClientId=self.id,
        )

        self.furniture.set_friction(self.furniture.base, friction=0.5)
        self.human.set_whole_body_frictions(
            lateral_friction=1, spinning_friction=1, rolling_friction=1
        )
        self.human.set_gravity(0, 0, -0.1)
        # Setup human in the air and let them settle into a resting pose on the bed
        joints_positions = []
        self.human.setup_joints(
            joints_positions, use_static_joints=False, reactive_force=4.8
        )
        self.human.set_base_pos_orient([0, 0, 0], [-np.pi / 2.0, 0, 0])
        # Add small variation in human joint positions
        (
            motor_indices,
            motor_positions,
            motor_velocities,
            motor_torques,
        ) = self.human.get_motor_joint_states()
        self.human.set_joint_angles(
            motor_indices,
            motor_positions
            + self.np_random.uniform(-0.1, 0.1, size=len(motor_indices)),
        )

        for joints_j in self.human.right_arm_joints:
            self.human.enable_force_torque_sensor(joints_j)

        self.setup_camera_rpy(
            camera_target=[0, -0.2, 3.101],
            distance=0.01,
            rpy=[0, -90, 0],
            fov=60,
            camera_width=1920,
            camera_height=1080,
        )  # camera_width=1920//2, camera_height=1080//2

        # Enable rendering
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1, physicsClientId=self.id)
        self.load_smpl_model()

        # opts_joints=[self.human.j_left_shoulder_x, self.human.j_left_shoulder_y, self.human.j_left_shoulder_z,self.human.j_left_elbow ]
        # #sol_point = self.cma_optimizer(opts_joints=[self.human.j_left_shoulder_x, self.human.j_left_shoulder_y, self.human.j_left_shoulder_z,self.human.j_left_elbow ])
        # # opts_joints = [10, 11, 12, 13, 14, 15, 16]
        # #self.human.set_joint_angles(opts_joints, sol_point)
        # #human_jts = self.human_pose_smpl_format()
        # radius = 0.01
        # mass = 0.001
        # #spheres1 = self.create_spheres(radius=radius, mass=mass, batch_positions=henry_jts, visual=True, collision=False, rgba=[0, 1, 1, 1])
        # #spheres2 = self.create_spheres(radius=radius, mass=mass, batch_positions=human_jts, visual=True, collision=False, rgba=[1, 0, 1, 1])

        p.setGravity(0, 0, -1.15, physicsClientId=self.id)
        self.human.set_gravity(0, 0, -1.15)

        # joint_pose = p.calculateInverseKinematics2(self.human.body, tar_toe_pos)

        # for _ in range(20):
        #     p.stepSimulation(physicsClientId=self.id)

        # time.sleep(10)
        # # Lock the person in place
        # self.human.control(self.human.all_joint_indices, self.human.get_joint_angles(), 0.025, 5)
        # self.human.set_mass(self.human.base, mass=10.1)
        # self.human.set_mass(self.human.waist, mass=10.1)
        # self.human.set_mass(self.human.head, mass=10.1)
        # self.human.set_mass(self.human.left_knee, mass=10.1)
        # self.human.set_mass(self.human.right_knee, mass=10.1)
        # self.human.set_mass(self.human.left_ankle, mass=10.1)
        # self.human.set_mass(self.human.right_ankle, mass=10.1)
        # self.human.set_mass(self.human.left_shoulder, mass=10.1)
        # self.human.set_mass(self.human.right_shoulder, mass=10.1)
        # self.human.set_mass(self.human.left_wrist, mass=10.1)
        # self.human.set_base_velocity(linear_velocity=[0, 0, 0], angular_velocity=[0, 0, 0])
        # self.human.set_base_velocity(linear_velocity=[0, 0, -0.3], angular_velocity=[0, 0, 0])
        self.convert_smpl_body_to_gym()

        # self.convert_smpl_body_to_gym()
        # print('Completed reset function')
        # self.human.set_joint_angles(opts_joints, sol_point)
        self.human.set_mass(self.human.base, mass=10.1)
        self.human.set_mass(self.human.head, mass=10.1)
        # self.sample_pkl = self.sample_pkl+1
        # #self.convert_smpl_body_to_gym()
        # if self.sample_pkl in self.sample_pkl_list:
        #     self.sample_pkl = self.sample_pkl+1

        # mid_angle = np.array([-1.28484584e-14,  1.74532925e-01,  1.24455499e-14, -8.98027383e-02, -1.10828408e+00,
        #       6.82478992e-14, 0, 0,  7.85398163e-02,  1.74532925e-01])

        # self.human.set_joint_angles(self.human.left_arm_joints, mid_angle)

        # self.init_env_variables()
        return self._get_obs()

    def human_bed_collision(self):
        # when human fall on bed flag will change from false to true
        collision_objects = [self.furniture]
        dists_list = []
        for obj in collision_objects:
            dists = self.human.get_closest_points(obj, distance=0)[-1]
            dists_list.append(dists)
            # print('obj ', obj, ' return ',dists )

        collision_flag = True
        if all(not d for d in dists_list):
            collision_flag = False

        # print('--collision: ',collision_flag)
        return collision_flag

    def load_smpl_model(self):
        # print('Smple sample', self.sample_pkl)
        with open(self.f_name, "rb") as handle:
            data = pickle.load(handle)

        print("data keys", data.keys())
        print(
            "body_pose: ",
            data["body_pose"].shape,
            "betas: ",
            data["betas"].shape,
            "global_orient: ",
            data["global_orient"].shape,
        )
        # print('data', data)
        df = torch.from_numpy(np.array(data["body_pose"]))
        dt = torch.reshape(df, (1, 24, 3))  # TODO: Why is this 24?
        db = dt[:, :, :]  # for real time model #TODO: wtf is this
        # dt = torch.reshape(df, (1, 23, 3)) # for simulation model
        # db = dt[:,:21,:] # for simulation model
        self.m_pose = dt
        # print('-------bodyshape---',dt)
        self.smpl_body_pose = db[0].numpy()
        (
            self.human_pos_offset,
            self.human_orient_offset,
        ) = self.human.get_base_pos_orient()
        # self.henry_joints = np.array(data["human_joints_3D_est"]) + np.array(
        #     [-0.35, -1.3, 1.2]
        # )

        orient_tensor = torch.from_numpy(np.array(data["global_orient"]))
        self.orient_body = orient_tensor.numpy()
        # ang = self.orient_body[2]
        # self.human.set_base_pos_orient(
        #     [0, 0.02, 0.99195], [-np.pi / 2.0, ang, 0]
        # )  # p.getQuaternionFromEuler(np.array(euler))
        self.human.set_base_pos_orient(
            [0, 0.02, 0.99195], [-np.pi / 2.0, 0, 0]
        ) 


    ##--------------------------------------
    # -- Joint Legend --

    # 0-2 right_pecs x,y,z 14 (collar bone)
    # 3-5 right_shoulder_socket x,y,z 17 (shoulder)
    # 6 right_elbow x 19 (elbow)
    # 7 right_forearm z (no transform)
    # 8-9 right_wrist x,y 21 (wrist)
    # 10-12 left_shoulder x,y,z 13
    # 13-15 left_shoulder_socket x,y,z 16
    # 16 left_elbow x 18
    # 17 left_forearm_roll z 20
    # 18-19 left_hand x,y 20
    # 20 neck x 12
    # 21-23 head x,y,z 15
    # 24 stomach x 3
    # 25-27 waist x,y,z 0
    # 28-30 right_hip x,y,z 2
    # 31 right_knee x 5
    # 32-34 right_ankle x,y,z 8
    # 35-37 left_hip x,y,z 1
    # 38 left_knee x 4
    # 39-41 left_ankle x,y,z 7

    # corresponding_smpl_orders = [14, 17, 19, 21, 21, 13, 16, 18, 20, 20, 12, 15, 3, 0, 2, 5, 8, 1, 4, 7]

    def convert_smpl_body_to_gym(self):
        pose = self.smpl_body_pose
        print("pose: ", pose.shape)

        # R.from_rotvec(pr).as_quat()

        smpl_bp = np.zeros((24, 3))
        for i in range(len(pose)):
            mat = t3d.axis_angle_to_matrix(torch.from_numpy(pose[i]))
            print ("mat", mat)
            quats = t3d.matrix_to_quaternion(mat)
            euler = t3d.matrix_to_euler_angles(mat, "XYZ")
            # rot_mat = t3d.quaternion_to_matrix(quats)
            smpl_bp[i] = euler
            # smpl_bp[i] = R.from_rotvec(pose[i]).as_euler('xyz')
            print ("i", i, "pose[i]", pose[i], "smpl_bp[i]", smpl_bp[i], "quats", quats, 'mat', mat, 'euler', euler, 'deg', np.rad2deg(euler))
            # smpl_bp[i] = R.from_rotvec(pose[i]).as_euler('xyz')
            # smpl_bp[i] = np.rad2deg(smpl_bp[i])
        # right hand
        joints_angles = [
            smpl_bp[14, 2],
            smpl_bp[14, 1],
            smpl_bp[14, 0],
            smpl_bp[17, 0],
            smpl_bp[17, 1],
            smpl_bp[17, 2],
            smpl_bp[19, 0],
            smpl_bp[21][2],
            smpl_bp[21][0],
            smpl_bp[21][1],
        ]
        print("joints_angles", joints_angles)
        self.human.set_joint_angles(self.human.right_arm_joints, joints_angles, use_limits=True)

        # left hand
        joints_angles = [
            smpl_bp[13, 0],
            smpl_bp[13, 1],
            smpl_bp[13, 2],
            smpl_bp[16, 0],
            smpl_bp[16, 1],
            smpl_bp[16, 2],
            smpl_bp[18, 0],
            smpl_bp[20][2],
            smpl_bp[20][0],
            smpl_bp[20][1],
        ]
        self.human.set_joint_angles(self.human.left_arm_joints, joints_angles, use_limits=True)

        # right leg
        joints_angles = [
            smpl_bp[2, 0],
            smpl_bp[2, 1],
            smpl_bp[2, 2],
            smpl_bp[5, 0],
            smpl_bp[8, 0],
            smpl_bp[8, 1],
            smpl_bp[8, 2],
        ]
        self.human.set_joint_angles(self.human.right_leg_joints, joints_angles, use_limits=True)

        # #left leg
        joints_angles = [
            smpl_bp[1, 0],
            smpl_bp[1, 1],
            smpl_bp[1, 2],
            smpl_bp[4, 0],
            smpl_bp[7, 0],
            smpl_bp[7, 1],
            smpl_bp[7, 2],
        ]
        self.human.set_joint_angles(self.human.left_leg_joints, joints_angles, use_limits=True)

        # # head
        joints_angles = [smpl_bp[12, 0], smpl_bp[15, 0], smpl_bp[15, 1], smpl_bp[15, 2]]
        self.human.set_joint_angles(self.human.head_joints, joints_angles)

        # # waist
        opts_joints = [24, 25, 26, 27]
        joints_angles = [smpl_bp[3, 0], smpl_bp[0, 0], smpl_bp[0, 1], smpl_bp[0, 2]]
        self.human.set_joint_angles(opts_joints, joints_angles)

    def get_human_joint_position(self):
        agym_jt = np.zeros((20, 3))
        # agym_jt
        agym_jt[0, :] = self.human.get_pos_orient(self.human.head)[0]
        agym_jt[1, :] = self.human.get_pos_orient(self.human.j_neck)[
            0
        ]  # try lower_neck
        agym_jt[2, :] = self.human.get_pos_orient(self.human.stomach)[0]
        agym_jt[3, :] = self.human.get_pos_orient(self.human.waist)[0]
        # agym_jt[4,:] = self.human.get_pos_orient(self.human.upper_chest)[0]
        # agym_jt[5,:] = self.human.get_pos_orient(self.human.j_upper_chest_x)[0]

        # arms - adding this updated version of the arms but leaving the original in case anything goes wrong
        agym_jt[6, :] = self.human.get_pos_orient(self.human.right_shoulder)[0]
        agym_jt[7, :] = self.human.get_pos_orient(self.human.left_shoulder)[0]
        agym_jt[8, :] = self.human.get_pos_orient(self.human.right_elbow)[0]
        agym_jt[9, :] = self.human.get_pos_orient(self.human.left_elbow)[0]
        agym_jt[10, :] = self.human.get_pos_orient(self.human.right_wrist)[0]
        agym_jt[11, :] = self.human.get_pos_orient(self.human.left_wrist)[0]
        agym_jt[12, :] = self.human.get_pos_orient(self.human.right_pecs)[0]
        agym_jt[13, :] = self.human.get_pos_orient(self.human.left_pecs)[0]

        # 7,9,11,13 left arm
        # 6,8,10,12 right arm
        # legs - same with arms
        agym_jt[14, :] = self.human.get_pos_orient(self.human.right_knee)[0]
        agym_jt[15, :] = self.human.get_pos_orient(self.human.left_knee)[0]
        agym_jt[16, :] = self.human.get_pos_orient(self.human.right_hip)[0]
        agym_jt[17, :] = self.human.get_pos_orient(self.human.left_hip)[0]
        agym_jt[18, :] = self.human.get_pos_orient(self.human.right_ankle)[0]
        agym_jt[19, :] = self.human.get_pos_orient(self.human.left_ankle)[0]

        # for i_ in range(20):
        #     agym_jt[i_][2] = agym_jt[i_][2]+0.3

        return agym_jt
