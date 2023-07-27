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
from .utils.smpl_dict import SMPLDict
from torch import tensor

# human_controllable_joint_indices = human.motion_right_arm_joints
smpl_dict = SMPLDict()
# join all lists
# controllable_joints = (
#     human.right_arm_joints
#     + human.left_arm_joints
#     + human.right_leg_joints
#     + human.left_leg_joints
#     + human.head_joints
# )
controllable_joints = [0,1]

def load_smpl(filename):
    with open(filename, "rb") as handle:
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

    return data

def configure_human(human):
    human.impairment = None
    human.set_all_joints_stiffness(0.02)
    human.set_whole_body_frictions(lateral_friction=50., spinning_friction=10., rolling_friction=10.)

    # joint_pos = default_sitting_pose(human)
    # pose = armchair_seated()
    # pose = test_pose()
    # frames = Frames().armchair_frames
    # frame = "armchair001_stageII.pkl"
    # pkl_path = os.path.expanduser("~") + "/HRL/SAMP/pkl/" + frame
    # _, _, pose, _, _ = load_pose_from_pkl(pkl_path, frames[frame])
    # pose = pose.reshape(165, 1)

    # pose = pose.reshape(165, 1)
    data = load_smpl( os.path.join(os.getcwd(), "examples/data/smpl_bp_ros_smpl_3.pkl"))
    pose = data["body_pose"]
    joint_pos = unpack_smplx_pose(human, pose)
    human.setup_joints(joint_pos, use_static_joints=False, reactive_force=None)

    start_pos = [0., 0., 0.85]
    # start_pos = [0, 0, 1.2]
    # start_orient = [0, 0, 0, 1]

    start_angles = [-np.pi/2., 0., np.pi]  # XYZ Euler Angles
    start_orient = t3d.matrix_to_quaternion(t3d.euler_angles_to_matrix(tensor(start_angles), 'XYZ')).numpy()

    # start_orient = [-0.0898752, 0, 0.7058752, 0.7026113]

    # commented: pure x axis rotation by -0.2rad to -0.5rad
    # start_orient = [ -0.0998334, 0, 0, 0.9950042 ]
    # start_orient = [ -0.1494381, 0, 0, 0.9887711 ]
    # start_orient = [ -0.1986693, 0, 0, 0.9800666 ]
    # start_orient = [ -0.247404, 0, 0, 0.9689124 ]

    human.set_base_pos_orient(start_pos, start_orient)
    # human.set_on_ground()

    joint_i = [pose[0] for pose in joint_pos]
    joint_th = [pose[1] for pose in joint_pos]
    joint_gains = [0.] * len(joint_i)
    # forces = [50.] * len(joint_i)
    forces = [0.] * len(joint_i)

    # tweak joint control
    for i in range(len(joint_gains)):
        if i not in controllable_joints or i in human.right_arm_joints:
            joint_gains[i] = 0.
            forces[i] = 0.

    human.control(joint_i, joint_th, joint_gains, forces)
    human.is_controllable = True

    human.set_base_pos_orient(
            [0, 0.02, 0.99195], [-np.pi / 2.0, 0, 0]
    
    ) 
    # human.set_on_ground()

# def set_joint_stiffnesses(human):
#     human.set_joint_stiffness(human.j_)


def default_sitting_pose(human):
    # Arms
    joint_pos = [(human.j_right_shoulder_x, 30.),
                 (human.j_left_shoulder_x, -30.),
                 (human.j_right_shoulder_y, 0.),
                 (human.j_left_shoulder_y, 0.),
                 (human.j_right_elbow, -90.),
                 (human.j_left_elbow, -90.)]

    # Legs
    joint_pos += [(human.j_right_knee, 90.),
                  (human.j_left_knee, 90.),
                  (human.j_right_hip_x, -90.),
                  (human.j_left_hip_x, -90.)]

    # Torso
    joint_pos += [(human.j_waist_x, 0.)]
    return joint_pos


def convert_aa_to_euler(aa):
    aa = np.array(aa)
    mat = t3d.axis_angle_to_matrix(torch.from_numpy(aa))
    # print ("mat", mat)
    quats = t3d.matrix_to_quaternion(mat)
    euler = t3d.matrix_to_euler_angles(mat, "XYZ")
    return euler

def unpack_smplx_pose(human, pose):
    print (pose[smpl_dict.get_pose_ids("right_shoulder")])
    # unpack smplx pose
    right_shoulder = convert_aa_to_euler(pose[smpl_dict.get_pose_ids("right_shoulder")])
    left_shoulder = convert_aa_to_euler(pose[smpl_dict.get_pose_ids("left_shoulder")])
    right_elbow = convert_aa_to_euler(pose[smpl_dict.get_pose_ids("right_elbow")])
    left_elbow = convert_aa_to_euler(pose[smpl_dict.get_pose_ids("left_elbow")])
    right_wrist = convert_aa_to_euler(pose[smpl_dict.get_pose_ids("right_wrist")])
    left_wrist = convert_aa_to_euler(pose[smpl_dict.get_pose_ids("left_wrist")])
    right_hip = convert_aa_to_euler(pose[smpl_dict.get_pose_ids("right_hip")])
    left_hip = convert_aa_to_euler(pose[smpl_dict.get_pose_ids("left_hip")])
    right_knee = convert_aa_to_euler(pose[smpl_dict.get_pose_ids("right_knee")])
    left_knee = convert_aa_to_euler(pose[smpl_dict.get_pose_ids("left_knee")])
    lower_spine = convert_aa_to_euler(pose[smpl_dict.get_pose_ids("lower_spine")])
    right_collar = convert_aa_to_euler(pose[smpl_dict.get_pose_ids("right_collar")])
    left_collar = convert_aa_to_euler(pose[smpl_dict.get_pose_ids("left_collar")])
    neck = convert_aa_to_euler(pose[smpl_dict.get_pose_ids("neck")])
    head = convert_aa_to_euler(pose[smpl_dict.get_pose_ids("head")])

    # Arms
    # joint_pos = [(human.j_right_shoulder_x, -right_shoulder[2] + pi/2.),
    #              (human.j_right_shoulder_y, -right_shoulder[1]),
    #              (human.j_right_shoulder_z, right_shoulder[0]),
    #              (human.j_left_shoulder_x, -left_shoulder[2] - pi/2.),
    #              (human.j_left_shoulder_y, left_shoulder[1]),
    #              (human.j_left_shoulder_z, -left_shoulder[0]),
    #              (human.j_right_elbow, -right_elbow[1]),
    #              (human.j_left_elbow, left_elbow[1])]
    #              # (human.j_right_wrist_x, right_wrist[0]),   
    #              # (human.j_left_wrist_x, left_wrist[0])]

    # print (right_shoulder, left_shoulder    )
    joint_pos = [(human.j_right_shoulder_x, -right_shoulder[2]-np.pi/2.),
                 (human.j_right_shoulder_y, -right_shoulder[0]),
                 (human.j_right_shoulder_z, right_shoulder[1]),
                 (human.j_left_shoulder_x, -left_shoulder[2]+np.pi/2.),
                 (human.j_left_shoulder_y, -left_shoulder[0]),
                 (human.j_left_shoulder_z, left_shoulder[1]),
                 (human.j_right_elbow, -right_elbow[1]),
                 (human.j_left_elbow, left_elbow[1])]
                 # (human.j_right_wrist_x, right_wrist[0]),
                 # (human.j_left_wrist_x, left_wrist[0])]

    # Legs
    # joint_pos += [(human.j_left_hip_x, left_hip[0]),
    #               (human.j_left_hip_y, -left_hip[2]),
    #               (human.j_left_hip_z, left_hip[1]),
    #               (human.j_right_hip_x, right_hip[0]),
    #               (human.j_right_hip_y, -right_hip[2]),
    #               (human.j_right_hip_z, right_hip[1]),
    #               (human.j_left_knee, left_knee[0]),
    #               (human.j_right_knee, right_knee[0])]

    joint_pos += [(human.j_left_hip_x, left_hip[0]),
                  (human.j_left_hip_y, -left_hip[2]),
                  (human.j_left_hip_z, left_hip[1]),
                  (human.j_right_hip_x, right_hip[0]),
                  (human.j_right_hip_y, -right_hip[2]),
                  (human.j_right_hip_z, right_hip[1]),
                  (human.j_left_knee, left_knee[0]),
                  (human.j_right_knee, right_knee[0])]
    print ("left_hip: ", left_hip/np.pi*180, "right_hip: ", right_hip/np.pi*180, "left_knee: ", left_knee/np.pi*180, "right_knee: ", right_knee/np.pi*180)
    # Torso
    joint_pos += [(human.j_waist_x, lower_spine[0]),
                  (human.j_waist_y, lower_spine[2]),
                  (human.j_waist_z, lower_spine[1])]
                  # (human.j_right_pecs_x, -right_collar[2]),
                  # (human.j_right_pecs_y, -right_collar[1]),
                  # (human.j_right_pecs_z, right_collar[0]),
    #               (human.j_left_pecs_x, left_collar[0]),
    #               (human.j_left_pecs_y, left_collar[1]),
    #               (human.j_left_pecs_z, left_collar[2])]

    # Head
    joint_pos += [(human.j_neck,   neck[2]),
                  (human.j_head_x, head[2]),
                  (human.j_head_y, -head[0]),
                  (human.j_head_z, head[1])]

    for i in range(len(joint_pos)):
        joint = joint_pos[i]
        # we need to convert to deg since this setup joint method use deg. also, joint[0] here is joint index and joint[1] is joint value
        joint_pos[i] = (joint[0], np.rad2deg(joint[1])) 

    return joint_pos

class HumanLyingEnv(AssistiveEnv):
    def __init__(self, use_mesh=False):
        super(HumanLyingEnv, self).__init__(
            robot=None,
            human=Human(controllable_joints, controllable=True),
            task="human_lying",
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


    def step(self, action):
        # self.take_step(action, action_multiplier=0.003)
        # self.count += 1
        # print('-----------------------step--------------------------', self.count)
        # self.convert_smpl_body_to_gym()
    
        return np.zeros(1), 0, False, {}


    def _get_obs(self, agent=None):
        # self.human.set_joint_angles([self.human.j_left_elbow], [0.0])
        # self.convert_smpl_body_to_gym()
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
        super(HumanLyingEnv, self).reset()

        # with open(self.f_name, "rb") as handle:
        #     data = pickle.load(handle)

        # print('--data', data)
        # set the human body shape by beta
        # self.human.body_shape = torch.from_numpy(np.array(data["betas"])) #TODO: clean

        self.build_assistive_env(
            furniture_type="hospital_bed",
            fixed_human_base=False,
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
        # self.human.set_whole_body_frictions(
        #     lateral_friction=1, spinning_friction=1, rolling_friction=1
        # )
        # self.human.set_gravity(0, 0, -0.1)
        # # Setup human in the air and let them settle into a resting pose on the bed
        # joints_positions = []
        # self.human.setup_joints(
        #     joints_positions, use_static_joints=False, reactive_force=4.8
        # )
        # self.human.set_base_pos_orient([0, 0, 0], [-np.pi / 2.0, 0, 0])
        # # Add small variation in human joint positions
        # (
        #     motor_indices,
        #     motor_positions,
        #     motor_velocities,
        #     motor_torques,
        # ) = self.human.get_motor_joint_states()
        # self.human.set_joint_angles(
        #     motor_indices,
        #     motor_positions
        #     + self.np_random.uniform(-0.1, 0.1, size=len(motor_indices)),
        # )

        # for joints_j in self.human.right_arm_joints:
        #     self.human.enable_force_torque_sensor(joints_j)

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

        p.setGravity(0, 0, -1.15, physicsClientId=self.id)
        self.human.set_gravity(0, 0, -1.15)

        # self.convert_smpl_body_to_gym()

        # self.convert_smpl_body_to_gym()
        # print('Completed reset function')
        # self.human.set_joint_angles(opts_joints, sol_point)
        # self.human.set_mass(self.human.base, mass=10.1)
        # self.human.set_mass(self.human.head, mass=10.1)

        configure_human(self.human)
        
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

    # def load_smpl_model(self):
    #     # print('Smple sample', self.sample_pkl)
    #     with open(self.f_name, "rb") as handle:
    #         data = pickle.load(handle)

    #     print("data keys", data.keys())
    #     print(
    #         "body_pose: ",
    #         data["body_pose"].shape,
    #         "betas: ",
    #         data["betas"].shape,
    #         "global_orient: ",
    #         data["global_orient"].shape,
    #     )
    #     # print('data', data)
    #     df = torch.from_numpy(np.array(data["body_pose"]))
    #     dt = torch.reshape(df, (1, 24, 3))  # TODO: Why is this 24?
    #     db = dt[:, :, :]  # for real time model #TODO: wtf is this
    #     self.m_pose = dt
    #     # print('-------bodyshape---',dt)
    #     self.smpl_body_pose = db[0].numpy()
    #     (
    #         self.human_pos_offset,
    #         self.human_orient_offset,
    #     ) = self.human.get_base_pos_orient()
    #     orient_tensor = torch.from_numpy(np.array(data["global_orient"]))
    #     self.orient_body = orient_tensor.numpy()
    #     # ang = self.orient_body[2]
    #     # self.human.set_base_pos_orient(
    #     #     [0, 0.02, 0.99195], [-np.pi / 2.0, ang, 0]
    #     # )  # p.getQuaternionFromEuler(np.array(euler))
    #     self.human.set_base_pos_orient(
    #         [0, 0.02, 0.99195], [-np.pi / 2.0, 0, 0]
    #     ) 


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
