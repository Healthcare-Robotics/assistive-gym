import colorsys
import os

import numpy as np
import pybullet as p
import pybullet_data
from gym.utils import seeding
from kinpy import Transform

from assistive_gym.envs.agents.agent import Agent
from assistive_gym.envs.utils.human_urdf_dict import HumanUrdfDict
from assistive_gym.envs.utils.human_utils import set_self_collisions, change_dynamic_properties, check_collision
from assistive_gym.envs.utils.smpl_dict import SMPLDict

from assistive_gym.envs.utils.smpl_geom import generate_geom, show_human_mesh
from assistive_gym.envs.utils.urdf_utils import convert_aa_to_euler_quat, load_smpl, generate_urdf, SMPLData
import kinpy as kp

SMPL_PATH = os.path.join(os.getcwd(), "examples/data/smpl_bp_ros_smpl_re2.pkl")
URDF_FILE = "test_mesh.urdf"

# generated by running self.print_all_joints(). TODO: automate this
all_controllable_joint_indices = [1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19, 21, 22, 23, 25, 26, 27,
                                       29, 30, 31, 33, 34, 35, 37, 38, 39, 41, 42, 43, 45, 46, 47, 49, 50, 51, 53,
                                       54, 55, 57, 58, 59, 61, 62, 63, 65, 66, 67, 69, 70, 71, 73, 74, 75, 77, 78,
                                       79, 81, 82, 83, 85, 86, 87, 89, 90, 91]
left_leg_joint_indices = [1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15]
right_leg_joint_indices = [17, 18, 19, 21, 22, 23, 25, 26, 27, 29, 30, 31]
# left_arm_joint_indices = [53, 54, 55, 57, 58, 59, 61, 62, 63, 65, 66, 67, 69, 70, 71]
left_arm_joint_indices = [57, 58, 59, 61, 62, 63, 65, 66, 67, 69, 70, 71]
# right_arm_joint_indices =  [77, 78, 79, 81, 82, 83, 85, 86, 87, 89, 90, 91]
right_arm_joint_indices =  [73, 74, 75, 77, 78, 79, 81, 82, 83, 85, 86, 87, 89, 90, 91] # with clavicle
body_joint_indices = [33, 34, 35, 37, 38, 39, 41, 42, 43]

all_joint_indices = list(range(0, 93))
class HumanUrdf(Agent):
    def __init__(self):
        super(HumanUrdf, self).__init__()
        self.smpl_dict = SMPLDict()
        self.human_dict = HumanUrdfDict()
        self.controllable_joint_indices = right_arm_joint_indices
        print("controllable_joint_indices: ", len(self.controllable_joint_indices))
        self.controllable = True
        self.motor_forces = 1.0
        self.motor_gains = 0.005
        self.end_effectors = ['right_hand', 'left_hand', 'right_foot', 'left_foot', 'head']
        # self.chain=  kp.build_serial_chain_from_urdf(open("test_mesh.urdf").read(), end_link_name="right_hand_limb")
        self.initial_collisions = set() # collision due to initial pose
        self.chain = None

    def find_ik_joint_indices(self):
        ik_indices = []
        for i in self.controllable_joint_indices:
            counter = 0
            for j in all_joint_indices:
                if i == j:
                    ik_indices.append(counter)
                joint_type = p.getJointInfo(self.body, j, physicsClientId=self.id)[2]
                if joint_type != p.JOINT_FIXED:
                    counter += 1
        return ik_indices

    def change_color(self, color):
        r"""
        Change the color of a robot.
        :param color: Vector4 for rgba.
        """
        for j in range(p.getNumJoints(self.id)):
            p.changeVisualShape(self.id, j, rgbaColor=color, specularColor=[0.1, 0.1, 0.1])

    def set_joint_angles_with_smpl(self, smpl_data: SMPLData):

        print("global_orient", smpl_data.global_orient)
        print("pelvis", smpl_data.body_pose[0:3])
        pose = smpl_data.body_pose

        self.set_global_angle(self.body, pose)

        self._set_joint_angle(self.body, pose, "Spine1", "spine_2")
        self._set_joint_angle(self.body, pose, "Spine2", "spine_3")
        self._set_joint_angle(self.body, pose, "Spine3", "spine_4")

        self._set_joint_angle(self.body, pose, "L_Hip", "left_hip")
        self._set_joint_angle(self.body, pose, "L_Knee", "left_knee")
        self._set_joint_angle(self.body, pose, "L_Ankle", "left_ankle")
        self._set_joint_angle(self.body, pose, "L_Foot", "left_foot")

        self._set_joint_angle(self.body, pose, "R_Hip", "right_hip")
        self._set_joint_angle(self.body, pose, "R_Knee", "right_knee")
        self._set_joint_angle(self.body, pose, "R_Ankle", "right_ankle")
        self._set_joint_angle(self.body, pose, "R_Foot", "right_foot")

        self._set_joint_angle(self.body, pose, "R_Collar", "right_clavicle")
        self._set_joint_angle(self.body, pose, "R_Shoulder", "right_shoulder")
        self._set_joint_angle(self.body, pose, "R_Elbow", "right_elbow")
        self._set_joint_angle(self.body, pose, "R_Wrist", "right_lowarm")
        self._set_joint_angle(self.body, pose, "R_Hand", "right_hand")

        self._set_joint_angle(self.body, pose, "L_Collar", "left_clavicle")
        self._set_joint_angle(self.body, pose, "L_Shoulder", "left_shoulder")
        self._set_joint_angle(self.body, pose, "L_Elbow", "left_elbow")
        self._set_joint_angle(self.body, pose, "L_Wrist", "left_lowarm")
        self._set_joint_angle(self.body, pose, "L_Hand", "left_hand")

        self._set_joint_angle(self.body, pose, "Neck", "neck")
        self._set_joint_angle(self.body, pose, "Head", "head")

        self.initial_collisions = self.check_self_collision() # collision due to initial pose

    def get_skin_color(self):
        hsv = list(colorsys.rgb_to_hsv(0.8, 0.6, 0.4))
        hsv[-1] = np.random.uniform(0.4, 0.8)
        skin_color = list(colorsys.hsv_to_rgb(*hsv)) + [1]
        return skin_color

    def set_global_angle(self, human_id, pose):
        _, quat = convert_aa_to_euler_quat(pose[self.smpl_dict.get_pose_ids("Pelvis")])
        # quat = np.array(p.getQuaternionFromEuler(np.array(euler)))
        p.resetBasePositionAndOrientation(human_id, [0, 0, 0], quat)

    def _set_joint_angle(self, human_id, pose, smpl_joint_name, robot_joint_name):
        smpl_angles, _ = convert_aa_to_euler_quat(pose[self.smpl_dict.get_pose_ids(smpl_joint_name)])

        # smpl_angles = pose[smpl_dict.get_pose_ids(smpl_joint_name)]
        robot_joints = self.human_dict.get_joint_ids(robot_joint_name)
        for i in range(0, 3):
            p.resetJointState(human_id, robot_joints[i], smpl_angles[i])

    def generate_human_mesh(self, id, physic_id, model_path):
        hull_dict, joint_pos_dict, _ = generate_geom(model_path, smpl_data)
        # now trying to scale the urdf file
        generate_urdf(id, physic_id, hull_dict, joint_pos_dict)
        # p.loadURDF("test_mesh.urdf", [0, 0, 0])

    def init(self, physics_id, np_random):
        self.body = p.loadURDF(URDF_FILE, [0, 0, 1],
                               flags=p.URDF_USE_SELF_COLLISION,
                               useFixedBase=False)
        self._init_kinematic_chain()

        set_self_collisions(self.body, physics_id)

        # set contact damping
        num_joints = p.getNumJoints(self.body, physicsClientId=physics_id)
        change_dynamic_properties(self.body, list(range(0, num_joints)))

        # enable force torque sensor
        for i in self.controllable_joint_indices:
            p.enableJointForceTorqueSensor(self.body, i, enableSensor=True, physicsClientId=physics_id)

        super(HumanUrdf, self).init(self.body, physics_id, np_random)

    def _get_end_and_root_link(self, ee: str):
        if ee == "right_hand":
          end_link_name, root_link_name= "right_hand_limb", "spine_4_limb"
        elif ee == "left_hand":
            end_link_name, root_link_name= "left_hand_limb", "spine_4_limb"
        elif ee == "right_foot":
            end_link_name, root_link_name= "right_foot_limb", "pelvis_limb"
        elif ee == "left_foot":
            end_link_name, root_link_name= "left_foot_limb", "pelvis_limb"
        elif ee == "head":
            end_link_name, root_link_name= "head_limb", "spine_4_limb"
        else:
            raise NotImplementedError
        return end_link_name, root_link_name

    def _init_kinematic_chain(self):
        chain = {}
        chain['whole_body'] = kp.build_chain_from_urdf(open(URDF_FILE).read())
        for ee in self.end_effectors:
            end_link_name, root_link_name = self._get_end_and_root_link(ee)
            chain[ee] = kp.build_serial_chain_from_urdf(open(URDF_FILE).read(),
                                                        end_link_name=end_link_name,
                                                        root_link_name=root_link_name)
        self.chain = chain

    def _get_end_effector_indexes(self, ee_names):
        ee_idxs = []
        for ee in ee_names:
            ee_idxs.append(self.human_dict.get_dammy_joint_id(ee))  # TODO: check if this is correct
        return ee_idxs

    def _get_controllable_joints(self, joints=None):
        joint_states = p.getJointStates(self.body, self.all_joint_indices if joints is None else joints,
                                        physicsClientId=self.id)
        joint_infos = [p.getJointInfo(self.body, i, physicsClientId=self.id) for i in
                       (self.all_joint_indices if joints is None else joints)]
        motor_indices = [i[0] for j, i in zip(joint_states, joint_infos) if i[2] != p.JOINT_FIXED]
        return motor_indices

    # this method wont be precise as the control api only move the joint in the time defined by 1 step only, so not neccessary the end position
    def forward_kinematic(self, ee_idxs, joint_angles):
        """
        :param link_idx: target link
        :param joint_angles: new joint angles
        :param joints: optional, if not given, use all controllable joints
        :return: Cartesian position of center of mass of the target link
        """
        original_angles = self.get_joint_angles(self.controllable_joint_indices)
        print ("original_angles", original_angles.shape, "joint_angles", joint_angles.shape)
        self.control(self.controllable_joint_indices, joint_angles, 0, 0)  # forward to calculate fk
        self.step_simulation()
        _, motor_positions, _, _ = self.get_motor_joint_states()

        ee_positions = []
        for ee in ee_idxs:
            ee_pos = p.getLinkState(self.body, ee, computeLinkVelocity=True, computeForwardKinematics=True,
                                                physicsClientId=self.id)[0]
            ee_positions.append(ee_pos)  # TODO: check if motor_positions change after computeForwardKinematics
        self.set_joint_angles(self.controllable_joint_indices, original_angles, velocities=0.0)  # reset to original angles
        self.step_simulation()
        return ee_positions, motor_positions

    # inverse kinematic using kinpy
    # most of the case it wont be able to find a solution as good as pybullet
    def ik_chain(self, ee_pos, ee_quat = [1, 0, 0, 0]):
        """
        :param ee_pos:
        :param ee_quat:
        :return: ik solutions (angles) for all joints in chain
        """
        t = Transform(ee_quat, ee_pos)
        print (self.right_hand_chain.get_joint_parameter_names())
        return self.right_hand_chain.inverse_kinematics(t)

    # fk for a chain using kinpy
    def fk_chain(self, target_angles, ee: str):
        """
        :param target_angles:
        :return: pos of end effector
        """
        th = {}
        chain = self.chain[ee]
        for idx, joint in enumerate(chain.get_joint_parameter_names()):
            th[joint]=  target_angles[idx]
        # print(th)
        ret = chain.forward_kinematics(th)
        return ret.pos

    # forward kinematics using kinpy
    def fk(self, ee_names, target_angles):
        th = {}
        chain = self.chain["whole_body"]
        for idx, joint in enumerate(chain.get_joint_parameter_names()):
            th[joint]=  target_angles[idx]
        # print(th)
        g_pos, g_orient = p.getBasePositionAndOrientation(self.body, physicsClientId=self.id)
        # [x y z w] to [w x y z] format
        g_quat =  [g_orient[3]] + list(g_orient[:3])
        # print (g_pos, g_orient, g_quat)
        ret = chain.forward_kinematics(th, world = Transform(g_quat, list(g_pos)))
        # ret = chain.forward_kinematics(th)
        print(ret)

        j_angles = []
        for key in ret:
            q_rot = ret[key].rot
            # w x y z to x y z w
            rot = list(q_rot[1:]) + [q_rot[0]]
            j_angles.append(rot)
        ee_pos = []
        for ee_name in ee_names:
            ee_pos.append(ret[ee_name].pos)
        # J= self.chain.jacobian(target_angles)
        return ee_pos, j_angles, None

    def step_simulation(self):
        for _ in range(5): # 5 is the number of skip steps
            p.stepSimulation(physicsClientId=self.id)

    def cal_chain_manipulibility(self, joint_angles, ee: str):
        chain = self.chain[ee]
        J = chain.jacobian(joint_angles, end_only=True)
        J = np.array(J)
        # print("J: ", J.shape)
        # J = J[:, 6:]
        m = np.linalg.det(np.dot(J, J.T))
        return m

    # might need to remove this
    def cal_manipulibility(self, joint_angles, ee_pos_arr, manipulibity_ee_names = None):
        J_arr = []
        m_arr = []  # manipulibility
        ee_idxes = self._get_end_effector_indexes(manipulibity_ee_names)

        for i in range(0, len(ee_idxes)):
            ee = ee_idxes[i]
            ee_pos = ee_pos_arr[i]
            print ("ee_pos: ", ee_pos)
            J_linear, J_angular = p.calculateJacobian(self.body, ee, localPosition=ee_pos,
                                                      objPositions=joint_angles, objVelocities=joint_velocities,
                                                      objAccelerations=joint_accelerations, physicsClientId=self.id)
            # print("J linear: ", J_linear)
            J_linear = np.array(J_linear)
            J_angular = np.array(J_angular)  # TODO: check if we only need part of it (right now it is 3* 75)
            J = np.concatenate([J_linear, J_angular], axis=0)
            m = np.sqrt(np.linalg.det(J @ J.T))
            J_arr.append(J)
            m_arr.append(m)
            print ("End effector idx: ", ee, "Jacobian_l: ", J_linear.shape, "Jacobian_r: ", J_angular.shape, "Manipulibility: ", m)
        avg_manipubility = np.mean(m_arr)

        return avg_manipubility


    def check_self_collision(self):
        """
        Check self collision
        :return: set of collision pairs
        """
        p.performCollisionDetection(physicsClientId=self.id)
        return check_collision( self.body, self.body) # TODO: Check with initial collision

    def check_env_collision(self, body_ids):
        """
        Check self collision
        :return: set of collision pairs
        """
        collision_pairs = set()
        p.performCollisionDetection(physicsClientId=self.id)
        # print ("env_objects: ", body_ids, [p.getBodyInfo(i, physicsClientId=self.id)[1].decode('UTF-8') for i in body_ids])
        for env_body in body_ids:
            collision_pairs.update(check_collision( self.body, env_body))
        return collision_pairs

    def _print_joint_indices(self):
        """
        Getting the joint index for debugging purpose
        TODO: refactor for programmatically generate the joint ind
        :return:
        """
        print(self._get_controllable_joints())
        human_dict = self.human_dict

        left_arms = human_dict.get_joint_ids("left_clavicle")
        for name in human_dict.joint_chain_dict["left_arm"]:
            left_arms.extend(human_dict.get_joint_ids(name))

        right_arms = human_dict.get_joint_ids("right_clavicle")
        for name in human_dict.joint_chain_dict["right_arm"]:
            right_arms.extend(human_dict.get_joint_ids(name))

        left_legs = []
        for name in human_dict.joint_chain_dict["left_leg"]:
            left_legs.extend(human_dict.get_joint_ids(name))

        right_legs = []
        for name in human_dict.joint_chain_dict["right_leg"]:
            right_legs.extend(human_dict.get_joint_ids(name))

        print("left arms: ", left_arms)
        print("right arms: ", right_arms)
        print("left legs: ", left_legs)
        print("right legs: ", right_legs)


if __name__ == "__main__":
    # Start the simulation engine
    physic_client_id = p.connect(p.DIRECT)

    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    np_random, seed = seeding.np_random(1001)

    # plane
    planeId = p.loadURDF("assistive_gym/envs/assets/plane/plane.urdf", [0, 0, 0])

    # bed
    # bed = Furniture()
    # bed.init("hospital_bed","assistive_gym/envs/assets/", physic_client_id, np_random)
    # bed.set_on_ground()

    # human
    human = HumanUrdf()
    human.init(physic_client_id, np_random)
    # human.change_color(human.get_skin_color())

    # print all the joints
    # for j in range(p.getNumJoints(id)):
    #     print (p.getJointInfo(id, j))
    # Set the simulation parameters
    p.setGravity(0, 0, -9.81)

    # bed_height, bed_base_height = bed.get_heights(set_on_ground=True)
    smpl_data = load_smpl(SMPL_PATH)
    # human.set_joint_angles_with_smpl(smpl_data)
    # human.set_on_ground(bed_height)
    human.generate_human_mesh(human.body, physic_client_id, SMPL_PATH)
    show_human_mesh(SMPL_PATH)
    human._print_joint_indices()

    # Set the camera view
    cameraDistance = 3
    cameraYaw = 0
    cameraPitch = -30
    cameraTargetPosition = [0, 0, 1]
    p.resetDebugVisualizerCamera(cameraDistance, cameraYaw, cameraPitch, cameraTargetPosition)

    # Disconnect from the simulation
    p.disconnect()
