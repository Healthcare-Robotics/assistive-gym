class HumanUrdfDict:
    def __init__(self):
        # TODO: dynamically generate this based on the URDF
        # currently, manually generate this based on the URDF

        self.end_effectors = ['right_hand', 'left_hand', 'right_foot', 'left_foot', 'head']

        self.limb_name_dict = {
            "pelvis_limb": 0,
            "left_hip_limb": 4,
            "left_knee_limb": 8,
            "left_ankle_limb": 12,
            "left_foot_limb": 16,
            "right_hip_limb": 20,
            "right_knee_limb": 24,
            "right_ankle_limb": 28,
            "right_foot_limb": 32,
            "spine_2_limb": 36,
            "spine_3_limb": 40,
            "spine_4_limb": 44,
            "neck_limb": 48,
            "head_limb": 52,
            "left_clavicle_limb": 56,
            "left_shoulder_limb": 60,
            "left_elbow_limb": 64,
            "left_lowarm_limb": 68,
            "left_hand_limb": 72,
            "right_clavicle_limb": 76,
            "right_shoulder_limb": 80,
            "right_elbow_limb": 84,
            "right_lowarm_limb": 88,
            "right_hand_limb": 92,
        }

        # create a dictionary of indices to limb names
        self.limb_index_dict = {v: k for k, v in self.limb_name_dict.items()}

        self.joint_dict = {
            "pelvis": 0,
            "left_hip": 1,
            "left_knee": 5,
            "left_ankle": 9,
            "left_foot": 13,
            "right_hip": 17,
            "right_knee": 21,
            "right_ankle": 25,
            "right_foot": 29,
            "spine_2": 33,
            "spine_3": 37,
            "spine_4": 41,
            "neck": 45,
            "head": 49,
            "left_clavicle": 53,
            "left_shoulder": 57,
            "left_elbow": 61,
            "left_lowarm": 65,
            "left_hand": 69,
            "right_clavicle": 73,
            "right_shoulder": 77,
            "right_elbow": 81,
            "right_lowarm": 85,
            "right_hand": 89,
        }

        self.urdf_to_smpl_dict = {
            "pelvis": "Pelvis",
            "left_hip": "L_Hip",
            "left_knee": "L_Knee",
            "left_ankle": "L_Ankle",
            "left_foot": "L_Foot",
            "right_hip": "R_Hip",
            "right_knee": "R_Knee",
            "right_ankle": "R_Ankle",
            "right_foot": "R_Foot",
            "spine_2": "Spine1",
            "spine_3": "Spine2",
            "spine_4": "Spine3",
            "neck": "Neck",
            "head": "Head",
            "left_clavicle": "L_Collar",
            "left_shoulder": "L_Shoulder",
            "left_elbow": "L_Elbow",
            "left_lowarm": "L_Wrist",
            "left_hand": "L_Hand",
            "right_clavicle": "R_Collar",
            "right_shoulder": "R_Shoulder",
            "right_elbow": "R_Elbow",
            "right_lowarm": "R_Wrist",
            "right_hand": "R_Hand"
        }
        # TODO: change the smpl and robot joint to same name
        self.joint_to_parent_joint_dict = {
            "pelvis": "pelvis",
            "left_hip": "pelvis",
            "left_knee": "left_hip",
            "left_ankle": "left_knee",
            "left_foot": "left_ankle",
            "right_hip": "pelvis",
            "right_knee": "right_hip",
            "right_ankle": "right_knee",
            "right_foot": "right_ankle",
            "spine_2": "pelvis",
            "spine_3": "spine_2",
            "spine_4": "spine_3",
            "neck": "spine_4",
            "head": "neck",
            "left_clavicle": "spine_4",
            "left_shoulder": "left_clavicle",
            "left_elbow": "left_shoulder",
            "left_lowarm": "left_elbow",
            "left_hand": "left_lowarm",
            "right_clavicle": "spine_4",
            "right_shoulder": "right_clavicle",
            "right_elbow": "right_shoulder",
            "right_lowarm": "right_elbow",
            "right_hand": "right_lowarm"
        }
        self.joint_to_child_joint_dict = {v: k for k, v in self.joint_to_parent_joint_dict.items()}

        self.joint_chain_dict = {
            "head": ["neck", "head"],
            "right_hand": ["right_clavicle", "right_shoulder", "right_elbow", "right_lowarm", "right_hand"],
            "left_hand": ["left_clavicle", "left_shoulder", "left_elbow", "left_lowarm", "left_hand"],
            "right_foot": ["right_hip", "right_knee", "right_ankle", "right_foot"],
            "left_foot": ["left_hip", "left_knee", "left_ankle", "left_foot"]
        }

        self.joint_collision_ignore_dict = {
            "head": [],
            "right_hand": ["spine_4"],
            "left_hand": ["spine_4"],
            "right_foot": ["pelvis"],
            "left_foot": ["pelvis"]
        }

    # TODO: solve the issue with fixed joint in URDF
    def get_joint_ids(self, joint_name):
        """
        Obtain the joint ids (x, y, z) for the given joint name (revolute joint only)
        :param joint_name:
        :return:
        """
        joint_id = self.joint_dict[joint_name]
        return [joint_id, joint_id + 1, joint_id + 2]

    def get_dammy_joint_id(self, joint_name):
        """
        Obtain the dammy joint id for the given joint name (revolute joint only)
        :param joint_name:
        :return:
        """
        joint_id = self.joint_dict[joint_name]
        return joint_id + 3

    def get_fixed_joint_id(self, joint_name):  # TODO: rename of refactor. The function name is confusing
        """
        Obtain the joint id for the given joint name (fixed joint)
        :param joint_name:
        :return:
        """
        return self.joint_dict[joint_name]

    def get_real_link_indices(self, end_effector):
        if end_effector not in self.joint_chain_dict:
            raise Exception("The end effector {} is not in the joint chain dict".format(end_effector))
        real_link_indices = []
        for j in self.joint_chain_dict[end_effector]:
            real_link_indices.append(self.get_dammy_joint_id(j))
        return real_link_indices
