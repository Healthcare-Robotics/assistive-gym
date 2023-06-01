class HumanUrdfDict:
    def __init__(self):
        # TODO: dynamically generate this based on the URDF
        # currently, manually generate this based on the URDF
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
            "head":  49,
            "left_clavicle": 53,
            "left_shoulder": 57,
            "left_elbow":  61,
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

        self.joint_chain_dict = {
            "body": ["pelvis", "spine_2", "spine_3", "spine_4"],
            "head": ["neck", "head"],
            "right_arm": ["right_shoulder", "right_elbow", "right_lowarm", "right_hand"],
            "left_arm":  ["left_shoulder", "left_elbow", "left_lowarm", "left_hand"],
            "right_leg": ["right_hip", "right_knee", "right_ankle", "right_foot"],
            "left_leg": ["left_hip", "left_knee", "left_ankle", "left_foot"]
        }

        self.joint_collision_ignore_dict = {
            "right_arm": ["right_clavicle", "spine_4"],
            "left_arm": ["left_clavicle", "spine_4"],
            "right_leg": ["pelvis"],
            "left_leg": ["pelvis"]
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

    def get_joint_id(self, joint_name): # TODO: rename of refactor. The function name is confusing
        """
        Obtain the joint id for the given joint name (fixed joint)
        :param joint_name:
        :return:
        """
        return self.joint_dict[joint_name]