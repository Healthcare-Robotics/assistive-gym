class SMPLDict:
    def __init__(self):
        #TODO: generate this one based on index of joint names
        self.joint_dict = {
            "Pelvis": 0,
            "L_Hip": 1,
            "R_Hip": 2,
            "Spine1": 3,
            "L_Knee": 4,
            "R_Knee": 5,
            "Spine2": 6,
            "L_Ankle": 7,
            "R_Ankle": 8,
            "Spine3": 9,
            "L_Foot": 10,
            "R_Foot": 11,
            "Neck": 12,
            "L_Collar": 13,
            "R_Collar": 14,
            "Head": 15,
            "L_Shoulder": 16,
            "R_Shoulder": 17,
            "L_Elbow": 18,
            "R_Elbow": 19,
            "L_Wrist": 20,
            "R_Wrist": 21,
            "L_Hand": 22,
            "R_Hand": 23,
        }

    def get_pose_ids(self, joint_name):
        joint_id = self.joint_dict[joint_name]
        base = 3 * joint_id
        return [base, base + 1, base + 2]
