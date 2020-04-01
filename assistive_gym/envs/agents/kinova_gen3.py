import os
import pybullet as p
from .robot import Robot

class KinovaGen3(Robot):
    def __init__(self, arm='right'):
        self.right_arm_joint_indices = [0, 1, 2, 3, 4, 5, 6]
        self.left_arm_joint_indices = [0, 1, 2, 3, 4, 5, 6]
        self.right_end_effector = 7
        self.left_end_effector = 7
        super(KinovaGen3, self).__init__(arm, self.right_arm_joint_indices, self.left_arm_joint_indices, self.right_end_effector, self.left_end_effector, self.toc_base_pos_offset, self.toc_ee_orient_rpy, self.wheelchair_mounted)

    def init(self, directory, id):
        self.body = p.loadURDF(os.path.join(directory, 'kinova_gen3', 'GEN3_URDF_V12.urdf'), useFixedBase=True, basePosition=[-2, -2, 0.975], flags=p.URDF_USE_SELF_COLLISION, physicsClientId=id)
        super(KinovaGen3, self).init(self.body, id)

