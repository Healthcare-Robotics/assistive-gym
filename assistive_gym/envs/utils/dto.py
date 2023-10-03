from typing import Optional

import numpy as np
from enum import Enum

class HandoverObject(Enum):
    PILL = "pill"
    CUP = "cup"
    CANE = "cane"

    @staticmethod
    def from_string(label):
        if label == "pill":
            return HandoverObject.PILL
        elif label == "cup":
            return HandoverObject.CUP
        elif label == "cane":
            return HandoverObject.CANE
        else:
            raise ValueError(f"Invalid handover object label: {label}")


class OriginalHumanInfo:
    def __init__(self, original_angles: np.ndarray, original_link_positions: np.ndarray, original_self_collisions,
                 original_env_collisions):
        self.link_positions = original_link_positions  # should be array of tuples that are the link positions
        self.angles = original_angles
        self.self_collisions = original_self_collisions
        self.env_collisions = original_env_collisions


class MaximumHumanDynamics:
    def __init__(self, max_torque, max_manipulibility, max_energy):
        self.torque = max_torque
        self.manipulibility = max_manipulibility
        self.energy = max_energy

class RobotSetting:
    def __init__(self, base_pos, base_orient, robot_joint_angles, robot_side, gripper_orient):
        self.base_pos = base_pos
        self.base_orient = base_orient
        self.robot_joint_angles = robot_joint_angles if robot_joint_angles is not None else np.array([])
        self.robot_side = robot_side
        self.gripper_orient = gripper_orient

class InitRobotSetting:
    def __init__(self, base_pos, base_orient, side):
        self.base_pos = base_pos
        self.base_orient = base_orient
        self.robot_side = side
class HandoverObjectConfig:
    def __init__(self, object_type: HandoverObject, weights: list, limits: list, end_effector: Optional[str]):
        self.object_type = object_type
        self.weights = weights
        self.limits = limits
        self.end_effector = end_effector
