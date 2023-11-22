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


class EnvConfig:
    def __init__(self, env_name, person_id, smpl_file, handover_obj, end_effector, coop):
        self.env_name = env_name
        self.person_id = person_id
        self.smpl_file = smpl_file
        self.handover_obj = handover_obj
        self.end_effector = end_effector
        self.coop = coop


class SearchConfig:
    def __init__(self, robot_ik, env_object_ids, original_info, max_dynamics, handover_obj, handover_obj_config,
                 initial_robot_setting):
        self.robot_ik = robot_ik
        self.env_object_ids = env_object_ids  # TODO: check
        self.original_info = original_info
        self.max_dynamics = max_dynamics
        self.handover_obj = handover_obj
        self.handover_obj_config = handover_obj_config
        self.initial_robot_setting = initial_robot_setting


class SearchResult:
    def __init__(self, joint_angles, cost, manipulability, dist, energy, torque, robot_setting):
        self.joint_angles = joint_angles  # just for reference, in case multithread messed up the order
        self.cost = cost
        self.dist = dist
        self.manipulability = manipulability
        self.energy = energy
        self.torque = torque
        self.robot_setting = robot_setting


class MainEnvInitResult:
    def __init__(self, original_info: OriginalHumanInfo, max_dynamics: MaximumHumanDynamics, env_object_ids,
                 human_link_robot_collision, end_effector, handover_obj_config,
                 joint_lower_limits, joint_upper_limits, robot_setting: InitRobotSetting):
        self.original_info = original_info
        self.max_dynamics = max_dynamics
        self.env_object_ids = env_object_ids
        self.human_link_robot_collision = human_link_robot_collision
        self.end_effector = end_effector
        self.handover_obj_config = handover_obj_config
        self.joint_lower_limits = joint_lower_limits
        self.joint_upper_limits = joint_upper_limits
        self.robot_setting = robot_setting

class MainEnvProcessTaskType(Enum):
    INIT = "init"
    RENDER_STEP = "render_step"
    GET_HUMAN_ROBOT_INFO = "get_human_robot_info"


class MainEnvProcessTask:
    def __init__(self, task_type: MainEnvProcessTaskType):
        self.task_type = task_type


class MainEnvProcessInitTask(MainEnvProcessTask):
    def __init__(self):
        super().__init__(MainEnvProcessTaskType.INIT)


class MainEnvProcessRenderTask(MainEnvProcessTask):
    def __init__(self, joint_angle, robot_setting: RobotSetting):
        super().__init__(MainEnvProcessTaskType.RENDER_STEP)
        self.joint_angle = joint_angle
        self.robot_setting = robot_setting


class MainEnvProcessGetHumanRobotInfoTask(MainEnvProcessTask):
    def __init__(self, joint_angle, robot_setting: RobotSetting, end_effector: str):
        super().__init__(MainEnvProcessTaskType.GET_HUMAN_ROBOT_INFO)
        self.joint_angle = joint_angle
        self.robot_setting = robot_setting
        self.end_effector = end_effector
