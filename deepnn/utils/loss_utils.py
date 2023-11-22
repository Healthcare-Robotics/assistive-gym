import numpy as np

from deepnn.utils.data_parser import ModelOutput


def cal_joint_angle_loss(joints_gt, joints_pred):
    joints_gt, joints_pred = np.array(joints_gt), np.array(joints_pred)
    return np.linalg.norm(joints_gt - joints_pred)/joints_gt.shape[0] * 180/np.pi


def cal_robot_base_pos_loss(pos_gt, pos_pred):
    pos_gt, pos_pred = np.array(pos_gt), np.array(pos_pred)
    return np.linalg.norm(pos_gt - pos_pred)/pos_gt.shape[0]


def cal_robot_base_orient_loss(orient_gt, orient_pred):
    orient_gt, orient_pred = np.array(orient_gt), np.array(orient_pred)
    return np.linalg.norm(orient_gt - orient_pred)/orient_gt.shape[0]


def cal_loss(gt:ModelOutput, pred:ModelOutput):
    human_joint_angle_loss = cal_joint_angle_loss(gt.human_joint_angles, pred.human_joint_angles)
    robot_joint_angle_loss = cal_joint_angle_loss(gt.robot_joint_angles, pred.robot_joint_angles)
    robot_base_loss = cal_robot_base_pos_loss(gt.robot_base_pos, pred.robot_base_pos)
    robot_base_rot_loss = cal_robot_base_orient_loss(gt.robot_base_orient, pred.robot_base_orient)

    return human_joint_angle_loss, robot_joint_angle_loss, robot_base_loss, robot_base_rot_loss