# This script is borrowed and extended from https://github.com/nkolot/SPIN/blob/master/models/hmr.py
# Adhere to their licence to use this script

import torch
import numpy as np
from smplx import SMPL as _SMPL

SMPL_BONE_ORDER_NAMES = [
    "Pelvis",
    "L_Hip",
    "R_Hip",
    "Spine1",
    "L_Knee",
    "R_Knee",
    "Spine2",
    "L_Ankle",
    "R_Ankle",
    "Spine3",
    "L_Foot",
    "R_Foot",
    "Neck",
    "L_Collar",
    "R_Collar",
    "Head",
    "L_Shoulder",
    "R_Shoulder",
    "L_Elbow",
    "R_Elbow",
    "L_Wrist",
    "R_Wrist",
    "L_Hand",
    "R_Hand",
]

SMPL_BONE_KINTREE_NAMES = [
    'Pelvis', 'L_Hip', 'L_Knee', 'L_Ankle', 'L_Toe', 'R_Hip', 'R_Knee', 'R_Ankle', 'R_Toe', 'Torso', 'Spine', 'Chest',
    'Neck', 'Head', 'L_Thorax', 'L_Shoulder', 'L_Elbow', 'L_Wrist', 'L_Hand', 'R_Thorax', 'R_Shoulder', 'R_Elbow',
    'R_Wrist', 'R_Hand'
]


class SMPL_Parser(_SMPL):
    def __init__(self, *args, **kwargs):
        """SMPL model constructor
        Parameters
        ----------
        model_path: str
            The path to the folder or to the file where the model
            parameters are stored
        data_struct: Strct
            A struct object. If given, then the parameters of the model are
            read from the object. Otherwise, the model tries to read the
            parameters from the given `model_path`. (default = None)
        create_global_orient: bool, optional
            Flag for creating a member variable for the global orientation
            of the body. (default = True)
        global_orient: torch.tensor, optional, Bx3
            The default value for the global orientation variable.
            (default = None)
        create_body_pose: bool, optional
            Flag for creating a member variable for the pose of the body.
            (default = True)
        body_pose: torch.tensor, optional, Bx(Body Joints * 3)
            The default value for the body pose variable.
            (default = None)
        create_betas: bool, optional
            Flag for creating a member variable for the shape space
            (default = True).
        betas: torch.tensor, optional, Bx10
            The default value for the shape member variable.
            (default = None)
        create_transl: bool, optional
            Flag for creating a member variable for the translation
            of the body. (default = True)
        transl: torch.tensor, optional, Bx3
            The default value for the transl variable.
            (default = None)
        dtype: torch.dtype, optional
            The data type for the created variables
        batch_size: int, optional
            The batch size used for creating the member variables
        joint_mapper: object, optional
            An object that re-maps the joints. Useful if one wants to
            re-order the SMPL joints to some other convention (e.g. MSCOCO)
            (default = None)
        gender: str, optional
            Which gender to load
        vertex_ids: dict, optional
            A dictionary containing the indices of the extra vertices that
            will be selected
        """
        super(SMPL_Parser, self).__init__(*args, **kwargs)
        self.device = next(self.parameters()).device
        self.joint_names = SMPL_BONE_ORDER_NAMES
        self.joint_axes = {x: np.identity(3) for x in self.joint_names}
        self.joint_dofs = {x: ["z", "y", "x"] for x in self.joint_names}
        self.joint_range = {
            x: np.hstack([np.ones([3, 1]) * -np.pi, np.ones([3, 1]) * np.pi])
            for x in self.joint_names
        }
        self.joint_range["L_Elbow"] *= 4
        self.joint_range["R_Elbow"] *= 4

        self.contype = {1: self.joint_names}
        self.conaffinity = {1: self.joint_names}
        self.zero_pose = torch.zeros(1, 72).float()

    def forward(self, *args, **kwargs):
        smpl_output = super(SMPL_Parser, self).forward(*args, **kwargs)
        return smpl_output

    def get_joints_verts(self, pose, th_betas=None, th_trans=None):
        """
        Pose should be batch_size x 72
        """
        if pose.shape[0] != 72:
            pose = pose.reshape(-1, 72)

        pose = pose.float()
        if th_betas is not None:
            th_betas = th_betas.float()

            if th_betas.shape[-1] == 16:
                th_betas = th_betas[:, :10]

        smpl_output = self.forward(
            betas=th_betas,
            transl=th_trans,
            body_pose=pose[:, 3:],
            global_orient=pose[:, :3],
        )
        vertices = smpl_output.vertices
        joints = smpl_output.joints[:, :24]
        betas = smpl_output.betas
        print("betas: ", betas)
        # joints = smpl_output.joints[:,JOINST_TO_USE]
        return vertices, joints

    def get_mesh_offsets(self, pose, betas=torch.zeros(1, 10), flatfoot=False, transl=None):
        with torch.no_grad():
            joint_names = self.joint_names
            if pose is not None:
                verts, Jtr = self.get_joints_verts(pose, th_betas=betas, th_trans=transl)
            else:
                verts, Jtr = self.get_joints_verts(self.zero_pose, th_betas=betas, th_trans=transl)

            verts_np = verts.detach().cpu().numpy()
            verts = verts_np[0]

            if flatfoot:
                feet_subset = verts[:, 1] < np.min(verts[:, 1]) + 0.01
                verts[feet_subset, 1] = np.mean(verts[feet_subset][:, 1])

            smpl_joint_parents = self.parents.cpu().numpy()

            joint_pos = Jtr[0].numpy()
            joint_offsets = {
                joint_names[c]: (joint_pos[c] - joint_pos[p]) if c > 0 else joint_pos[c]
                for c, p in enumerate(smpl_joint_parents)
            }
            joint_parents = {
                x: joint_names[i] if i >= 0 else None
                for x, i in zip(joint_names, smpl_joint_parents)
            }
            # override joint limit with our limit setting
            for j in range(0, len(self.joint_names)):
                joint_name = self.joint_names[j]
                # if joint_name in JOINT_SETTING:
                #     joint_limit = np.array(JOINT_SETTING[joint_name].limit)
                #     self.joint_range[joint_name] = joint_limit / 180.0 * np.pi
            print(self.joint_range)
            # skin_weights = smpl_layer.th_weights.numpy()
            skin_weights = self.lbs_weights.numpy()

            return (
                verts,
                joint_pos,
                skin_weights,
                joint_names,
                joint_offsets,
                joint_parents,
                self.joint_axes,
                self.joint_dofs,
                self.joint_range,
                self.contype,
                self.conaffinity,
            )

