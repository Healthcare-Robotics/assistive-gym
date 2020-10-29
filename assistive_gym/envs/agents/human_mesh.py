import os, pickle, torch, smplx, trimesh, colorsys, tempfile, gc
import numpy as np
from scipy.spatial.transform import Rotation as R
import pybullet as p
from .agent import Agent

right_arm_joints = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
left_arm_joints = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
right_leg_joints = [28, 29, 30, 31, 32, 33, 34]
left_leg_joints = [35, 36, 37, 38, 39, 40, 41]
head_joints = [20, 21, 22, 23]

class HumanMesh(Agent):
    def __init__(self):
        super(HumanMesh, self).__init__()
        self.controllable_joint_indices = []
        self.controllable = False
        # Joint indices (raw mesh vertex # .obj vertex)
        self.waist = 0
        self.left_hip = 1
        self.right_hip = 2
        self.chest = 3
        self.left_knee = 4
        self.right_knee = 5
        self.upper_chest = 6
        self.left_ankle = 7
        self.right_ankle = 8
        self.upper_chest_2 = 9
        self.left_toes = 10
        self.right_toes = 11
        self.lower_neck = 12
        self.left_pecs = 13
        self.right_pecs = 14
        self.head_center = self.head = 15
        self.left_shoulder = 16
        self.right_shoulder = 17
        self.left_elbow = 18
        self.right_elbow = 19
        self.left_wrist = 20
        self.right_wrist = 21
        self.left_eye = 23
        self.right_eye = 24
        self.nose = 55
        self.right_ear = 58
        self.left_ear = 59
        self.mouth = 121

        # Joints
        self.j_left_hip_x, self.j_left_hip_y, self.j_left_hip_z = 0, 1, 2
        self.j_right_hip_x, self.j_right_hip_y, self.j_right_hip_z = 3, 4, 5
        self.j_waist_x, self.j_waist_y, self.j_waist_z = 6, 7, 8
        self.j_left_knee_x, self.j_left_knee_y, self.j_left_knee_z = 9, 10, 11
        self.j_right_knee_x, self.j_right_knee_y, self.j_right_knee_z = 12, 13, 14
        self.j_chest_x, self.j_chest_y, self.j_chest_z = 15, 16, 17
        self.j_left_ankle_x, self.j_left_ankle_y, self.j_left_ankle_z = 18, 19, 20
        self.j_right_ankle_x, self.j_right_ankle_y, self.j_right_ankle_z = 21, 22, 23
        self.j_upper_chest_x, self.j_upper_chest_y, self.j_upper_chest_z = 24, 25, 26
        self.j_left_toes_x, self.j_left_toes_y, self.j_left_toes_z = 27, 28, 29
        self.j_right_toes_x, self.j_right_toes_y, self.j_right_toes_z = 30, 31, 32
        self.j_lower_neck_x, self.j_lower_neck_y, self.j_lower_neck_z = 33, 34, 35
        self.j_left_pecs_x, self.j_left_pecs_y, self.j_left_pecs_z = 36, 37, 38
        self.j_right_pecs_x, self.j_right_pecs_y, self.j_right_pecs_z = 39, 40, 41
        self.j_upper_neck_x, self.j_upper_neck_y, self.j_upper_neck_z = 42, 43, 44
        self.j_left_shoulder_x, self.j_left_shoulder_y, self.j_left_shoulder_z = 45, 46, 47
        self.j_right_shoulder_x, self.j_right_shoulder_y, self.j_right_shoulder_z = 48, 49, 50
        self.j_left_elbow_x, self.j_left_elbow_y, self.j_left_elbow_z = 51, 52, 53
        self.j_right_elbow_x, self.j_right_elbow_y, self.j_right_elbow_z = 54, 55, 56
        self.j_left_wrist_x, self.j_left_wrist_y, self.j_left_wrist_z = 57, 58, 59
        self.j_right_wrist_x, self.j_right_wrist_y, self.j_right_wrist_z = 60, 61, 62
        # SMPL only joints (not SMPL-X)
        # self.j_left_fingers_x, self.j_left_fingers_y, self.j_left_fingers_z = 63, 64, 65
        # self.j_right_fingers_x, self.j_right_fingers_y, self.j_right_fingers_z = 66, 67, 68

        self.num_body_shape = 10
        self.vertex_positions = None
        self.obj_verts = None
        self.joint_positions = None
        self.right_arm_vertex_indices = None
        self.bottom_index = 5574

    def create_smplx_body(self, directory, id, np_random, gender='female', height=None, body_shape=None, joint_angles=[], position=[0, 0, 0], orientation=[0, 0, 0], body_pose=None):
        # Choose gender
        self.gender = gender
        if self.gender not in ['male', 'female']:
            self.gender = np_random.choice(['male', 'female'])

        # Create SMPL-X model
        model_folder = os.path.join(directory, 'smpl_models')
        model = smplx.create(model_folder, model_type='smplx', gender=self.gender)

        # Define body shape
        if type(body_shape) == str:
            params_filename = os.path.join(model_folder, 'human_params', body_shape)
            with open(params_filename, 'rb') as f:
                params = pickle.load(f)
            betas = torch.Tensor(params['betas'])
        elif body_shape is None:
            betas = torch.Tensor(np_random.uniform(-1, 5, (1, self.num_body_shape)))
        else:
            betas = torch.Tensor(body_shape)
            # betas = torch.Tensor(np.zeros((1, 10)))
        # betas = torch.Tensor(np.zeros((1, 10)))

        # Set human body pose
        if body_pose is None:
            body_pose = np.zeros((1, model.NUM_BODY_JOINTS*3))
            for joint_index, angle in joint_angles:
                body_pose[0, joint_index] = np.deg2rad(angle)

        # Generate standing human mesh and determine default height of the mesh
        output = model(betas=betas, body_pose=torch.Tensor(np.zeros((1, model.NUM_BODY_JOINTS*3))), return_verts=True)
        vertices = output.vertices.detach().cpu().numpy().squeeze()
        out_mesh = trimesh.Trimesh(vertices, model.faces)
        rot = trimesh.transformations.rotation_matrix(np.deg2rad(90), [1, 0, 0])
        out_mesh.apply_transform(rot)
        # Find indices for right arm vertices and save to file
        # self.vert_indices = np.where(np.logical_and(np.logical_and(vertices[:, 0] < -0.17, vertices[:, 1] > -0.1), vertices[:, 0] > -0.64))
        # self.vert_indices = np.where(np.logical_and(np.logical_and(np.logical_and(np.logical_and(vertices[:, 1] > -0.4, vertices[:, 1] < -0.37), vertices[:, 0] < 0.02), vertices[:, 0] > -0.02), vertices[:, 2] < 0))
        # np.savetxt('right_arm_vertex_indices.csv', self.vert_indices, delimiter=',', fm1='%d')

        # Generate human mesh with correct height scaling
        height_scale = height/out_mesh.extents[-1] if height is not None else 1.0
        # print('Scale:', height_scale, '=', height, '/', out_mesh.extents[-1])
        output = model(betas=betas, body_pose=torch.Tensor(body_pose), return_verts=True)
        vertices = output.vertices.detach().cpu().numpy().squeeze()
        joints = output.joints.detach().cpu().numpy().squeeze()
        # Scale vertices and rotate
        orient_quat = p.getQuaternionFromEuler(orientation, physicsClientId=id)
        vertices = vertices*height_scale
        vertices = vertices.dot(R.from_euler('x', -90, degrees=True).as_matrix())
        vertices = vertices.dot(R.from_quat(orient_quat).as_matrix())
        joints = joints*height_scale
        joints = joints.dot(R.from_euler('x', -90, degrees=True).as_matrix())
        joints = joints.dot(R.from_quat(orient_quat).as_matrix())
        out_mesh = trimesh.Trimesh(vertices, model.faces)
        # scale = trimesh.transformations.scale_matrix(height_scale, [0, 0, 0])
        # out_mesh.apply_transform(scale)
        # rot = trimesh.transformations.rotation_matrix(np.deg2rad(90), [1, 0, 0])
        # out_mesh.apply_transform(rot)

        return out_mesh, vertices, joints

    def init(self, directory, id, np_random, gender='female', height=None, body_shape=None, joint_angles=[], position=[0, 0, 0], orientation=[0, 0, 0], skin_color='random', specular_color=[0.1, 0.1, 0.1], body_pose=None, out_mesh=None, vertices=None, joints=None):
        if out_mesh is None:
            # Create mesh
            out_mesh, vertices, joints = self.create_smplx_body(directory, id, np_random, gender, height, body_shape, joint_angles, position, orientation, body_pose)

        model_folder = os.path.join(directory, 'smpl_models')
        self.skin_color = skin_color
        if self.skin_color == 'random':
            hsv = list(colorsys.rgb_to_hsv(0.8, 0.6, 0.4))
            hsv[-1] = np_random.uniform(0.4, 0.8)
            self.skin_color = list(colorsys.hsv_to_rgb(*hsv)) + [1.0]

        if self.right_arm_vertex_indices is None:
            self.right_arm_vertex_indices = np.loadtxt(os.path.join(model_folder, 'right_arm_vertex_indices.csv'), delimiter=',', dtype=np.int)

        # Load mesh into environment
        with tempfile.NamedTemporaryFile(suffix='.obj') as f:
            out_mesh.export(f.name)
            human_visual = p.createVisualShape(shapeType=p.GEOM_MESH, fileName=f.name, meshScale=1.0, rgbaColor=self.skin_color, specularColor=specular_color, physicsClientId=id)
            human_collision = p.createCollisionShape(shapeType=p.GEOM_MESH, fileName=f.name, meshScale=1.0, flags=p.GEOM_FORCE_CONCAVE_TRIMESH, physicsClientId=id)
            self.body = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=human_collision, baseVisualShapeIndex=human_visual, basePosition=position, baseOrientation=[0, 0, 0, 1], useMaximalCoordinates=False, physicsClientId=id)
            # self.body = p.createMultiBody(baseMass=0, baseVisualShapeIndex=human_visual, basePosition=position, baseOrientation=[0, 0, 0, 1], useMaximalCoordinates=False, physicsClientId=id)

        super(HumanMesh, self).init(self.body, id, np_random, indices=-1)

        self.vertex_positions = vertices
        self.obj_verts = out_mesh.vertices
        self.joint_positions = joints

        gc.collect()
        # human_height, human_base_height = self.get_heights()
        # print('Mesh size:', out_mesh.extents, human_height, 'Target:', height)

    def get_pos_orient(self, joint):
        if joint == self.base:
            return super(HumanMesh, self).get_pos_orient(joint)
        return self.get_joint_positions([joint])[0], [0, 0, 0, 1]

    def get_joint_positions(self, joints):
        pos, _ = self.get_base_pos_orient()
        return self.joint_positions[joints] + pos

    def get_vertex_positions(self, vertices):
        pos, _ = self.get_base_pos_orient()
        return self.vertex_positions[vertices] + pos

