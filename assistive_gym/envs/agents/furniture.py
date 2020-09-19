import os
import pybullet as p
import numpy as np
from .agent import Agent

class Furniture(Agent):
    def __init__(self):
        super(Furniture, self).__init__()

    def init(self, furniture_type, directory, id, np_random, wheelchair_mounted=False):
        if 'wheelchair' in furniture_type:
            left = False
            if 'left' in furniture_type:
                furniture_type = 'wheelchair'
                left = True
            furniture = p.loadURDF(os.path.join(directory, furniture_type, 'wheelchair.urdf' if not wheelchair_mounted else ('wheelchair_jaco.urdf' if not left else 'wheelchair_jaco_left.urdf')), basePosition=[0, 0, 0.06], baseOrientation=[0, 0, 0, 1], physicsClientId=id)
        elif furniture_type == 'bed':
            furniture = p.loadURDF(os.path.join(directory, 'bed', 'bed.urdf'), basePosition=[-0.1, 0, 0], baseOrientation=[0, 0, 0, 1], physicsClientId=id)

            # mesh_scale = [1.1]*3
            # bed_visual = p.createVisualShape(shapeType=p.GEOM_MESH, fileName=os.path.join(self.directory, 'bed', 'bed_single_reduced.obj'), rgbaColor=[1, 1, 1, 1], specularColor=[0.2, 0.2, 0.2], meshScale=mesh_scale, physicsClientId=self.id)
            # bed_collision = p.createCollisionShape(shapeType=p.GEOM_MESH, fileName=os.path.join(self.directory, 'bed', 'bed_single_reduced_vhacd.obj'), meshScale=mesh_scale, flags=p.GEOM_FORCE_CONCAVE_TRIMESH, physicsClientId=self.id)
            # furniture = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=bed_collision, baseVisualShapeIndex=bed_visual, basePosition=[0, 0, 0], useMaximalCoordinates=True, physicsClientId=self.id)
            # # Initialize bed position
            # p.resetBasePositionAndOrientation(furniture, [-0.1, 0, 0], p.getQuaternionFromEuler([np.pi/2.0, 0, 0], physicsClientId=self.id), physicsClientId=self.id)
        elif furniture_type == 'hospital_bed':
            furniture = p.loadURDF(os.path.join(directory, 'bed', 'hospital_bed.urdf'), basePosition=[0, 0, 0], baseOrientation=[0, 0, 0, 1], physicsClientId=id)
            self.controllable_joint_indices = [1]
            super(Furniture, self).init(furniture, id, np_random)
        elif furniture_type == 'table':
            furniture = p.loadURDF(os.path.join(directory, 'table', 'table_tall.urdf'), basePosition=[0.25, -1.0, 0], baseOrientation=[0, 0, 0, 1], physicsClientId=id)
        elif furniture_type == 'bowl':
            bowl_pos = np.array([-0.15, -0.65, 0.75]) + np.array([np_random.uniform(-0.05, 0.05), np_random.uniform(-0.05, 0.05), 0])
            furniture = p.loadURDF(os.path.join(directory, 'dinnerware', 'bowl.urdf'), basePosition=bowl_pos, baseOrientation=[0, 0, 0, 1], physicsClientId=id)
        elif furniture_type == 'nightstand':
            furniture = p.loadURDF(os.path.join(directory, 'nightstand', 'nightstand.urdf'), basePosition=np.array([-0.9, 0.7, 0]), baseOrientation=[0, 0, 0, 1], physicsClientId=id)
        else:
            furniture = None

        super(Furniture, self).init(furniture, id, np_random, indices=-1)

