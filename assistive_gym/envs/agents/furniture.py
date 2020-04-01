import os
import pybullet as p
import numpy as np
from .agent import Agent

class Furniture(Agent):
    def __init__(self):
        super(Furniture, self).__init__()

    def init(self, furniture_type, directory, id, np_random):
        if 'wheelchair' in furniture_type:
            furniture = p.loadURDF(os.path.join(directory, 'wheelchair', furniture_type + '.urdf'), basePosition=[0, 0, 0.06], baseOrientation=p.getQuaternionFromEuler([np.pi/2.0, 0, np.pi], physicsClientId=id), physicsClientId=id)
        elif furniture_type == 'bed':
            furniture = p.loadURDF(os.path.join(directory, 'bed', 'bed.urdf'), basePosition=[-0.1, 0, 0], baseOrientation=p.getQuaternionFromEuler([np.pi/2.0, 0, 0], physicsClientId=id), physicsClientId=id)

            # mesh_scale = [1.1]*3
            # bed_visual = p.createVisualShape(shapeType=p.GEOM_MESH, fileName=os.path.join(self.directory, 'bed', 'bed_single_reduced.obj'), rgbaColor=[1, 1, 1, 1], specularColor=[0.2, 0.2, 0.2], meshScale=mesh_scale, physicsClientId=self.id)
            # bed_collision = p.createCollisionShape(shapeType=p.GEOM_MESH, fileName=os.path.join(self.directory, 'bed', 'bed_single_reduced_vhacd.obj'), meshScale=mesh_scale, flags=p.GEOM_FORCE_CONCAVE_TRIMESH, physicsClientId=self.id)
            # furniture = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=bed_collision, baseVisualShapeIndex=bed_visual, basePosition=[0, 0, 0], useMaximalCoordinates=True, physicsClientId=self.id)
            # # Initialize bed position
            # p.resetBasePositionAndOrientation(furniture, [-0.1, 0, 0], p.getQuaternionFromEuler([np.pi/2.0, 0, 0], physicsClientId=self.id), physicsClientId=self.id)
        elif furniture_type == 'table':
            furniture = p.loadURDF(os.path.join(directory, 'table', 'table.urdf'), basePosition=[0, -0.35, 0], baseOrientation=p.getQuaternionFromEuler([0, 0, np.pi/2.0], physicsClientId=id), physicsClientId=id)
        else:
            furniture = None

        super(Furniture, self).init(furniture, id, np_random, indices=-1)

