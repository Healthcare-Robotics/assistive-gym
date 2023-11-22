import os
import pybullet as p
import numpy as np
from .agent import Agent

class Tool(Agent):
    def __init__(self):
        super(Tool, self).__init__()

    def init(self, robot, task, directory, id, np_random, right=True, mesh_scale=[1]*3, maximal=False, alpha=1.0, mass=1):
        self.robot = robot
        self.task = task
        self.right = right
        self.id = id
        transform_pos, transform_orient = self.get_transform()

        # Instantiate the tool mesh
        if task == 'scratch_itch':
            tool = p.loadURDF(os.path.join(directory, 'scratcher', 'tool_scratch.urdf'), basePosition=transform_pos, baseOrientation=transform_orient, physicsClientId=id)
        elif task == 'bed_bathing':
            tool = p.loadURDF(os.path.join(directory, 'bed_bathing', 'wiper.urdf'), basePosition=transform_pos, baseOrientation=transform_orient, physicsClientId=id)
        elif task in ['drinking', 'feeding', 'arm_manipulation', 'comfort_standing_up', 'comfort_drinking']:
            if task == 'drinking':
                visual_filename = os.path.join(directory, 'dinnerware', 'plastic_coffee_cup.obj')
                collision_filename = os.path.join(directory, 'dinnerware', 'plastic_coffee_cup_vhacd.obj')
            elif task == 'feeding':
                visual_filename = os.path.join(directory, 'dinnerware', 'spoon.obj')
                collision_filename = os.path.join(directory, 'dinnerware', 'spoon_vhacd.obj')
            elif task == 'arm_manipulation':
                visual_filename = os.path.join(directory, 'arm_manipulation', 'arm_manipulation_scooper.obj')
                collision_filename = os.path.join(directory, 'arm_manipulation', 'arm_manipulation_scooper_vhacd.obj')
            elif task == 'comfort_standing_up':
                visual_filename = os.path.join(directory, 'cane', 'meshes', 'cane.stl')
                collision_filename = os.path.join(directory, 'cane', 'meshes', 'cane.stl')
                mesh_scale = [0.001] * 3
            elif task == 'comfort_drinking':
                visual_filename = os.path.join(directory, 'dinnerware', 'plastic_coffee_cup.obj')
                collision_filename = os.path.join(directory, 'dinnerware', 'plastic_coffee_cup_vhacd.obj')

            tool_visual = p.createVisualShape(shapeType=p.GEOM_MESH, fileName=visual_filename, meshScale=mesh_scale,
                                              rgbaColor=[1, 1, 1, alpha], physicsClientId=id)
            tool_collision = p.createCollisionShape(shapeType=p.GEOM_MESH, fileName=collision_filename,
                                                    meshScale=mesh_scale, physicsClientId=id)
            tool = p.createMultiBody(baseMass=mass, baseCollisionShapeIndex=tool_collision,
                                     baseVisualShapeIndex=tool_visual, basePosition=transform_pos,
                                     baseOrientation=transform_orient, useMaximalCoordinates=maximal,
                                     physicsClientId=id)

        elif task == 'comfort_taking_medicine':
            tool_visual, tool_collision  = self.create_capsule( radius=0.005, length=0.02, position_offset=[0, 0, 0.0], orientation=[0, 0, 0, 1])
            tool = p.createMultiBody(baseMass=mass, baseCollisionShapeIndex=tool_collision,
                                     baseVisualShapeIndex=tool_visual, basePosition=transform_pos,
                                     baseOrientation=transform_orient, useMaximalCoordinates=maximal,
                                     physicsClientId=id)
        else:
            tool = None

        super(Tool, self).init(tool, id, np_random, indices=-1)

        if robot is not None:
            # Disable collisions between the tool and robot
            for j in (robot.right_gripper_collision_indices if right else robot.left_gripper_collision_indices):
                for tj in self.all_joint_indices + [self.base]:
                    p.setCollisionFilterPair(robot.body, self.body, j, tj, False, physicsClientId=id)
            # Create constraint that keeps the tool in the gripper
            constraint = p.createConstraint(robot.body, robot.right_tool_joint if right else robot.left_tool_joint, self.body, -1, p.JOINT_FIXED, [0, 0, 0], parentFramePosition=self.pos_offset, childFramePosition=[0, 0, 0], parentFrameOrientation=self.orient_offset, childFrameOrientation=[0, 0, 0, 1], physicsClientId=id)
            p.changeConstraint(constraint, maxForce=500, physicsClientId=id)

    def get_transform(self):
        if self.robot is not None:
            self.pos_offset = self.robot.tool_pos_offset[self.task]
            self.orient_offset = self.get_quaternion(self.robot.tool_orient_offset[self.task])
            gripper_pos, gripper_orient = self.robot.get_pos_orient(self.robot.right_tool_joint if self.right else self.robot.left_tool_joint, center_of_mass=True)
            transform_pos, transform_orient = p.multiplyTransforms(positionA=gripper_pos, orientationA=gripper_orient, positionB=self.pos_offset, orientationB=self.orient_offset, physicsClientId=self.id)
        else:
            transform_pos = [0, 0, 0]
            transform_orient = [0, 0, 0, 1]
        return transform_pos, transform_orient

    def reset_pos_orient(self):
        """
        Reset the position and orientation of the tool to the gripper position
        :return:
        """
        transform_pos, transform_orient = self.get_transform()
        self.set_base_pos_orient(transform_pos, transform_orient)

    def create_capsule(self, radius=0, length=0, position_offset=[0, 0, 0], orientation=[0, 0, 0, 1]):
        visual_shape = p.createVisualShape(p.GEOM_CAPSULE, radius=radius, length=length,  visualFramePosition=position_offset,
                                           visualFrameOrientation=orientation, physicsClientId=self.id)
        collision_shape = p.createCollisionShape(p.GEOM_CAPSULE, radius=radius, height=length,
                                                 collisionFramePosition=position_offset,
                                                 collisionFrameOrientation=orientation, physicsClientId=self.id)
        return visual_shape, collision_shape

