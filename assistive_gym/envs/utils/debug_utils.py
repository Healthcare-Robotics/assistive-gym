import pybullet as p
import numpy as np

def axiscreator(bodyId, linkId = -1):
    print(f'axis creator at bodyId = {bodyId} and linkId = {linkId} as XYZ->RGB')
    x_axis = p.addUserDebugLine(lineFromXYZ=[0, 0, 0],
                                lineToXYZ=[0.2, 0, 0],
                                lineColorRGB=[1, 0, 0],
                                lineWidth=10,
                                lifeTime=2,
                                parentObjectUniqueId=bodyId,
                                parentLinkIndex=linkId)

    y_axis = p.addUserDebugLine(lineFromXYZ=[0, 0, 0],
                                lineToXYZ=[0, 0.2, 0],
                                lineColorRGB=[0, 1, 0],
                                lineWidth=10,
                                lifeTime=2,
                                parentObjectUniqueId=bodyId,
                                parentLinkIndex=linkId)

    z_axis = p.addUserDebugLine(lineFromXYZ=[0, 0, 0],
                                lineToXYZ=[0, 0, 0.2],
                                lineColorRGB=[0, 0, 1],
                                lineWidth=10,
                                lifeTime=2,
                                parentObjectUniqueId=bodyId,
                                parentLinkIndex=linkId)
    return [x_axis, y_axis, z_axis]

def add_debug_line_wrt_parent_frame(pos, orient, parent_id, parent_link_id):
    ee_rot_matrix = np.array(p.getMatrixFromQuaternion(np.array(orient))).reshape(3, 3)
    x_axis= p.addUserDebugLine(np.array(pos), np.array(pos) + ee_rot_matrix[:, 0] , [1, 0, 0], 0.1, 2,
                       parentObjectUniqueId=parent_id,
                       parentLinkIndex=parent_link_id)
    y_axis = p.addUserDebugLine(np.array(pos), np.array(pos) + ee_rot_matrix[:, 1], [0, 1, 0], 0.1, 2,
                                parentObjectUniqueId=parent_id,
                                parentLinkIndex=parent_link_id)
    z_axis = p.addUserDebugLine(np.array(pos), np.array(pos) + ee_rot_matrix[:, 2], [0, 0, 1], 0.1, 2,
                                parentObjectUniqueId=parent_id,
                                parentLinkIndex=parent_link_id)

