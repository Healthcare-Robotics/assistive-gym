# -*- coding: utf-8 -*-
# ---------------------

import matplotlib


from matplotlib import figure
from matplotlib import cm
import numpy as np
from typing import *
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from random import randint


def show_skeleton(joints, swap_y=False, title=None):

    if joints.shape[1] == 14:
        bone_list = [[0, 1], [1, 2], [2, 3], [3, 4], [1, 5], [5, 6], [6, 7],
                     [1, 8], [8, 9], [9, 10], [1, 11], [11, 12], [12, 13]]

    if joints.shape[1]  == 16:
        bone_list = [[0, 1], [1, 2], [2, 3], [3, 4], [4, 14], [1, 5], [5, 6], [6, 7], [7, 15],
                     [1, 8], [8, 9], [9, 10], [1, 11], [11, 12], [12, 13]]

    elif joints.shape[1]  == 24:
        bone_list = [[0, 1], [1, 4], [4, 7], [7, 10], [0, 2], [2, 5], [5, 8],
                     [8, 11], [0, 3], [3, 6], [6, 9], [9, 14], [14, 17], [17, 19],
                     [19, 21], [21, 23], [9, 13], [13, 16], [16, 18], [18, 20], [20, 22],
                     [16, 12], [17, 12], [12, 15]
                     ]

    elif joints.shape[1]  == 32:
        bone_list=[[0, 1], [1, 2], [2, 3], [3, 4], [4, 5],
                    [0, 6], [6, 7], [7, 8], [8, 9], [9, 10],
                    [0, 11], [11, 12], [12, 13], [13, 14],
                    [14, 15],
                    [16, 17], [17, 18], [18, 19], [19, 20],
                    [20, 21], [21, 22],
                    [22, 23],
                    [16, 24], [24, 25], [25, 26], [26, 27],
                    [27, 28], [28, 29], [29, 30], [30, 31]
                    ]

    plt.figure()
    ax = plt.axes(projection='3d')
    if title is not None:
        plt.title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_xlim(-0.7, 0.7)
    ax.set_ylim(0.7, -0.7) if swap_y else ax.set_ylim(-0.7, 0.7)
    ax.set_zlim(-0.7, 0.7)
    ax.view_init(azim=-90, elev=100)

    for joint in joints:
        color = (randint(64, 255) / 255, randint(64, 255) / 255, randint(64, 255) / 255)
        x, y, z = joint.T[0], joint.T[1], joint.T[2]

        ax.scatter3D(x, y, z, color=color)
        for bone in bone_list:
            ax.plot3D([x[bone[0]], x[bone[1]]], [y[bone[0]], y[bone[1]]], [z[bone[0]], z[bone[1]]], color=color)

    plt.show()


def rotate_pose(data, rotation_joint=8, rotation_matrix=None, m_coeff=None):

    if rotation_matrix is None:
        rotation_matrix = np.zeros((data.shape[0], 3, 3))

        for i in range(data.shape[0]):

            # X, Z coordinates of ankle joint
            m = np.arctan2(data[i, rotation_joint, 0], data[i, rotation_joint, 2])

            if m_coeff is not None:
                m=m_coeff

            # Rotation Matrix
            R = np.array(([np.cos(m), 0, np.sin(m)],
                          [0, 1, 0],
                          [-np.sin(m), 0, np.cos(m)]))

            rotation_matrix[i] = R

    data = np.matmul(data, rotation_matrix)
    return data, rotation_matrix
