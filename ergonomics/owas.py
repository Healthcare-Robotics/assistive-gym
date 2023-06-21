# -*- coding: utf-8 -*-
# ---------------------

import numpy as np
import ergonomics.utils as utils

class OwasScore:
    '''
    Class to compute OWAS metrics

    Pose:
          [0]: Head
          [1]: Neck
          [2, 3, 4, 14]: Left arm + (optional)left hand
          [5, 6, 7, 15]: Right arm + (optional)right hand
          [8, 9, 10]: Left leg
          [11, 12, 13]: Right leg
    '''

    def __init__(self):

        # Body Params
        self.body_params = {'trunk_angle': 0, 'trunk_side_angle': 0, 'arm_step_left': 0, 'arm_step_right': 0,
                            'legs_angle_left': 0, 'legs_angle_right': 0, 'knee_angle_left': 0, 'knee_angle_right': 0,
                            'knee_offset_left': 0, 'knee_offset_right': 0, 'step_size_y': 0, 'step_size_x': 0
                            }

        # Init lookup tables
        self.table = np.ones((4, 3, 7, 3)).astype(int)
        self.init_table()

    def init_table(self):
        '''
        Table used to compute OWAS score

        :return: None
        '''

        self.table[0] = [
                            [[1, 1, 1], [1, 1, 1], [1, 1, 1], [2, 2, 2], [2, 2, 2], [1, 1, 1], [1, 1, 1]],
                            [[1, 1, 1], [1, 1, 1], [1, 1, 1], [2, 2, 2], [2, 2, 2], [1, 1, 1], [1, 1, 1]],
                            [[1, 1, 1], [1, 1, 1], [1, 1, 1], [2, 2, 3], [2, 2, 3], [1, 1, 1], [1, 1, 2]]
                        ]

        self.table[1] = [
                            [[2, 2, 3], [2, 2, 3], [2, 2, 3], [3, 3, 3], [3, 3, 3], [2, 2, 2], [2, 3, 3]],
                            [[2, 2, 3], [2, 2, 3], [2, 3, 3], [3, 4, 4], [3, 4, 4], [3, 3, 4], [2, 3, 4]],
                            [[3, 3, 4], [2, 2, 3], [3, 3, 3], [3, 4, 4], [4, 4, 4], [4, 4, 4], [2, 3, 4]]
                        ]

        self.table[2] = [
                            [[1, 1, 1], [1, 1, 1], [1, 1, 2], [3, 3, 3], [4, 4, 4], [1, 1, 1], [1, 1, 1]],
                            [[2, 2, 3], [1, 1, 1], [1, 1, 2], [4, 4, 4], [4, 4, 4], [3, 3, 3], [1, 1, 1]],
                            [[2, 2, 3], [1, 1, 1], [2, 3, 3], [4, 4, 4], [4, 4, 4], [4, 4, 4], [1, 1, 2]]
                        ]

        self.table[3] = [
                            [[2, 3, 3], [2, 2, 3], [2, 3, 3], [4, 4, 4], [4, 4, 4], [4, 4, 4], [2, 3, 4]],
                            [[3, 3, 4], [2, 3, 4], [3, 3, 4], [4, 4, 4], [4, 4, 4], [4, 4, 4], [2, 3, 4]],
                            [[4, 4, 4], [2, 3, 4], [3, 3, 4], [4, 4, 4], [4, 4, 4], [4, 4, 4], [2, 3, 4]]
                        ]

    def set_body_params(self, values):
        # type: (np.ndarray) -> None
        '''
        Set body params

        :param values: [trunk_angle, trunk_side_angle, arm_step_left, arm_step_right, legs_angle_left,
                        legs_angle_right, knee_angle_left, knee_angle_right, knee_offset_left,
                        knee_offset_right, step_size_y, step_size_x]
        :return: None
        '''
        assert len(values) == len(self.body_params)

        for i, (key, _) in enumerate(self.body_params.items()):
            self.body_params[key] = values[i]

    def compute_score(self):
        # type: (OwasScore) -> (np.ndarray, np.ndarray)
        '''
        Compute OWAS score
        >>> owascore = OwasScore()
        >>> owascore.set_body_params(np.array([10, 1, -0.1, 0.1, 10, 10, 160, 160, 0.4, 0.4, 0.1, 0.1]))
        >>> owascore.compute_score()
        (1, array([1, 2, 2, 1]))

        :return: OWAS_score, [neck_score, trunk_score, leg_score]
        '''

        trunk, arms, legs, load = 0, 1, 0, 0
        trunk = 1 if self.body_params['trunk_angle'] < 20 else 2

        if self.body_params['trunk_side_angle'] > 30:
            if trunk == 1:
                trunk = 3
            elif trunk == 2:
                trunk = 4

        if self.body_params['arm_step_left'] > 0:
            arms += 1

        if self.body_params['arm_step_right'] > 0:
            arms += 1

        # Is walking?
        if self.body_params['step_size_x'] > 0.2:
            legs = 7  # yes
        else:
            legs = 2  # no

        # On knee
        if self.body_params['knee_offset_left'] < 0.1 or self.body_params['knee_offset_right'] < 0.1:
            legs = 6

        # Sitting
        if self.body_params['legs_angle_left'] > 80 and self.body_params['legs_angle_right'] > 80:
            legs = 1

        # Both legs bent
        if self.body_params['knee_angle_left'] < 95 and self.body_params['knee_angle_right'] < 95:
            legs = 4

        # Weight on one leg + eventual bent
        if self.body_params['step_size_y'] > 0.1:
            legs = 5 if self.body_params['knee_angle_left'] < 95 or self.body_params['knee_angle_right'] < 95 else 3

        # Load
        load = 1

        assert 0 < trunk < 4 and 0 < arms < 3 and 0 < legs < 7 and 0 < load < 3
        owas_score = self.table[trunk-1][arms-1][legs-1][load-1]
        return owas_score, np.array([trunk, arms, legs, load])

    @staticmethod
    def get_param_from_pose(pose, verbose=False):
        # type: (np.ndarray, bool) -> np.ndarray
        '''
        Get body params from pose

        :param pose: Pose (Joints coordinates)
        :param verbose: If true show each pose for debugging

        :return: Body params [trunk_angle, trunk_side_angle, arm_step_left, arm_step_right, legs_angle_left,
                              legs_angle_right, knee_angle_left, knee_angle_right, knee_offset_left,
                              knee_offset_right, step_size_y, step_size_x]
        '''
        pose = np.expand_dims(np.copy(pose), 0)

        if verbose:
            utils.show_skeleton(pose, title="GT pose")

        # Trunk position
        pose, _ = utils.rotate_pose(pose, rotation_joint=8)
        pose -= (pose[:, 8] + pose[:, 11]) / 2
        trunk_angle = np.rad2deg(np.arctan2(pose[0, 1, 1], pose[0, 1, 0]) - (np.pi / 2))

        if verbose:
            utils.show_skeleton(pose, title="Trunk angle: " +str(round(trunk_angle, 2)))

        # Trunk twist
        pose, _ = utils.rotate_pose(pose, rotation_joint=8, m_coeff=np.pi/2)
        trunk_side_angle = abs(np.rad2deg(np.arctan2(pose[0, 1, 1], pose[0, 1, 0]) - (np.pi / 2)))

        if verbose:
            utils.show_skeleton(pose, title="Trunk side angle: " + str(round(trunk_side_angle, 2)))

        # Arms position
        arm_step_left =  pose[0, 3, 1] - pose[0, 2, 1]
        arm_step_right = pose[0, 6, 1] - pose[0, 5, 1]

        # Legs position
        pose, _ = utils.rotate_pose(pose, rotation_joint=8, m_coeff=-np.pi / 2)
        pose -= pose[:, 8]
        legs_angle_left = -np.rad2deg(np.arctan2(pose[0, 9, 1], pose[0, 9, 0]) + (np.pi / 2))
        pose -= pose[:, 11]
        legs_angle_right = -np.rad2deg(np.arctan2(pose[0, 12, 1], pose[0, 12, 0]) + (np.pi / 2))

        if verbose:
            utils.show_skeleton(pose, title="Legs angle left: " + str(round(legs_angle_left, 2))
                                             + " Legs angle right: " + str(round(legs_angle_right, 2)))

        pose -= pose[:, 9]
        a = np.rad2deg(np.arctan2(pose[0, 10, 1], pose[0, 10, 0]))
        b = np.rad2deg(np.arctan2(pose[0, 8, 1], pose[0, 8, 0]))
        knee_angle_left = abs(a) + abs(b)
        pose -= pose[:, 12]
        a = np.rad2deg(np.arctan2(pose[0, 13, 1], pose[0, 13, 0]))
        b = np.rad2deg(np.arctan2(pose[0, 11, 1], pose[0, 11, 0]))
        knee_angle_right = abs(a) + abs(b)

        if verbose:
            utils.show_skeleton(pose, title="Knee angle left: " + str(round(knee_angle_left, 2))
                                             + " Knee angle right: " + str(round(knee_angle_right, 2)))

        knee_offset_left = abs(pose[0, 9, 1] - pose[0, 10, 1])
        knee_offset_right = abs(pose[0, 12, 1] - pose[0, 13, 1])

        step_size_y = abs(pose[0, 10, 1] - pose[0, 13, 1])
        step_size_x = abs(pose[0, 10, 0] - pose[0, 13, 0])

        return np.array([trunk_angle, trunk_side_angle, arm_step_left, arm_step_right, legs_angle_left,
                         legs_angle_right, knee_angle_left, knee_angle_right, knee_offset_left,
                         knee_offset_right, step_size_y, step_size_x])


if __name__ == '__main__':

    import doctest
    doctest.testmod()

    sample_pose = np.array([[ 0.08533354,  1.03611605,  0.09013124],
                              [ 0.15391247,  0.91162637, -0.00353906],
                              [ 0.22379057,  0.87361878,  0.11541229],
                              [ 0.4084777 ,  0.69462843,  0.1775224 ],
                              [ 0.31665226,  0.46389668,  0.16556387],
                              [ 0.1239769 ,  0.82994377, -0.11715403],
                              [ 0.08302169,  0.58146328, -0.19830338],
                              [-0.06767788,  0.53928527, -0.00511249],
                              [ 0.11368726,  0.49372503,  0.21275574],
                              [ 0.069179  ,  0.07140968,  0.26841402],
                              [ 0.10831762, -0.36339359,  0.34032449],
                              [ 0.11368726,  0.41275504, -0.01171348],
                              [ 0.        ,  0.        ,  0.        ],
                              [ 0.02535541, -0.43954643,  0.04373671],
                              [ 0.26709431,  0.33643749,  0.17985192],
                              [-0.15117603,  0.49462711,  0.02703403]])

    owasScore = OwasScore()

    body_params = owasScore.get_param_from_pose(sample_pose, verbose=False)
    owasScore.set_body_params(body_params)
    owas_score, partial_score = owasScore.compute_score()

    print("Owas Score:", owas_score)
    print("Trunk, Arms, Legs, Load :", partial_score)

