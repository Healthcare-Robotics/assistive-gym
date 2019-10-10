import numpy as np
import pybullet as p

class Util:
    def __init__(self, pid, np_random):
        self.id = pid
        self.ik_lower_limits = {}
        self.ik_upper_limits = {}
        self.ik_joint_ranges = {}
        self.ik_rest_poses = {}
        self.np_random = np_random

    def ik_random_restarts(self, body, target_joint, target_pos, target_orient, world_creation, robot_arm_joint_indices, robot_lower_limits, robot_upper_limits, ik_indices=range(29, 29+7), max_iterations=1000, max_ik_random_restarts=50, random_restart_threshold=0.01, half_range=False, step_sim=False, check_env_collisions=False):
        orient_orig = target_orient
        best_ik_joints = None
        best_ik_distance = 0
        for r in range(max_ik_random_restarts):
            target_joint_positions = self.ik(body, target_joint, target_pos, target_orient, ik_indices=ik_indices, max_iterations=max_iterations, half_range=half_range)
            world_creation.setup_robot_joints(body, robot_arm_joint_indices, robot_lower_limits, robot_upper_limits, randomize_joint_positions=False, default_positions=np.array(target_joint_positions), tool=None)
            if step_sim:
                for _ in range(5):
                    p.stepSimulation(physicsClientId=self.id)
                if len(p.getContactPoints(bodyA=body, bodyB=body, physicsClientId=self.id)) > 0 and orient_orig is not None:
                    # The robot's arm is in contact with itself. Continually randomize end effector orientation until a solution is found
                    target_orient = p.getQuaternionFromEuler(p.getEulerFromQuaternion(orient_orig, physicsClientId=self.id) + np.deg2rad(self.np_random.uniform(-45, 45, size=3)), physicsClientId=self.id)
            if check_env_collisions:
                for _ in range(25):
                    p.stepSimulation(physicsClientId=self.id)
            gripper_pos, gripper_orient = p.getLinkState(body, target_joint, computeForwardKinematics=True, physicsClientId=self.id)[:2]
            if np.linalg.norm(target_pos - np.array(gripper_pos)) < random_restart_threshold and (target_orient is None or np.linalg.norm(target_orient - np.array(gripper_orient)) < random_restart_threshold or np.isclose(np.linalg.norm(target_orient - np.array(gripper_orient)), 2, atol=random_restart_threshold)):
                return True, np.array(target_joint_positions)
            if best_ik_joints is None or np.linalg.norm(target_pos - np.array(gripper_pos)) < best_ik_distance:
                best_ik_joints = target_joint_positions
                best_ik_distance = np.linalg.norm(target_pos - np.array(gripper_pos))
        world_creation.setup_robot_joints(body, robot_arm_joint_indices, robot_lower_limits, robot_upper_limits, randomize_joint_positions=False, default_positions=np.array(best_ik_joints), tool=None)
        return False, np.array(best_ik_joints)

    def ik_jlwki(self, body, target_joint, target_pos, target_orient, world_creation, robot_arm_joint_indices, robot_lower_limits, robot_upper_limits, ik_indices=range(29, 29+7), max_iterations=100, success_threshold=0.03, half_range=False, step_sim=False, check_env_collisions=False):
        target_joint_positions = self.ik(body, target_joint, target_pos, target_orient, ik_indices=ik_indices, max_iterations=max_iterations, half_range=half_range)
        world_creation.setup_robot_joints(body, robot_arm_joint_indices, robot_lower_limits, robot_upper_limits, randomize_joint_positions=False, default_positions=np.array(target_joint_positions), tool=None)
        if step_sim:
            for _ in range(5):
                p.stepSimulation(physicsClientId=self.id)
            if len(p.getContactPoints(bodyA=body, bodyB=body, physicsClientId=self.id)) > 0:
                # The robot's arm is in contact with itself.
                return False, np.array(target_joint_positions)
        if check_env_collisions:
            for _ in range(25):
                p.stepSimulation(physicsClientId=self.id)
        gripper_pos, gripper_orient = p.getLinkState(body, target_joint, computeForwardKinematics=True, physicsClientId=self.id)[:2]
        if np.linalg.norm(target_pos - np.array(gripper_pos)) < success_threshold and (target_orient is None or np.linalg.norm(target_orient - np.array(gripper_orient)) < success_threshold or np.isclose(np.linalg.norm(target_orient - np.array(gripper_orient)), 2, atol=success_threshold)):
            return True, np.array(target_joint_positions)
        return False, np.array(target_joint_positions)

    def ik(self, body, target_joint, target_pos, target_orient, ik_indices=range(29, 29+7), max_iterations=1000, half_range=False):
        key = '%d_%d' % (body, target_joint)
        if key not in self.ik_lower_limits:
            self.ik_lower_limits[key] = []
            self.ik_upper_limits[key] = []
            self.ik_joint_ranges[key] = []
            self.ik_rest_poses[key] = []
            j_names = []
            for j in range(p.getNumJoints(body, physicsClientId=self.id)):
                if p.getJointInfo(body, j, physicsClientId=self.id)[2] != p.JOINT_FIXED:
                    joint_info = p.getJointInfo(body, j, physicsClientId=self.id)
                    lower_limit = joint_info[8]
                    upper_limit = joint_info[9]
                    # print(len(self.ik_lower_limits[key]), joint_info[1], lower_limit, upper_limit)
                    if lower_limit == 0 and upper_limit == -1:
                        # lower_limit = -1e10
                        # upper_limit = 1e10
                        lower_limit = -2*np.pi
                        upper_limit = 2*np.pi
                    self.ik_lower_limits[key].append(lower_limit)
                    self.ik_upper_limits[key].append(upper_limit)
                    if not half_range:
                        self.ik_joint_ranges[key].append(upper_limit - lower_limit)
                    else:
                        self.ik_joint_ranges[key].append((upper_limit - lower_limit)/2.0)
                    # self.ik_rest_poses[key].append((upper_limit + lower_limit)/2.0)
                    j_names.append([len(j_names)] + list(joint_info[:2]))
        self.ik_rest_poses[key] = self.np_random.uniform(self.ik_lower_limits[key], self.ik_upper_limits[key]).tolist()
        if target_orient is not None:
            ik_joint_poses = np.array(p.calculateInverseKinematics(body, target_joint, targetPosition=target_pos, targetOrientation=target_orient, lowerLimits=self.ik_lower_limits[key], upperLimits=self.ik_upper_limits[key], jointRanges=self.ik_joint_ranges[key], restPoses=self.ik_rest_poses[key], maxNumIterations=max_iterations, physicsClientId=self.id))
        else:
            ik_joint_poses = np.array(p.calculateInverseKinematics(body, target_joint, targetPosition=target_pos, lowerLimits=self.ik_lower_limits[key], upperLimits=self.ik_upper_limits[key], jointRanges=self.ik_joint_ranges[key], restPoses=self.ik_rest_poses[key], maxNumIterations=max_iterations, physicsClientId=self.id))
        # print(j_names)
        # print(ik_joint_poses)
        # exit()
        target_joint_positions = ik_joint_poses[ik_indices]
        return target_joint_positions

    def points_in_cylinder(self, pt1, pt2, r, q):
        vec = pt2 - pt1
        const = r * np.linalg.norm(vec)
        return np.dot(q - pt1, vec) >= 0 and np.dot(q - pt2, vec) <= 0 and np.linalg.norm(np.cross(q - pt1, vec)) <= const

    def point_on_capsule(self, p1, p2, radius, theta_range=(0, np.pi*2)):
        '''
        Pick a random point along the outer surface of a capsule (cylinder)
        '''
        # Pick a random point along the length of the capsule
        axis_vector = p2 - p1
        random_length = self.np_random.uniform(radius, np.linalg.norm(axis_vector))

        # Normalize axis vector to unit length
        axis_vector = axis_vector / np.linalg.norm(axis_vector)
        ortho_vector = self.orthogonal_vector(axis_vector)
        # Normalize orthogonal vector to unit length
        ortho_vector = ortho_vector / np.linalg.norm(ortho_vector)
        # Determine normal vector through cross product (this will be of unit length)
        normal_vector = np.cross(axis_vector, ortho_vector)

        # Pick a random rotation along the cylinder
        theta = self.np_random.uniform(theta_range[0], theta_range[1])

        point = p1 + random_length*axis_vector + radius*np.cos(theta)*ortho_vector + radius*np.sin(theta)*normal_vector
        return point

    def capsule_points(self, p1, p2, radius, distance_between_points=0.05):
        '''
        Creates a set of points around a capsule.
        Check out: http://mathworld.wolfram.com/ConicalFrustum.html
        and: http://math.stackexchange.com/questions/73237/parametric-equation-of-a-circle-in-3d-space
        sphere = [x, y, z, r]
        '''
        points = []

        p1, p2 = np.array(p1), np.array(p2)
        axis_vector = p2 - p1
        # Normalize axis vector to unit length
        axis_vector = axis_vector / np.linalg.norm(axis_vector)
        ortho_vector = self.orthogonal_vector(axis_vector)
        # Normalize orthogonal vector to unit length
        ortho_vector = ortho_vector / np.linalg.norm(ortho_vector)
        # Determine normal vector through cross product (this will be of unit length)
        normal_vector = np.cross(axis_vector, ortho_vector)

        # Determine the section positions along the frustum at which we will create point around in a circular fashion
        sections = int(np.linalg.norm(p2 - p1) / distance_between_points)
        section_positions = [(p2 - p1) / (sections + 1) * (i + 1) for i in range(sections)]
        for i, section_pos in enumerate(section_positions):
            # Determine radius and circumference of this section
            circumference = 2*np.pi*radius
            # Determine the angle difference (in radians) between points
            theta_dist = distance_between_points / radius
            for j in range(int(circumference / distance_between_points)):
                theta = theta_dist * j
                # Determine cartesian coordinates for the point along the circular section of the frustum
                point_on_circle = p1 + section_pos + radius*np.cos(theta)*ortho_vector + radius*np.sin(theta)*normal_vector
                points.append(point_on_circle)

        return points

    def orthogonal_vector(self, v):
        '''
        Two Euclidean vectors are orthogonal if and only if their dot product is zero.
        '''
        # Find first element in v that is nonzero
        m = np.argmax(np.abs(v))
        y = np.zeros(len(v))
        y[(m+1) % len(v)] = 1
        return np.cross(v, y)

    def line_intersects_triangle(self, p0, p1, p2, q0, q1):
        # Check that the arm line segment intersects two different triangles defined by points around the sleeve.
        # https://stackoverflow.com/questions/42740765/intersection-between-line-and-triangle-in-3d
        signed_volume = lambda a, b, c, d: (1.0/6.0) * np.dot(np.cross(b-a, c-a), d-a)
        if np.sign(signed_volume(q0, p0, p1, p2)) != np.sign(signed_volume(q1, p0, p1, p2)):
            if np.sign(signed_volume(q0, q1, p0, p1)) == np.sign(signed_volume(q0, q1, p1, p2)) == np.sign(signed_volume(q0, q1, p2, p0)):
                return True
        return False

    def sleeve_on_arm_reward(self, triangle1_points, triangle2_points, human, hand_radius, elbow_radius, shoulder_radius):
        shoulder_pos, shoulder_orient = p.getLinkState(human, 15, computeForwardKinematics=True, physicsClientId=self.id)[:2]
        elbow_pos, elbow_orient = p.getLinkState(human, 17, computeForwardKinematics=True, physicsClientId=self.id)[:2]
        wrist_pos, wrist_orient = p.getLinkState(human, 19, computeForwardKinematics=True, physicsClientId=self.id)[4:6]
        # Use full length of arm, rather than from hand center to elbow center
        wrist_pos, elbow_pos, shoulder_pos = np.array(wrist_pos), np.array(elbow_pos), np.array(shoulder_pos)
        hand_end_pos = wrist_pos + (wrist_pos - elbow_pos) / np.linalg.norm(wrist_pos - elbow_pos) * hand_radius*2
        elbow_end_pos = elbow_pos + (elbow_pos - wrist_pos) / np.linalg.norm(wrist_pos - elbow_pos) * elbow_radius
        shoulder_end_pos = shoulder_pos + (shoulder_pos - elbow_pos) / np.linalg.norm(shoulder_pos - elbow_pos) * shoulder_radius

        # Given the central axis of the arm, find the plane through the axis and one vector perpendicular to the axis
        # and the plane through the axis and the second vector perpendicular to the other two.
        # There must be points above and below both of these two planes
        # https://math.stackexchange.com/questions/7931/point-below-a-plane
        normal_forearm = hand_end_pos - elbow_end_pos
        normal_forearm = normal_forearm / np.linalg.norm(normal_forearm)
        # Normalized Tangent Vector, assumes arm axis not parallel to vector [1, 1, 0]
        tangent_forearm = np.cross(np.array([1, 1, 0]), normal_forearm)
        tangent_forearm = tangent_forearm / np.linalg.norm(tangent_forearm)
        # Normalized Binormal_forearm or Bitangent_forearm vector
        binormal_forearm = np.cross(tangent_forearm, normal_forearm)
        binormal_forearm = binormal_forearm / np.linalg.norm(binormal_forearm)

        # Check if at least one point exists above and below both planes
        # v.dot(p - p0), p0 on plane, v is normal_forearm of a plane. v = tangent_forearm, v = binormal_forearm, p0 = elbow_end_pos
        all_points = np.concatenate([triangle1_points, triangle2_points], axis=0)
        tangent_forearm_points = np.dot(tangent_forearm, (all_points - elbow_end_pos).T)
        binormal_forearm_points = np.dot(binormal_forearm, (all_points - elbow_end_pos).T)
        points_above_below_forearm = np.any(tangent_forearm_points > 0) and np.any(tangent_forearm_points < 0) and np.any(binormal_forearm_points > 0) and np.any(binormal_forearm_points < 0)

        normal_upperarm = elbow_end_pos - shoulder_end_pos
        normal_upperarm = normal_upperarm / np.linalg.norm(normal_upperarm)
        tangent_upperarm = np.cross(np.array([1, 1, 0]), normal_upperarm)
        tangent_upperarm = tangent_upperarm / np.linalg.norm(tangent_upperarm)
        binormal_upperarm = np.cross(tangent_upperarm, normal_upperarm)
        binormal_upperarm = binormal_upperarm / np.linalg.norm(binormal_upperarm)
        tangent_upperarm_points = np.dot(tangent_upperarm, (all_points - shoulder_end_pos).T)
        binormal_upperarm_points = np.dot(binormal_upperarm, (all_points - shoulder_end_pos).T)
        points_above_below_upperarm = np.any(tangent_upperarm_points > 0) and np.any(tangent_upperarm_points < 0) and np.any(binormal_upperarm_points > 0) and np.any(binormal_upperarm_points < 0)

        # Check that the arm line segment intersects two different triangles defined by points around the sleeve.
        # https://stackoverflow.com/questions/42740765/intersection-between-line-and-triangle-in-3d
        forearm_intersects_triangle1 = self.line_intersects_triangle(triangle1_points[0], triangle1_points[1], triangle1_points[2], hand_end_pos, elbow_end_pos)
        forearm_intersects_triangle2 = self.line_intersects_triangle(triangle2_points[0], triangle2_points[1], triangle2_points[2], hand_end_pos, elbow_end_pos)
        upperarm_intersects_triangle1 = self.line_intersects_triangle(triangle1_points[0], triangle1_points[1], triangle1_points[2], elbow_end_pos, shoulder_end_pos)
        upperarm_intersects_triangle2 = self.line_intersects_triangle(triangle2_points[0], triangle2_points[1], triangle2_points[2], elbow_end_pos, shoulder_end_pos)
        sleeve_center = np.mean(all_points, axis=0)
        distance_to_shoulder = np.linalg.norm(shoulder_end_pos - sleeve_center)
        distance_to_elbow = np.linalg.norm(elbow_end_pos - sleeve_center)
        distance_to_hand = np.linalg.norm(hand_end_pos - sleeve_center)

        # Reward forward movement along the arm, away from the hand (pulling the sleeve onto the arm)
        distance_along_forearm = np.linalg.norm(sleeve_center - hand_end_pos)
        distance_along_upperarm = np.linalg.norm(sleeve_center - elbow_pos)

        forearm_in_sleeve = points_above_below_forearm and (forearm_intersects_triangle1 or forearm_intersects_triangle2)
        upperarm_in_sleeve = points_above_below_upperarm and (upperarm_intersects_triangle1 or upperarm_intersects_triangle2)

        # Find the point at which the arm central axis intersects one of the triangles
        # p0, p1, p2 = triangle1_points
        # N = np.cross(p1-p0, p2-p0)
        # t = -np.dot(hand_end_pos, N-p0) / np.dot(hand_end_pos, elbow_end_pos-hand_end_pos)
        # intersection_point = hand_end_pos + t*(elbow_end_pos-hand_end_pos)

        return forearm_in_sleeve, upperarm_in_sleeve, distance_along_forearm, distance_along_upperarm, distance_to_hand, distance_to_elbow, distance_to_shoulder, np.linalg.norm(hand_end_pos - elbow_end_pos), np.linalg.norm(elbow_pos - shoulder_pos)

