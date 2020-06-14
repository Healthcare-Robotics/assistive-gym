import numpy as np
import pybullet as p

class Agent:
    def __init__(self):
        self.base = -1
        self.lower_limits = None
        self.upper_limits = None
        self.ik_lower_limits = None
        self.ik_upper_limits = None
        self.ik_joint_names = None

    def init(self, body, id, np_random, indices=None):
        self.body = body
        self.id = id
        self.np_random = np_random
        self.all_joint_indices = list(range(p.getNumJoints(body, physicsClientId=id)))
        if indices != -1:
            self.update_joint_limits()
            self.enforce_joint_limits(indices)
            self.controllable_joint_lower_limits = [self.lower_limits[i] for i in self.controllable_joint_indices]
            self.controllable_joint_upper_limits = [self.upper_limits[i] for i in self.controllable_joint_indices]

    def control(self, indices, target_angles, gains, forces):
        if type(gains) in [int, float]:
            gains = [gains]*len(indices)
        if type(forces) in [int, float]:
            forces = [forces]*len(indices)
        p.setJointMotorControlArray(self.body, jointIndices=indices, controlMode=p.POSITION_CONTROL, targetPositions=target_angles, positionGains=gains, forces=forces, physicsClientId=self.id)

    def get_joint_angles(self, indices=None):
        if indices is None:
            indices = self.all_joint_indices
        robot_joint_states = p.getJointStates(self.body, jointIndices=indices, physicsClientId=self.id)
        return np.array([x[0] for x in robot_joint_states])

    def get_joint_angles_dict(self, indices=None):
        return {j: a for j, a in zip(indices, self.get_joint_angles(indices))}

    def get_pos_orient(self, link, center_of_mass=False, convert_to_realworld=False):
        # Get the 3D position and orientation (4D quaternion) of a specific link on the body
        if link == self.base:
            pos, orient = p.getBasePositionAndOrientation(self.body, physicsClientId=self.id)
        else:
            if not center_of_mass:
                pos, orient = p.getLinkState(self.body, link, computeForwardKinematics=True, physicsClientId=self.id)[4:6]
            else:
                pos, orient = p.getLinkState(self.body, link, computeForwardKinematics=True, physicsClientId=self.id)[:2]
        if convert_to_realworld:
            return self.convert_to_realworld(pos, orient)
        else:
            return np.array(pos), np.array(orient)

    def convert_to_realworld(self, pos, orient=[0, 0, 0, 1]):
        base_pos, base_orient = self.get_base_pos_orient()
        base_pos_inv, base_orient_inv = p.invertTransform(base_pos, base_orient, physicsClientId=self.id)
        real_pos, real_orient = p.multiplyTransforms(base_pos_inv, base_orient_inv, pos, orient, physicsClientId=self.id)
        return np.array(real_pos), np.array(real_orient)

    def get_base_pos_orient(self):
        return self.get_pos_orient(self.base)

    def get_velocity(self, link):
        if link == self.base:
            return p.getBaseVelocity(self.body, physicsClientId=self.id)[0]
        return p.getLinkState(self.body, link, computeForwardKinematics=True, computeLinkVelocity=True, physicsClientId=self.id)[6]

    def get_motor_joint_states(self):
        # Get the position, velocity, and torque for nonfixed joint motors
        joint_states = p.getJointStates(self.body, self.all_joint_indices, physicsClientId=self.id)
        joint_infos = [p.getJointInfo(self.body, i, physicsClientId=self.id) for i in self.all_joint_indices]
        motor_states = [j for j, i in zip(joint_states, joint_infos) if i[2] != p.JOINT_FIXED]
        motor_indices = [i[0] for j, i in zip(joint_states, joint_infos) if i[2] != p.JOINT_FIXED]
        motor_positions = [state[0] for state in motor_states]
        motor_velocities = [state[1] for state in motor_states]
        motor_torques = [state[3] for state in motor_states]
        return motor_indices, motor_positions, motor_velocities, motor_torques

    def get_joint_max_force(self, indices=None):
        if indices is None:
            indices = self.all_joint_indices
        joint_infos = [p.getJointInfo(self.body, i, physicsClientId=self.id) for i in indices]
        return [j[10] for j in joint_infos]

    def get_contact_points(self, agentB=None):
        if agentB is None:
            cp = p.getContactPoints(bodyA=self.body, physicsClientId=self.id)
        else:
            cp = p.getContactPoints(bodyA=self.body, bodyB=agentB.body, physicsClientId=self.id)
        if cp is None:
            return [], [], [], [], []
        linkA = [c[3] for c in cp]
        linkB = [c[4] for c in cp]
        posA = [c[5] for c in cp]
        posB = [c[6] for c in cp]
        force = [c[9] for c in cp]
        return linkA, linkB, posA, posB, force

    def get_closest_points(self, agentB, distance=4.0):
        cp = p.getClosestPoints(bodyA=self.body, bodyB=agentB.body, distance=distance, physicsClientId=self.id)
        linkA = [c[3] for c in cp]
        linkB = [c[4] for c in cp]
        posA = [c[5] for c in cp]
        posB = [c[6] for c in cp]
        contact_distance = [c[8] for c in cp]
        return linkA, linkB, posA, posB, contact_distance

    def get_force_torque_sensor(self, joint):
        return np.array(p.getJointState(self.body, joint, physicsClientId=self.id)[2])

    def set_base_pos_orient(self, pos, orient, euler=False):
        p.resetBasePositionAndOrientation(self.body, pos, orient if not euler else p.getQuaternionFromEuler(orient, physicsClientId=self.id), physicsClientId=self.id)

    def set_base_velocity(self, linear_velocity, angular_velocity):
        p.resetBaseVelocity(self.body, linearVelocity=linear_velocity, angularVelocity=angular_velocity, physicsClientId=self.id)

    def set_joint_angles(self, indices, angles, use_limits=True):
        for j, a in zip(indices, angles):
            p.resetJointState(self.body, jointIndex=j, targetValue=min(max(a, self.lower_limits[j]), self.upper_limits[j]) if use_limits else a, targetVelocity=0, physicsClientId=self.id)

    def reset_joints(self):
        # Reset all joints to 0 position, 0 velocity
        self.set_joint_angles(self.all_joint_indices, [0]*len(self.all_joint_indices))

    def set_friction(self, link, friction):
        p.changeDynamics(self.body, link, lateralFriction=friction, spinningFriction=friction, rollingFriction=friction, physicsClientId=self.id)

    def set_mass(self, link, mass):
        p.changeDynamics(self.body, link, mass=mass, physicsClientId=self.id)

    def enable_force_torque_sensor(self, joint):
        p.enableJointForceTorqueSensor(self.body, joint, enableSensor=True, physicsClientId=self.id)

    def create_constraint(self, parent_link, child, child_link, joint_type=p.JOINT_FIXED, joint_axis=[0, 0, 0], parent_pos=[0, 0, 0], child_pos=[0, 0, 0], parent_orient=[0, 0, 0], child_orient=[0, 0, 0]):
        return p.createConstraint(self.body, parent_link, child.body, child_link, joint_type, joint_axis, parent_pos, child_pos, p.getQuaternionFromEuler(parent_orient, physicsClientId=self.id), p.getQuaternionFromEuler(child_orient, physicsClientId=self.id), physicsClientId=self.id)

    def update_joint_limits(self, indices=None):
        if indices is None:
            indices = self.all_joint_indices
        self.lower_limits = dict()
        self.upper_limits = dict()
        self.ik_lower_limits = []
        self.ik_upper_limits = []
        self.ik_joint_names = []
        for j in indices:
            joint_info = p.getJointInfo(self.body, j, physicsClientId=self.id)
            joint_name = joint_info[1]
            joint_type = joint_info[2]
            lower_limit = joint_info[8]
            upper_limit = joint_info[9]
            if lower_limit == 0 and upper_limit == -1:
                lower_limit = -1e10
                upper_limit = 1e10
                if joint_type != p.JOINT_FIXED:
                    # NOTE: IK only works on non fixed joints, so we build special joint limit lists for IK
                    self.ik_lower_limits.append(-2*np.pi)
                    self.ik_upper_limits.append(2*np.pi)
                    self.ik_joint_names.append([len(self.ik_joint_names)] + list(joint_info[:2]))
            elif joint_type != p.JOINT_FIXED:
                self.ik_lower_limits.append(lower_limit)
                self.ik_upper_limits.append(upper_limit)
                self.ik_joint_names.append([len(self.ik_joint_names)] + list(joint_info[:2]))
            self.lower_limits[j] = lower_limit
            self.upper_limits[j] = upper_limit
        self.ik_lower_limits = np.array(self.ik_lower_limits)
        self.ik_upper_limits = np.array(self.ik_upper_limits)

    def enforce_joint_limits(self, indices=None):
        if indices is None:
            indices = self.all_joint_indices
        joint_angles = self.get_joint_angles_dict(indices)
        if self.lower_limits is None or len(indices) > len(self.lower_limits):
            self.update_joint_limits()
        for j in indices:
            if joint_angles[j] < self.lower_limits[j]:
                p.resetJointState(self.body, jointIndex=j, targetValue=self.lower_limits[j], targetVelocity=0, physicsClientId=self.id)
            elif joint_angles[j] > self.upper_limits[j]:
                p.resetJointState(self.body, jointIndex=j, targetValue=self.upper_limits[j], targetVelocity=0, physicsClientId=self.id)

    def ik(self, target_joint, target_pos, target_orient, ik_indices, max_iterations=1000, half_range=False):
        ik_joint_ranges = self.ik_upper_limits - self.ik_lower_limits
        if half_range:
            ik_joint_ranges /= 2.0
        ik_rest_poses = self.np_random.uniform(self.ik_lower_limits, self.ik_upper_limits)

        # print('JPO:', target_joint, target_pos, target_orient)
        # print('Lower:', self.ik_lower_limits)
        # print('Upper:', self.ik_upper_limits)
        # print('Range:', ik_joint_ranges)
        # print('Rest:', ik_rest_poses)
        if target_orient is not None:
            ik_joint_poses = np.array(p.calculateInverseKinematics(self.body, target_joint, targetPosition=target_pos, targetOrientation=target_orient, lowerLimits=self.ik_lower_limits.tolist(), upperLimits=self.ik_upper_limits.tolist(), jointRanges=ik_joint_ranges.tolist(), restPoses=ik_rest_poses.tolist(), maxNumIterations=max_iterations, physicsClientId=self.id))
        else:
            ik_joint_poses = np.array(p.calculateInverseKinematics(self.body, target_joint, targetPosition=target_pos, lowerLimits=self.ik_lower_limits.tolist(), upperLimits=self.ik_upper_limits.tolist(), jointRanges=ik_joint_ranges.tolist(), restPoses=ik_rest_poses.tolist(), maxNumIterations=max_iterations, physicsClientId=self.id))
        return ik_joint_poses[ik_indices]

    def print_joint_info(self, show_fixed=True):
        joint_names = []
        for j in self.all_joint_indices:
            info = p.getJointInfo(self.body, j, physicsClientId=self.id)
            if show_fixed or info[2] != p.JOINT_FIXED:
                print(info)
                joint_names.append((j, info[1]))
        print(joint_names)

