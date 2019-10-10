from .arm_manipulation import ArmManipulationEnv

class ArmManipulationPR2Env(ArmManipulationEnv):
    def __init__(self):
        super(ArmManipulationPR2Env, self).__init__(robot_type='pr2', human_control=False)

class ArmManipulationBaxterEnv(ArmManipulationEnv):
    def __init__(self):
        super(ArmManipulationBaxterEnv, self).__init__(robot_type='baxter', human_control=False)

class ArmManipulationSawyerEnv(ArmManipulationEnv):
    def __init__(self):
        super(ArmManipulationSawyerEnv, self).__init__(robot_type='sawyer', human_control=False)

class ArmManipulationJacoEnv(ArmManipulationEnv):
    def __init__(self):
        super(ArmManipulationJacoEnv, self).__init__(robot_type='jaco', human_control=False)

class ArmManipulationKinovaGen3Env(ArmManipulationEnv):
    def __init__(self):
        super(ArmManipulationKinovaGen3Env, self).__init__(robot_type='kinova_gen3', human_control=False)

class ArmManipulationPR2HumanEnv(ArmManipulationEnv):
    def __init__(self):
        super(ArmManipulationPR2HumanEnv, self).__init__(robot_type='pr2', human_control=True)

class ArmManipulationBaxterHumanEnv(ArmManipulationEnv):
    def __init__(self):
        super(ArmManipulationBaxterHumanEnv, self).__init__(robot_type='baxter', human_control=True)

class ArmManipulationSawyerHumanEnv(ArmManipulationEnv):
    def __init__(self):
        super(ArmManipulationSawyerHumanEnv, self).__init__(robot_type='sawyer', human_control=True)

class ArmManipulationJacoHumanEnv(ArmManipulationEnv):
    def __init__(self):
        super(ArmManipulationJacoHumanEnv, self).__init__(robot_type='jaco', human_control=True)

class ArmManipulationKinovaGen3HumanEnv(ArmManipulationEnv):
    def __init__(self):
        super(ArmManipulationKinovaGen3HumanEnv, self).__init__(robot_type='kinova_gen3', human_control=True)

