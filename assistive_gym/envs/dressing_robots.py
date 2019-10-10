from .dressing import DressingEnv

class DressingPR2Env(DressingEnv):
    def __init__(self):
        super(DressingPR2Env, self).__init__(robot_type='pr2', human_control=False)

class DressingBaxterEnv(DressingEnv):
    def __init__(self):
        super(DressingBaxterEnv, self).__init__(robot_type='baxter', human_control=False)

class DressingSawyerEnv(DressingEnv):
    def __init__(self):
        super(DressingSawyerEnv, self).__init__(robot_type='sawyer', human_control=False)

class DressingJacoEnv(DressingEnv):
    def __init__(self):
        super(DressingJacoEnv, self).__init__(robot_type='jaco', human_control=False)

class DressingPR2HumanEnv(DressingEnv):
    def __init__(self):
        super(DressingPR2HumanEnv, self).__init__(robot_type='pr2', human_control=True)

class DressingBaxterHumanEnv(DressingEnv):
    def __init__(self):
        super(DressingBaxterHumanEnv, self).__init__(robot_type='baxter', human_control=True)

class DressingSawyerHumanEnv(DressingEnv):
    def __init__(self):
        super(DressingSawyerHumanEnv, self).__init__(robot_type='sawyer', human_control=True)

class DressingJacoHumanEnv(DressingEnv):
    def __init__(self):
        super(DressingJacoHumanEnv, self).__init__(robot_type='jaco', human_control=True)

