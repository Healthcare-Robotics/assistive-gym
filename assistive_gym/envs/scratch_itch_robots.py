from .scratch_itch import ScratchItchEnv

class ScratchItchPR2Env(ScratchItchEnv):
    def __init__(self):
        super(ScratchItchPR2Env, self).__init__(robot_type='pr2', human_control=False)

class ScratchItchBaxterEnv(ScratchItchEnv):
    def __init__(self):
        super(ScratchItchBaxterEnv, self).__init__(robot_type='baxter', human_control=False)

class ScratchItchSawyerEnv(ScratchItchEnv):
    def __init__(self):
        super(ScratchItchSawyerEnv, self).__init__(robot_type='sawyer', human_control=False)

class ScratchItchJacoEnv(ScratchItchEnv):
    def __init__(self):
        super(ScratchItchJacoEnv, self).__init__(robot_type='jaco', human_control=False)

class ScratchItchPR2HumanEnv(ScratchItchEnv):
    def __init__(self):
        super(ScratchItchPR2HumanEnv, self).__init__(robot_type='pr2', human_control=True)

class ScratchItchBaxterHumanEnv(ScratchItchEnv):
    def __init__(self):
        super(ScratchItchBaxterHumanEnv, self).__init__(robot_type='baxter', human_control=True)

class ScratchItchSawyerHumanEnv(ScratchItchEnv):
    def __init__(self):
        super(ScratchItchSawyerHumanEnv, self).__init__(robot_type='sawyer', human_control=True)

class ScratchItchJacoHumanEnv(ScratchItchEnv):
    def __init__(self):
        super(ScratchItchJacoHumanEnv, self).__init__(robot_type='jaco', human_control=True)

