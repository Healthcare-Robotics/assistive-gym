from .bed_bathing import BedBathingEnv

class BedBathingPR2Env(BedBathingEnv):
    def __init__(self):
        super(BedBathingPR2Env, self).__init__(robot_type='pr2', human_control=False)

class BedBathingBaxterEnv(BedBathingEnv):
    def __init__(self):
        super(BedBathingBaxterEnv, self).__init__(robot_type='baxter', human_control=False)

class BedBathingSawyerEnv(BedBathingEnv):
    def __init__(self):
        super(BedBathingSawyerEnv, self).__init__(robot_type='sawyer', human_control=False)

class BedBathingJacoEnv(BedBathingEnv):
    def __init__(self):
        super(BedBathingJacoEnv, self).__init__(robot_type='jaco', human_control=False)

class BedBathingPR2HumanEnv(BedBathingEnv):
    def __init__(self):
        super(BedBathingPR2HumanEnv, self).__init__(robot_type='pr2', human_control=True)

class BedBathingBaxterHumanEnv(BedBathingEnv):
    def __init__(self):
        super(BedBathingBaxterHumanEnv, self).__init__(robot_type='baxter', human_control=True)

class BedBathingSawyerHumanEnv(BedBathingEnv):
    def __init__(self):
        super(BedBathingSawyerHumanEnv, self).__init__(robot_type='sawyer', human_control=True)

class BedBathingJacoHumanEnv(BedBathingEnv):
    def __init__(self):
        super(BedBathingJacoHumanEnv, self).__init__(robot_type='jaco', human_control=True)

