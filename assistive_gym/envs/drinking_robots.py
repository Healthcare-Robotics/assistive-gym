from .drinking import DrinkingEnv

class DrinkingPR2Env(DrinkingEnv):
    def __init__(self):
        super(DrinkingPR2Env, self).__init__(robot_type='pr2', human_control=False)

class DrinkingBaxterEnv(DrinkingEnv):
    def __init__(self):
        super(DrinkingBaxterEnv, self).__init__(robot_type='baxter', human_control=False)

class DrinkingSawyerEnv(DrinkingEnv):
    def __init__(self):
        super(DrinkingSawyerEnv, self).__init__(robot_type='sawyer', human_control=False)

class DrinkingJacoEnv(DrinkingEnv):
    def __init__(self):
        super(DrinkingJacoEnv, self).__init__(robot_type='jaco', human_control=False)

class DrinkingPR2HumanEnv(DrinkingEnv):
    def __init__(self):
        super(DrinkingPR2HumanEnv, self).__init__(robot_type='pr2', human_control=True)

class DrinkingBaxterHumanEnv(DrinkingEnv):
    def __init__(self):
        super(DrinkingBaxterHumanEnv, self).__init__(robot_type='baxter', human_control=True)

class DrinkingSawyerHumanEnv(DrinkingEnv):
    def __init__(self):
        super(DrinkingSawyerHumanEnv, self).__init__(robot_type='sawyer', human_control=True)

class DrinkingJacoHumanEnv(DrinkingEnv):
    def __init__(self):
        super(DrinkingJacoHumanEnv, self).__init__(robot_type='jaco', human_control=True)

