from .drinking import DrinkingEnv
from .agents import pr2, baxter, sawyer, jaco
from .agents.pr2 import PR2
from .agents.baxter import Baxter
from .agents.sawyer import Sawyer
from .agents.jaco import Jaco

robot_arm = 'right'
class DrinkingPR2Env(DrinkingEnv):
    def __init__(self):
        super(DrinkingPR2Env, self).__init__(robot=PR2(robot_arm), human_control=False)

class DrinkingBaxterEnv(DrinkingEnv):
    def __init__(self):
        super(DrinkingBaxterEnv, self).__init__(robot=Baxter(robot_arm), human_control=False)

class DrinkingSawyerEnv(DrinkingEnv):
    def __init__(self):
        super(DrinkingSawyerEnv, self).__init__(robot=Sawyer(robot_arm), human_control=False)

class DrinkingJacoEnv(DrinkingEnv):
    def __init__(self):
        super(DrinkingJacoEnv, self).__init__(robot=Jaco(robot_arm), human_control=False)

class DrinkingPR2HumanEnv(DrinkingEnv):
    def __init__(self):
        super(DrinkingPR2HumanEnv, self).__init__(robot=PR2(robot_arm), human_control=True)

class DrinkingBaxterHumanEnv(DrinkingEnv):
    def __init__(self):
        super(DrinkingBaxterHumanEnv, self).__init__(robot=Baxter(robot_arm), human_control=True)

class DrinkingSawyerHumanEnv(DrinkingEnv):
    def __init__(self):
        super(DrinkingSawyerHumanEnv, self).__init__(robot=Sawyer(robot_arm), human_control=True)

class DrinkingJacoHumanEnv(DrinkingEnv):
    def __init__(self):
        super(DrinkingJacoHumanEnv, self).__init__(robot=Jaco(robot_arm), human_control=True)

