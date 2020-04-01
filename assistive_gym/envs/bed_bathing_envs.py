from .bed_bathing import BedBathingEnv
from agents.pr2 import PR2
from agents.baxter import Baxter
from agents.sawyer import Sawyer
from agents.jaco import Jaco

robot_arm = 'left'
class BedBathingPR2Env(BedBathingEnv):
    def __init__(self):
        super(BedBathingPR2Env, self).__init__(robot=PR2(robot_arm), human_control=False)

class BedBathingBaxterEnv(BedBathingEnv):
    def __init__(self):
        super(BedBathingBaxterEnv, self).__init__(robot=Baxter(robot_arm), human_control=False)

class BedBathingSawyerEnv(BedBathingEnv):
    def __init__(self):
        super(BedBathingSawyerEnv, self).__init__(robot=Sawyer(robot_arm), human_control=False)

class BedBathingJacoEnv(BedBathingEnv):
    def __init__(self):
        super(BedBathingJacoEnv, self).__init__(robot=Jaco(robot_arm), human_control=False)

class BedBathingPR2HumanEnv(BedBathingEnv):
    def __init__(self):
        super(BedBathingPR2HumanEnv, self).__init__(robot=PR2(robot_arm), human_control=True)

class BedBathingBaxterHumanEnv(BedBathingEnv):
    def __init__(self):
        super(BedBathingBaxterHumanEnv, self).__init__(robot=Baxter(robot_arm), human_control=True)

class BedBathingSawyerHumanEnv(BedBathingEnv):
    def __init__(self):
        super(BedBathingSawyerHumanEnv, self).__init__(robot=Sawyer(robot_arm), human_control=True)

class BedBathingJacoHumanEnv(BedBathingEnv):
    def __init__(self):
        super(BedBathingJacoHumanEnv, self).__init__(robot=Jaco(robot_arm), human_control=True)

