from .scratch_itch import ScratchItchEnv
from .agents import pr2, baxter, sawyer, jaco
from .agents.pr2 import PR2
from .agents.baxter import Baxter
from .agents.sawyer import Sawyer
from .agents.jaco import Jaco

robot_arm = 'left'
class ScratchItchPR2Env(ScratchItchEnv):
    def __init__(self):
        super(ScratchItchPR2Env, self).__init__(robot=PR2(robot_arm), human_control=False)

class ScratchItchBaxterEnv(ScratchItchEnv):
    def __init__(self):
        super(ScratchItchBaxterEnv, self).__init__(robot=Baxter(robot_arm), human_control=False)

class ScratchItchSawyerEnv(ScratchItchEnv):
    def __init__(self):
        super(ScratchItchSawyerEnv, self).__init__(robot=Sawyer(robot_arm), human_control=False)

class ScratchItchJacoEnv(ScratchItchEnv):
    def __init__(self):
        super(ScratchItchJacoEnv, self).__init__(robot=Jaco(robot_arm), human_control=False)

class ScratchItchPR2HumanEnv(ScratchItchEnv):
    def __init__(self):
        super(ScratchItchPR2HumanEnv, self).__init__(robot=PR2(robot_arm), human_control=True)

class ScratchItchBaxterHumanEnv(ScratchItchEnv):
    def __init__(self):
        super(ScratchItchBaxterHumanEnv, self).__init__(robot=Baxter(robot_arm), human_control=True)

class ScratchItchSawyerHumanEnv(ScratchItchEnv):
    def __init__(self):
        super(ScratchItchSawyerHumanEnv, self).__init__(robot=Sawyer(robot_arm), human_control=True)

class ScratchItchJacoHumanEnv(ScratchItchEnv):
    def __init__(self):
        super(ScratchItchJacoHumanEnv, self).__init__(robot=Jaco(robot_arm), human_control=True)

