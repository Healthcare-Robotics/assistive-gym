from .dressing import DressingEnv
from .agents import pr2, baxter, sawyer, jaco
from .agents.pr2 import PR2
from .agents.baxter import Baxter
from .agents.sawyer import Sawyer
from .agents.jaco import Jaco

robot_arm = 'left'
class DressingPR2Env(DressingEnv):
    def __init__(self):
        super(DressingPR2Env, self).__init__(robot=PR2(robot_arm), human_control=False)

class DressingBaxterEnv(DressingEnv):
    def __init__(self):
        super(DressingBaxterEnv, self).__init__(robot=Baxter(robot_arm), human_control=False)

class DressingSawyerEnv(DressingEnv):
    def __init__(self):
        super(DressingSawyerEnv, self).__init__(robot=Sawyer(robot_arm), human_control=False)

class DressingJacoEnv(DressingEnv):
    def __init__(self):
        super(DressingJacoEnv, self).__init__(robot=Jaco(robot_arm), human_control=False)

class DressingPR2HumanEnv(DressingEnv):
    def __init__(self):
        super(DressingPR2HumanEnv, self).__init__(robot=PR2(robot_arm), human_control=True)

class DressingBaxterHumanEnv(DressingEnv):
    def __init__(self):
        super(DressingBaxterHumanEnv, self).__init__(robot=Baxter(robot_arm), human_control=True)

class DressingSawyerHumanEnv(DressingEnv):
    def __init__(self):
        super(DressingSawyerHumanEnv, self).__init__(robot=Sawyer(robot_arm), human_control=True)

class DressingJacoHumanEnv(DressingEnv):
    def __init__(self):
        super(DressingJacoHumanEnv, self).__init__(robot=Jaco(robot_arm), human_control=True)

