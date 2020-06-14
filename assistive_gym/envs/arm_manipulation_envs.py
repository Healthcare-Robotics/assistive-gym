from .arm_manipulation import ArmManipulationEnv
from .agents import pr2, baxter, sawyer, jaco
from .agents.pr2 import PR2
from .agents.baxter import Baxter
from .agents.sawyer import Sawyer
from .agents.jaco import Jaco

robot_arm = 'both'
class ArmManipulationPR2Env(ArmManipulationEnv):
    def __init__(self):
        super(ArmManipulationPR2Env, self).__init__(robot=PR2(robot_arm), human_control=False)

class ArmManipulationBaxterEnv(ArmManipulationEnv):
    def __init__(self):
        super(ArmManipulationBaxterEnv, self).__init__(robot=Baxter(robot_arm), human_control=False)

class ArmManipulationSawyerEnv(ArmManipulationEnv):
    def __init__(self):
        super(ArmManipulationSawyerEnv, self).__init__(robot=Sawyer(robot_arm), human_control=False)

class ArmManipulationJacoEnv(ArmManipulationEnv):
    def __init__(self):
        super(ArmManipulationJacoEnv, self).__init__(robot=Jaco(robot_arm), human_control=False)

class ArmManipulationPR2HumanEnv(ArmManipulationEnv):
    def __init__(self):
        super(ArmManipulationPR2HumanEnv, self).__init__(robot=PR2(robot_arm), human_control=True)

class ArmManipulationBaxterHumanEnv(ArmManipulationEnv):
    def __init__(self):
        super(ArmManipulationBaxterHumanEnv, self).__init__(robot=Baxter(robot_arm), human_control=True)

class ArmManipulationSawyerHumanEnv(ArmManipulationEnv):
    def __init__(self):
        super(ArmManipulationSawyerHumanEnv, self).__init__(robot=Sawyer(robot_arm), human_control=True)

class ArmManipulationJacoHumanEnv(ArmManipulationEnv):
    def __init__(self):
        super(ArmManipulationJacoHumanEnv, self).__init__(robot=Jaco(robot_arm), human_control=True)

