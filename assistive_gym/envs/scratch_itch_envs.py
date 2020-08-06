from .scratch_itch import ScratchItchEnv
from .agents import pr2, baxter, sawyer, jaco, human
from .agents.pr2 import PR2
from .agents.baxter import Baxter
from .agents.sawyer import Sawyer
from .agents.jaco import Jaco
from .agents.stretch import Stretch
from .agents.human import Human

robot_arm = 'left'
human_controllable_joint_indices = human.right_arm_joints
class ScratchItchPR2Env(ScratchItchEnv):
    def __init__(self):
        super(ScratchItchPR2Env, self).__init__(robot=PR2(robot_arm), human=Human(human_controllable_joint_indices, controllable=False))

class ScratchItchBaxterEnv(ScratchItchEnv):
    def __init__(self):
        super(ScratchItchBaxterEnv, self).__init__(robot=Baxter(robot_arm), human=Human(human_controllable_joint_indices, controllable=False))

class ScratchItchSawyerEnv(ScratchItchEnv):
    def __init__(self):
        super(ScratchItchSawyerEnv, self).__init__(robot=Sawyer(robot_arm), human=Human(human_controllable_joint_indices, controllable=False))

class ScratchItchJacoEnv(ScratchItchEnv):
    def __init__(self):
        super(ScratchItchJacoEnv, self).__init__(robot=Jaco(robot_arm), human=Human(human_controllable_joint_indices, controllable=False))

class ScratchItchStretchEnv(ScratchItchEnv):
    def __init__(self):
        super(ScratchItchStretchEnv, self).__init__(robot=Stretch('wheel_'+robot_arm), human=Human(human_controllable_joint_indices, controllable=False))

class ScratchItchPR2HumanEnv(ScratchItchEnv):
    def __init__(self):
        super(ScratchItchPR2HumanEnv, self).__init__(robot=PR2(robot_arm), human=Human(human_controllable_joint_indices, controllable=True))

class ScratchItchBaxterHumanEnv(ScratchItchEnv):
    def __init__(self):
        super(ScratchItchBaxterHumanEnv, self).__init__(robot=Baxter(robot_arm), human=Human(human_controllable_joint_indices, controllable=True))

class ScratchItchSawyerHumanEnv(ScratchItchEnv):
    def __init__(self):
        super(ScratchItchSawyerHumanEnv, self).__init__(robot=Sawyer(robot_arm), human=Human(human_controllable_joint_indices, controllable=True))

class ScratchItchJacoHumanEnv(ScratchItchEnv):
    def __init__(self):
        super(ScratchItchJacoHumanEnv, self).__init__(robot=Jaco(robot_arm), human=Human(human_controllable_joint_indices, controllable=True))

class ScratchItchStretchHumanEnv(ScratchItchEnv):
    def __init__(self):
        super(ScratchItchStretchHumanEnv, self).__init__(robot=Stretch('wheel_'+robot_arm), human=Human(human_controllable_joint_indices, controllable=True))

