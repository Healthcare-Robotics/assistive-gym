from .dressing import DressingEnv
from .agents import pr2, baxter, sawyer, jaco, stretch, panda, human
from .agents.pr2 import PR2
from .agents.baxter import Baxter
from .agents.sawyer import Sawyer
from .agents.jaco import Jaco
from .agents.stretch import Stretch
from .agents.panda import Panda
from .agents.human import Human
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.tune.registry import register_env

robot_arm = 'left'
human_controllable_joint_indices = human.left_arm_joints
class DressingPR2Env(DressingEnv):
    def __init__(self):
        super(DressingPR2Env, self).__init__(robot=PR2(robot_arm), human=Human(human_controllable_joint_indices, controllable=False))

class DressingBaxterEnv(DressingEnv):
    def __init__(self):
        super(DressingBaxterEnv, self).__init__(robot=Baxter(robot_arm), human=Human(human_controllable_joint_indices, controllable=False))

class DressingSawyerEnv(DressingEnv):
    def __init__(self):
        super(DressingSawyerEnv, self).__init__(robot=Sawyer(robot_arm), human=Human(human_controllable_joint_indices, controllable=False))

class DressingJacoEnv(DressingEnv):
    def __init__(self):
        super(DressingJacoEnv, self).__init__(robot=Jaco(robot_arm), human=Human(human_controllable_joint_indices, controllable=False))

class DressingStretchEnv(DressingEnv):
    def __init__(self):
        super(DressingStretchEnv, self).__init__(robot=Stretch('wheel_'+robot_arm), human=Human(human_controllable_joint_indices, controllable=False))

class DressingPandaEnv(DressingEnv):
    def __init__(self):
        super(DressingPandaEnv, self).__init__(robot=Panda(robot_arm), human=Human(human_controllable_joint_indices, controllable=False))

class DressingPR2HumanEnv(DressingEnv, MultiAgentEnv):
    def __init__(self):
        super(DressingPR2HumanEnv, self).__init__(robot=PR2(robot_arm), human=Human(human_controllable_joint_indices, controllable=True))
register_env('assistive_gym:DressingPR2Human-v1', lambda config: DressingPR2HumanEnv())

class DressingBaxterHumanEnv(DressingEnv, MultiAgentEnv):
    def __init__(self):
        super(DressingBaxterHumanEnv, self).__init__(robot=Baxter(robot_arm), human=Human(human_controllable_joint_indices, controllable=True))
register_env('assistive_gym:DressingBaxterHuman-v1', lambda config: DressingBaxterHumanEnv())

class DressingSawyerHumanEnv(DressingEnv, MultiAgentEnv):
    def __init__(self):
        super(DressingSawyerHumanEnv, self).__init__(robot=Sawyer(robot_arm), human=Human(human_controllable_joint_indices, controllable=True))
register_env('assistive_gym:DressingSawyerHuman-v1', lambda config: DressingSawyerHumanEnv())

class DressingJacoHumanEnv(DressingEnv, MultiAgentEnv):
    def __init__(self):
        super(DressingJacoHumanEnv, self).__init__(robot=Jaco(robot_arm), human=Human(human_controllable_joint_indices, controllable=True))
register_env('assistive_gym:DressingJacoHuman-v1', lambda config: DressingJacoHumanEnv())

class DressingStretchHumanEnv(DressingEnv, MultiAgentEnv):
    def __init__(self):
        super(DressingStretchHumanEnv, self).__init__(robot=Stretch('wheel_'+robot_arm), human=Human(human_controllable_joint_indices, controllable=True))
register_env('assistive_gym:DressingStretchHuman-v1', lambda config: DressingStretchHumanEnv())

class DressingPandaHumanEnv(DressingEnv, MultiAgentEnv):
    def __init__(self):
        super(DressingPandaHumanEnv, self).__init__(robot=Panda(robot_arm), human=Human(human_controllable_joint_indices, controllable=True))
register_env('assistive_gym:DressingPandaHuman-v1', lambda config: DressingPandaHumanEnv())

