from .drinking import DrinkingEnv
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

robot_arm = 'right'
human_controllable_joint_indices = human.head_joints
class DrinkingPR2Env(DrinkingEnv):
    def __init__(self):
        super(DrinkingPR2Env, self).__init__(robot=PR2(robot_arm), human=Human(human_controllable_joint_indices, controllable=False))

class DrinkingBaxterEnv(DrinkingEnv):
    def __init__(self):
        super(DrinkingBaxterEnv, self).__init__(robot=Baxter(robot_arm), human=Human(human_controllable_joint_indices, controllable=False))

class DrinkingSawyerEnv(DrinkingEnv):
    def __init__(self):
        super(DrinkingSawyerEnv, self).__init__(robot=Sawyer(robot_arm), human=Human(human_controllable_joint_indices, controllable=False))

class DrinkingJacoEnv(DrinkingEnv):
    def __init__(self):
        super(DrinkingJacoEnv, self).__init__(robot=Jaco(robot_arm), human=Human(human_controllable_joint_indices, controllable=False))

class DrinkingStretchEnv(DrinkingEnv):
    def __init__(self):
        super(DrinkingStretchEnv, self).__init__(robot=Stretch('wheel_'+robot_arm), human=Human(human_controllable_joint_indices, controllable=False))

class DrinkingPandaEnv(DrinkingEnv):
    def __init__(self):
        super(DrinkingPandaEnv, self).__init__(robot=Panda(robot_arm), human=Human(human_controllable_joint_indices, controllable=False))

class DrinkingPR2HumanEnv(DrinkingEnv, MultiAgentEnv):
    def __init__(self):
        super(DrinkingPR2HumanEnv, self).__init__(robot=PR2(robot_arm), human=Human(human_controllable_joint_indices, controllable=True))
register_env('assistive_gym:DrinkingPR2Human-v1', lambda config: DrinkingPR2HumanEnv())

class DrinkingBaxterHumanEnv(DrinkingEnv, MultiAgentEnv):
    def __init__(self):
        super(DrinkingBaxterHumanEnv, self).__init__(robot=Baxter(robot_arm), human=Human(human_controllable_joint_indices, controllable=True))
register_env('assistive_gym:DrinkingBaxterHuman-v1', lambda config: DrinkingBaxterHumanEnv())

class DrinkingSawyerHumanEnv(DrinkingEnv, MultiAgentEnv):
    def __init__(self):
        super(DrinkingSawyerHumanEnv, self).__init__(robot=Sawyer(robot_arm), human=Human(human_controllable_joint_indices, controllable=True))
register_env('assistive_gym:DrinkingSawyerHuman-v1', lambda config: DrinkingSawyerHumanEnv())

class DrinkingJacoHumanEnv(DrinkingEnv, MultiAgentEnv):
    def __init__(self):
        super(DrinkingJacoHumanEnv, self).__init__(robot=Jaco(robot_arm), human=Human(human_controllable_joint_indices, controllable=True))
register_env('assistive_gym:DrinkingJacoHuman-v1', lambda config: DrinkingJacoHumanEnv())

class DrinkingStretchHumanEnv(DrinkingEnv, MultiAgentEnv):
    def __init__(self):
        super(DrinkingStretchHumanEnv, self).__init__(robot=Stretch('wheel_'+robot_arm), human=Human(human_controllable_joint_indices, controllable=True))
register_env('assistive_gym:DrinkingStretchHuman-v1', lambda config: DrinkingStretchHumanEnv())

class DrinkingPandaHumanEnv(DrinkingEnv, MultiAgentEnv):
    def __init__(self):
        super(DrinkingPandaHumanEnv, self).__init__(robot=Panda(robot_arm), human=Human(human_controllable_joint_indices, controllable=True))
register_env('assistive_gym:DrinkingPandaHuman-v1', lambda config: DrinkingPandaHumanEnv())

