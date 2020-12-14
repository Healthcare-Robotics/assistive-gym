from .arm_manipulation import ArmManipulationEnv
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

robot_arm = 'both'
human_controllable_joint_indices = human.right_arm_joints
class ArmManipulationPR2Env(ArmManipulationEnv):
    def __init__(self):
        super(ArmManipulationPR2Env, self).__init__(robot=PR2(robot_arm), human=Human(human_controllable_joint_indices, controllable=False))

class ArmManipulationBaxterEnv(ArmManipulationEnv):
    def __init__(self):
        super(ArmManipulationBaxterEnv, self).__init__(robot=Baxter(robot_arm), human=Human(human_controllable_joint_indices, controllable=False))

class ArmManipulationSawyerEnv(ArmManipulationEnv):
    def __init__(self):
        super(ArmManipulationSawyerEnv, self).__init__(robot=Sawyer(robot_arm), human=Human(human_controllable_joint_indices, controllable=False))

class ArmManipulationJacoEnv(ArmManipulationEnv):
    def __init__(self):
        super(ArmManipulationJacoEnv, self).__init__(robot=Jaco(robot_arm), human=Human(human_controllable_joint_indices, controllable=False))

class ArmManipulationStretchEnv(ArmManipulationEnv):
    def __init__(self):
        super(ArmManipulationStretchEnv, self).__init__(robot=Stretch('wheel_'+robot_arm), human=Human(human_controllable_joint_indices, controllable=False))

class ArmManipulationPandaEnv(ArmManipulationEnv):
    def __init__(self):
        super(ArmManipulationPandaEnv, self).__init__(robot=Panda(robot_arm), human=Human(human_controllable_joint_indices, controllable=False))

class ArmManipulationPR2HumanEnv(ArmManipulationEnv, MultiAgentEnv):
    def __init__(self):
        super(ArmManipulationPR2HumanEnv, self).__init__(robot=PR2(robot_arm), human=Human(human_controllable_joint_indices, controllable=True))
register_env('assistive_gym:ArmManipulationPR2Human-v1', lambda config: ArmManipulationPR2HumanEnv())

class ArmManipulationBaxterHumanEnv(ArmManipulationEnv, MultiAgentEnv):
    def __init__(self):
        super(ArmManipulationBaxterHumanEnv, self).__init__(robot=Baxter(robot_arm), human=Human(human_controllable_joint_indices, controllable=True))
register_env('assistive_gym:ArmManipulationBaxterHuman-v1', lambda config: ArmManipulationBaxterHumanEnv())

class ArmManipulationSawyerHumanEnv(ArmManipulationEnv, MultiAgentEnv):
    def __init__(self):
        super(ArmManipulationSawyerHumanEnv, self).__init__(robot=Sawyer(robot_arm), human=Human(human_controllable_joint_indices, controllable=True))
register_env('assistive_gym:ArmManipulationSawyerHuman-v1', lambda config: ArmManipulationSawyerHumanEnv())

class ArmManipulationJacoHumanEnv(ArmManipulationEnv, MultiAgentEnv):
    def __init__(self):
        super(ArmManipulationJacoHumanEnv, self).__init__(robot=Jaco(robot_arm), human=Human(human_controllable_joint_indices, controllable=True))
register_env('assistive_gym:ArmManipulationJacoHuman-v1', lambda config: ArmManipulationJacoHumanEnv())

class ArmManipulationStretchHumanEnv(ArmManipulationEnv, MultiAgentEnv):
    def __init__(self):
        super(ArmManipulationStretchHumanEnv, self).__init__(robot=Stretch('wheel_'+robot_arm), human=Human(human_controllable_joint_indices, controllable=True))
register_env('assistive_gym:ArmManipulationStretchHuman-v1', lambda config: ArmManipulationStretchHumanEnv())

class ArmManipulationPandaHumanEnv(ArmManipulationEnv, MultiAgentEnv):
    def __init__(self):
        super(ArmManipulationPandaHumanEnv, self).__init__(robot=Panda(robot_arm), human=Human(human_controllable_joint_indices, controllable=True))
register_env('assistive_gym:ArmManipulationPandaHuman-v1', lambda config: ArmManipulationPandaHumanEnv())

