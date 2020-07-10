from .feeding import FeedingEnv
from .agents import pr2, baxter, sawyer, jaco, human
from .agents.pr2 import PR2
from .agents.baxter import Baxter
from .agents.sawyer import Sawyer
from .agents.jaco import Jaco
from .agents.human import Human

robot_arm = 'right'
human_controllable_joint_indices = human.head_joints
class FeedingPR2Env(FeedingEnv):
    def __init__(self):
        super(FeedingPR2Env, self).__init__(robot=PR2(robot_arm), human=Human(human_controllable_joint_indices, controllable=False))

class FeedingBaxterEnv(FeedingEnv):
    def __init__(self):
        super(FeedingBaxterEnv, self).__init__(robot=Baxter(robot_arm), human=Human(human_controllable_joint_indices, controllable=False))

class FeedingSawyerEnv(FeedingEnv):
    def __init__(self):
        super(FeedingSawyerEnv, self).__init__(robot=Sawyer(robot_arm), human=Human(human_controllable_joint_indices, controllable=False))

class FeedingJacoEnv(FeedingEnv):
    def __init__(self):
        super(FeedingJacoEnv, self).__init__(robot=Jaco(robot_arm), human=Human(human_controllable_joint_indices, controllable=False))

class FeedingPR2HumanEnv(FeedingEnv):
    def __init__(self):
        super(FeedingPR2HumanEnv, self).__init__(robot=PR2(robot_arm), human=Human(human_controllable_joint_indices, controllable=True))

class FeedingBaxterHumanEnv(FeedingEnv):
    def __init__(self):
        super(FeedingBaxterHumanEnv, self).__init__(robot=Baxter(robot_arm), human=Human(human_controllable_joint_indices, controllable=True))

class FeedingSawyerHumanEnv(FeedingEnv):
    def __init__(self):
        super(FeedingSawyerHumanEnv, self).__init__(robot=Sawyer(robot_arm), human=Human(human_controllable_joint_indices, controllable=True))

class FeedingJacoHumanEnv(FeedingEnv):
    def __init__(self):
        super(FeedingJacoHumanEnv, self).__init__(robot=Jaco(robot_arm), human=Human(human_controllable_joint_indices, controllable=True))

