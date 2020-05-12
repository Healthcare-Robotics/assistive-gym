from .feeding import FeedingEnv
from .agents import pr2, baxter, sawyer, jaco
from .agents.pr2 import PR2
from .agents.baxter import Baxter
from .agents.sawyer import Sawyer
from .agents.jaco import Jaco

robot_arm = 'right'
class FeedingPR2Env(FeedingEnv):
    def __init__(self):
        super(FeedingPR2Env, self).__init__(robot=PR2(robot_arm), human_control=False)

class FeedingBaxterEnv(FeedingEnv):
    def __init__(self):
        super(FeedingBaxterEnv, self).__init__(robot=Baxter(robot_arm), human_control=False)

class FeedingSawyerEnv(FeedingEnv):
    def __init__(self):
        super(FeedingSawyerEnv, self).__init__(robot=Sawyer(robot_arm), human_control=False)

class FeedingJacoEnv(FeedingEnv):
    def __init__(self):
        super(FeedingJacoEnv, self).__init__(robot=Jaco(robot_arm), human_control=False)

class FeedingPR2HumanEnv(FeedingEnv):
    def __init__(self):
        super(FeedingPR2HumanEnv, self).__init__(robot=PR2(robot_arm), human_control=True)

class FeedingBaxterHumanEnv(FeedingEnv):
    def __init__(self):
        super(FeedingBaxterHumanEnv, self).__init__(robot=Baxter(robot_arm), human_control=True)

class FeedingSawyerHumanEnv(FeedingEnv):
    def __init__(self):
        super(FeedingSawyerHumanEnv, self).__init__(robot=Sawyer(robot_arm), human_control=True)

class FeedingJacoHumanEnv(FeedingEnv):
    def __init__(self):
        super(FeedingJacoHumanEnv, self).__init__(robot=Jaco(robot_arm), human_control=True)

