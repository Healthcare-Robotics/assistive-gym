from .scratch_itch import ScratchItchEnv
from .scratch_itch_mesh import ScratchItchMeshEnv
from .agents import pr2, baxter, sawyer, jaco, human, human_mesh
from .agents.pr2 import PR2
from .agents.baxter import Baxter
from .agents.sawyer import Sawyer
from .agents.jaco import Jaco
from .agents.stretch import Stretch
from .agents.panda import Panda
from .agents.human import Human
from .agents.human_mesh import HumanMesh

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

class ScratchItchPandaEnv(ScratchItchEnv):
    def __init__(self):
        super(ScratchItchPandaEnv, self).__init__(robot=Panda(robot_arm), human=Human(human_controllable_joint_indices, controllable=False))

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

class ScratchItchPandaHumanEnv(ScratchItchEnv):
    def __init__(self):
        super(ScratchItchPandaHumanEnv, self).__init__(robot=Panda(robot_arm), human=Human(human_controllable_joint_indices, controllable=True))

class ScratchItchPR2MeshEnv(ScratchItchMeshEnv):
    def __init__(self):
        super(ScratchItchPR2MeshEnv, self).__init__(robot=PR2(robot_arm), human=HumanMesh())

class ScratchItchBaxterMeshEnv(ScratchItchMeshEnv):
    def __init__(self):
        super(ScratchItchBaxterMeshEnv, self).__init__(robot=Baxter(robot_arm), human=HumanMesh())

class ScratchItchSawyerMeshEnv(ScratchItchMeshEnv):
    def __init__(self):
        super(ScratchItchSawyerMeshEnv, self).__init__(robot=Sawyer(robot_arm), human=HumanMesh())

class ScratchItchJacoMeshEnv(ScratchItchMeshEnv):
    def __init__(self):
        super(ScratchItchJacoMeshEnv, self).__init__(robot=Jaco(robot_arm), human=HumanMesh())

class ScratchItchStretchMeshEnv(ScratchItchMeshEnv):
    def __init__(self):
        super(ScratchItchStretchMeshEnv, self).__init__(robot=Stretch('wheel_'+robot_arm), human=HumanMesh())

class ScratchItchPandaMeshEnv(ScratchItchMeshEnv):
    def __init__(self):
        super(ScratchItchPandaMeshEnv, self).__init__(robot=Panda(robot_arm), human=HumanMesh())

