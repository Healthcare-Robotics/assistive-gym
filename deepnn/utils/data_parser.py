import torch


class ModelInput:
    def __init__(self, pose, betas):
        self.pose = pose  # len 72
        self.betas = betas  # len 10
        assert len(pose) == 72, "pose should be len 72"
        assert len(betas) == 10, "betas should be len 10"

    def to_tensor(self):
        data = self.pose + self.betas
        return torch.tensor(data, dtype=torch.float)

    @classmethod
    def from_tensor(cls, tensor: torch.Tensor):
        # expect len tensor = 72 + 10
        tensor = tensor.tolist()
        pose = tensor[:72]
        betas = tensor[72:]
        return cls(pose, betas)


class ModelOutput:
    def __init__(self, human_joint_angles, robot_base_pos, robot_base_orient, robot_joint_angles):
        self.human_joint_angles = human_joint_angles  # len 15
        self.robot_base_pos = robot_base_pos  # len 3
        self.robot_base_orient = robot_base_orient  # len 4
        self.robot_joint_angles = robot_joint_angles  # len 10
        assert len(human_joint_angles) == 15, "human_joint_angles should be len 15"
        assert len(robot_base_pos) == 3, "robot_base_pos should be len 3"
        assert len(robot_base_orient) == 4, "robot_base_orient should be len 4"
        assert len(robot_joint_angles) == 10, "robot_joint_angles should be len 10"

    def to_tensor(self):
        data = self.human_joint_angles + self.robot_base_pos + self.robot_base_orient + self.robot_joint_angles
        # data = self.human_joint_angles
        return torch.tensor(data, dtype=torch.float)

    @classmethod
    def from_tensor(cls, tensor: torch.Tensor):
        # expect len tensor = 15 + 3 + 4 + 10
        # convert tensor to list
        tensor = tensor.tolist()
        human_joint_angles = tensor[:15]
        robot_base_pos = tensor[15:18]
        robot_base_orient = tensor[18:22]
        robot_joint_angles = tensor[22:]
        return cls(human_joint_angles, robot_base_pos, robot_base_orient, robot_joint_angles)

    def convert_to_dict(self):
        return {
            'human': {
                'joint_angles': self.human_joint_angles,
            },
            'robot': {
                'base': [self.robot_base_pos, self.robot_base_orient],
                'joint_angles': self.robot_joint_angles
            }
        }
