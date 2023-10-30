import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import json
import os
import numpy as np

class MyNet(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        """
        In the constructor, we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(MyNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)  # First layer
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)  # Second layer
        self.fc3 = nn.Linear(hidden_size2, output_size)  # Ouput layer

    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


