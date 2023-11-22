import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import json
import os
import numpy as np

class MyNet(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, hidden_size3, output_size):
        """
        2 hidden layers
        """
        super(MyNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)  # First layer
        # self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)  # Second layer
        self.fc3 = nn.Linear(hidden_size2, hidden_size3)  # Third layer
        self.fc4 = nn.Linear(hidden_size3, output_size)  # Ouput layer

    def forward(self,x):
        # TODO: add dropout & regularization
        x = F.relu(self.fc1(x))
        # x = self.dropout(x)
        x = F.relu(self.fc2(x))

        # x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x


