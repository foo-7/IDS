import torch
import torch.nn as nn
import torch.nn.functional as F

class OldNet(nn.Module):

    def __init__(self):
        super(OldNet, self).__init__()
        self.convLayer1 = nn.Conv1d(1, 16, kernel_size=3)
        self.convLayer2 = nn.Conv1d(16, 32, kernel_size=3)
        self.conv2_drop = nn.Dropout1d(0.2)
        self.fc1 = None # will be set in forward pass
        self.fc2 = nn.Linear(128, 1)
        
    def forward(self, x):
        x = x.unsqueeze(1)
        x = F.relu(F.max_pool1d(self.convLayer1(x), 2))
        x = F.relu(F.max_pool1d(self.conv2_drop(self.convLayer2(x)), 2))

        # Unconventional, but I don't know what input to have for the first dense layer.
        fc1_input_size = x.view(x.size(0), -1).size(1)
        self.fc1 = nn.Linear(fc1_input_size, 128)
        self.fc1 = self.fc1.to(x.device)

        # view() flattens your tensor from multi-dimensional to 2D
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))


        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return torch.sigmoid(x)