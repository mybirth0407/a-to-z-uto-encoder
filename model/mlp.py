import torch
from torch import nn

class MLP(nn.Module):
    """Some Information about MLP"""
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        if type(self.hidden_size) == int:
            self.hidden_size = [self.hidden_size]
        elif type(self.hidden_size) == list:
            self.hidden_size = self.hidden_size

        self.output_size = output_size

        self.fc1 = nn.Linear(self.input_size, self.hidden_size)
        self.fc2 = nn.Linear(self.input_size, self.hidden_size)
        self.fc3 = nn.Linear(self.input_size, self.hidden_size)

    def forward(self, x):

        return x