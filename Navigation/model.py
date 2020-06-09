import torch
import torch.nn as nn
import torch.nn.functional as F

class network(nn.Module):

    def __init__(self,state_size,action_size,h1 = 64,h2 = 64):
        super(network,self).__init__()
        self.fc1 = nn.Linear(state_size,h1)
        self.fc2 = nn.Linear(h1,h2)
        self.fc3 = nn.Linear(h2,action_size)
    
    def forward(self,state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x