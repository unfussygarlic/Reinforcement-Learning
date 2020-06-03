import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class Actor(nn.Module):

    def __init__(self,state_size,action_size,h1 = 256,h2 = 128):
        super(Actor,self).__init__()
        self.fc1 = nn.Linear(state_size,h1)
        self.fc2 = nn.Linear(h1,h2)
        self.fc3 = nn.Linear(h2,action_size)
        self.weight_init()
    
    def weight_init(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3.0e-3,3.0e-3)
    
    def forward(self,state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        return x

class Critic(nn.Module):

    def __init__(self,state_size,action_size,h1 = 256,h2 = 128):
        super(Critic,self).__init__()
        self.fc1 = nn.Linear(state_size,h1)
        self.fc2 = nn.Linear(h1+action_size,h2)
        self.fc3 = nn.Linear(h2,1)
        self.weight_init()
    
    def weight_init(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3.0e-3,3.0e-3)
    
    def forward(self,state,action):
        x_state = F.relu(self.fc1(state))
        concat = torch.cat((x_state,action),dim = 1)
        x = F.relu(self.fc2(concat))
        x = self.fc3(x)
        return x