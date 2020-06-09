import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np


def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class Critic(nn.Module):

    def __init__(self,state_size,action_size,seed,h1 = 128,h2 = 128):
        super(Critic,self).__init__()
        self.seed = torch.manual_seed(seed)

        self.fc1 = nn.Linear(state_size,h1)
        self.fc2 = nn.Linear(h1 + action_size,h2)
        self.fc3 = nn.Linear(h2,1)

        self.fc4 = nn.Linear(state_size,h1)
        self.fc5 = nn.Linear(h1 + action_size,h2)
        self.fc6 = nn.Linear(h2,1)

        self.reset_weights()
    
    def reset_weights(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

        self.fc4.weight.data.uniform_(*hidden_init(self.fc4))
        self.fc5.weight.data.uniform_(*hidden_init(self.fc5))
        self.fc6.weight.data.uniform_(-3e-3, 3e-3)      

    def forward(self,state,action):
        x = F.relu(self.fc1(state))
        x = torch.cat((x,action),dim = 1)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        y = F.relu(self.fc4(state))
        y = torch.cat((y,action),dim = 1)
        y = F.relu(self.fc5(y))
        y = self.fc6(y)

        return x,y
    
    def Q1(self,state,action):
        x = F.relu(self.fc1(state))
        x = torch.cat((x,action),dim = 1)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

class TD3Policy(nn.Module):

    def __init__(self,state_size,action_size,seed,h1 = 128,h2 = 128):
        super(TD3Policy,self).__init__()
        self.seed = torch.manual_seed(seed)

        self.fc1 = nn.Linear(state_size,h1)
        self.fc2 = nn.Linear(h1,h2)
        self.fc3 = nn.Linear(h2,action_size)

        self.reset_weights()
    
    def reset_weights(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)
    
    def forward(self,state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        return x

class SACPolicy(nn.Module):

    def __init__(self,state_size,action_size,seed,h1 = 128,h2 = 128):
        super(SACPolicy,self).__init__()

        self.sig_max = 2
        self.sig_min = -20

        self.fc1 = nn.Linear(state_size,h1)
        self.fc2 = nn.Linear(h1,h2)
        self.mean = nn.Linear(h2,action_size)
        self.std = nn.Linear(h2,action_size)
        self.reset_weights()
    
    def reset_weights(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.mean.weight.data.uniform_(-3e-3, 3e-3)
        self.std.weight.data.uniform_(-3e-3,3e-3)
    
    def forward(self,state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        mean = self.mean(x)
        log_std = self.std(x)
        log_std = torch.clamp(log_std,min = self.sig_min,max = self.sig_max)

        return mean,log_std
    
    def sample(self,state):
        epsilon = 1e-6
        mu,log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mu,std)

        z = normal.rsample()
        action = torch.tanh(z)

        log_prob = (normal.log_prob(z) - torch.log(1 - (torch.tanh(z)).pow(2) + epsilon))
        log_prob = log_prob.sum(1,keepdims = True)

        return action,log_prob
    
    def get_action(self,state):
        mu,log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mu,std)
        z = normal.sample()
        action = torch.tanh(z)
        return action
