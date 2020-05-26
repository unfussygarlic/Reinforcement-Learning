from model import network
import torch
import torch.optim as opt
from PrioritizedReplay import ReplayBuffer
import numpy as np
import random

class DQAgent:

    def __init__(self,state_size,action_size,lr = 5e-4):
        self.action_size = action_size
        self.local_network = network(state_size,action_size)
        self.target_network = network(state_size,action_size)
        self.optimizer = opt.Adam(self.local_network.parameters(),lr = lr)
        
        self.memory = ReplayBuffer()
        self.eps = 1.0
        self.decay = 0.99
        self.eps_min = 0.01
        self.update_every = 4
        self.t_step = 1
        self.gamma = 0.99
        self.TAU = 1e-3

    def act(self,state):
        state = torch.from_numpy(state).float().unsqueeze(0)

        self.local_network.eval()
        with torch.no_grad():
            actions = self.local_network(state)
        self.local_network.train()

        greedy_action = np.argmax(actions.data.numpy())
        random_action = random.choice(np.arange(self.action_size))

        action = random_action if random.random() < self.eps else greedy_action
        return action
    
    def step(self,experience):
        self.memory.add(experience)
        self.eps = max(self.eps*self.decay,self.eps_min) if experience[-1] else self.eps
        self.t_step = (self.t_step+1) % self.update_every
        
        if self.t_step == 0:
            experiences,weights,indices = self.memory.sample()
            self.learn(experiences,weights,indices)
    
    def learn(self,experiences,weights,indices):
        state,action,reward,next_state,done = experiences

        self.target_network.eval()
        q_expected = self.local_network(state).gather(1,action)

        self.local_network.eval()
        with torch.no_grad():
            local_action = self.local_network(next_state).max(1)[1].unsqueeze(1).long()
            q_update = self.target_network(next_state).gather(1,local_action)
        self.local_network.train()
        self.target_network.train()

        beta = 0 if self.eps <= 0.01 else self.eps
        weights = weights**(1-beta)
        q_target = reward + (self.gamma*q_update*(1-done))
        squared_loss = ((q_expected - q_target)**2) * weights
        ms_loss = torch.mean(squared_loss)

        self.optimizer.zero_grad()
        ms_loss.backward()
        self.optimizer.step()

        self.memory.update_priority(indices,squared_loss.cpu().data.numpy())
        self.soft_update()
    
    def soft_update(self):
        for l,t in zip(self.local_network.parameters(),self.target_network.parameters()):
            t.data.copy_(self.TAU*l.data + (1.0-self.TAU)*t.data)