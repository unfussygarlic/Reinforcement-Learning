from model import Actor,Critic
from ExperienceReplay import ReplayBuffer
import torch
import torch.optim as opt
import torch.nn.functional as F
import numpy as np
import random
import copy

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class ddpgagent:

    def __init__(self,state_size,action_size,num_agents,buffer_size = int(1e6),batch_size = 64,seed = 42):
        self.memory = ReplayBuffer(buffer_size,batch_size)

        self.local_actor = Actor(state_size,action_size).to(device)
        self.target_actor = Actor(state_size,action_size).to(device)

        self.local_critic = Critic(state_size,action_size).to(device)
        self.target_critic = Critic(state_size,action_size)

        self.actor_opt = opt.Adam(self.local_actor.parameters(),lr = 1e-4,weight_decay = 0.0)
        self.critic_opt = opt.Adam(self.local_critic.parameters(),lr = 1e-4,weight_decay = 0.0)

        self.noise = OUNoise((num_agents,action_size),seed)

        self.gamma = 0.99
        self.TAU = 0.001
        self.learn_every = 10
        self.t = 0
        self.num_agents = num_agents
        self.batch_size = batch_size
        self.noise_value = 0
        self.train_steps = 20
    
    def act(self,state,eps,add_noise = True):
        state = torch.from_numpy(state).float()
        self.local_actor.eval()
        with torch.no_grad():
            action = self.local_actor(state).cpu().data.numpy()
        self.local_actor.train()
        self.action_type = 'GREEDY'
        if add_noise and (np.random.random() < eps):
            self.action_type = 'RANDOM'
            self.noise_value = self.noise.sample()
            action = action + self.noise_value
        return np.clip(action,-1,1)
    
    def step(self,experience):
        state,action,reward,next_state,done = experience
        for i in range(self.num_agents):
            self.memory.add((state[i],action[i],reward[i],next_state[i],done[i]))
        self.t = (self.t + 1) % self.learn_every
        if self.t == 0:
            for _ in range(self.train_steps):
                experiences = self.memory.sample()
                self.learn(experiences)
    
    def learn(self,experience):
        state,action,reward,next_state,done = experience

        next_action = self.target_actor(next_state)
        Q_next = self.target_critic(next_state,next_action)
        Q_target = reward + (self.gamma * Q_next * (1-done))
        Q_expected = self.local_critic(state,action)
        critic_loss = F.mse_loss(Q_expected,Q_target)
        self.critic_opt.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm(self.local_critic.parameters(), 1)
        self.critic_opt.step()

        expected_action = self.local_actor(state)
        actor_loss = -self.local_critic(state,expected_action).mean()
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        self.soft_update(self.local_critic,self.target_critic)
        self.soft_update(self.local_actor,self.target_actor)
    
    def soft_update(self,local,target):
        for l,t in zip(local.parameters(),target.parameters()):
            t.data.copy_(self.TAU * l.data + (1 - self.TAU) * t.data)
    
    def reset(self):
        self.noise.reset()

class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.size = size
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.standard_normal(self.size)
        self.state = x + dx
        return self.state
