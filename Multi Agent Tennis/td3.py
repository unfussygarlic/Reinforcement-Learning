from models import TD3Policy,Critic
from ExperienceReplay import ReplayBuffer
import torch
import torch.optim as opt
import torch.nn.functional as F
import numpy as np

class agent(object):

    def __init__(self,state_size,action_size,num_agents,lr,seed):
        self.actor = TD3Policy(state_size,action_size,seed)
        self.actor_opt = opt.Adam(self.actor.parameters(),lr = lr)
        self.actor_target = TD3Policy(state_size,action_size,seed)

        self.critic = Critic(state_size,action_size,seed)
        self.critic_opt = opt.Adam(self.critic.parameters(),lr = lr)
        self.critic_target = Critic(state_size,action_size,seed)

        self.num_agents = num_agents
        self.policy_update = 2
        self.step = 0
        self.noise_clip = 0.5
        self.policy_noise = 0.2
        self.gamma = 0.998
        self.TAU = 0.005
        self.batch_size = 64

        self.memory = ReplayBuffer(int(1e6),self.batch_size)

        self.hard_update(self.critic,self.critic_target)
    
    def act(self,state,add_noise = True):
        state = torch.from_numpy(state).float()
        self.actor.eval()
        with torch.no_grad():
            actions =   self.actor(state).cpu().data.numpy()
        self.actor.train()
        return actions
    
    def update(self,experience):
        self.memory.add(experience)
        if len(self.memory) > self.batch_size:
            experiences = self.memory.sample()
            self.train(experiences)    
    
    def train(self,observations):
        states,actions,rewards,next_states,dones = observations
        self.step = (self.step+1) % self.policy_update

        with torch.no_grad():
            noise = (torch.randn_like(actions)*self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_actions = (self.actor_target(next_states) + noise).clamp(-1,1)
            Q1_target,Q2_target = self.critic_target(next_states,next_actions)
            Q_target = torch.min(Q1_target,Q2_target)
            Q_target = rewards + (self.gamma*Q_target*(1-dones))
        
        Q1_expected,Q2_expected = self.critic(states,actions)

        critic_loss = F.mse_loss(Q1_expected,Q_target) + F.mse_loss(Q2_expected,Q_target)

        self.critic_opt.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1)
        self.critic_opt.step()

        if self.step == 0:
            expected_actions = self.actor(states)
            actor_loss = -self.critic.Q1(states,expected_actions).mean()

            self.actor_opt.zero_grad()
            actor_loss.backward()
            self.actor_opt.step()

            self.soft_update(self.actor,self.actor_target)
            self.soft_update(self.critic,self.critic_target)
    
    def soft_update(self,local,target):
        for l,t in zip(local.parameters(),target.parameters()):
            t.data.copy_(self.TAU*l.data + (1-self.TAU)*t.data)
    
    def hard_update(self,local,target):
        for l,t in zip(local.parameters(),target.parameters()):
            t.data.copy_(l.data)