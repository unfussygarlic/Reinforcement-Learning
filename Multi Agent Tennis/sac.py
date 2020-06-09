from models import SACPolicy,Critic
from ExperienceReplay import ReplayBuffer
import torch
import torch.optim as opt
import torch.nn.functional as F
import numpy as np

class agent(object):

    def __init__(self,state_size,action_size,num_agents,lr,seed):
        self.num_agents = num_agents

        self.actor = SACPolicy(state_size,action_size,seed)
        self.actor_opt = opt.Adam(self.actor.parameters(),lr = lr)

        self.critic = Critic(state_size,action_size,seed)
        self.critic_target = Critic(state_size,action_size,seed)
        self.critic_opt = opt.Adam(self.critic.parameters(),lr = lr)

        self.alpha = 0.2
        self.target_entropy = -4
        self.log_alpha = torch.zeros(1,requires_grad = True)
        self.alpha_opt = opt.Adam([self.log_alpha],lr = lr)

        self.step = 0
        self.policy_update = 2
        self.gamma = 0.99
        self.TAU = 0.005
        self.batch_size = 64

        self.memory = ReplayBuffer(int(1e6),self.batch_size)

        self.hard_update(self.critic,self.critic_target)
    
    def act(self,state):
        state = torch.from_numpy(state).float()
        action = self.actor.get_action(state)
        return action.cpu().detach().numpy()
    
    def update(self,experience):
        self.memory.add(experience)
        if len(self.memory) > self.batch_size:
            experiences = self.memory.sample()
            self.train(experiences)   
        
    def train(self,experiences):
        self.step = (self.step+1)%self.policy_update
        states,actions,rewards,next_states,dones = experiences

        with torch.no_grad():
            next_actions,next_log_probs = self.actor.sample(next_states)
            Q1_tar,Q2_tar = self.critic_target(next_states,next_actions)
            Q_target = torch.min(Q1_tar,Q2_tar) - (self.alpha*next_log_probs)
            Q_target = rewards + (self.gamma*Q_target*(1-dones))

        Q1_ex,Q2_ex = self.critic(states,actions)

        critic_loss = F.mse_loss(Q1_ex,Q_target) + F.mse_loss(Q2_ex,Q_target)
        self.critic_opt.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(),1)
        self.critic_opt.step()

        new_actions,new_log_probs = self.actor.sample(states)

        alpha_loss = (self.log_alpha * (-new_log_probs - self.target_entropy).detach()).mean()    
        self.alpha_opt.zero_grad()
        alpha_loss.backward()
        self.alpha_opt.step()

        if self.step == 0:
            Q1,Q2 = self.critic(states,new_actions)
            Q = torch.min(Q1,Q2)

            actor_loss = (self.alpha*new_log_probs - Q).mean()
            self.actor_opt.zero_grad()
            actor_loss.backward()
            self.actor_opt.step()

            self.soft_update(self.critic,self.critic_target)
    
        self.alpha = self.log_alpha.exp()

    def soft_update(self,local,target):
        for l,t in zip(local.parameters(),target.parameters()):
            t.data.copy_(self.TAU*l.data + (1-self.TAU)*t.data)
    
    def hard_update(self,local,target):
        for l,t in zip(local.parameters(),target.parameters()):
            t.data.copy_(l.data)