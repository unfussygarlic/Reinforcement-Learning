from models import SACPolicy,Critic
from ExperienceReplay import ReplayBuffer
import torch
import torch.optim as opt
import torch.nn.functional as F
import numpy as np

class agent(object):

    def __init__(self,state_size,action_size,num_agents,lr,seed):
        """ 
        Soft Actor Critic Agent.

        Arguments:
        state_size : Size of the state from environment.
        action_size : Size of the action taken by the agent
        num_agents : Total number of agents
        lr : Common learning rate for both agents
        seed : Seed value for reproducibility
        """
        self.num_agents = num_agents

        #Initializing Policy network with Adam optimizer
        #SAC doesn't use target Policy network
        self.actor = SACPolicy(state_size,action_size,seed)
        self.actor_opt = opt.Adam(self.actor.parameters(),lr = lr)

        #Initializing local and target Critic networks with Adam optimizer
        self.critic = Critic(state_size,action_size,seed)
        self.critic_target = Critic(state_size,action_size,seed)
        self.critic_opt = opt.Adam(self.critic.parameters(),lr = lr)

        """
        Regularization parameter alpha.
        Alpha is automatically optimized using Adam optimizer.
        """
        self.alpha = 0.2
        self.target_entropy = -4
        self.log_alpha = torch.zeros(1,requires_grad = True)
        self.alpha_opt = opt.Adam([self.log_alpha],lr = lr)

        #Agent hyper parameters
        self.step = 0
        self.policy_update = 2
        self.gamma = 0.99
        self.TAU = 0.005
        self.batch_size = 64

        #Initializing buffer for 'COMPETE' mode
        self.memory = ReplayBuffer(int(1e6),self.batch_size)

        #Weights of local network are copied to the target network
        self.hard_update(self.critic,self.critic_target)
    
    def act(self,state):
        """
        Returns action for the given state

        Argument:
        state : State vector containing state variables
        """
        state = torch.from_numpy(state).float()
        action = self.actor.get_action(state)
        return action.cpu().detach().numpy()
    
    def update(self,experience):
        """
        Updates the agent's replay buffer and trains the agent
        in 'COMPETE' mode.

        Arguments:
        experience : Tuple containing current experience
        """
        self.memory.add(experience)
        if len(self.memory) > self.batch_size:
            experiences = self.memory.sample()
            self.train(experiences)   
        
    def train(self,experiences):
        """
        Trains the agent using the experiences

        Arguments:
        experience : Tuple containing current experience
        """
        #Step is updated at each timestep for policy updation
        self.step = (self.step+1)%self.policy_update

        states,actions,rewards,next_states,dones = experiences

        """
        Actions and log probs are sampled from the network
        and used for obtaining clipped Q values from target critic network.
        Log probs introduces entropy to the Q_target encouraging
        exploration for the agent.
        """
        with torch.no_grad():
            next_actions,next_log_probs = self.actor.sample(next_states)
            Q1_tar,Q2_tar = self.critic_target(next_states,next_actions)
            Q_target = torch.min(Q1_tar,Q2_tar) - (self.alpha*next_log_probs)
            Q_target = rewards + (self.gamma*Q_target*(1-dones))

        #Current Q values using local critic network
        Q1_ex,Q2_ex = self.critic(states,actions)

        #Critic loss using both the Q values and Updation of local critic network
        critic_loss = F.mse_loss(Q1_ex,Q_target) + F.mse_loss(Q2_ex,Q_target)
        self.critic_opt.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(),1)
        self.critic_opt.step()

        #Current actions and log probs obtained using current states
        new_actions,new_log_probs = self.actor.sample(states)

        #Alpha optimizer is updated using current log probs
        alpha_loss = (self.log_alpha * (-new_log_probs - self.target_entropy).detach()).mean()    
        self.alpha_opt.zero_grad()
        alpha_loss.backward()
        self.alpha_opt.step()

        if self.step == 0:
            """ 
            Policy network updation.
            Q values obtained using current state and action values
            are subtracted by parametrized entropy for exploration.
            """
            Q1,Q2 = self.critic(states,new_actions)
            Q = torch.min(Q1,Q2)

            actor_loss = (self.alpha*new_log_probs - Q).mean()
            self.actor_opt.zero_grad()
            actor_loss.backward()
            self.actor_opt.step()

            #Target network are updated using TAU constant
            self.soft_update(self.critic,self.critic_target)
        
        #Alpha is updated for each timestep
        self.alpha = self.log_alpha.exp()

    def soft_update(self,local,target):
        """ 
        Soft update of target network parameters using TAU
        """
        for l,t in zip(local.parameters(),target.parameters()):
            t.data.copy_(self.TAU*l.data + (1-self.TAU)*t.data)
    
    def hard_update(self,local,target):
        """
        Hard update which copies local network parameters to target network
        """
        for l,t in zip(local.parameters(),target.parameters()):
            t.data.copy_(l.data)